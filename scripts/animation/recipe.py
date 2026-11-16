"""Layer existing animation, bake contacts and carry rigid props in a GLB."""

import numpy as np
from scipy.spatial.transform import Rotation

from .scene import Scene, SLICES, blend_quats, decompose, matrices
from .solve import solve
from .articulation import arm_limits, limited, distribute_twist
from .floor import SurfaceFloor
from .surface import Surface
from .collision import capsule_setup
from transfer.motion import retarget, sample_motion
from roster.write_face_clips import write_clips


def local_transform(spec):
    return matrices(
        np.array(
            [
                [
                    *spec.get("translation", [0] * 3),
                    *spec.get("rotation", [0, 0, 0, 1]),
                    *spec.get("scale", [1] * 3),
                ]
            ]
        )
    )[0]


def attachment_chain(scene, joint, chain_root):
    chain = [scene.joint(joint)]
    root = scene.joint(chain_root)
    while chain[-1] != root:
        if chain[-1] not in scene.parents:
            raise ValueError(f"{root} is outside the attachment joint's ancestor chain")
        chain.append(scene.parents[chain[-1]])
    return chain


def grip_turn(world, prop, driver, followers, limit):
    pivot = world[driver, :3, 3]
    normal = np.eye(3) * 1e-9
    rhs = np.zeros(3)
    for follower, anchor in followers:
        arm = follower["arm_chain"]
        point = (prop @ anchor)[:3, 3]
        reach = np.linalg.norm(np.diff(world[arm, :3, 3], axis=0), axis=1).sum()
        ray = point - world[arm[-1], :3, 3]
        excess = max(0, 1 - reach / max(np.linalg.norm(ray), 1e-12))
        lever = point - pivot
        normal += (lever @ lever) * np.eye(3) - np.outer(lever, lever)
        rhs += np.cross(lever, -ray * excess)
    turn = np.linalg.solve(normal, rhs)
    return Rotation.from_rotvec(
        turn * min(1.0, limit / max(np.linalg.norm(turn), 1e-12))
    )


def build(model, recipe, base, scratch):
    from .props import add_props

    if recipe.get("schema_version") != "0.1":
        raise ValueError("Expected an animation recipe with schema_version 0.1")
    supported = {
        "schema_version",
        "asset",
        "name",
        "duration",
        "fps",
        "layers",
        "contacts",
        "solver",
        "props",
        "attachments",
        "surface_floor",
        "distribute_twist",
        "arm_articulations",
        "body_clearance",
        "temporal_correction",
        "rigid_regions",
        "hand_shapes",
    }
    extra = set(recipe) - supported
    if extra:
        raise ValueError(f"Unsupported recipe options: {sorted(extra)}")
    add_props(model, recipe.get("props", []), base)
    layers = []
    for index, layer in enumerate(recipe["layers"]):
        if "source" in layer:
            samples = scratch / f"layer-{index}.npz"
            clip = f"__transfer_{index}"
            config = dict(fps=recipe.get("fps", 30)) | layer.get("retarget", {})
            sample_motion(base / layer["source"], samples, config, layer["clip"])
            with np.load(samples) as take:
                retarget(take, model, clip, config)
        else:
            clip = layer["clip"]
        layers.append((layer, clip))
    scene = Scene(model)
    from .regions import prepare_regions, HandGuard

    solver = dict(recipe.get("solver", {}))
    if recipe.get("rigid_regions") or recipe.get("hand_shapes"):
        prepare_regions(model, recipe)
        scene = Scene(model)
    surface = (
        Surface(model)
        if any(
            recipe.get(key)
            for key in ("surface_floor", "body_clearance", "hand_shapes")
        )
        else None
    )
    floor = (
        SurfaceFloor(surface, recipe["surface_floor"])
        if recipe.get("surface_floor")
        else None
    )
    if recipe.get("body_clearance"):
        solver["collision"] = capsule_setup(surface, recipe["body_clearance"])
    articulations = list(solver.get("articulations", []))
    for arm in recipe.get("arm_articulations", []):
        articulations.extend(arm_limits(scene, arm))
    solver["articulations"] = articulations
    twists = [scene.joint(n) for n in recipe.get("distribute_twist", [])]
    hands = [HandGuard(surface, h) for h in recipe.get("hand_shapes", [])]
    temporal = recipe.get("temporal_correction")
    smoothed = [scene.joint(n) for n in temporal["joints"]] if temporal else []
    if temporal and temporal["decay_seconds"] <= 0:
        raise ValueError("Temporal correction needs a positive decay")
    prepared = []
    for layer, clip in layers:
        lo, hi = scene.interval(clip)
        lo, hi = layer.get("source_start", lo), layer.get("source_end", hi)
        mask = {scene.joint(j) for j in layer.get("joints", [])}
        speed, weight = layer.get("speed", 1.0), layer.get("weight", 1.0)
        if not hi > lo or speed <= 0 or not 0 <= weight <= 1:
            raise ValueError(f"Invalid layer range, speed or weight for {clip}")
        prepared.append((layer, clip, lo, hi, mask))
    attachments = []
    for a in recipe.get("attachments", []):
        node, driver = scene.joint(a["node"]), scene.joint(a["driver"])
        ref = (
            scene.sample(a["reference_clip"], a.get("reference_time", 0))[0]
            if a.get("reference_clip")
            else scene.rest
        )
        world = scene.world(ref)
        offset = (
            local_transform(a["driver_offset"])
            if a.get("driver_offset")
            else np.linalg.solve(world[driver], world[node])
        )
        followers = []
        for f in a.get("followers", []):
            anchor = (
                local_transform(f["offset"])
                if f.get("offset")
                else np.linalg.solve(world[node], world[scene.joint(f["joint"])])
            )
            followers.append(
                (
                    dict(
                        f,
                        arm_chain=attachment_chain(scene, f["joint"], f["chain_root"]),
                    ),
                    anchor,
                )
            )
        if not 0 <= a.get("max_driver_rotation_rad", 0) <= np.pi:
            raise ValueError("Driver rotation budget must be between zero and pi")
        speed = a.get("max_joint_angular_speed_rad_s")
        if speed is not None and (not np.isfinite(speed) or speed <= 0):
            raise ValueError("Attachment angular speed must be positive and finite")
        if a.get("driver_chain_root"):
            a = dict(
                a, arm_chain=attachment_chain(scene, driver, a["driver_chain_root"])
            )
        attachments.append((a, node, driver, offset, followers))
    fps, duration = recipe.get("fps", 30.0), recipe["duration"]
    if not np.isfinite([fps, duration]).all() or min(fps, duration) <= 0:
        raise ValueError("Recipe duration and fps must be positive")
    boundaries = [
        k
        for c in recipe.get("contacts", []) + recipe.get("attachments", [])
        for k in (c["start"], c["end"])
    ]
    if any(t < 0 or t > duration + 1e-6 for t in boundaries):
        raise ValueError(
            "An attachment or contact interval is outside the recipe duration"
        )
    times = np.unique(
        np.r_[np.arange(0, duration, 1 / fps), duration, boundaries].astype(np.float32)
    )
    channels = set()
    defaults = {}
    for _, clip, _, _, mask in prepared:
        for i, prop in scene.clips[clip]:
            if not mask or i in mask:
                channels.add((i, prop))
                if prop == "weights":
                    defaults[i] = scene.morph_rest(i)
    frames, shape_frames, locks, held = [], [], {}, {}
    errors = {}
    previous_time = None
    previous_corrections = {}
    grip_failures = 0
    previous_grips = {}
    max_driver_correction = 0.0
    max_floor_correction = 0.0
    for t in times:
        pose = scene.rest.copy()
        morphs = {i: val.copy() for i, val in defaults.items()}
        for layer, clip, lo, hi, mask in prepared:
            speed = layer.get("speed", 1.0)
            elapsed = float(t) - layer.get("timeline_start", 0.0)
            duration = (hi - lo) / speed
            repeat = layer.get("repeat", False)
            if elapsed < -1e-8 or not repeat and elapsed > duration + 1e-6:
                continue
            target = lo + (
                elapsed * speed % (hi - lo)
                if repeat
                else np.clip(elapsed * speed, 0, hi - lo)
            )
            weight = layer.get("weight", 1.0)
            if layer.get("fade_in", 0) > 0:
                weight *= np.clip(elapsed / layer["fade_in"], 0, 1)
            if not repeat and layer.get("fade_out", 0) > 0:
                weight *= np.clip((duration - elapsed) / layer["fade_out"], 0, 1)
            layer_pose, layer_shapes = scene.sample(clip, target)
            additive = layer.get("additive", False)
            for i, prop in scene.clips[clip]:
                if mask and i not in mask:
                    continue
                if prop == "weights":
                    reference = defaults[i] if additive else morphs[i]
                    morphs[i] += weight * (layer_shapes[i] - reference)
                    continue
                col = SLICES[prop]
                current = pose[i, col]
                new = layer_pose[i, col]
                if prop == "rotation":
                    if additive:
                        new = (
                            Rotation.from_quat(current)
                            * Rotation.from_quat(scene.rest[i, col]).inv()
                            * Rotation.from_quat(new)
                        ).as_quat()
                    pose[i, col] = blend_quats(current, new, weight)
                else:
                    pose[i, col] += weight * (
                        new - (scene.rest[i, col] if additive else current)
                    )
        original_height = (
            floor.height(pose, morphs)
            if floor and floor.config.get("preserve_source_height")
            else None
        )
        source_rotations = {j: Rotation.from_quat(pose[j, 3:7]) for j in smoothed}
        if previous_corrections:
            decay = np.exp(-float(t - previous_time) / temporal["decay_seconds"])
            for j, corr in previous_corrections.items():
                pose[j, 3:7] = (
                    Rotation.from_rotvec(corr.as_rotvec() * decay) * source_rotations[j]
                ).as_quat()
        for limit in articulations:
            j = scene.joint(limit["joint"])
            pose[j, 3:7] = limited(Rotation.from_quat(pose[j, 3:7]), limit).as_quat()
        world = scene.world(pose)
        goals = []
        for index, c in enumerate(recipe.get("contacts", [])):
            if c["start"] - 1e-7 <= t <= c["end"] + 1e-7:
                target = locks.setdefault(
                    index, world[scene.joint(c["joint"]), :3, 3].copy()
                )
                goals.append(dict(c, position=c.get("position", target)))
        if goals or solver.get("collision"):
            pose, reports = solve(scene, pose, goals, solver)
            for r in reports:
                errors[r["joint"]] = max(
                    errors.get(r["joint"], 0), r["position_error_m"]
                )
        for a, node, driver, offset, followers in attachments:
            if t < a["start"] - 1e-7:
                continue
            world = scene.world(pose)
            if t <= a["end"] + 1e-7:
                prop = world[driver] @ offset
                if followers and a.get("driver_chain_root"):
                    # Move the driving hand enough to bring calibrated follower targets
                    # into reach. Keep its original orientation and bound the translation.
                    rotation = Rotation.from_quat(decompose(world[driver])[3:7])
                    turn = (
                        grip_turn(
                            world, prop, driver, followers, a["max_driver_rotation_rad"]
                        )
                        if a.get("max_driver_rotation_rad")
                        else Rotation.identity()
                    )
                    shift = np.zeros(3)
                    for f, anchor in followers:
                        joint = scene.joint(f["joint"])
                        start = scene.joint(f["chain_root"])
                        length = 0.0
                        while joint != start:
                            parent = scene.parents[joint]
                            length += np.linalg.norm(
                                world[parent, :3, 3] - world[joint, :3, 3]
                            )
                            joint = parent
                        offset_world = (
                            world[driver, :3, 3]
                            + turn.apply((prop @ anchor)[:3, 3] - world[driver, :3, 3])
                            + shift
                            - world[start, :3, 3]
                        )
                        distance = np.linalg.norm(offset_world)
                        if distance > length:
                            shift -= offset_world * (1.0 - length / distance)
                    max_shift = a.get("max_driver_translation_m", 0.25)
                    shift *= min(1.0, max_shift / max(np.linalg.norm(shift), 1e-12))
                    if np.linalg.norm(shift) > 1e-8 or turn.magnitude() > 1e-8:
                        before = world[driver, :3, 3].copy()
                        driver_goal = dict(
                            joint=driver,
                            chain_root=a["driver_chain_root"],
                            position=before + shift,
                            rotation=(turn * rotation).as_quat(),
                        )
                        pose, _ = solve(scene, pose, [driver_goal], solver)
                        world = scene.world(pose)
                        max_driver_correction = max(
                            max_driver_correction,
                            float(np.linalg.norm(world[driver, :3, 3] - before)),
                        )
                        prop = world[driver] @ offset
                follow = []
                for f, anchor in followers:
                    dest = decompose(prop @ anchor)
                    follow.append(
                        dict(
                            f,
                            rotation_weight=a.get("orientation_weight", 1.0),
                            position=dest[:3],
                            rotation=dest[3:7] if f.get("orientation", True) else None,
                        )
                    )
                if follow:
                    pose, _ = solve(scene, pose, follow, solver)
                speed = a.get("max_joint_angular_speed_rad_s")
                if speed:
                    active = sorted(
                        set(a.get("arm_chain", [])).union(
                            *(f["arm_chain"] for f, _ in followers)
                        )
                    )
                    prev = previous_grips.get(node)
                    current = pose[active, 3:7]
                    if prev is not None and len(active):
                        last_time, last = prev
                        rotation = Rotation.from_quat(last)
                        delta = (
                            rotation.inv() * Rotation.from_quat(current)
                        ).as_rotvec()
                        fraction = np.minimum(
                            1,
                            speed
                            * (t - last_time)
                            / np.maximum(np.linalg.norm(delta, axis=1), 1e-12),
                        )
                        pose[active, 3:7] = (
                            rotation * Rotation.from_rotvec(delta * fraction[:, None])
                        ).as_quat()
                    previous_grips[node] = (t, pose[active, 3:7].copy())
                world = scene.world(pose)
                prop = world[driver] @ offset
                held[node] = prop
                failed = False
                for f, anchor in followers:
                    target = prop @ anchor
                    actual = world[f["arm_chain"][0]]
                    error = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
                    errors[f["joint"]] = max(errors.get(f["joint"], 0), error)
                    failed |= error > solver.get("tolerance_m", 0.001)
                    if f.get("orientation", True):
                        delta = Rotation.from_quat(
                            decompose(target)[3:7]
                        ).inv() * Rotation.from_quat(decompose(actual)[3:7])
                        failed |= delta.magnitude() > solver.get(
                            "rotation_tolerance_rad", 0.01
                        )
                grip_failures += int(failed)
            elif a.get("release", "hold_world") == "hold_world" and node in held:
                prop = held[node]
            else:
                continue
            parent = scene.parents.get(node)
            pose[node] = decompose(
                np.linalg.solve(world[parent], prop) if parent is not None else prop
            )
        for j in twists:
            distribute_twist(scene, pose, j, scene.rest[j, 3:7])
        for guard in hands:
            guard.apply(pose, morphs)
        if floor:
            max_floor_correction = max(
                max_floor_correction, abs(floor.apply(pose, morphs, original_height))
            )
        previous_corrections = {
            j: Rotation.from_quat(pose[j, 3:7]) * source_rotations[j].inv()
            for j in smoothed
        }
        previous_time = t
        frames.append(pose)
        shape_frames.append(morphs)
    frames = np.stack(frames)
    tracks = []
    for prop, col in SLICES.items():
        changed = np.any(
            np.abs(frames[:, :, col] - scene.rest[:, col]) > 1e-7, axis=(0, 2)
        )
        for i in sorted(
            set(np.flatnonzero(changed)) | {j for j, c in channels if c == prop}
        ):
            tracks.append(
                dict(joint=int(i), property=prop, times=times, values=frames[:, i, col])
            )
    shapes = [
        dict(node=i, times=times, values=[s[i] for s in shape_frames]) for i in defaults
    ]
    # Intermediate external-layer clips are implementation details of this bake.
    transient = {clip for layer, clip in layers if "source" in layer}
    if transient:
        g = scene.glb
        g.doc["animations"] = [
            a for a in g.doc["animations"] if a["name"] not in transient
        ]
        g.write(model)
    write_clips(model, [dict(name=recipe["name"], tracks=tracks, morph_tracks=shapes)])
    from .hand_surface import add_correctives

    corrective_reports = [
        add_correctives(model, recipe["name"], hand)
        for hand in recipe.get("hand_shapes", [])
        if hand.get("surface_corrective")
    ]
    if scene.glb.doc.get("extras", {}).get("charloomClothChains"):
        from transfer.cloth import bake

        bake(model, recipe["name"])
    verified_surface = Surface(model) if corrective_reports else None
    final_hand_edges = [
        HandGuard(verified_surface, hand).measure(recipe["name"])
        if corrective_reports
        else guard.peak_violations
        for guard, hand in zip(hands, recipe.get("hand_shapes", []))
    ]
    return dict(
        clip=recipe["name"],
        frames=len(times),
        tracks=len(tracks) + len(shapes),
        peak_position_error_m=errors,
        grip_nonconverged_frames=grip_failures,
        max_driver_correction_m=max_driver_correction,
        max_floor_correction_m=max_floor_correction,
        hand_correctives=corrective_reports,
        hand_violating_edges=final_hand_edges,
    )
