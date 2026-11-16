"""Bounded fixed-pose skin repair and clip-scoped corrective shapes for hands."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.spatial.transform import Rotation

from roster.mesh_output import add
from roster.write_face_clips import write_clips
from .surface import Surface
from .scene import Scene


def hand_joints(scene, hand):
    return [scene.joint(n) for chain in hand["digits"] for n in chain["joints"]]


def repair_weights(surface, part, dense, hand):
    options = hand["surface_repair"]
    digit_ids = hand_joints(surface.scene, hand)
    wrist = surface.scene.joint(hand["wrist"])
    family = np.isin(part.bones, [wrist, *digit_ids])
    active = dense[:, np.isin(part.bones, digit_ids)].sum(1) > 0.05
    if not active.any():
        return
    count = options["samples_per_clip"]
    budget = options["max_weight_change_l1"]
    if (
        not 3 <= count <= 65
        or not 0 < budget <= 2
        or not 1 <= options["iterations"] <= 256
    ):
        raise ValueError("Invalid fixed-pose weight repair settings")
    edges = part.edges[active[part.edges].any(1)]
    rows, edge = np.unique(edges, return_inverse=True)
    edge = edge.reshape(-1, 2)
    slots = np.flatnonzero(family)
    weights = dense[np.ix_(rows, slots)].copy()
    original = weights.copy()
    mass = original.sum(1)
    inside = active[rows]
    poses = []
    scene = surface.scene
    for clip in options["reference_clips"]:
        start, end = scene.interval(clip)
        times = np.linspace(start, end, count)
        if options.get("include_keys_and_midpoints"):
            keys = np.unique(np.concatenate([v[0] for v in scene.clips[clip].values()]))
            times = np.unique(np.r_[times, keys, (keys[:-1] + keys[1:]) / 2])
        poses.extend(scene.sample(clip, float(t)) for t in times)
    if not poses:
        raise ValueError("Hand weight repair needs reference animation")
    rest = part.points(
        scene.world(scene.rest),
        scene.morph_rest(part.node) if part.deltas else [],
        rows,
    )
    a, b = edge.T
    limit = np.linalg.norm(rest[a] - rest[b], axis=1) * hand.get(
        "max_edge_ratio", 1.2
    ) + hand.get("max_edge_growth_m", 0.004)
    samples = []
    for pose, shapes in poses:
        world = scene.world(pose)
        matrix = (world[part.bones] @ part.inverse)[slots]
        points = part.positions[rows].copy()
        shapew = shapes.get(
            part.node, scene.morph_rest(part.node) if part.deltas else []
        )
        for w, d in zip(shapew, part.deltas):
            points += w * d[rows]
        moved = (
            np.einsum("bij,vj->vbi", matrix[:, :3, :3], points) + matrix[None, :, :3, 3]
        )
        initial = part.points(world, shapew, rows)
        fixed = initial - np.einsum("vb,vbi->vi", original, moved)
        samples.append((moved, fixed))

    def error(candidate, gradient=False):
        grad = np.zeros_like(candidate)
        cost = 0.0
        for moved, fixed in samples:
            points = fixed + np.einsum("vb,vbi->vi", candidate, moved)
            d = points[a] - points[b]
            length = np.linalg.norm(d, axis=1)
            excess = np.maximum(length - limit, 0.0)
            cost += excess @ excess
            if gradient:
                force = d * (2 * excess / np.maximum(length, 1e-12))[:, None]
                acc = np.zeros_like(points)
                np.add.at(acc, a, force)
                np.add.at(acc, b, -force)
                grad += np.einsum("vi,vbi->vb", acc, moved)
        return cost, grad

    for _ in range(options["iterations"]):
        cost, grad = error(weights, True)
        grad -= grad.mean(1, keepdims=True)
        step = 0.05 / max(np.max(np.abs(grad)), 1e-12)
        for _ in range(12):
            trial = np.maximum(weights - step * grad, 0.0)
            total = trial.sum(1)
            trial = np.where(
                (total > 1e-14)[:, None],
                trial * (mass / np.maximum(total, 1e-14))[:, None],
                weights,
            )
            delta = trial - original
            trial = (
                original
                + delta
                * np.minimum(1.0, budget / np.maximum(abs(delta).sum(1), 1e-12))[
                    :, None
                ]
            )
            trial[~inside] = original[~inside]
            # Respect the GLB's four/eight-influence capacity before accepting the objective.
            candidates = dense[rows].copy()
            candidates[:, slots] = trial
            retained = np.argsort(-candidates, axis=1, kind="stable")[
                :, : part.palette.shape[1]
            ]
            keep = np.zeros_like(candidates, dtype=bool)
            np.put_along_axis(keep, retained, True, axis=1)
            if np.any((candidates[:, ~family] > 1e-8) & ~keep[:, ~family]):
                step *= 0.5
                continue
            trial *= keep[:, slots]
            trial *= (mass / np.maximum(trial.sum(1), 1e-15))[:, None]
            if (
                np.max(abs(trial - original).sum(1)) <= budget + 1e-7
                and error(trial)[0] < cost - 1e-15
            ):
                weights = trial
                break
            step *= 0.5
        else:
            break
    dense[np.ix_(rows, slots)] = weights


def corrective_delta(part, rest, world, shape_weights, ids, cfg):
    zone = (part.weights * np.isin(part.bones[part.palette], ids)).sum(1) > 0.05
    if not zone.any():
        return None
    edges = part.edges[zone[part.edges].any(1)]
    rows, e = np.unique(edges, return_inverse=True)
    a, b = e.reshape(-1, 2).T
    posed = part.points(world, shape_weights, rows)
    # Local/global as-rigid-as-possible reconstruction with anchored boundary vertices.
    reference = rest[rows]
    restedge = reference[a] - reference[b]
    degree = np.bincount(np.r_[a, b], minlength=len(rows))
    pin = np.where(zone[rows], max(cfg["source_weight"], 1e-6), 1e6)
    adj = sparse.coo_matrix(
        (np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(len(rows), len(rows))
    ).tocsc()
    solve = factorized(sparse.diags(degree + pin, format="csc") - adj)
    current = posed.copy()
    for _ in range(cfg["iterations"]):
        edges_now = current[a] - current[b]
        cov = np.zeros((len(rows), 3, 3))
        outer = np.einsum("ni,nj->nij", edges_now, restedge)
        np.add.at(cov, a, outer)
        np.add.at(cov, b, outer)
        u, _, vh = np.linalg.svd(cov)
        sign = np.linalg.det(u @ vh)
        u[:, :, 2] *= sign[:, None]
        rotations = u @ vh
        target = np.einsum("nij,nj->ni", (rotations[a] + rotations[b]) / 2, restedge)
        rhs = pin[:, None] * posed
        np.add.at(rhs, a, target)
        np.add.at(rhs, b, -target)
        for _ in range(cfg["global_iterations"]):
            current = solve(rhs)
            for obstacle in cfg.get("obstacles", []):
                mat = world[obstacle["node"]]
                s = mat[:3, :3] @ obstacle["start"] + mat[:3, 3]
                v = mat[:3, :3] @ (np.asarray(obstacle["end"]) - obstacle["start"])
                center = (
                    s
                    + np.clip((current - s) @ v / max(v @ v, 1e-10), 0.0, 1.0)[:, None]
                    * v
                )
                offset = current - center
                r = np.linalg.norm(offset, axis=1)
                gap = np.maximum(obstacle["radius_m"] - r, 0.0)
                rhs += pin[:, None] * offset * (gap / np.maximum(r, 1e-10))[:, None]
            shift = current - posed
            shift *= np.minimum(
                1.0,
                cfg["max_correction_m"]
                / np.maximum(np.linalg.norm(shift, axis=1), 1e-10),
            )[:, None]
            shift[~zone[rows]] = 0.0
            current = posed + shift
            if not cfg.get("obstacles"):
                break
    skin = (world[part.bones] @ part.inverse)[:, :3, :3]
    jac = np.einsum("vk,vkij->vij", part.weights[rows], skin[part.palette[rows]])
    delta = np.zeros_like(part.positions, dtype="<f4")
    delta[rows] = np.einsum("nij,nj->ni", np.linalg.pinv(jac), current - posed)
    return delta


def add_correctives(model, clip, hand):
    surface = Surface(model)
    scene = surface.scene
    g = scene.glb
    cfg = dict(hand["surface_corrective"])
    cfg["obstacles"] = [
        dict(o, node=scene.joint(o["node"])) for o in cfg.get("obstacles", [])
    ]
    if (
        cfg["max_correction_m"] <= 0
        or min(cfg["iterations"], cfg["global_iterations"]) < 1
    ):
        raise ValueError("Invalid hand surface correction budget")
    ids = hand_joints(scene, hand)
    times = np.unique(np.concatenate([v[0] for v in scene.clips[clip].values()]))
    chosen = [int(np.argmin(np.abs(times - cfg["reference_time"])))]
    angles = (
        Rotation.from_quat(
            np.concatenate([scene.sample(clip, t)[0][ids, 3:7] for t in times])
        )
        .as_quat()
        .reshape(len(times), len(ids), 4)
    )

    def distances():
        return (
            2
            * np.arccos(
                np.clip(abs(np.einsum("tjq,sjq->tsj", angles, angles[chosen])), 0, 1)
            )
        ).max(2)

    opts = cfg.get("pose_sampling")
    if opts:
        while len(chosen) < opts["max_samples"]:
            residual = distances().min(1)
            if residual.max() <= opts["max_pose_error_rad"]:
                break
            chosen.append(int(residual.argmax()))
    dist = distances()
    mixes = 1.0 / np.maximum(dist, 1e-8) ** 2
    mixes /= mixes.sum(1, keepdims=True)
    nodes = g.doc["nodes"]
    by_mesh = {}
    for p in surface.parts:
        if (
            p.weights is not None
            and (np.isin(p.bones[p.palette], ids) * p.weights).sum(1).max() > 0.05
        ):
            by_mesh.setdefault(nodes[p.node]["mesh"], p.node)
    for mesh_id, active in by_mesh.items():
        mesh = g.doc["meshes"][mesh_id]
        parts = [(i, p) for i, p in enumerate(surface.parts) if p.node == active]
        old = len(mesh["primitives"][0].get("targets", []))
        original_rest_weights = scene.morph_rest(active).copy()
        count = len(chosen)
        mesh["weights"] = [*mesh.get("weights", [0.0] * old), *[0.0] * count]
        names = mesh.setdefault("extras", {}).setdefault(
            "targetNames", [f"shape_{i}" for i in range(old)]
        )
        names.extend(f"{clip}_{hand['wrist']}_corrective_{k}" for k in range(count))
        users = {i for i, n in enumerate(nodes) if n.get("mesh") == mesh_id}
        for i in users:
            if "weights" in nodes[i]:
                nodes[i]["weights"] += [0.0] * count
        for i, p in parts:
            for frame in chosen:
                pose, shapes = scene.sample(clip, times[frame])
                sw = shapes.get(active, original_rest_weights)
                delta = (
                    corrective_delta(
                        p, surface.rest[i], scene.world(pose), sw, ids, cfg
                    )
                    if p.weights is not None
                    else None
                )
                if delta is None:
                    delta = np.zeros_like(p.positions, dtype="<f4")
                p.prim.setdefault("targets", []).append(
                    dict(POSITION=add(g, delta, "VEC3"))
                )
        # Every primitive in a mesh must have the same morph target count.
        for p in mesh["primitives"]:
            if len(p.get("targets", [])) == old:
                zeros = np.zeros_like(g.array(p["attributes"]["POSITION"]), dtype="<f4")
                acc = add(g, zeros, "VEC3")
                p.setdefault("targets", []).extend([dict(POSITION=acc) for _ in chosen])
        for anim in g.doc.get("animations", []):
            for ch in anim["channels"]:
                if (
                    ch["target"]["node"] not in users
                    or ch["target"]["path"] != "weights"
                ):
                    continue
                sampler = dict(anim["samplers"][ch["sampler"]])
                data = g.array(sampler["output"]).reshape(-1, old)
                sampler["output"] = add(
                    g,
                    np.pad(data, ((0, 0), (0, count))).astype("<f4").ravel(),
                    "SCALAR",
                )
                ch["sampler"] = len(anim["samplers"])
                anim["samplers"].append(sampler)
        old_values = []
        for t in times:
            _, shapes = scene.sample(clip, float(t))
            old_values.append(shapes.get(active, original_rest_weights))
        values = np.column_stack(
            [np.asarray(old_values).reshape(len(times), old), mixes]
        )
        g.doc["buffers"][0]["byteLength"] = len(g.blob)
        g.write(model)
        write_clips(
            model,
            [
                dict(
                    name=clip,
                    tracks=[],
                    morph_tracks=[dict(node=active, times=times, values=values)],
                )
            ],
            extend=True,
        )
        # The next mesh's edits must include the newly written animation channel.
        from roster.mesh import Glb

        updated = Glb(model)
        g.blob = updated.blob
        for key in ("buffers", "bufferViews", "accessors", "animations"):
            g.doc[key] = updated.doc[key]
    return dict(samples=len(chosen), meshes=len(by_mesh))
