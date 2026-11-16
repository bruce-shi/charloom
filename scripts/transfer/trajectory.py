"""Adapt an explicitly characterized limb trajectory and distribute wrist twist."""

import numpy as np
from scipy.spatial.transform import Rotation

from animation.articulation import two_bone, distribute_twist
from animation.scene import Scene, decompose
from roster.write_face_clips import write_clips


def adapt(model, clip, samples, mapping, options, align, scale):
    scene = Scene(model)
    rest = scene.world(scene.rest)
    src = samples["rest"]
    trajectories = []
    for spec in options.get("limb_trajectories", []):
        root, mid, orient, end = [scene.joint(n) for n in spec["target_chain"]]
        anchor = scene.joint(spec["target_anchor"])
        if (
            scene.parents.get(mid) != root
            or scene.parents.get(orient) != mid
            or scene.parents.get(end) != orient
            or anchor not in mapping
            or end not in mapping
        ):
            raise ValueError(
                "Limb trajectory needs a direct four-joint chain and mapped anchor and endpoint"
            )
        trajectories.append((spec, anchor, root, mid, orient, end))
    twists = []
    for names in options.get("wrist_twist_chains", []):
        chain = [scene.joint(n) for n in names]
        if len(chain) < 2 or any(
            scene.parents.get(c) != p for p, c in zip(chain, chain[1:])
        ):
            raise ValueError(
                "Wrist twist supports must form a direct chain to the hand"
            )
        twists.append(chain)
    if options.get("distribute_palm_twist") or options.get("distribute_wrist_twist"):
        from transfer.motion import key

        for i in mapping:
            if key(scene.names[i]) in ("left_hand", "right_hand") and not any(
                i in c for c in twists
            ):
                twists.append([i])

    def rot(m):
        return Rotation.from_quat(decompose(m)[3:7])

    def cap(actual, reference, bound):
        delta = (reference.inv() * actual).as_rotvec()
        return (
            reference
            * Rotation.from_rotvec(
                delta * min(1.0, bound / max(np.linalg.norm(delta), 1e-12))
            )
            if bound is not None
            else actual
        )

    tracks = {}
    peaks = [0.0] * len(trajectories)
    for t, sw in zip(samples["times"], samples["poses"]):
        pose, _ = scene.sample(clip, float(t))
        touched = set()
        for idx, (cfg, anchor, root, mid, orient, end) in enumerate(trajectories):
            world = scene.world(pose)
            se, sa = mapping[end], mapping[anchor]
            relative = sw[se, :3, 3] - sw[sa, :3, 3] - (src[se, :3, 3] - src[sa, :3, 3])
            goal = (
                world[anchor, :3, 3]
                + rest[end, :3, 3]
                - rest[anchor, :3, 3]
                + align.apply(relative) * cfg.get("position_scale", scale)
            )
            anchor_delta = rot(world[anchor]) * rot(rest[anchor]).inv()
            wanted = cap(
                rot(world[orient]),
                anchor_delta * rot(rest[orient]),
                cfg.get("max_orientation_deviation_rad"),
            )
            end_rot = cap(
                rot(world[end]),
                anchor_delta * rot(rest[end]),
                cfg.get("max_end_orientation_deviation_rad"),
            )
            ankle = goal - wanted.apply(pose[end, :3] * pose[orient, 7:])
            two_bone(scene, pose, [orient, mid, root], ankle, None)
            updated = scene.world(pose)
            pose[orient, 3:7] = (rot(updated[mid]).inv() * wanted).as_quat()
            pose[end, 3:7] = (wanted.inv() * end_rot).as_quat()
            peaks[idx] = max(
                peaks[idx], float(np.linalg.norm(scene.world(pose)[end, :3, 3] - goal))
            )
            touched.update([root, mid, orient, end])
        for chain in twists:
            for level, child in reversed(list(enumerate(chain))):
                fraction = (
                    (level + 1) / (level + 2)
                    if len(chain) > 1 or options.get("distribute_palm_twist")
                    else 1.0
                )
                distribute_twist(scene, pose, child, scene.rest[child, 3:7], fraction)
                touched.update([child, scene.parents[child]])
        for i in touched:
            tracks.setdefault(i, []).append(pose[i, 3:7].copy())
    if tracks:
        write_clips(
            model,
            [
                dict(
                    name=clip,
                    morph_tracks=[],
                    tracks=[
                        dict(
                            joint=i,
                            property="rotation",
                            times=samples["times"],
                            values=values,
                        )
                        for i, values in tracks.items()
                    ],
                )
            ],
            extend=True,
        )
    return peaks
