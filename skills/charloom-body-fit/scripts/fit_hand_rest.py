"""Set the wrist rest direction from measured wrist-to-fingertip vectors before retargeting."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills/charloom-body-measure/scripts"))
from measurement_common import read, write, checked_file, unit, digest
from animation.scene import decompose, matrices
from animation.surface import Surface
from roster.mesh_output import add, finish, normals
from refit_sections import world_to_local


def descendants(nodes, root):
    pending = [root]
    found = set()
    while pending:
        index = pending.pop()
        if index not in found:
            found.add(index)
            pending.extend(nodes[index].get("children", []))
    return sorted(found)


def run(path, config, out, rebuild_clips):
    if out.exists():
        raise FileExistsError(out)
    cfg = read(config)
    surface = Surface(path)
    scene = surface.scene
    g = scene.glb
    nodes = g.doc["nodes"]
    parts = [
        (i, part)
        for i, part in enumerate(surface.parts)
        if scene.names[part.node] == cfg["body_node"]
    ]
    if len(parts) != 1 or parts[0][1].deltas:
        raise ValueError("Expected one skinned body before body morphs")
    index, part = parts[0]
    points = surface.rest[index]
    # Face visibility differs by clothing stage, but both have identically
    # ordered body positions, UVs and a common rest skeleton.
    import hashlib

    if (
        hashlib.sha256(part.positions.astype("<f4").tobytes()).hexdigest()
        != cfg["body_positions_sha256"]
    ):
        raise ValueError("Body vertex positions differ from the reviewed rest body")
    evidence = checked_file(config.parent, cfg["evidence"])
    original = scene.world(scene.rest)
    desired = original.copy()
    moved = points.copy()
    changed = set()
    records = []
    for hand in cfg["hands"]:
        wrist = scene.joint(hand["wrist"])
        joint_ids = descendants(nodes, wrist)
        if changed.intersection(joint_ids):
            raise ValueError("Overlapping joint subtrees")
        changed.update(joint_ids)
        weight = (part.weights * np.isin(part.bones[part.palette], joint_ids)).sum(
            axis=1
        )
        pivot = original[wrist, :3, 3]
        tip = hand["tip_vertex"]
        if not isinstance(tip, int) or not 0 <= tip < len(points) or weight[tip] < 0.99:
            raise ValueError("Expected a measured fingertip in the wrist subtree")
        start = points[tip] - pivot
        target = np.array(hand["wrist_to_tip_m"], float)
        direction = unit(start)
        ratio = np.linalg.norm(target) / np.linalg.norm(start)
        target_direction = unit(target)
        if np.dot(direction, target_direction) < -0.9999:
            raise ValueError(
                "Opposed directions need an additional palm orientation landmark"
            )
        rot = Rotation.align_vectors([target_direction], [direction])[0].as_matrix()
        stretch = np.eye(3) + (ratio - 1) * np.outer(direction, direction)
        transform = rot @ stretch
        mapped = pivot + (points - pivot) @ transform.T
        moved += (mapped - points) * weight[:, None]
        desired[joint_ids, :3, :3] = rot @ original[joint_ids, :3, :3]
        desired[joint_ids, :3, 3] = (
            pivot + (original[joint_ids, :3, 3] - pivot) @ transform.T
        )
        records.append(
            {
                "wrist": hand["wrist"],
                "tip_vertex": tip,
                "original_m": start.tolist(),
                "target_m": target.tolist(),
                "axis_scale": ratio,
                "rotation_degrees": float(
                    np.degrees(Rotation.from_matrix(rot).magnitude())
                ),
            }
        )
    clips = g.doc.get("animations", [])
    clip_names = {clip.get("name") for clip in clips}
    if set(rebuild_clips) - clip_names:
        raise ValueError("A requested clip to rebuild is absent")
    for clip in clips:
        if clip.get("name") not in rebuild_clips and any(
            ch["target"].get("node") in changed for ch in clip["channels"]
        ):
            raise ValueError(
                f"Retarget {clip.get('name')!r} again after changing its animated wrist rest pose"
            )
    g.doc["animations"] = [
        clip for clip in clips if clip.get("name") not in rebuild_clips
    ]
    if not g.doc["animations"]:
        del g.doc["animations"]
    local = world_to_local(surface, part, moved)
    attr = part.prim["attributes"]
    attr["POSITION"] = add(g, local.astype("<f4"), "VEC3")
    _, first, weld = np.unique(
        local.round(6), axis=0, return_index=True, return_inverse=True
    )
    new_normal = normals(local[first], weld[part.faces])[0][weld]
    active = np.linalg.norm(moved - points, axis=1) > 1.0e-8
    visible = np.zeros(len(active), bool)
    visible[np.unique(part.faces)] = True
    attr["NORMAL"] = add(
        g,
        np.where(
            (active & visible)[:, None], new_normal, g.array(attr["NORMAL"])
        ).astype("<f4"),
        "VEC3",
    )
    for i in changed:
        if "mesh" in nodes[i]:
            raise ValueError(
                "Attached mesh node in wrist subtree needs explicit review"
            )
        parent = scene.parents.get(i)
        matrix = (
            np.linalg.solve(desired[parent], desired[i])
            if parent is not None
            else desired[i]
        )
        trs = decompose(matrix)
        if not np.allclose(matrices(trs.reshape(1, 10))[0], matrix, atol=1.0e-6):
            raise ValueError("Updated wrist requires a sheared bind transform")
        nodes[i].pop("matrix", None)
        nodes[i].update(
            translation=trs[:3].tolist(),
            rotation=trs[3:7].tolist(),
            scale=trs[7:].tolist(),
        )
    for skin in g.doc["skins"]:
        joints = np.array(skin["joints"])
        if not changed.intersection(joints):
            continue
        old_inverse = (
            g.array(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
            if "inverseBindMatrices" in skin
            else np.broadcast_to(np.eye(4), (len(joints), 4, 4))
        )
        new_inverse = np.linalg.solve(desired[joints], original[joints] @ old_inverse)
        skin["inverseBindMatrices"] = add(
            g, new_inverse.transpose(0, 2, 1).reshape(-1, 16).astype("<f4"), "MAT4"
        )
    finish(g, out)
    return {
        "input_sha256": digest(path),
        "output_sha256": digest(out),
        "config_sha256": digest(config),
        "evidence_sha256": digest(evidence),
        "hands": records,
        "rebuild_clips": list(rebuild_clips),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rebuild-clip", action="append", default=[])
    args = parser.parse_args()
    report = run(args.model, args.config, args.out, args.rebuild_clip)
    write(args.out.with_suffix(".hand-rest.json"), report)
