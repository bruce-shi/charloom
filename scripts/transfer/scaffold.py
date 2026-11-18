"""Fit the MakeHuman bind pose to marked joints and retain the template's skin palette."""

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial.transform import Rotation
from .scan import transforms


JOINTS = {
    "shoulder": "upperarm01",
    "elbow": "lowerarm01",
    "wrist": "wrist",
    "hip": "upperleg01",
    "knee": "lowerleg01",
    "ankle": "foot",
    "eye": "eye",
}


def landmark_warp(old, wanted, mode="spatial"):
    """Interpolate measured landmark displacements with an explicit depth policy.

    Frontal measurements constrain where the body axis sits in depth, but do
    not measure the anterior/posterior thickness around that axis. Opting into
    the frontal field preserves the native depth difference at each XY point.
    Spatial fitting remains the default for overlapping or nonfrontal poses.
    """
    if mode not in ("spatial", "frontal_depth_preserving"):
        raise ValueError(f"Unknown template_warp mode: {mode}")
    dimensions = slice(None) if mode == "spatial" else slice(0, 2)
    coordinates = old[:, dimensions]
    if mode == "frontal_depth_preserving":
        if np.linalg.matrix_rank(coordinates - coordinates.mean(0)) < 2:
            raise ValueError("Frontal template warp needs distinct 2D landmark positions")
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        if np.any(distances < 1e-5):
            raise ValueError("Overlapping frontal landmarks require spatial template warp")
    field = RBFInterpolator(
        coordinates, wanted - old, kernel="thin_plate_spline", smoothing=0.00005
    )

    def warp(points):
        return points + field(points[:, dimensions])

    return warp


def fit_template(g, resolved):
    from roster.mesh_output import add, normals

    world, parents = transforms(g.doc)
    skin = g.doc["skins"][0]
    joints = skin["joints"]
    nodes = g.doc["nodes"]
    named = {nodes[i]["name"]: i for i in joints}
    marks = {
        k: np.asarray(v["point"]) for k, v in resolved["joint_section_depth"].items()
    }
    marks.update(resolved["landmarks"])
    anchors = {"neck01": marks["neck"]}
    for key, bone in JOINTS.items():
        for side in ("l", "r"):
            label = key + "_" + side
            if label in marks:
                anchors[bone + "." + side.upper()] = marks[label]
    for side in ("l", "r"):
        for digit in range(1, 6):
            for joint, number in (("base", 1), ("mid", 2)):
                label = f"finger{digit}_{side}_{joint}"
                if label in marks:
                    anchors[f"finger{digit}-{number}.{side.upper()}"] = marks[label]
    old = np.array([world[named[k], :3, 3] for k in anchors])
    wanted = np.array(list(anchors.values()))
    # The paired oris03 pivots sit on the upper lip, halfway to the
    # commissures. Pulling both pivots out to the corners widens the teeth.
    # Use a central mouth translation and preserve the authored dental arches.
    mouth_old = (world[named["oris05"], :3, 3] + world[named["oris01"], :3, 3]) / 2
    mouth_new = (np.asarray(marks["mouth_l"]) + marks["mouth_r"]) / 2
    old = np.vstack((old, mouth_old))
    wanted = np.vstack((wanted, mouth_new))
    eye_old = world[named["eye.L"], :3, 3] - world[named["eye.R"], :3, 3]
    eye_new = np.asarray(marks["eye_l"]) - marks["eye_r"]
    dental_scale = np.linalg.norm(eye_new) / np.linalg.norm(eye_old)
    dental_rotation, _ = Rotation.align_vectors(
        [eye_new / np.linalg.norm(eye_new)], [eye_old / np.linalg.norm(eye_old)]
    )

    def dental(points):
        return mouth_new + dental_rotation.apply(points - mouth_old) * dental_scale

    # The caller explicitly selects frontal depth preservation only after
    # reviewing the pose and measuring hidden body sections independently.
    warp_mode = resolved.get("subject", {}).get("template_warp", "spatial")
    warp = landmark_warp(old, wanted, warp_mode)

    orig_joints = world[joints, :3, 3]
    moved = warp(orig_joints)
    rest = world.copy()
    for i, pos, original, node in zip(
        joints, moved, orig_joints, (nodes[j] for j in joints)
    ):
        children = [child for child in node.get("children", []) if child in joints]
        if children:
            main = max(
                children, key=lambda c: np.linalg.norm(world[c, :3, 3] - original)
            )
            a, b = world[main, :3, 3] - original, moved[joints.index(main)] - pos
            if min(np.linalg.norm(a), np.linalg.norm(b)) > 1e-8:
                rot, _ = Rotation.align_vectors(
                    [b / np.linalg.norm(b)], [a / np.linalg.norm(a)]
                )
                rest[i, :3, :3] = rot.as_matrix() @ world[i, :3, :3]
        rest[i, :3, 3] = pos
    for i in joints:
        local = np.linalg.solve(rest[parents[i]], rest[i]) if i in parents else rest[i]
        rot, scale = local[:3, :3], np.linalg.norm(local[:3, :3], axis=0)
        nodes[i].pop("matrix", None)
        nodes[i].update(
            translation=local[:3, 3].tolist(),
            rotation=Rotation.from_matrix(rot / scale).as_quat().tolist(),
            scale=scale.tolist(),
        )
    g.array(skin["inverseBindMatrices"])[:] = (
        np.linalg.inv(rest[joints]).transpose(0, 2, 1).reshape(-1, 16)
    )
    body = g.body()
    for node in nodes:
        if "mesh" not in node:
            continue
        mesh = g.doc["meshes"][node["mesh"]]
        source_name = mesh.get("name", "").lower()
        if body is mesh["primitives"][0]:
            name = "body"
        elif "teeth" in source_name:
            name = "teeth"
        elif "tongue" in source_name:
            name = "tongue"
        elif (
            "low-poly" in source_name
            or "eye" in source_name
            and "lash" not in source_name
            and "brow" not in source_name
        ):
            name = "eyes"
        else:
            # The completed face carries the observed lashes and brows.
            node.pop("mesh")
            node.pop("skin", None)
            continue
        mesh["name"] = node["name"] = name
        for prim in mesh["primitives"]:
            attr = prim["attributes"]
            original = g.array(attr["POSITION"]).copy()
            deform = dental if name in ("teeth", "tongue") else warp
            points = deform(original)
            # Appending attributes can relocate the GLB byte buffer.
            tri = g.array(prim["indices"]).reshape(-1, 3).copy()
            attr["POSITION"] = add(g, points.astype("<f4"), "VEC3")
            attr["NORMAL"] = add(g, normals(points, tri)[0].astype("<f4"), "VEC3")
            for target in prim.get("targets", []):
                delta = g.array(target["POSITION"])
                target["POSITION"] = add(
                    g, (deform(original + delta) - points).astype("<f4"), "VEC3"
                )
                target.pop("NORMAL", None)
    return {
        "template_warp": warp_mode,
        "joints": len(joints),
        "anchors": len(old),
        "max_anchor_error_mm": float(
            1000 * np.linalg.norm(warp(old) - wanted, axis=1).max()
        ),
    }


def bary_weights(joints, weights, triangles, bary):
    """Interpolate the body's four influences and reduce the sum to the top four."""
    js, ws = [], []
    for start in range(0, len(triangles), 16384):
        ids = triangles[start : start + 16384]
        bones = joints[ids].reshape(-1, 12)
        values = (weights[ids] * bary[start : start + len(ids), :, None]).reshape(
            -1, 12
        )
        values = np.maximum(values, 0.0)
        combined = np.zeros((len(ids), int(joints.max()) + 1))
        np.add.at(combined, (np.arange(len(ids))[:, None], bones), values)
        top = np.argsort(-combined, axis=1, kind="stable")[:, :4]
        total = np.take_along_axis(combined, top, axis=1)
        if np.any(total.sum(1) <= 0):
            raise ValueError("No template weights for garment vertex")
        js.append(top.astype("<u2"))
        ws.append((total / total.sum(1, keepdims=True)).astype("<f4"))
    return np.concatenate(js), np.concatenate(ws)
