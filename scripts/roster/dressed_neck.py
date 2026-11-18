"""Retain the visible lower scan neck inside a reviewed high-collar garment."""

import numpy as np


def lower_surface(vertices, faces, normal, offset, source_normals, source_uv):
    """Clip textured source triangles below the cut with per-corner interpolation."""
    positions, normals, uvs, indices = [], [], [], []
    signed = vertices @ normal - offset
    for face in faces[signed[faces].min(1) < 0]:
        weights = []
        for i in range(3):
            j = (i + 1) % 3
            a, b = signed[face[i]], signed[face[j]]
            if a <= 0:
                weights.append(np.eye(3)[i])
            if (a < 0 < b) or (b < 0 < a):
                t = a / (a - b)
                weights.append((1 - t) * np.eye(3)[i] + t * np.eye(3)[j])
        if len(weights) < 3:
            continue
        bary = np.asarray(weights)
        points = bary @ vertices[face]
        base = len(positions)
        positions.extend(points)
        normals.extend(bary @ source_normals[face])
        uvs.extend(bary @ source_uv[face])
        indices.extend(
            (base, base + k, base + k + 1) for k in range(1, len(points) - 1)
        )
    if not indices:
        raise ValueError("Reviewed source neck has no textured surface below the cut")
    normals = np.asarray(normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    return np.asarray(positions), np.asarray(indices), normals, np.asarray(uvs)


def bind_shared_collar(g, nv, sv, sf, skin_faces, uv, normal, offset, garments, head):
    """Match the lower neck to existing garment vertices on its original boundary."""
    from scipy.spatial import cKDTree

    # Split atlas vertices may share a geometric edge without sharing a UV.
    xyz, weld = np.unique(sv.round(7), axis=0, return_inverse=True)
    edges = weld[sf[skin_faces]][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edge, count = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    border = np.unique(edge[count == 1])
    border = border[xyz[border] @ normal < offset]
    on_border = np.isin(weld[sf], border)
    matches = {}
    for face in np.flatnonzero(
        on_border.any(1) & ~np.isin(np.arange(len(sf)), skin_faces)
    ):
        matches.setdefault(uv[sf[face]].astype("<f4").tobytes(), []).append(face)

    anchors = {}
    for node in g.doc["nodes"]:
        if node.get("name") not in garments or "mesh" not in node:
            continue
        for part in g.doc["meshes"][node["mesh"]]["primitives"]:
            attrs = part["attributes"]
            corners = g.array(part["indices"]).reshape(-1, 3)
            tex = g.array(attrs["TEXCOORD_0"])
            pos = g.array(attrs["POSITION"])
            joint = g.array(attrs["JOINTS_0"])
            weight = g.array(attrs["WEIGHTS_0"])
            for tri in corners:
                candidates = matches.get(tex[tri].astype("<f4").tobytes(), [])
                if not candidates:
                    continue
                face = min(
                    candidates, key=lambda f: np.linalg.norm(pos[tri] - sv[sf[f]])
                )
                for k in np.flatnonzero(on_border[face]):
                    vid = weld[sf[face, k]]
                    i = tri[k]
                    anchors[vid] = (pos[i] - sv[sf[face, k]], joint[i], weight[i])
    if len(anchors) < 3:
        raise ValueError(
            "Could not match the original lower-neck boundary to its worn collar"
        )
    ids = list(anchors)
    distance, nearest = cKDTree(xyz[ids]).query(nv)
    height = np.maximum(offset - nv @ normal, 0)
    influence = height / np.maximum(distance + height, 1.0e-12)
    influence = influence**2 * (3 - 2 * influence)
    delta, joint, weight = map(np.asarray, zip(*(anchors[ids[k]] for k in nearest)))
    values = np.zeros((len(nv), max(4, int(joint.max()) + 1, head + 1)))
    np.add.at(values, (np.arange(len(nv))[:, None], joint), influence[:, None] * weight)
    values[:, head] += 1 - influence
    palette = np.argsort(-values, axis=1, kind="stable")[:, :4]
    weights = np.take_along_axis(values, palette, axis=1)
    weights /= weights.sum(1, keepdims=True)
    palette[palette > max(int(joint.max()), head)] = head
    return nv + influence[:, None] * delta, palette, weights, len(anchors)


def seam_controls(g, neck, normal, offset, garments):
    """Pair coincident, rendered neck edges for motion checks on the dressed GLB."""
    from scipy.spatial import cKDTree

    border = abs(neck @ normal - offset) < 1.e-6
    strips = []
    coverage = set()
    for node in g.doc["nodes"]:
        name = node.get("name")
        upper = name in ("FaceSurface", "fitted_head")
        if "mesh" not in node or (not upper and name not in garments):
            continue
        for index, prim in enumerate(g.doc["meshes"][node["mesh"]]["primitives"]):
            ids = np.unique(g.array(prim["indices"]))
            xyz = g.array(prim["attributes"]["POSITION"])[ids]
            distance, local = cKDTree(xyz).query(neck)
            hit = np.flatnonzero((distance < 1.e-6) & (border if upper else ~border))
            if len(hit):
                pairs = np.column_stack((hit, ids[local[hit]]))
                strips.append(dict(mesh=name, primitive=index, pairs=pairs.tolist()))
                coverage.add("upper" if upper else "garment")
    if coverage != {"upper", "garment"}:
        raise ValueError("Dressed neck lacks verifiable seams with head and garment")
    return strips
