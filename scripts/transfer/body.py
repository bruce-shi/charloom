"""Fit the body's silhouette to registered bodysuit measurements on calibrated views."""

import json

import cv2
import igl
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .guide import project, rasterize
from .scan import transform_points
from roster.mesh import adjacency


def closest_contour(points, vertices, edges, confidence):
    """Project onto the closed contour edges without attracting skin to sample vertices."""
    distance2, selected, closest = igl.point_mesh_squared_distance(
        np.column_stack((points, np.zeros(len(points)))),
        np.column_stack((vertices, np.zeros(len(vertices)))),
        edges,
    )
    start, end = vertices[edges[selected]].transpose(1, 0, 2)
    tangent = end - start
    fraction = np.sum((closest[:, :2] - start) * tangent, axis=1) / np.maximum(
        np.sum(tangent**2, axis=1), 1.e-20
    )
    a, b = confidence[edges[selected]].T
    return np.sqrt(distance2), closest[:, :2], a + fraction * (b - a)


def fit_body(glb, guide_path, cams, frame):
    from roster.mesh_output import add, normals

    folder = guide_path.parent / "views"
    contours = next(
        (
            p
            for p in (
                folder / "suit_contours.ensemble.json",
                folder / "suit_contours.json",
            )
            if p.is_file()
        ),
        None,
    )
    if contours is None:
        return {"registered_views": [], "reason": "no registered body contours"}
    guide = json.loads(guide_path.read_text())
    evidence = json.loads(contours.read_text())
    p = next(m["primitives"][0] for m in glb.doc["meshes"] if m.get("name") == "body")
    a = p["attributes"]
    positions = glb.array(a["POSITION"]).copy()
    faces = glb.array(p["indices"]).reshape(-1, 3).astype(np.int64)
    _, first, weld = np.unique(
        positions, axis=0, return_index=True, return_inverse=True
    )
    points = positions[first].astype(float)
    triangles = weld[faces]
    weights = glb.array(a["WEIGHTS_0"])
    joints = glb.array(a["JOINTS_0"])
    names = [glb.doc["nodes"][j]["name"].lower() for j in glb.doc["skins"][0]["joints"]]
    editable_prefix = (
        "spine",
        "pelvis",
        "breast",
        "shoulder",
        "upperarm",
        "lowerarm",
        "upperleg",
        "lowerleg",
        "clavicle",
        "root",
    )
    eligible = np.array([n.startswith(editable_prefix) for n in names])[joints]
    free = np.zeros(len(points), bool)
    np.logical_or.at(free, weld, (weights * eligible).sum(1) > 0.75)
    if not free.any():
        raise ValueError("No torso or limb controls on the fitted template")
    _, lap = adjacency(triangles, len(points))
    eye = sparse.eye(len(points), format="csr")
    regular = (
        lap.T @ lap * 5 + eye * 0.008 + sparse.diags((~free).astype(float) * 1000.0)
    )
    anchors = []
    for view, loops in evidence["views"].items():
        if view not in cams or not loops:
            continue
        if not view.startswith("body_"):
            continue
        cam = cams[view]
        uv = np.concatenate(loops).astype(float)
        weight = np.concatenate(
            evidence.get("weights", {}).get(
                view, [np.ones(len(loop)) for loop in loops]
            )
        )
        if (
            len(weight) != len(uv)
            or not np.isfinite(uv).all()
            or not np.isfinite(weight).all()
        ):
            raise ValueError(f"Invalid registered silhouette in {view}")
        edges = []
        start = 0
        for loop in loops:
            indices = np.arange(start, start + len(loop))
            edges.append(np.column_stack((indices, np.roll(indices, -1))))
            start += len(loop)
        anchors.append((cam, uv, weight, np.concatenate(edges)))
    inverse = np.linalg.inv(frame)
    base = points.copy()
    stats = []
    for iteration in range(5):
        # Each view supplies only image-plane components. World-space depth is
        # determined by the orthogonal view and by the regularized rest surface.
        diag = [np.zeros(len(points)) for _ in range(3)]
        rhs = np.zeros_like(points)
        errors = []
        for cam, uv, confidence, edges in anchors:
            pix = project(cam, transform_points(points, inverse))
            ids, bary = rasterize(pix, triangles, cam["width"], cam["height"])
            mask = (ids >= 0).astype(np.uint8)
            boundary = mask - cv2.erode(mask, np.ones((3, 3), np.uint8))
            ys, xs = np.where(boundary != 0)
            # Uniformly sample the silhouette to avoid weighting by tessellation density.
            ys, xs = ys[::3], xs[::3]
            at = np.column_stack((xs + 0.5, ys + 0.5))
            dist, target, trust = closest_contour(at, uv, edges, confidence)
            tri = triangles[ids[ys, xs]]
            blend = bary[ys, xs]
            good = (free[tri] * blend).sum(1) > 0.90
            trust = trust.clip(0, 1)
            good &= trust > 0.05
            if not good.any():
                continue
            offset = target[good] - at[good]
            norm = np.maximum(1.0, np.linalg.norm(offset, axis=1) / 12.0)
            scale = cam["meters_per_pixel"]
            basis = (
                frame[:3, :3] @ np.array([cam["right"], -np.array(cam["up"])]).T
            ) * scale
            step = (offset / norm[:, None]) @ basis.T
            weight = trust[good] / np.maximum(1.0, dist[good] / 18.0)
            image_axes = basis / np.linalg.norm(basis, axis=0)
            influence = np.sum(image_axes**2, axis=1)
            tri = tri[good]
            for k in range(3):
                w = weight * blend[good, k]
                np.add.at(rhs, tri[:, k], w[:, None] * step)
                for axis in range(3):
                    np.add.at(diag[axis], tri[:, k], w * influence[axis])
            errors.extend(dist[good].tolist())
        if not errors:
            break
        stats.append(float(np.median(errors)))
        if iteration == 4:
            break
        for axis in range(3):
            matrix = regular + sparse.diags(diag[axis])
            displacement = spsolve(
                matrix.tocsc(), rhs[:, axis] - (regular @ (points - base)[:, axis])
            )
            points[free, axis] += 0.65 * displacement[free]
    fitted = points[weld].astype("<f4")
    a["POSITION"] = add(glb, fitted, "VEC3")
    a["NORMAL"] = add(glb, normals(fitted, faces)[0].astype("<f4"), "VEC3")
    return dict(
        registered_views=[cam["id"] for cam, _, _, _ in anchors],
        contours=str(contours),
        source_hash=guide["source_hash"],
        median_silhouette_distance_px=stats,
        max_displacement_mm=float(1000 * np.linalg.norm(points - base, axis=1).max()),
    )
