"""Fit welded body controls to observed skin, or fair an audited repair region."""
import igl
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .mesh import adjacency


def solve_surface(positions, triangles, active, *, scan=None, smoothness=1., data_weight=10., rounds=60):
    """The scan tuple contains vertex positions, labelled skin faces and all scan faces."""
    rest = np.asarray(positions, dtype=float)
    active = np.asarray(active, dtype=bool)
    triangles = np.asarray(triangles, dtype=int)
    if (rest.ndim != 2 or rest.shape[1] != 3 or not rest.size or not np.isfinite(rest).all()
            or active.shape != (len(rest),) or triangles.ndim != 2 or triangles.shape[1] != 3
            or np.any(triangles < 0) or np.any(triangles >= len(rest))):
        raise ValueError("Invalid body controls")
    if smoothness < 0 or data_weight < 0 or rounds < 0:
        raise ValueError("Invalid fit settings")
    _, laplacian = adjacency(triangles, len(rest))
    free = laplacian[:, active]
    strength = smoothness if scan is not None else 1.
    matrix = (strength * (free.T @ free)).tocsc()
    boundary = -strength * (free.T @ (laplacian[:, ~active] @ rest[~active]))
    x = rest.copy()

    def solve(weight, rhs, step):
        if not active.any():
            return
        system = matrix + diags(np.broadcast_to(weight, (active.sum(),)))
        result = spsolve(system.tocsc(), rhs + boundary)
        if not np.isfinite(result).all():
            raise ValueError("Body surface solve produced nonfinite coordinates")
        x[active] += (result.reshape(-1, 3) - x[active]) * step

    if scan is None:
        solve(.002, .002 * rest[active], 1.)
        return x, np.empty(0, dtype=int), []
    vertices, skin, faces = scan
    def nondegenerate(rows):
        points = vertices[rows]
        areas = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        return rows[np.einsum("ij,ij->i", areas, areas) >= 1e-24]
    skin, faces = nondegenerate(skin), nondegenerate(faces)
    if not len(skin):
        raise ValueError("No observed skin in segmentation")
    skin_tree, full_tree = igl.AABB(), igl.AABB()
    skin_tree.init(vertices, skin)
    full_tree.init(vertices, faces)
    skin_tris = vertices[skin]
    normal = np.cross(skin_tris[:, 1] - skin_tris[:, 0], skin_tris[:, 2] - skin_tris[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-20)

    def hits():
        d, index, points = skin_tree.squared_distance(vertices, skin, x[active])
        full, _, _ = full_tree.squared_distance(vertices, faces, x[active])
        return np.sqrt(d), index, points, np.sqrt(d) < np.sqrt(full) + .006

    history = []
    _, _, _, observed = hits()
    anchor = np.where(observed, .3, 3.)
    for i in range(rounds):
        distances, index, points, visible = hits()
        tri = x[triangles]
        face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        vertex_normals = np.zeros_like(x)
        for corner in triangles.T:
            np.add.at(vertex_normals, corner, face_normals)
        lengths = np.linalg.norm(vertex_normals, axis=1)
        vertex_normals[lengths <= 1e-20] = (0, 1, 0)
        vertex_normals[lengths > 1e-20] /= lengths[lengths > 1e-20, None]
        compatible = np.einsum("ij,ij->i", vertex_normals[active], normal[index]) > .1
        good = observed & visible & (distances < .12) & compatible
        weight = data_weight * good
        solve(weight + anchor, points * weight[:, None] + rest[active] * anchor[:, None], .5)
        history.append({"round": i, "observed": int(good.sum())})
    return x, np.flatnonzero(active)[observed], history
