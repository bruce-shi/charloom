"""Preserve a cut scan seam when its two garments use different skin weights."""

import igl
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .scaffold import bary_weights


def border_edges(faces):
    edges = np.sort(np.vstack((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(
        edges[edges[:, 0] != edges[:, 1]], axis=0, return_counts=True
    )
    return edges, edges[counts == 1]


def match_layer_seam(outer, inner):
    """Return outer skin rows blended into the lower layer at shared boundary edges.

    Both meshes must be in the same rest space and joint palette. Point-only
    contacts, independent overlapping surfaces and disconnected panels stay free.
    """
    p, f, joints, weights = outer
    q, tri, inner_joints, inner_weights = inner
    verts, inv = np.unique(np.vstack((p, q)), axis=0, return_inverse=True)
    outer_ids, inner_ids = inv[: len(p)], inv[len(p) :]
    edges, border = border_edges(outer_ids[f])
    _, lower_border = border_edges(inner_ids[tri])
    shared = set(map(tuple, border)).intersection(map(tuple, lower_border))
    if not shared:
        return None
    seam = np.unique(np.asarray(list(shared)))
    # Blend over a narrow surface band; the sleeve on the other side of a
    # fold can be near the hem in space while far away along the garment.
    width = np.max(np.ptp(verts[seam], axis=0)) / 3
    if width <= 0:
        return None
    a, b = edges.T
    lengths = np.linalg.norm(verts[a] - verts[b], axis=1)
    graph = coo_matrix((lengths, (a, b)), shape=(len(verts), len(verts))).tocsr()
    distance = dijkstra(
        graph, directed=False, indices=seam, min_only=True, limit=width
    )[outer_ids]
    use = distance < width
    squared, near, closest = igl.point_mesh_squared_distance(p[use], q, tri)
    neighbors = tri[near]
    bary = igl.barycentric_coordinates(closest, *(q[neighbors[:, i]] for i in range(3)))
    ji, wt = bary_weights(
        inner_joints, inner_weights, neighbors, np.nan_to_num(bary, nan=1 / 3)
    )
    t = np.clip(np.maximum(distance[use], np.sqrt(squared)) / width, 0, 1)
    keep = (t * t * (3 - 2 * t))[:, None]
    slots = max(4, int(joints.max()) + 1, int(inner_joints.max()) + 1)
    dense = np.zeros((int(use.sum()), slots))
    rows = np.arange(len(dense))[:, None]
    np.add.at(dense, (rows, joints[use]), weights[use] * keep)
    np.add.at(dense, (rows, ji), wt * (1 - keep))
    top = np.argsort(-dense, axis=1, kind="stable")[:, :4]
    out_j, out_w = joints.copy(), weights.copy()
    mass = np.take_along_axis(dense, top, axis=1)
    out_j[use] = top
    out_w[use] = mass / mass.sum(1, keepdims=True)
    return out_j, out_w
