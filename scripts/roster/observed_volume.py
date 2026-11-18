"""Identify the scan's head surface from resolved face observations in the working frame."""

import igl
import manifold3d as manifold
import numpy as np


def head_solid(vertices, triangles, marks, parts=None, neck_floor=None):
    # GLB atlas seams duplicate positions. Face adjacency has anatomical meaning
    # only after welding them. Ignore features carried by separate eye meshes.
    vertices, weld = np.unique(vertices, axis=0, return_inverse=True)
    faces = weld[triangles]
    parts = parts or {}
    original_faces = faces
    eligible = parts.get("skin", np.arange(len(faces)))
    if neck_floor is not None:
        eligible = eligible[vertices[faces[eligible], 1].max(axis=1) > neck_floor]
    faces = faces[eligible]
    count, labels = igl.facet_components(faces)
    anchors = np.array([marks[k] for k in ("nose_tip", "mouth_l", "mouth_r", "eye_l_inner", "eye_r_inner")])
    distances, nearest, _ = igl.point_mesh_squared_distance(anchors, vertices, faces)
    votes = np.bincount(labels[nearest], minlength=count)
    winner = votes.argmax()
    if votes[winner] <= len(anchors) // 2:
        raise ValueError("Face observations do not identify a single scan surface")
    eye_parts = {}
    source_id = manifold.Manifold.reserve_ids(1)
    source_face_ids = eligible[labels == winner]
    # A detached eyeball must be the frontmost shell at the marked eye and
    # its interior samples, bounded on the same scale as the observed opening.
    tree = igl.AABB()
    tree.init(vertices, original_faces)
    # The visibility test includes the entire scan, including accessories.
    original_labels = labels if len(eligible) == len(original_faces) else igl.facet_components(original_faces)[1]
    for side in ("l", "r"):
        if "eye_" + side in parts:
            eye_parts[side] = parts["eye_" + side]
            continue
        center = np.array(marks["eye_" + side])
        corners = np.array([marks[f"eye_{side}_{k}"] for k in ("inner", "outer", "top", "bottom")])
        probes = np.vstack([center, (corners + center) / 2])
        probes[:, 2] = vertices[:, 2].max() + 1
        hit, _, _ = tree.intersect_ray_first(vertices, original_faces, probes, np.tile([0., 0., -1.], (len(probes), 1)))
        if np.any(hit < 0) or len(np.unique(original_labels[hit])) != 1 or original_labels[hit[0]] == original_labels[source_face_ids[0]]:
            continue
        ids = np.flatnonzero(original_labels == original_labels[hit[0]])
        pts = vertices[np.unique(original_faces[ids])]
        span = np.ptp(pts, axis=0)
        aperture = np.ptp(corners, axis=0)
        if (span[:2] < aperture[:2] * 0.8).any() or (span > 2.5 * max(aperture[:2])).any():
            continue
        if igl.boundary_loop_all(original_faces[ids]) or span[2] < 0.2 * max(span[:2]):
            continue
        eye_parts[side] = ids
    if len(eye_parts) == 2 and eye_parts["l"][0] == eye_parts["r"][0]:
        eye_parts.clear()
    used, local = np.unique(faces[labels == winner], return_inverse=True)
    positions = vertices[used]
    faces = local.reshape(-1, 3)
    if not igl.is_edge_manifold(faces)[0] or not np.all(igl.is_vertex_manifold(faces)):
        # Atlas welds can join disjoint surface fans at a vertex. Split the
        # fans before walking their boundaries; face IDs and UVs stay intact.
        positions, faces, _ = igl.split_nonmanifold(positions, faces)
    faces, _ = igl.bfs_orient(faces)
    # Close authored eye, lip and neck boundaries temporarily for the booleans.
    # Original surface vertices stay in place. The assembly forms its own ports.
    loops = igl.boundary_loop_all(faces)
    cap_faces, centers = [], []
    for loop in loops:
        ring = np.asarray(loop)
        center = len(positions) + len(centers)
        centers.append(positions[ring].mean(0))
        cap_faces.append(np.column_stack((np.roll(ring, -1), ring, np.full(len(ring), center))))
    if centers:
        positions = np.vstack([positions, centers])
        faces = np.vstack([faces, *cap_faces])
    # Keep ancestry through booleans so artificial caps over an authored eye
    # opening can be removed without discarding the actual eyelid or iris.
    face_ids = np.arange(len(faces), dtype=np.uint32)
    def make_volume(tris):
        return manifold.Manifold(manifold.Mesh(
            np.ascontiguousarray(positions, np.float32), np.ascontiguousarray(tris, np.uint32),
            run_index=np.array([0, 3 * len(tris)], np.uint32),
            run_original_id=np.array([source_id], np.uint32), face_id=face_ids))
    volume = make_volume(faces)
    if volume.status() != manifold.Error.NoError:
        raise ValueError(f"Observed head cannot form a closed solid: {volume.status()}")
    if volume.volume() < 0:
        volume = make_volume(faces[:, ::-1])
    eye_caps = {}
    offset = len(source_face_ids)
    for loop in loops:
        points = positions[np.asarray(loop)]
        for side in eye_parts:
            center = np.asarray(marks["eye_" + side])
            width = abs(marks[f"eye_{side}_inner"][0] - marks[f"eye_{side}_outer"][0])
            if (np.linalg.norm(points.mean(0) - center) < width * 0.8
                    and np.ptp(points[:, 0]) < 1.8 * width):
                eye_caps[side] = np.arange(offset, offset + len(points))
        offset += len(points)
    surface = dict(source_id=source_id, head_faces=source_face_ids, eye_parts=eye_parts, eye_caps=eye_caps)
    return volume, surface, {
        "method": "observed_face_component",
        "components": count,
        "face_anchor_votes": int(votes[winner]),
        "face_anchor_distance_mm": (np.sqrt(distances) * 1000).tolist(),
        "source_vertices": len(used),
        "capped_boundaries": list(map(len, loops)),
    }


def observed_neck(resolved):
    point = resolved["landmarks"].get("neck")
    if point is None:
        point = resolved["joint_section_depth"].get("neck", {}).get("point")
    if point is None:
        raise ValueError("Missing measured neck in the transfer guide")
    return point


def collar_cut(body_ring, resolved):
    # Seat the body port below the observed neck.
    point = observed_neck(resolved)
    spacing = np.linalg.norm(body_ring - np.roll(body_ring, 1, axis=0), axis=1)
    plane = float(point[1]) + np.median(spacing)
    return float(plane)


def stitch_neck(vertices, upper, lower):
    """Connect corresponding distances along the two oriented collar loops."""
    def perimeter(ids):
        p = vertices[ids]
        edges = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        return np.cumsum(edges) / edges.sum()

    a, b = perimeter(upper), perimeter(lower)
    i = j = 0
    triangles = []
    while i < len(a) or j < len(b):
        u, l = upper[i % len(a)], lower[j % len(b)]
        if j == len(b) or (i < len(a) and a[i] < b[j]):
            i += 1
            triangles.append([u, upper[i % len(a)], l])
        else:
            j += 1
            triangles.append([u, lower[j % len(b)], l])
    return triangles


def fair_collar_band(body, faces, collar, weld, radius):
    """Relax the fitted neck between a fixed seam and the surrounding shoulders."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import dijkstra
    from scipy.sparse.linalg import spsolve

    _, first = np.unique(weld, return_index=True)
    points = body[first].astype(float)
    tri = weld[faces]
    edges = np.unique(np.sort(tri[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    i, j = np.vstack((edges, edges[:, ::-1])).T
    length = np.linalg.norm(points[i] - points[j], axis=1)
    shape = (len(points), len(points))
    graph = coo_matrix((length, (i, j)), shape=shape).tocsr()
    distance = dijkstra(graph, indices=weld[collar], min_only=True, limit=radius)
    free = (distance > 0) & (distance < radius)
    if not free.any():
        return body
    # Positive weights and fixed ports keep folded interior rings from
    # projecting back onto a collar triangle or dragging the shoulder.
    weights = coo_matrix((np.ones(len(i)), (i, j)), shape=shape).tocsr()
    lap = diags(weights.sum(axis=1).A1) - weights
    points[free] = spsolve(lap[free][:, free], -(lap[free][:, ~free] @ points[~free]))
    result = body.copy()
    result[free[weld]] = points[weld[free[weld]]]
    return result


def follow_neck_surface(body, faces, collar, head_ring, weld, source, source_faces, resolved):
    """Seat the fitted body's collar under the observed neck, blending into adjacent body rings."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    def perimeter(points):
        edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        return np.r_[0., np.cumsum(edges)] / edges.sum()

    body_ring = body[collar].astype(float)
    perimeter_body = perimeter(body_ring)[:-1]
    perimeter_head = perimeter(head_ring)
    closed = np.vstack([head_ring, head_ring[0]])
    target = np.column_stack([np.interp(perimeter_body, perimeter_head, axis)
                              for axis in closed.T])
    edge = np.median(np.linalg.norm(np.roll(target, -1, axis=0) - target, axis=1))
    target[:, 1] -= edge
    # Interpolate displacements over geodesic distances. Nearby arms or other
    # disconnected surfaces must not move with the collar.
    pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ends, keep = np.unique(np.sort(weld[pairs], axis=1), axis=0, return_index=True)
    lengths = np.linalg.norm(body[pairs[keep, 0]] - body[pairs[keep, 1]], axis=1)
    count = int(weld.max()) + 1
    graph = coo_matrix((lengths, ends.T), shape=(count, count)).tocsr()
    radius = max(np.ptp(target[:, 0]), np.ptp(target[:, 2]))
    distances = dijkstra(graph, directed=False, indices=weld[collar], limit=radius)[:, weld]
    influence = np.maximum(1. - distances / radius, 0.)
    influence = influence**2 * (3. - 2. * influence)
    weights = influence / np.maximum(distances, edge * 0.1)**2
    delta = weights.T @ (target - body_ring) / np.maximum(weights.sum(0)[:, None], 1.e-20)
    result = body + delta * influence.max(0)[:, None]
    for i, point in zip(collar, target):
        result[weld == weld[i]] = point
    # A registered port alone does not recover the skin below it. Follow the
    # observed cut surface down towards the shoulders while it still exists.
    def station(name):
        return resolved["landmarks"].get(name, resolved["joint_section_depth"].get(name, {}).get("point"))
    shoulders = [station("shoulder_" + side) for side in ("l", "r")]
    stats = {"observed_vertices": 0}
    if source_faces is None or any(q is None for q in shoulders):
        return result.astype(np.float32), stats
    shoulder = np.mean(shoulders, axis=0)
    top = head_ring.mean(0)
    span = top[1] - shoulder[1]
    if span <= edge:
        return result.astype(np.float32), stats
    near = (distances.min(0) <= radius) & (result[:, 1] >= shoulder[1]) & (result[:, 1] <= top[1])
    ids = np.flatnonzero(near)
    height = np.clip((result[ids, 1] - shoulder[1]) / span, 0., 1.)
    center = shoulder + height[:, None] * (top - shoulder)
    direction = result[ids] - center
    direction[:, 1] = 0.
    length = np.linalg.norm(direction, axis=1)
    direction /= np.maximum(length[:, None], 1.e-12)
    # Start outside the reference head, so a mouth/throat lining inside the
    # neck cannot win ahead of the exterior skin.
    reach = np.linalg.norm(np.ptp(source[source_faces], axis=(0, 1)))
    origin = center + reach * direction
    tree = igl.AABB()
    tree.init(source, source_faces)
    hit, distance, _ = tree.intersect_ray_first(source, source_faces, origin, -direction)
    valid = (hit >= 0) & (distance < reach) & (length > 0)
    # Reject hits past a shoulder (the silhouette of a larger character can
    # cross this plane) and stop when the reference head no longer spans it.
    valid &= reach - distance < np.linalg.norm(np.subtract(*shoulders)) * 0.5
    weight = height**2 * (3. - 2. * height) * influence.max(0)[ids]
    q = ids[valid]
    move = np.zeros_like(result)
    move[q] = (weight[valid] * (reach - distance[valid] - length[valid]))[:, None] * direction[valid]
    # Reduce local projection where it would turn over a fitted triangle.
    # One factor per welded vertex keeps UV duplicates and the port coherent.
    active = faces[np.any(np.linalg.norm(move, axis=1)[faces] > 0., axis=1)]
    before = np.cross(result[active[:, 1]] - result[active[:, 0]], result[active[:, 2]] - result[active[:, 0]])
    area2 = np.einsum("ij,ij->i", before, before)
    scale = np.ones(count)
    for _ in range(24):
        trial = (result + move * scale[weld, None]).astype(np.float32)
        q0 = trial[active].astype(float)
        after = np.cross(q0[:, 1] - q0[:, 0], q0[:, 2] - q0[:, 0])
        bad = (area2 > 1.e-20) & (np.einsum("ij,ij->i", before, after) < area2 * 0.25)
        if not bad.any():
            stats["observed_vertices"] = int(np.count_nonzero(np.linalg.norm(trial - result, axis=1) > 1.e-7))
            return trial, stats
        scale[np.unique(weld[active[bad]])] *= 0.5
    return result.astype(np.float32), stats
