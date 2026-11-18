"""Refine flared sleeve weights using the connected surface between cuffs and hem.

Only NumPy is needed so the scan worker and the standalone Blender extension
can run the same binding on an aligned rest mesh.
"""

from heapq import heapify, heappop, heappush

import numpy as np

ARM = ("upperarm01", "upperarm02", "lowerarm01", "lowerarm02", "wrist")
SPINE = ("root", "spine04", "spine03", "spine02", "spine01")


def smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def openings(border):
    adjacent = {}
    for a, b in border:
        adjacent.setdefault(a, []).append(b)
        adjacent.setdefault(b, []).append(a)
    unseen = set(adjacent)
    while unseen:
        pending = [unseen.pop()]
        component = []
        while pending:
            current = pending.pop()
            component.append(current)
            for other in adjacent[current]:
                if other in unseen:
                    unseen.remove(other)
                    pending.append(other)
        yield np.asarray(component)


def surface_distance(neighbors, sources):
    distances = np.full(len(neighbors), np.inf)
    distances[sources] = 0
    pending = [(0.0, v) for v in sources]
    heapify(pending)
    while pending:
        distance, vertex = heappop(pending)
        if distance > distances[vertex]:
            continue
        for other, edge in neighbors[vertex]:
            candidate = distance + edge
            if candidate < distances[other]:
                distances[other] = candidate
                heappush(pending, (candidate, other))
    return distances


def diffuse(targets, pin, a, b, conductance):
    """Solve the pinned graph Laplacian with diagonal-preconditioned CG."""
    count = len(pin)
    diagonal = (
        pin
        + np.bincount(a, conductance, minlength=count)
        + np.bincount(b, conductance, minlength=count)
    )

    def multiply(x):
        return (
            diagonal * x
            - np.bincount(a, conductance * x[b], minlength=count)
            - np.bincount(b, conductance * x[a], minlength=count)
        )

    fields = []
    for target in targets:
        value = target.copy()
        residual = pin * target - multiply(value)
        direction = residual / diagonal
        energy = residual @ direction
        tolerance = max(energy * 1e-16, 1e-28)
        for _ in range(count):
            if energy <= tolerance:
                break
            product = multiply(direction)
            step = energy / (direction @ product)
            value += step * direction
            residual -= step * product
            correction = residual / diagonal
            next_energy = residual @ correction
            direction = correction + (next_energy / energy) * direction
            energy = next_energy
        fields.append(value)
    return fields


def joint_pair(knots, bones, coordinate, amount):
    k = np.clip(np.searchsorted(knots, coordinate) - 1, 0, len(knots) - 2)
    mix = np.clip((coordinate - knots[k]) / (knots[k + 1] - knots[k]), 0, 1)
    return np.column_stack((bones[k], bones[k + 1])), np.column_stack(
        (1 - mix, mix)
    ) * amount[:, None]


def sleeve_weights(points, faces, joints, weights, distance_to_body, names, centers):
    """Refine four body influences only on flared, cuff-to-hem components."""
    arms = [[f"{bone}.{side}" for bone in ARM] for side in ("L", "R")]
    if not len(faces) or not all(n in names for n in [*SPINE, *arms[0], *arms[1]]):
        return None
    arm_ids = np.array([[names.index(n) for n in limb] for limb in arms])
    spine_ids = np.array([names.index(n) for n in SPINE])
    vertices, first, inverse = np.unique(
        points, axis=0, return_index=True, return_inverse=True
    )
    triangles = inverse[faces]
    lateral = centers[arm_ids[0, 0]] - centers[arm_ids[1, 0]]
    width = np.linalg.norm(lateral)
    up = centers[spine_ids[-1]] - centers[spine_ids[0]]
    if min(width, np.linalg.norm(up)) < 1e-9:
        return None
    up /= np.linalg.norm(up)
    chest = centers[arm_ids[:, 0]].mean(0)
    x = (vertices - chest) @ (lateral / width)
    levels = centers[spine_ids] @ up
    if np.any(np.diff(levels) <= 0):
        return None
    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edges = np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    loops = list(openings(edges[counts == 1]))
    hems = [
        r
        for r in loops
        if x[r].min() < -width * 0.25
        and x[r].max() > width * 0.25
        and np.mean(vertices[r] @ up) < levels[-2]
    ]
    if not hems:
        return None
    a, b = edges.T
    lengths = np.linalg.norm(vertices[a] - vertices[b], axis=1)
    neighbors = [[] for _ in vertices]
    for i, j, length in zip(a, b, lengths):
        neighbors[i].append((j, length))
        neighbors[j].append((i, length))
    hem = surface_distance(neighbors, np.concatenate(hems))
    area = (
        np.linalg.norm(
            np.cross(
                vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
            ),
            axis=1,
        )
        / 6
    )
    mass = np.bincount(
        triangles.ravel(), weights=np.repeat(area, 3), minlength=len(vertices)
    )
    conductance = np.minimum(
        (np.sqrt(mass[a]) + np.sqrt(mass[b])) / (2 * np.maximum(lengths, 1e-12)), 8
    )
    result = None
    for sign, arm in zip((1, -1), arm_ids):
        shoulder, elbow, wrist = centers[arm[::2]]
        lower = wrist - elbow
        upper = elbow - shoulder
        length = np.linalg.norm(lower)
        if min(length, np.linalg.norm(upper)) < 1e-8:
            continue
        # A flared or oblique opening can extend past the wrist on one edge.
        # Locate it by its center and exclude boundaries spanning the elbow.
        rims = [
            r
            for r in loops
            if len(r) >= 6
            and np.linalg.norm(vertices[r].mean(0) - wrist) < length / 2
            and np.max(np.linalg.norm(vertices[r] - wrist, axis=1)) < length
        ]
        if not rims:
            continue
        cuff = surface_distance(neighbors, np.concatenate(rims))
        supported = np.isfinite(hem) & np.isfinite(cuff) & (x * sign >= 0)
        along_upper = np.clip((vertices - shoulder) @ upper / (upper @ upper), 0, 1)
        along_lower = np.clip((vertices - elbow) @ lower / (lower @ lower), 0, 1)
        upper_gap = np.linalg.norm(
            vertices - shoulder - along_upper[:, None] * upper, axis=1
        )
        lower_gap = np.linalg.norm(
            vertices - elbow - along_lower[:, None] * lower, axis=1
        )
        distal = supported & (cuff < hem) & (along_lower > 0.55)
        # Tight sleeves retain their original, more local body donors.
        if not distal.any() or np.quantile(lower_gap[distal], 0.9) < 0.48 * length:
            continue
        centerline = np.flatnonzero((x * sign < width * 0.28) & np.isfinite(cuff))
        core = np.minimum(surface_distance(neighbors, centerline), hem)
        fraction = np.zeros(len(vertices))
        fraction[supported] = smooth(
            (
                core[supported] / np.maximum(core[supported] + cuff[supported], 1e-12)
                - 0.12
            )
            / 0.60
        )
        contact = smooth(1 - distance_to_body[first] / (0.48 * length))
        donor = (weights[first] * np.isin(joints[first], arm)).sum(1)
        fraction = (1 - contact) * fraction + contact * donor
        fraction *= smooth(hem / (0.55 * length)) * smooth(x * sign / (0.34 * width))
        fraction = np.maximum(fraction, 1 - smooth(cuff / (0.85 * length)))
        fraction[~supported] = 0
        station = np.where(upper_gap < lower_gap, along_upper, 1 + along_lower)
        confidence = 0.35 + 5 * contact
        confidence[(hem < length * 0.15) | (cuff < length * 0.22)] = 35
        pin = np.maximum(mass, 1e-14) * confidence / (0.14 * width) ** 2
        # Smooth the arm share and joint coordinate over the same cloth surface.
        share, weighted_station = diffuse(
            [fraction, fraction * station], pin, a, b, conductance
        )
        station = weighted_station / np.maximum(share, 1e-12)
        cuff_anchor = 1 - smooth(cuff / (0.48 * length))
        share = (
            np.clip(share, 0, 1)
            * smooth(hem / (0.32 * length))
            * smooth(x * sign / (width * 0.25))
        )
        share = np.maximum(share, cuff_anchor)
        station = np.maximum(station, 2 * cuff_anchor)
        aj, aw = joint_pair(np.array([0.18, 0.68, 1.2, 1.75]), arm[:4], station, share)
        bj, bw = joint_pair(levels, spine_ids, vertices @ up, 1 - share)
        if result is None:
            result = joints.copy(), weights.copy()
        rows = supported[inverse]
        result[0][rows] = np.column_stack((aj, bj))[inverse[rows]]
        result[1][rows] = np.column_stack((aw, bw))[inverse[rows]]
    return result
