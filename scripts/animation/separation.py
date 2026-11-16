"""Adjust existing skin influences on sampled triangle crossings."""

import numpy as np
from scipy.sparse import csr_matrix


def constraints(surface, contact):
    # Capture all offending pairs, including pairs omitted from a short human report.
    frames = surface.contacts({**contact, "maximum_reported_pairs": None})["frames"]
    constraints_by_part = [[] for _ in surface.parts]
    for frame in frames:
        pose, shapes = surface.scene.sample(contact["clip"], frame["time"])
        points = surface.evaluate(pose, shapes)
        world = surface.scene.world(pose)
        for pair in frame["pairs"]:
            for (moving, face), (other, barrier) in (pair, reversed(pair)):
                part = surface.parts[moving]
                if part.weights is None:
                    continue
                ids = part.faces[face]
                obstacle = surface.parts[other].faces[barrier]
                ref = surface.rest[other][obstacle]
                now = points[other][obstacle]
                rest_normal = np.cross(ref[1] - ref[0], ref[2] - ref[0])
                side = np.sign(
                    (surface.rest[moving][ids].mean(0) - ref[0]) @ rest_normal
                )
                normal = side * np.cross(now[1] - now[0], now[2] - now[0])
                magnitude = np.linalg.norm(normal)
                if magnitude < 1e-14:
                    continue
                normal /= magnitude
                local = part.positions[ids].copy()
                for weight, delta in zip(
                    shapes.get(
                        part.node,
                        surface.scene.morph_rest(part.node) if part.deltas else [],
                    ),
                    part.deltas,
                ):
                    local += weight * delta[ids]
                skin = world[part.bones] @ part.inverse
                distances = np.einsum("bi,vi->vb", normal @ skin[:, :3, :3], local)
                distances += (skin[:, :3, 3] - now[0]) @ normal
                constraints_by_part[moving].extend(zip(ids, distances))
    return constraints_by_part


def apply(part, weights, rows, editable, protected, plan, margin):
    initial = weights.copy()
    radius = min(
        plan.get("contact_weight_change_l1", 0.05),
        plan.get("max_weight_change_l1", 0.35),
    )
    smoothing = plan.get("contact_smoothing", 0.5)
    if not 0 < radius <= 2 or not 0 <= smoothing <= 1:
        raise ValueError("Invalid surface separation weight budget or smoothing")
    a, b = part.edges.T
    adjacency = csr_matrix(
        (np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])),
        shape=(len(weights), len(weights)),
    )
    degree = np.maximum(np.asarray(adjacency.sum(1)), 1)
    # Keep original bone slots, protected joints and the total weight at each vertex.
    for _ in range(plan.get("iterations", 32)):
        update = np.zeros_like(weights)
        for vertex, distances in rows:
            if not editable[vertex]:
                continue
            gap = margin - distances @ weights[vertex]
            allowed = (initial[vertex] > 0) & ~protected
            if gap <= 0 or allowed.sum() < 2:
                continue
            gradient = distances[allowed] - distances[allowed].mean()
            denom = gradient @ gradient
            if denom > 1e-18:
                update[vertex, allowed] += gap * gradient / denom
        if not np.any(update):
            break
        if smoothing:
            update += smoothing * (adjacency @ update / degree - update)
        allowed = (initial > 0) & ~protected & editable[:, None]
        update *= allowed
        update -= allowed * (update.sum(1) / np.maximum(allowed.sum(1), 1))[:, None]
        bound = np.min(
            np.divide(
                -weights, update, out=np.full_like(update, np.inf), where=update < 0
            ),
            axis=1,
        )
        delta = weights - initial + update * np.minimum(bound, 1)[:, None]
        weights[:] = (
            initial
            + delta
            * np.minimum(1, radius / np.maximum(np.abs(delta).sum(1), 1e-15))[:, None]
        )
