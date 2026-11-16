"""Rig-region weights and surface-aware finger pose limits used by animation recipes."""

import numpy as np
from scipy.spatial.transform import Rotation

from .repair import append_attr
from .surface import Surface


def smooth(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def store_weights(glb, part, weights):
    count = part.palette.shape[1]
    indices = np.argsort(-weights, axis=1, kind="stable")[:, :count]
    packed = np.take_along_axis(weights, indices, axis=1)
    packed /= np.maximum(packed.sum(1, keepdims=True), 1e-15)
    pad = count - indices.shape[1]
    if pad:
        indices = np.pad(indices, ((0, 0), (0, pad)))
        packed = np.pad(packed, ((0, 0), (0, pad)))
    attrs = part.prim["attributes"]
    for i in range(0, count, 4):
        attrs[f"JOINTS_{i // 4}"] = append_attr(
            glb, indices[:, i : i + 4].astype("<u2"), 5123
        )
        attrs[f"WEIGHTS_{i // 4}"] = append_attr(
            glb, packed[:, i : i + 4].astype("<f4"), 5126
        )


def prepare_regions(model, recipe):
    surface = Surface(model)
    scene = surface.scene
    edited = False
    for part in surface.parts:
        if part.weights is None:
            continue
        dense = np.zeros((len(part.positions), len(part.bones)))
        np.add.at(dense, (np.arange(len(dense))[:, None], part.palette), part.weights)
        original = dense.copy()
        slots = {scene.names[b]: k for k, b in enumerate(part.bones)}
        centers = np.linalg.inv(part.inverse)[:, :3, 3]
        for config in recipe.get("rigid_regions", []):
            if (
                not {config["joint"], config["end"], *config["influences"]}
                <= slots.keys()
            ):
                continue
            owner, end = slots[config["joint"]], slots[config["end"]]
            chosen = [slots[n] for n in config["influences"]]
            if owner not in chosen:
                raise ValueError(
                    "The owner must be included in a rigid region's influences"
                )
            segment = centers[end] - centers[owner]
            offset = part.positions - centers[owner]
            t = offset @ segment / (segment @ segment)
            r = np.linalg.norm(offset - t[:, None] * segment, axis=1)
            mass = dense[:, chosen].sum(1)
            lo, hi = config["interval"]
            left, right = config["feather"]
            eligible = mass >= config["minimum_region_weight"]
            inside = eligible & (t >= lo) & (t <= hi)
            if not inside.any():
                continue
            radius = np.quantile(r[inside], config["radius_quantile"])
            ratio = (
                eligible
                * smooth((t - lo + left) / left)
                * smooth((hi + right - t) / right)
            )
            ratio *= smooth((1.15 * radius - r) / max(0.15 * radius, 1e-9)) * smooth(
                dense[:, owner] / 0.05
            )
            dense[:, chosen] *= 1.0 - ratio[:, None]
            dense[:, owner] += ratio * mass
        for spec in recipe.get("hand_shapes", []):
            if not spec.get("rebind_missing_digits") or spec["wrist"] not in slots:
                continue
            chains = [
                [slots[n] for n in d["joints"] if n in slots] for d in spec["digits"]
            ]
            missing = any(c and dense[:, c].sum() < 1e-8 for c in chains)
            if not missing:
                continue
            wrist = slots[spec["wrist"]]
            candidates = [q for chain in chains for q in chain[:-1]]
            region = [wrist, *{q for chain in chains for q in chain}]
            mass = dense[:, region].sum(1)
            distances = []
            for chain in chains:
                for a, b in zip(chain, chain[1:]):
                    delta = centers[b] - centers[a]
                    t = np.clip(
                        (part.positions - centers[a])
                        @ delta
                        / max(delta @ delta, 1e-15),
                        0,
                        1,
                    )
                    distances.append(
                        np.linalg.norm(
                            part.positions - centers[a] - t[:, None] * delta, axis=1
                        )
                    )
            if not distances:
                raise ValueError(
                    "A digit needs at least two bound joints for repartition"
                )
            distances = np.column_stack(
                distances + [np.linalg.norm(part.positions - centers[wrist], axis=1)]
            )
            scores = 1.0 / np.maximum(distances, 1e-5) ** 4
            scores /= scores.sum(1, keepdims=True)
            dense[:, region] = 0.0
            dense[:, [*candidates, wrist]] += mass[:, None] * scores
        from .hand_surface import repair_weights

        for hand in recipe.get("hand_shapes", []):
            if hand.get("surface_repair"):
                repair_weights(surface, part, dense, hand)
        if not np.allclose(original, dense, atol=1e-10):
            store_weights(scene.glb, part, dense)
            edited = True
    if edited:
        scene.glb.doc["buffers"][0]["byteLength"] = len(scene.glb.blob)
        scene.glb.write(model)


class HandGuard:
    def __init__(self, surface, config):
        self.surface = surface
        scene = surface.scene
        self.config = config
        self.chains = []
        self.peak_violations = 0
        for spec in config["digits"]:
            chain = [scene.joint(n) for n in spec["joints"]]
            if len(chain) < 2 or any(
                scene.parents.get(c) != p
                for p, c in zip([scene.joint(config["wrist"]), *chain], chain)
            ):
                raise ValueError(f"Invalid digit chain {spec['name']}")
            regions = []
            for i, part in enumerate(surface.parts):
                if part.weights is None:
                    continue
                selected = (
                    part.weights * np.isin(part.bones[part.palette], chain)
                ).sum(1) > 0.05
                edges = part.edges[selected[part.edges].any(1)]
                if not len(edges):
                    continue
                ids, edge = np.unique(edges, return_inverse=True)
                edge = edge.reshape(-1, 2)
                lengths = np.linalg.norm(
                    np.diff(surface.rest[i][ids][edge], axis=1)[:, 0], axis=1
                )
                maxlen = lengths * config.get("max_edge_ratio", 1.2) + config.get(
                    "max_edge_growth_m", 0.004
                )
                regions.append((part, ids, edge, maxlen))
            self.chains.append((chain, regions))

    def apply(self, pose, shapes):
        scene = self.surface.scene
        for chain, regions in self.chains:
            pose[chain, :3] = scene.rest[chain, :3]
            pose[chain, 7:] = scene.rest[chain, 7:]
            rest = Rotation.from_quat(scene.rest[chain, 3:7])
            delta = (rest.inv() * Rotation.from_quat(pose[chain, 3:7])).as_rotvec()
            fractions = (
                [1.0, 0.75, 0.5, 0.25, 0.0]
                if self.config.get("allow_rotation_fallback", True)
                else [1.0]
            )
            for amount in fractions:
                pose[chain, 3:7] = (
                    rest * Rotation.from_rotvec(delta * amount)
                ).as_quat()
                world = scene.world(pose)
                violations = self.violations(world, shapes, regions)
                if not violations:
                    break
            self.peak_violations = max(self.peak_violations, violations)

    def violations(self, world, shapes, regions):
        scene = self.surface.scene
        violations = 0
        for part, ids, edge, maxlen in regions:
            weights = shapes.get(
                part.node, scene.morph_rest(part.node) if part.deltas else []
            )
            points = part.points(world, weights, ids)
            lengths = np.linalg.norm(np.diff(points[edge], axis=1)[:, 0], axis=1)
            violations += int((lengths > maxlen + 1e-9).sum())
        return violations

    def measure(self, clip):
        scene = self.surface.scene
        times = np.unique(np.concatenate([v[0] for v in scene.clips[clip].values()]))
        peak = 0
        for t in times:
            pose, shapes = scene.sample(clip, t)
            world = scene.world(pose)
            for _, regions in self.chains:
                peak = max(peak, self.violations(world, shapes, regions))
        return peak
