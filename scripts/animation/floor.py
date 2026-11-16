"""Keep the measured skinned support region above a world-space floor."""

import numpy as np


class SurfaceFloor:
    def __init__(self, surface, options):
        self.scene = surface.scene
        self.config = options
        self.root = self.scene.joint(options["root"])
        joints = [self.scene.joint(n) for n in options["support_joints"]]
        descendants = {self.root}
        for i in self.scene.order:
            if self.scene.parents.get(i) in descendants:
                descendants.add(i)
        if not set(joints) <= descendants or options["max_lift_m"] <= 0:
            raise ValueError("Invalid support region or floor lift limit")
        self.regions = []
        for part in surface.parts:
            if part.weights is None:
                continue
            bones = part.bones[part.palette]
            mask = (part.weights * np.isin(bones, joints)).sum(1) >= 0.5
            if np.any(mask):
                if np.any(
                    (part.weights[mask] > 0) & ~np.isin(bones[mask], list(descendants))
                ):
                    raise ValueError("Support surface spans independent roots")
                self.regions.append((part, np.flatnonzero(mask)))
        if not self.regions:
            raise ValueError("No weighted support surface for the selected joints")
        if options.get("preserve_source_height") and options.get("align_to_ground"):
            raise ValueError("Incompatible floor modes")

    def height(self, pose, shapes):
        w = self.scene.world(pose)
        return min(
            part.points(
                w,
                shapes.get(
                    part.node, self.scene.morph_rest(part.node) if part.deltas else []
                ),
                ids,
            )[:, 1].min()
            for part, ids in self.regions
        )

    def apply(self, pose, shapes, original):
        current = self.height(pose, shapes)
        desired = self.config["height_m"]
        if original is not None:
            desired = max(original, desired)
        delta = desired - current
        if original is None and not self.config.get("align_to_ground"):
            delta = max(0.0, delta)
        if abs(delta) > self.config["max_lift_m"]:
            raise ValueError(
                f"Required floor correction {delta:.4f} m exceeds the allowed travel"
            )
        translation = np.array([0.0, delta, 0.0])
        if self.root in self.scene.parents:
            parent = self.scene.parents[self.root]
            translation = np.linalg.solve(
                self.scene.world(pose)[parent, :3, :3], translation
            )
        pose[self.root, :3] += translation
        return float(delta)
