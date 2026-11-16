"""Evaluate skinned surfaces and track seam separation, edge strain and intersections."""

import hashlib

import numpy as np

from .scene import Scene


class Primitive:
    def __init__(self, scene, node, prim):
        self.node, self.prim = node, prim
        g = scene.glb
        attrs = prim["attributes"]
        self.positions = g.array(attrs["POSITION"]).astype(float)
        self.faces = (
            (
                g.array(prim["indices"]).ravel()
                if "indices" in prim
                else np.arange(len(self.positions))
            )
            .astype(np.int64)
            .reshape(-1, 3)
        )
        self.deltas = [
            g.array(t["POSITION"]).astype(float)
            if "POSITION" in t
            else np.zeros_like(self.positions)
            for t in prim.get("targets", [])
        ]
        self.weights = self.palette = self.inverse = None
        n = g.doc["nodes"][node]
        if "skin" in n:
            skin = g.doc["skins"][n["skin"]]
            self.bones = np.array(skin["joints"])
            self.inverse = (
                g.array(skin["inverseBindMatrices"])
                .reshape(-1, 4, 4)
                .transpose(0, 2, 1)
                if "inverseBindMatrices" in skin
                else np.repeat(np.eye(4)[None], len(self.bones), axis=0)
            )
            slots = sorted(k[7:] for k in attrs if k.startswith("JOINTS_"))
            self.palette = np.column_stack(
                [g.array(attrs["JOINTS_" + k]) for k in slots]
            ).astype(int)
            weights = []
            for slot in slots:
                acc = attrs["WEIGHTS_" + slot]
                raw = g.array(acc)
                weights.append(
                    raw.astype(float) / np.iinfo(raw.dtype).max
                    if raw.dtype.kind in "iu"
                    else raw
                )
            self.weights = np.column_stack(weights)
            self.weights = self.weights / np.maximum(
                self.weights.sum(1)[:, None], 1e-20
            )
        self.edges = np.unique(
            np.sort(
                np.concatenate(
                    [
                        self.faces[:, [0, 1]],
                        self.faces[:, [1, 2]],
                        self.faces[:, [2, 0]],
                    ]
                ),
                axis=1,
            ),
            axis=0,
        )
        h = hashlib.sha256()
        for arr, typ in ((self.positions, "<f8"), (self.faces, "<u4")):
            h.update(len(arr).to_bytes(8, "little"))
            h.update(arr.astype(typ).tobytes())
        self.hash = h.hexdigest()
        self.seams = self.seam_pairs()

    def seam_pairs(self):
        # Only opposing, two-sided geometric edges establish a shared surface.
        oriented = np.concatenate(
            [self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]
        )
        _, weld = np.unique(self.positions, axis=0, return_inverse=True)
        geo = weld[oriented]
        unordered = np.sort(geo, axis=1)
        order = np.lexsort(unordered.T[::-1])
        sorted_edges = unordered[order]
        start = np.r_[
            0, np.flatnonzero(np.any(sorted_edges[:-1] != sorted_edges[1:], axis=1)) + 1
        ]
        two = start[np.diff(np.r_[start, len(order)]) == 2]
        a, b = order[two], order[two + 1]
        opposite = np.all(geo[a] == geo[b, ::-1], axis=1)
        pairs = np.stack(
            [oriented[a[opposite]], oriented[b[opposite], ::-1]], axis=-1
        ).reshape(-1, 2)
        return np.unique(np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1), axis=0)

    def points(self, world, weights, ids=None):
        source = self.positions if ids is None else self.positions[ids]
        for weight, delta in zip(weights, self.deltas):
            if weight:
                source = source + weight * (delta if ids is None else delta[ids])
        if self.weights is None:
            return source @ world[self.node, :3, :3].T + world[self.node, :3, 3]
        ji = self.palette if ids is None else self.palette[ids]
        ws = self.weights if ids is None else self.weights[ids]
        skin = world[self.bones] @ self.inverse
        out = np.zeros_like(source)
        for k in range(ji.shape[1]):
            matrix = skin[ji[:, k]]
            moved = (
                np.einsum("ijk,ik->ij", matrix[:, :3, :3], source) + matrix[:, :3, 3]
            )
            out += moved * ws[:, k, None]
        return out


class Surface:
    def __init__(self, model):
        self.scene = Scene(model)
        g = self.scene.glb
        self.parts = [
            Primitive(self.scene, i, p)
            for i, n in enumerate(g.doc["nodes"])
            if "mesh" in n
            for p in g.doc["meshes"][n["mesh"]]["primitives"]
            if p.get("mode", 4) == 4
        ]
        self.rest = self.evaluate(self.scene.rest, {})

    def evaluate(self, pose, shapes):
        world = self.scene.world(pose)
        return [
            part.points(
                world,
                shapes.get(
                    part.node, self.scene.morph_rest(part.node) if part.deltas else []
                ),
            )
            for part in self.parts
        ]

    def mask(self, spec):
        part = self.parts[spec["mesh"]]
        ids = np.asarray(spec["vertices"], int)
        if (
            not len(ids)
            or spec["geometry_hash"] != part.hash
            or min(ids) < 0
            or max(ids) >= len(part.positions)
        ):
            raise ValueError("Surface mask geometry or vertex indices do not match")
        return ids

    def analyze(self, config):
        clips = config.get("clips") or list(self.scene.clips)
        samples = config.get("samples_per_clip", 33)
        if samples < 1:
            raise ValueError("samples_per_clip must be positive")
        peak_growth = peak_seam = 0.0
        worst = None
        baselines = [
            np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1)
            for p, e in zip(self.rest, (m.edges for m in self.parts))
        ]
        count = 0
        for clip in clips or [None]:
            times = (
                np.unique(
                    np.r_[
                        np.linspace(*self.scene.interval(clip), samples),
                        config.get("extra_times", []),
                    ]
                )
                if clip
                else [0.0]
            )
            for t in times:
                points = (
                    self.evaluate(*self.scene.sample(clip, t)) if clip else self.rest
                )
                count += 1
                for idx, (part, p, baseline) in enumerate(
                    zip(self.parts, points, baselines)
                ):
                    e = part.edges
                    growth = np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1) - baseline
                    if len(growth) and growth.max() > peak_growth:
                        i = int(growth.argmax())
                        peak_growth = float(growth[i])
                        worst = dict(
                            mesh=idx, vertices=e[i].tolist(), clip=clip, time=float(t)
                        )
                    if len(part.seams):
                        a, b = part.seams.T
                        peak_seam = max(
                            peak_seam,
                            float(np.max(np.linalg.norm(p[a] - p[b], axis=1))),
                        )
        return dict(
            samples=count,
            max_edge_growth_m=peak_growth,
            max_seam_gap_m=peak_seam,
            worst_edge=worst,
            meshes=[
                dict(
                    mesh=i,
                    node=self.scene.names[p.node],
                    vertices=len(p.positions),
                    geometry_hash=p.hash,
                    seam_pairs=len(p.seams),
                )
                for i, p in enumerate(self.parts)
            ],
        )

    def contacts(self, config):
        from igl.copyleft import cgal

        masks = config["regions"]
        if not masks:
            raise ValueError("Select at least one surface region")
        faces = []
        selected = []
        labels = []
        base = 0
        for i, part in enumerate(self.parts):
            ids = [self.mask(m) for m in masks if m["mesh"] == i]
            faces.append(part.faces + base)
            selected.extend(
                np.isin(part.faces, np.concatenate(ids)).any(1)
                if ids
                else np.zeros(len(part.faces), bool)
            )
            labels.extend((i, j) for j in range(len(part.faces)))
            base += len(part.positions)
        triangles = np.vstack(faces)
        selected = np.asarray(selected)
        eps = config.get("epsilon_m", 1e-6)

        def pairs(points):
            vertices = np.vstack(points)
            _, _, intersections, _, _ = cgal.remesh_self_intersections(
                vertices, triangles, detect_only=True
            )
            if not len(intersections):
                return set()
            ii = intersections[selected[intersections].any(1)]
            # Only transverse interior crossings, excluding coplanar overlap or
            # shared endpoints. Require both triangles to span the other's plane.
            a, b = (vertices[triangles[ii[:, k]]] for k in range(2))
            keep = np.ones(len(ii), bool)
            for test, other in ((a, b), (b, a)):
                n = np.cross(test[:, 1] - test[:, 0], test[:, 2] - test[:, 0])
                n /= np.maximum(1e-30, np.linalg.norm(n, axis=1))[:, None]
                distances = np.einsum("ijk,ik->ij", other - test[:, :1], n)
                keep &= (distances.min(1) < -eps) & (distances.max(1) > eps)
            return {tuple(sorted((labels[i], labels[j]))) for i, j in ii[keep]}

        baseline = pairs(self.rest)
        frames = []
        for t in config.get("times", []):
            points = self.evaluate(*self.scene.sample(config["clip"], t))
            current = pairs(points)
            new = sorted(current - baseline)
            frames.append(
                dict(
                    time=t,
                    crossings=len(current),
                    new_crossings=len(new),
                    pairs=new[: config.get("maximum_reported_pairs", 200)],
                )
            )
        return dict(rest_crossings=len(baseline), frames=frames)
