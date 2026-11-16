"""Assign scan triangles to reviewed skin and garment regions on calibrated images."""

from collections import defaultdict

import cv2
import igl
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .guide import project, rasterize


def segment(scan, guide, cams):
    names = ["skin"] + [p["name"] for p in guide.get("garments", [])]
    if len(set(names)) != len(names):
        raise ValueError("Each garment needs its own name")
    owners = np.full(len(scan.faces), -1, dtype=np.int32)
    strengths = np.zeros(len(scan.faces), dtype=np.float32)
    by_view = defaultdict(list)
    for label, (regions, strokes) in enumerate(
        [(guide.get("skin", []), guide.get("skin_strokes", []))]
        + [
            (p.get("regions", []), p.get("strokes", []))
            for p in guide.get("garments", [])
        ]
    ):
        for shape in [*regions, *strokes]:
            by_view[shape["view"]].append((label, shape))
    for name, shapes in by_view.items():
        cam = cams[name]
        ids, _ = rasterize(
            project(cam, scan.vertices), scan.faces, cam["width"], cam["height"]
        )
        for label, shape in shapes:
            mask = np.zeros(ids.shape, dtype=np.uint8)
            if "polygon" in shape:
                cv2.fillPoly(mask, [np.rint(shape["polygon"]).astype(np.int32)], 1)
                depth = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            else:
                pts = np.rint(shape["points"]).astype(np.int32)
                cv2.polylines(mask, [pts], False, 1, max(1, round(shape["width_px"])))
                depth = mask * 1000.0
            use = (mask != 0) & (ids >= 0)
            seen = ids[use]
            if not len(seen):
                continue
            evidence = np.zeros(len(owners), dtype=np.float32)
            np.maximum.at(evidence, seen, depth[use])
            stronger = evidence > strengths
            strengths[stronger] = evidence[stronger]
            owners[stronger] = label
    if len(names) == 1 and not by_view:
        return names, np.zeros(len(owners), dtype=np.int32)
    seeds = np.flatnonzero(owners >= 0)
    if not len(seeds):
        raise ValueError("Skin and garment annotations must hit the scan")
    # Preserving source triangle order lets reviewed parts be reused on a later build.
    _, weld = np.unique(np.round(scan.vertices, 7), axis=0, return_inverse=True)
    neighbors, _ = igl.triangle_triangle_adjacency(weld[scan.faces])
    rows, edges = np.where(neighbors >= 0)
    cols = neighbors[rows, edges]
    centers = scan.vertices[scan.faces].mean(1)
    rgb = scan.colors(np.arange(len(owners)), np.full((len(owners), 3), 1 / 3))
    lab = cv2.cvtColor(
        rgb.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3)
    jump = np.linalg.norm(lab[rows] - lab[cols], axis=1)
    distance = np.linalg.norm(centers[rows] - centers[cols], axis=1)
    graph = csr_matrix(
        (np.maximum(distance, 1e-12) * (1 + jump**2), (rows, cols)),
        shape=(len(owners),) * 2,
    )
    length, _, sources = dijkstra(
        graph, indices=seeds, min_only=True, return_predecessors=True
    )
    reached = np.isfinite(length)
    owners[reached] = owners[sources[reached]]
    return names, owners


def reviewed_owners(
    scan, owners, names, review_input, saved_evidence, output, vertices
):
    """Resolve disconnected components from agent decisions bound to source face indices."""
    import hashlib
    import json
    from scipy.sparse.csgraph import connected_components

    candidates = {}
    choices = {}
    for source, field in ((saved_evidence, "candidates"), (review_input, "decisions")):
        if source is None:
            continue
        doc = json.loads(source.read_text())
        if doc["source_hash"] != scan.source_hash:
            raise ValueError(f"Component review belongs to another scan: {source}")
        if field == "decisions":
            choices = doc[field]
        else:
            candidates = {c["id"]: c for c in doc[field]}

    def identity(rows):
        return hashlib.sha256(np.asarray(rows, dtype="<u8").tobytes()).hexdigest()

    def apply(rows, choice):
        action = choice["action"]
        if action == "piece":
            label = names.index(choice["name"])
        elif action in ("skin", "brow"):
            label = 0
        elif action == "internal":
            label = len(names)
        else:
            raise ValueError(f"Unknown component decision {action!r}")
        owners[rows] = label

    brow = []
    for key in choices.keys() & candidates.keys():
        ranges = candidates[key]["triangle_ranges"]
        if not all(0 <= a < b <= len(owners) for a, b in ranges):
            raise ValueError("Reviewed triangle range outside scan")
        rows = np.concatenate([np.arange(a, b) for a, b in ranges])
        if identity(rows) != key:
            raise ValueError(f"Review triangle hash mismatch for {key}")
        choice = choices[key]
        apply(rows, choice)
        if choice["action"] == "brow":
            brow.extend(ranges)

    pending = np.flatnonzero(owners < 0)
    if len(pending):
        _, weld = np.unique(np.round(scan.vertices, 7), axis=0, return_inverse=True)
        neighbors, _ = igl.triangle_triangle_adjacency(weld[scan.faces[pending]])
        rows, edges = np.where(neighbors >= 0)
        graph = csr_matrix(
            (np.ones(len(rows), bool), (rows, neighbors[rows, edges])),
            shape=(len(pending),) * 2,
        )
        count, groups = connected_components(graph, directed=False)
        for component in range(count):
            ids = pending[groups == component]
            key = identity(ids)
            splits = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)
            ranges = [[int(block[0]), int(block[-1] + 1)] for block in splits]
            coords = vertices[scan.faces[ids]].reshape(-1, 3)
            candidates[key] = dict(
                id=key,
                reason="unmarked_disconnected_surface",
                triangle_ranges=ranges,
                bounds_m=[coords.min(0).tolist(), coords.max(0).tolist()],
                status="uncertain",
            )
            if key in choices:
                apply(ids, choices[key])
                if choices[key]["action"] == "brow":
                    brow.extend(ranges)
    unknown = set(choices) - set(candidates)
    if unknown:
        raise ValueError(
            f"Missing source triangle evidence for reviewed components {sorted(unknown)}"
        )
    for key, choice in choices.items():
        candidates[key].update(
            status=choice["action"], assigned_piece=choice.get("name")
        )
    data = dict(
        source_hash=scan.source_hash,
        pieces=names[1:],
        candidates=list(candidates.values()),
        range_convention="half-open source triangle ranges",
    )
    if (owners < 0).any():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2))
        raise ValueError(
            f"Review unassigned components in {output} and return --segment-review decisions"
        )
    return data, brow
