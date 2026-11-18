"""Inventory source components and replay hash-bound part assignments without fitting a body.

    PY review_parts.py SCAN.glb --guide GUIDE --out NEW_DIRECTORY
    PY review_parts.py SCAN.glb --guide GUIDE --out NEW_DIRECTORY \
        --evidence COMPONENTS.json --review DECISIONS.json

The first run writes source triangle ranges, isolated views and a decision
template. A component may contain several anatomical/material parts: use guide
polygons or reviewed subranges instead of assigning an entire mixed component.
"""

import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
import sys

import igl
import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from transfer.guide import camera, load_annotations, render
from transfer.scan import Scan
from transfer.segment import segment, reviewed_owners
from roster.mesh_output import normals


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def ranges(ids):
    return [[int(x[0]), int(x[-1]+1)]
            for x in np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1) if len(x)]


def inventory(scan):
    _, weld = np.unique(np.round(scan.vertices, 7), axis=0, return_inverse=True)
    neighbors, _ = igl.triangle_triangle_adjacency(weld[scan.faces])
    rows, edges = np.where(neighbors >= 0)
    graph = coo_matrix((np.ones(len(rows)), (rows, neighbors[rows, edges])), shape=(len(scan.faces),)*2)
    count, groups = connected_components(graph, directed=False)
    candidates = []
    for number in range(count):
        ids = np.flatnonzero(groups == number)
        points = scan.vertices[scan.faces[ids]].reshape(-1, 3)
        candidates.append({
            "id": hashlib.sha256(ids.astype("<u8").tobytes()).hexdigest(),
            "triangle_ranges": ranges(ids), "triangles": len(ids),
            "bounds_source_units": [points.min(0).tolist(), points.max(0).tolist()],
            "status": "uncertain", "reason": "connected_source_component_requires_review",
        })
    return sorted(candidates, key=lambda c: c["triangles"], reverse=True)


def run(source, guide_dir, out, review=None, evidence=None):
    if out.exists():
        raise FileExistsError(out)
    scan = Scan(source)
    guide, _, cameras = load_annotations(scan, guide_dir / "guidance.json")
    names, owners = segment(scan, guide, cameras)
    out.mkdir(parents=True)
    candidates = inventory(scan)
    write(out / "components.json", {"source_hash": scan.source_hash, "pieces": names[1:],
                                     "range_convention": "half-open source triangle ranges", "candidates": candidates})
    write(out / "decisions.template.json", {"source_hash": scan.source_hash,
                                           "decisions": {c["id"]: {"action": None} for c in candidates}})
    vertex_normals = normals(scan.vertices, scan.faces)[0]
    tiles = []
    for index, item in enumerate(candidates):
        ids = np.concatenate([np.arange(a, b) for a, b in item["triangle_ranges"]])
        part = copy(scan)
        part.faces, part.face_material = scan.faces[ids], scan.face_material[ids]
        bounds = np.array(item["bounds_source_units"])
        center = bounds.mean(0)
        extent = max(float(np.max(bounds[1]-bounds[0]))*1.15, 1e-4)
        folder = out / f"part-{index:03d}-{item['id'][:8]}"
        folder.mkdir()
        for label, yaw in (("front", 0), ("side", 90), ("back", 180)):
            cam = camera(label, center, yaw, 256, 320, extent)
            render(part, cam, vertex_normals, folder)
        tile = Image.new("RGB", (256, 348), "#30343a")
        tile.paste(Image.open(folder / "front.png"), (0, 28))
        ImageDraw.Draw(tile).text((5, 5), f"{index}: {item['id'][:8]} / {item['triangles']} faces", fill="white")
        tiles.append(tile)
        item["renders"] = str(folder.resolve())
    columns = min(4, len(tiles))
    sheet = Image.new("RGB", (columns*256, ((len(tiles)+columns-1)//columns)*348), "#30343a")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns)*256, (i // columns)*348))
    sheet.save(out / "components.png")
    write(out / "components.json", {"source_hash": scan.source_hash, "pieces": names[1:],
                                     "range_convention": "half-open source triangle ranges", "candidates": candidates})
    report = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "source_hash": scan.source_hash, "component_count": len(candidates),
              "guide": str(guide_dir.resolve()), "status": "pending_part_review"}
    if review:
        if not evidence:
            raise ValueError("--review requires the corresponding --evidence")
        resolved, brows = reviewed_owners(scan, owners, names, review, evidence, out / "pending.json", scan.vertices)
        write(out / "review.json", resolved)
        np.save(out / "owners.npy", owners)
        write(out / "skin-regions.json", {"source_hash": scan.source_hash,
                                           "source_sha256": report["source_sha256"],
                                           "triangle_ranges": ranges(np.flatnonzero(owners == 0)),
                                           "brow_triangle_ranges": brows})
        report["status"] = "assignments_replayed_requires_visual_check"
    write(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--guide", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    result = run(args.source, args.guide, args.out, args.review, args.evidence)
    print(json.dumps(result))
