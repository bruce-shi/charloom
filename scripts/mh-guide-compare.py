#!/usr/bin/env python3
"""Compare two guidance.json files for the same scan (e.g. LLM vs. hand).

    scripts/mh-guide-compare.py A/guidance.json B/guidance.json [--json]

Each file is read with the views.json next to it, because the face
close-ups (`face_*`) are placed from that file's own eye marks and their
pixels differ between the two.

Landmarks: every observation becomes a point in the image plane of its
camera, in metres. Views that look the same way (body_y000, head_y000,
face_y000) share a plane. The distance per name is reported in mm, for
each viewing direction both files marked.

Garments and skin: polygons are rasterised on a 2 px grid per body/head view
(those cameras do not depend on the guidance). Garments are matched by best
overlap and reported as IoU, averaged over views either file marked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STEP = 2  # px grid for polygon rasterisation


def load(path: Path) -> tuple[dict, dict]:
    g = json.loads(path.read_text())
    views = json.loads((path.parent / "views.json").read_text())
    if views["source_hash"] != g["source_hash"]:
        raise SystemExit(f"{path}: source_hash does not match its views.json")
    return g, {c["id"]: c for c in views["cameras"]}


def plane_point(cam: dict, px: list[float]) -> tuple[tuple, tuple[float, float]]:
    """(direction key, (u, v) in metres along the camera's right/up axes from the origin)."""
    m = cam["meters_per_pixel"]
    right, up, c = cam["right"], cam["up"], cam["center"]
    # World point on the pixel's ray at the depth of the camera centre.
    du = (px[0] - cam["width"] * 0.5) * m
    dv = (cam["height"] * 0.5 - px[1]) * m
    world = [c[i] + right[i] * du + up[i] * dv for i in range(3)]
    key = tuple(round(x, 3) for x in cam["forward"])
    return key, (sum(world[i] * right[i] for i in range(3)), sum(world[i] * up[i] for i in range(3)))


def landmark_planes(g: dict, cams: dict) -> dict[str, dict[tuple, tuple[float, float]]]:
    out: dict[str, dict[tuple, list]] = {}
    for name, obs in g["landmarks"].items():
        for o in obs:
            if o["view"] not in cams:
                continue
            key, uv = plane_point(cams[o["view"]], o["px"])
            out.setdefault(name, {}).setdefault(key, []).append(uv)
    return {
        n: {k: (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps)) for k, ps in d.items()}
        for n, d in out.items()
    }


DIRECTION = {(-0.0, 0.0, -1.0): "y000", (0.0, 0.0, -1.0): "y000", (-1.0, 0.0, -0.0): "y090",
             (-1.0, 0.0, 0.0): "y090", (-0.0, 0.0, 1.0): "y180", (0.0, 0.0, 1.0): "y180",
             (1.0, 0.0, 0.0): "y270", (1.0, 0.0, -0.0): "y270"}


def mask(polys: list[list[list[float]]], w: int, h: int) -> set[tuple[int, int]]:
    cells = set()
    for poly in polys:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        for y in range(int(max(min(ys), 0)) // STEP * STEP, int(min(max(ys), h - 1)) + 1, STEP):
            yc = y + STEP * 0.5
            # Scanline crossings (even-odd rule).
            xs_cross = []
            n = len(poly)
            for i in range(n):
                (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
                if (y0 <= yc) != (y1 <= yc):
                    xs_cross.append(x0 + (yc - y0) * (x1 - x0) / (y1 - y0))
            xs_cross.sort()
            for a, b in zip(xs_cross[::2], xs_cross[1::2]):
                for x in range(int(max(a, 0)) // STEP * STEP, int(min(b, w - 1)) + 1, STEP):
                    if a <= x + STEP * 0.5 <= b:
                        cells.add((x, y))
    return cells


def region_masks(regions: list[dict], cams: dict) -> dict[str, set]:
    by_view: dict[str, list] = {}
    for r in regions:
        if r["view"].startswith(("body_", "head_")):
            by_view.setdefault(r["view"], []).append(r["polygon"])
    return {v: mask(p, cams[v]["width"], cams[v]["height"]) for v, p in by_view.items()}


def iou(a: dict[str, set], b: dict[str, set]) -> float | None:
    views = set(a) | set(b)
    if not views:
        return None
    vals = []
    for v in views:
        ma, mb = a.get(v, set()), b.get(v, set())
        u = len(ma | mb)
        vals.append(len(ma & mb) / u if u else 1.0)
    return sum(vals) / len(vals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args()
    (ga, ca), (gb, cb) = load(args.a), load(args.b)
    if ga["source_hash"] != gb["source_hash"]:
        raise SystemExit("the two files are for different scans")

    result: dict = {
        "front_view": [ga["front_view"], gb["front_view"]],
        "subject_kind": [ga["subject"].get("kind"), gb["subject"].get("kind")],
        "sex": [ga["subject"].get("sex"), gb["subject"].get("sex")],
    }
    la, lb = landmark_planes(ga, ca), landmark_planes(gb, cb)
    lm = {}
    for name in sorted(set(la) | set(lb)):
        if name not in la or name not in lb:
            lm[name] = "only in " + ("A" if name in la else "B")
            continue
        d = {}
        for key in set(la[name]) & set(lb[name]):
            (u0, v0), (u1, v1) = la[name][key], lb[name][key]
            d[DIRECTION.get(key, str(key))] = round(((u0 - u1) ** 2 + (v0 - v1) ** 2) ** 0.5 * 1000, 1)
        lm[name] = d or "no shared view direction"
    result["landmark_mm"] = lm

    ma = {g["name"]: region_masks(g["regions"], ca) for g in ga["garments"]}
    mb = {g["name"]: region_masks(g["regions"], cb) for g in gb["garments"]}
    pairs = sorted(((iou(ma[x], mb[y]) or 0.0, x, y) for x in ma for y in mb), reverse=True)
    used_a, used_b, garments = set(), set(), {}
    for score, x, y in pairs:
        if score > 0 and x not in used_a and y not in used_b:
            used_a.add(x)
            used_b.add(y)
            garments[x if x == y else f"{x} ~ {y}"] = round(score, 2)
    for x in ma:
        if x not in used_a:
            garments[x] = "only in A"
    for y in mb:
        if y not in used_b:
            garments[y] = "only in B"
    result["garment_iou"] = garments
    layers_a = {g["name"]: (g["label"], g["layer"], g["tight"]) for g in ga["garments"]}
    layers_b = {g["name"]: (g["label"], g["layer"], g["tight"]) for g in gb["garments"]}
    result["garment_flags_differ"] = {
        n: [layers_a[n], layers_b[n]] for n in set(layers_a) & set(layers_b) if layers_a[n] != layers_b[n]
    }
    for key in ("skin", "skin_cleanup"):
        s = iou(region_masks(ga[key], ca), region_masks(gb[key], cb))
        result[f"{key}_iou"] = None if s is None else round(s, 2)

    if args.json:
        print(json.dumps(result, indent=1))
        return
    print(f"front view {result['front_view']}, kind {result['subject_kind']}, sex {result['sex']}")
    print("landmarks (mm in the image plane, per viewing direction):")
    for name, d in lm.items():
        print(f"  {name:14s} {d}")
    print("garments (IoU):")
    for name, s in garments.items():
        print(f"  {name:24s} {s}")
    if result["garment_flags_differ"]:
        print("label/layer/tight differ:", result["garment_flags_differ"])
    print(f"skin IoU {result['skin_iou']}, skin_cleanup IoU {result['skin_cleanup_iou']}")


if __name__ == "__main__":
    main()
