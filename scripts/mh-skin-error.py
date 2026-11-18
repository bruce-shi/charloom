"""Where the fitted body misses the scan's skin.

    out/tools/cv/.venv/bin/python -I scripts/mh-skin-error.py RUN_DIR [RUN_DIR ...]

Reads the run's segmentation (`1-segment/segmentation.glb`, skin in its
skin colour) and fitted body (`2-body-rig/fit-diagnostic.glb`, same working
frame), measures each scan skin point's distance to the body's nearest
vertex, and draws front and side scatter views coloured by it (blue 0,
green 20 mm, yellow 40 mm, red 80 mm and over). Writes
`renders/evidence-5-skin-error.png` and prints the worst bands by height.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

match = import_module("mh-match")

SKIN = np.array([0.85, 0.66, 0.50])  # segmentation.glb's skin colour (approx.)


def colours(path):
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + n])
    off = 20 + n
    bn = struct.unpack_from("<I", data, off)[0]
    blob = data[off + 8:off + 8 + bn]
    acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]]
    view = doc["bufferViews"][acc["bufferView"]]
    comps = {"VEC3": 3, "VEC4": 4}[acc["type"]]
    dtype = {5126: np.float32, 5121: np.uint8, 5123: np.uint16}[acc["componentType"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    a = np.frombuffer(blob, dtype=dtype, count=acc["count"] * comps, offset=start).reshape(-1, comps)[:, :3].astype(float)
    if dtype == np.uint8:
        a /= 255
    elif dtype == np.uint16:
        a /= 65535
    return a


def ramp(d):
    stops = [(0, (40, 60, 220)), (0.02, (40, 200, 80)), (0.04, (240, 220, 40)), (0.08, (230, 40, 30))]
    out = np.zeros((len(d), 3))
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        m = (d >= x0) & (d < x1)
        t = ((d[m] - x0) / (x1 - x0))[:, None]
        out[m] = np.array(c0) * (1 - t) + np.array(c1) * t
    out[d >= stops[-1][0]] = stops[-1][1]
    return out.astype(np.uint8)


def main(run):
    seg = match.glb_meshes(run / "1-segment/segmentation.glb")
    pos = np.concatenate([m[1] for m in seg])
    col = colours(run / "1-segment/segmentation.glb")
    body = np.concatenate([m[1] for m in match.glb_meshes(run / "2-body-rig/fit-diagnostic.glb")])
    # Skin: the warm tan, not the blue/purple of garments and hair.
    skin = (col[:, 0] > col[:, 2] + 0.12) & (col[:, 0] > 0.45)
    pts = pos[skin]
    rng = np.random.default_rng(0)
    if len(pts) > 60000:
        pts = pts[rng.choice(len(pts), 60000, replace=False)]
    d = np.empty(len(pts))
    for i in range(0, len(pts), 2000):
        q = pts[i:i + 2000]
        d[i:i + 2000] = np.sqrt(((q[:, None, :] - body[None, :, :]) ** 2).sum(-1).min(1))
    lo, hi = pos.min(0), pos.max(0)
    h = hi[1] - lo[1]
    size = 900
    s = size * 0.9 / h
    img = Image.new("RGB", (size * 2, size), (30, 30, 34))
    dr = ImageDraw.Draw(img)
    c = ramp(d)
    for k, (axis, ox) in enumerate([(0, 0), (2, size)]):
        # Far points first so near ones draw over them.
        depth = pts[:, 2] if axis == 0 else -pts[:, 0]
        order = np.argsort(depth)
        for j in order:
            x = ox + size / 2 + (pts[j, axis] - (lo[axis] + hi[axis]) / 2) * s
            y = size * 0.95 - (pts[j, 1] - lo[1]) * s
            dr.point((x, y), fill=tuple(int(v) for v in c[j]))
    dr.text((10, 10), f"{run.name}: scan skin to body (blue 0, green 20, yellow 40, red 80+ mm)", fill=(230, 230, 230))
    out = run / "renders/evidence-5-skin-error.png"
    img.save(out)
    print(run.name, "skin points", len(pts), "p50 %.0f p95 %.0f mm" % (np.median(d) * 1000, np.percentile(d, 95) * 1000))
    bands = np.clip(((pts[:, 1] - lo[1]) / h * 20).astype(int), 0, 19)
    for b in range(20):
        m = bands == b
        if m.sum() > 50:
            print(f"  height {b * 5:3d}-{b * 5 + 5:3d} %  n {m.sum():6d}  p50 {np.median(d[m]) * 1000:5.0f}  p95 {np.percentile(d[m], 95) * 1000:5.0f} mm")
    print("  wrote", out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(Path(p))
