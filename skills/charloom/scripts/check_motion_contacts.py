"""Measure triangle intersections between selected surface regions over a GLB clip."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from igl.copyleft.cgal import intersect_other

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from animation.surface import Surface


def inventory(surface):
    return [
        dict(mesh=i, node=surface.scene.names[p.node], geometry_hash=p.hash,
             triangle_count=len(p.faces))
        for i, p in enumerate(surface.parts)
    ]


def region(surface, spec):
    mesh = spec["mesh"]
    if type(mesh) is not int or not 0 <= mesh < len(surface.parts):
        raise ValueError(f"Invalid primitive index {mesh}")
    part = surface.parts[mesh]
    if part.hash != spec["geometry_hash"]:
        raise ValueError(f"Geometry hash mismatch for primitive {mesh}")
    faces = np.asarray(spec["faces"])
    if faces.ndim == 0 and faces.item() == "all":
        faces = np.arange(len(part.faces))
    if (faces.ndim != 1 or not len(faces) or faces.dtype.kind not in "iu"
            or faces.min() < 0 or faces.max() >= len(part.faces)):
        raise ValueError(f"Expected triangle indices for primitive {mesh}")
    faces = np.unique(faces)
    return mesh, faces, part.faces[faces]


def inspect(surface, config, clip, samples):
    regions = {name: region(surface, spec) for name, spec in config["regions"].items()}
    pairs = config["pairs"]
    if not pairs or samples < 2:
        raise ValueError("Provide at least one region pair and at least two samples")
    scene = surface.scene
    start, end = scene.interval(clip)
    times = np.linspace(start, end, samples).tolist()
    results = []
    for a, b in pairs:
        ma, ia, _ = regions[a]
        mb, ib, _ = regions[b]
        if ma == mb and np.intersect1d(ia, ib).size:
            raise ValueError(f"Overlapping face selections for {a} and {b}")
        results.append(dict(regions=[a, b], counts=[], peak=None))

    def measure(points, time):
        for row in results:
            a, b = row["regions"]
            ma, ia, fa = regions[a]
            mb, ib, fb = regions[b]
            hits = intersect_other(points[ma], fa, points[mb], fb, detect_only=True)[0]
            count = len(hits)
            if time is None:
                row["rest_count"] = count
                continue
            row["counts"].append(count)
            if count and (row["peak"] is None or count > row["peak"]["count"]):
                row["peak"] = dict(time=time, count=count, face_pair_examples=[
                    [int(ia[i]), int(ib[j])] for i, j in hits[:12]
                ])

    measure(surface.rest, None)
    for time in times:
        measure(surface.evaluate(*scene.sample(clip, time)), time)
    return dict(clip=clip, times=times, pairs=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inventory", action="store_true")
    mode.add_argument("--config", type=Path)
    parser.add_argument("--clip")
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()
    if args.config and (not args.clip or args.samples is None):
        parser.error("--config requires --clip and --samples")
    surface = Surface(args.model)
    result = (inventory(surface) if args.inventory else
              inspect(surface, json.loads(args.config.read_text()), args.clip, args.samples))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
