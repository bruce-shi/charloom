"""Prepare or inspect calibrated scan views. All measurements refer to clean image pixels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from transfer.guide import (
    camera,
    cameras_for,
    load_annotations,
    prepare,
    refine,
    resolve,
    write_views,
)
from transfer.scan import Scan


def zoom(scan, guide, views, cameras, reference, box, name, yaw):
    x0, y0, x1, y1 = box
    cam = cameras[reference]
    if not (0 <= x0 < x1 <= cam["width"] and 0 <= y0 < y1 <= cam["height"]):
        raise ValueError("Zoom box lies outside the reference view")
    x, y = (x0 + x1) / 2, (y0 + y1) / 2
    points = resolve(
        scan,
        dict(landmarks={"zoom": [{"view": reference, "px": [x, y]}]}, subject={}),
        cameras,
        np.eye(4),
    )["landmarks"]
    front = np.rad2deg(np.arctan2(-cam["forward"][0], -cam["forward"][2]))
    extent = max(x1 - x0, y1 - y0) * cam["meters_per_pixel"]
    cams = [
        camera(
            name if angle == 0 else f"{name}_o{angle:g}",
            points["zoom"],
            front + angle,
            768,
            768,
            extent,
        )
        for angle in dict.fromkeys([0.0, *yaw])
    ]
    if any(c["id"] in cameras for c in cams):
        raise ValueError("Choose a new name to preserve previously marked views")
    write_views(scan, guide.parent, cams, views)
    return [c["id"] for c in cams]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="Render a new guide directory")
    mode.add_argument(
        "--views",
        type=Path,
        metavar="GUIDANCE",
        help="Add body/head views for the requested yaw angles",
    )
    mode.add_argument(
        "--refine",
        type=Path,
        metavar="GUIDANCE",
        help="Render face close-ups from marked eyes",
    )
    mode.add_argument(
        "--check",
        type=Path,
        metavar="GUIDANCE",
        help="Verify marked image hashes and resolve landmarks",
    )
    mode.add_argument(
        "--zoom",
        type=Path,
        metavar="GUIDANCE",
        help="Render a close-up of a marked box",
    )
    mode.add_argument(
        "--trace",
        type=Path,
        metavar="GUIDANCE",
        help="Bind a reviewed mouth polyline to its clean image",
    )
    p.add_argument("--view", help="View identifier for --zoom or --trace")
    p.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--name", default="zoom")
    p.add_argument("--yaw", type=float, nargs="*", default=[30.0])
    p.add_argument(
        "--pixels",
        type=Path,
        help="JSON array of mouth seam pixels in left-to-right order",
    )
    p.add_argument("--output", type=Path, help="New output trace file")
    args = p.parse_args()
    scan = Scan(args.source)
    if args.out:
        prepare(scan, args.out)
        return
    path = args.refine or args.check or args.zoom or args.trace or args.views
    guide, views, cams = load_annotations(scan, path)
    if args.views:
        additions = [c for c in cameras_for(scan, args.yaw) if c["id"] not in cams]
        write_views(scan, path.parent, additions, views)
        print(json.dumps({"views_added": additions}))
    elif args.refine:
        refine(scan, path)
    elif args.check:
        resolved = resolve(scan, guide, cams, np.eye(4))
        marks = resolved["landmarks"] | resolved["joint_section_depth"]
        required = {"neck", "nose_tip"}
        for side in ("l", "r"):
            required.update(
                f"{name}_{side}"
                for name in (
                    "eye",
                    "mouth",
                    "shoulder",
                    "elbow",
                    "wrist",
                    "hip",
                    "knee",
                    "ankle",
                )
            )
            required.update(
                f"eye_{side}_{part}" for part in ("inner", "outer", "top", "bottom")
            )
        missing = required - marks.keys()
        if missing:
            raise ValueError(f"Missing transfer landmarks: {sorted(missing)}")
        print(json.dumps(resolved))
    elif args.zoom:
        if not args.view or not args.box or not args.name.isidentifier():
            p.error("--zoom requires --view, --box and an identifier for --name")
        ids = zoom(scan, path, views, cams, args.view, args.box, args.name, args.yaw)
        print(json.dumps({"views_added": [{"id": i} for i in ids]}))
    else:
        if not args.pixels or not args.output or args.view not in cams:
            p.error("--trace requires --view, --pixels and --output")
        coords = np.array(json.loads(args.pixels.read_text()), float)
        cam = cams[args.view]
        if (
            coords.ndim != 2
            or coords.shape[1] != 2
            or len(coords) < 2
            or not np.isfinite(coords).all()
            or np.any(coords < 0)
            or np.any(coords >= [cam["width"], cam["height"]])
        ):
            raise ValueError("Every mouth seam point must be inside the clean image")
        image = path.parent / "views" / (args.view + ".png")
        with scan.path.open("rb") as f:
            source_hash = hashlib.file_digest(f, "sha256").hexdigest()
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest != views["images"][args.view]["sha256"]:
            raise ValueError("Trace image changed after calibration")
        with args.output.open("x") as f:
            json.dump(
                dict(
                    source_sha256=source_hash,
                    image=str(image.resolve()),
                    view=args.view,
                    image_sha256=digest,
                    pixels=coords.tolist(),
                ),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
