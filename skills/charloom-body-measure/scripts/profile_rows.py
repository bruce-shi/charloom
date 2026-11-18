"""Draw reviewed side-profile endpoints on the registered image at selected rows."""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from measurement_common import digest, matrix, read, write

suit = import_module("mh-bodysuit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guide", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("stations", type=Path)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registration = read(args.registration / "registration.json")
    spec = read(args.stations)
    frame = read(args.frame)
    views = read(args.guide / "views.json")
    registered = args.registration / "registered.png"
    if (
        not registration["anchors_within_tolerance"]
        or views["source_hash"] != registration["source_hash"]
        or frame["source_hash"] != views["source_hash"]
        or digest(registered) != spec["registered_sha256"]
    ):
        raise ValueError("Review input hashes, source frame and profile registration")
    view = registration["view"]
    base = args.guide / "views" / f"bodysuit_{view}-reg.png"
    if digest(base) != registration["base_sha256"]:
        raise ValueError("Registered base view changed")
    camera = next(c for c in views["cameras"] if c["id"] == view)
    scale = np.linalg.norm(matrix(frame["source_to_working"])[:3, :3] @ camera["right"])
    mm_per_px = float(1000 * scale * camera["meters_per_pixel"])
    if spec["anterior"] not in ("left", "right"):
        raise ValueError("Specify the anterior edge of this profile, left or right")
    masks = [suit.mask(image) for image in (base, registered)]
    canvas = Image.open(registered).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    rows = []
    for station in spec["stations"]:
        x, y = station["inside_px"]
        if not (
            isinstance(x, int)
            and isinstance(y, int)
            and 0 < x < canvas.width - 1
            and 0 < y < canvas.height - 1
        ):
            raise ValueError(
                "Expected a seed inside the torso in calibrated pixel coordinates"
            )
        edges = []
        for mask in masks:
            run = suit.run_at(mask[y], x)
            if (
                not mask[y, x]
                or run is None
                or run[0] == 0
                or run[1] == canvas.width - 1
            ):
                raise ValueError(f"Cannot isolate the selected row at {(x, y)}")
            edges.append(run if spec["anterior"] == "left" else run[::-1])
        anterior, posterior = edges[1]
        visible = [station[end + "_visible"] for end in ("anterior", "posterior")]
        if not all(isinstance(flag, bool) for flag in visible):
            raise ValueError("Each endpoint must have a reviewed Boolean visibility")
        if all(visible):
            draw.line([(anterior, y), (posterior, y)], fill="#e6d15d")
        for px, flag in zip(edges[1], visible):
            draw.ellipse(
                (px - 2, y - 2, px + 2, y + 2), fill="#36d7eb" if flag else "#ef685c"
            )
        rows.append(
            dict(
                station=station,
                old_edges_px=edges[0],
                new_edges_px=edges[1],
                projected_estimate_mm=abs(posterior - anterior) * mm_per_px
                if all(visible)
                else None,
            )
        )
    args.out.mkdir(parents=True, exist_ok=False)
    canvas.save(args.out / "profile-points.png")
    write(
        args.out / "profile-rows.json",
        dict(
            source_hash=views["source_hash"],
            view=view,
            registered_sha256=digest(registered),
            frame_sha256=digest(args.frame),
            stations_sha256=digest(args.stations),
            anterior=spec["anterior"],
            mm_per_pixel=mm_per_px,
            rows=rows,
        ),
    )


if __name__ == "__main__":
    main()
