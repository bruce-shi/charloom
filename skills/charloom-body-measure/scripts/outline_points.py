"""Inventory and draw the saved dense paired outline points; optionally compare an export.

    PY outline_points.py GUIDE --frame FRAME.json --out DIRECTORY
    PY outline_points.py GUIDE --frame FRAME.json --out DIRECTORY --model-spec MODEL.json

FRAME.json contains source_to_working as a row-major 4x4 (or the existing
transfer report's working_frame.source_to_working column-major array).
MODEL.json selects a GLB and reviewed per-anatomical-region faces. This command
measures image-plane spans, not inferred 3D cross-section diameters.
"""

import argparse
import re

import numpy as np
from PIL import Image, ImageDraw

from measurement_common import (
    digest, face_selection, load_mesh, matrix, read, write, project_working,
)
from transfer.guide import rasterize
from mh_sections import measure


def read_frame(path):
    data = read(path)
    if "working_frame" in data:
        return matrix(np.asarray(data["working_frame"]["source_to_working"]).reshape(4, 4).T)
    return matrix(data["source_to_working"])


def required_landmarks():
    names = {"neck", "nose_tip"}
    for side in ("l", "r"):
        names.update(f"{name}_{side}" for name in
                     ("eye", "mouth", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"))
        names.update(f"eye_{side}_{name}" for name in ("inner", "outer", "top", "bottom"))
    return names


def group_for(name):
    match = re.fullmatch(r"outline_(trunk|arm_[lr]|leg_[lr])\d+_[wd]", name)
    return match[1] if match else name.removeprefix("outline_").rsplit("_", 1)[0]


def inventory(folder, frame, out, model_spec=None):
    guide, views = read(folder / "guidance.json"), read(folder / "views.json")
    receipt = read(folder / "views/bodysuit.json")
    if guide["source_hash"] != views["source_hash"]:
        raise ValueError("Guide/source hash mismatch")
    if receipt.get("accepted") is not True:
        raise ValueError("Registered reference is not accepted; inspect registration first")
    missing = sorted(n for n in required_landmarks() if not guide["landmarks"].get(n))
    cameras = {c["id"]: c for c in views["cameras"]}
    marks = receipt["marks"]
    pairs, unpaired, rendered = {}, [], {}
    for key, (view, px) in marks.items():
        if not re.search(r"_[wd][12]$", key):
            continue
        mate = key[:-1] + ("2" if key.endswith("1") else "1")
        if mate not in marks:
            unpaired.append(key)
            continue
        if not key.endswith("1"):
            continue
        other_view, qx = marks[mate]
        if other_view != view:
            raise ValueError(f"Endpoints are in different views: {key}, {mate}")
        cam = cameras[view]
        points = np.asarray([px, qx], float)
        if (points.shape != (2, 2) or not np.isfinite(points).all()
                or (points < 0).any() or (points[:, 0] >= cam["width"]).any()
                or (points[:, 1] >= cam["height"]).any()):
            raise ValueError(f"Invalid outline endpoints: {key}")
        # Calibrated image-plane metric in the chosen working frame.
        basis = frame[:3, :3] @ np.array([cam["right"], -np.asarray(cam["up"])]).T
        step = (points[1] - points[0]) * cam["meters_per_pixel"]
        pairs[key[:-1]] = {
            "view": view, "group": group_for(key[:-1]), "reference_px": points.tolist(),
            "reference_projected_mm": float(np.linalg.norm(basis @ step) * 1000),
            "evidence": "registered_image_projection",
        }
    if unpaired:
        raise ValueError(f"Unpaired outline endpoints: {unpaired}")
    if not pairs:
        raise ValueError("No paired outline measurements")
    out.mkdir(parents=True, exist_ok=False)
    for view in sorted({v[0] for v in marks.values()}):
        path = folder / "views" / f"bodysuit_{view}-reg.png"
        cam = cameras[view]
        im = Image.open(path).convert("RGB")
        if im.size != (cam["width"], cam["height"]):
            raise ValueError(f"Reference image/camera dimensions disagree: {view}")
        draw = ImageDraw.Draw(im)
        count = 0
        for key, (panel, px) in marks.items():
            if panel != view:
                continue
            count += 1
            x, y = px
            if key.endswith("1") and key[:-1] in pairs:
                draw.line([tuple(p) for p in pairs[key[:-1]]["reference_px"]], fill="#f4d95d")
            draw.ellipse((x-2, y-2, x+2, y+2), fill="#ee5e47")
        im.save(out / f"points-{view}.png")
        rendered[view] = {"image": str(path.resolve()), "sha256": digest(path), "point_count": count}
    comparisons, failures = {}, []
    if model_spec:
        spec = read(model_spec)
        surface, part, vertices, topology = load_mesh(model_spec.parent, spec["mesh"])
        tolerance = float(spec["tolerance_mm"])
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Provide a positive comparison tolerance_mm")
        groups = spec["regions"]
        absent = sorted({p["group"] for p in pairs.values()} - groups.keys())
        if absent:
            raise ValueError(f"Missing reviewed model face regions: {absent}")
        for view in rendered:
            cam = cameras[view]
            # rasterize accepts screen XY; camera-space depth resolves overlap.
            from transfer.guide import project
            from transfer.scan import transform_points
            raster = project(cam, transform_points(vertices, np.linalg.inv(frame)))
            im = Image.open(rendered[view]["image"]).convert("RGB")
            draw = ImageDraw.Draw(im)
            for group, selection in groups.items():
                selected = face_selection(part, surface, selection)
                if not len(selected):
                    raise ValueError(f"Empty face region: {group}")
                ids, _ = rasterize(raster, part.faces[selected], cam["width"], cam["height"])
                stations = {k: v["reference_px"] for k, v in pairs.items()
                            if v["group"] == group and v["view"] == view}
                for key, ends in stations.items():
                    delta = np.diff(ends, axis=0)[0]
                    mm_per_px = pairs[key]["reference_projected_mm"] / np.linalg.norm(delta)
                    result = measure(ids >= 0, {key: ends}, mm_per_px)[key]
                    result["group"] = group
                    comparisons[key] = result
                    draw.line([tuple(p) for p in ends], fill="#f4d95d", width=2)
                    if result["status"] == "measured":
                        draw.line([tuple(p) for p in result["model_px"]], fill="#44e8ed", width=1)
                    if (result["status"] != "measured" or abs(result["error_mm"]) > tolerance
                            or max(result["endpoint_errors_mm"]) > tolerance):
                        failures.append(key)
            im.save(out / f"comparison-{view}.png")
    report = {
        "source_hash": guide["source_hash"], "guide_sha256": digest(folder / "guidance.json"),
        "reference_receipt_sha256": digest(folder / "views/bodysuit.json"),
        "source_to_working": frame.tolist(), "units": "working-model millimetres",
        "basic_required_landmarks": sorted(required_landmarks()), "missing_basic_landmarks": missing, "views": rendered,
        "pair_count": len(pairs), "pairs": pairs,
        "comparisons": comparisons, "failed": failures,
        "status": "comparison_failed" if failures else "comparison_passed" if model_spec else "targets_for_visual_review",
        "note": "Paired image projections; hidden anatomy remains estimated. Inspect arm/torso overlap and front/profile station correspondence.",
    }
    if model_spec:
        report.update(model=spec["mesh"], model_spec_sha256=digest(model_spec), triangle_sha256=topology)
    write(out / "outline-targets.json", report)
    return report


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guide", type=Path)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-spec", type=Path)
    args = parser.parse_args()
    report = inventory(args.guide, read_frame(args.frame), args.out, args.model_spec)
    print(f"{report['pair_count']} pairs; {report['status']}; {len(report['failed'])} failed")
    raise SystemExit(bool(report["failed"]))
