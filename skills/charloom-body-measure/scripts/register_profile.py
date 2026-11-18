"""Register a generated torso profile to its base camera from reviewed pixel pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def landmarks(items, size):
    pairs = np.array([[item["candidate_px"], item["base_px"]] for item in items], float)
    if (
        len(items) < 2
        or pairs.shape != (len(items), 2, 2)
        or not np.isfinite(pairs).all()
        or (pairs < 0).any()
        or (pairs >= size).any()
    ):
        raise ValueError("Supply at least two in-frame named point pairs per set")
    return pairs[:, 0], pairs[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="torso_profile.py output directory")
    parser.add_argument(
        "anchors", type=Path, help="Reviewed fit and independent check pairs"
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    receipt = json.loads((args.folder / "generation.json").read_text())
    marks = json.loads(args.anchors.read_text())
    source = Path(receipt["source"])
    candidate = args.folder / "profile-candidate.png"
    if (
        sha(source) != receipt["image_sha256"]
        or sha(candidate) != marks["candidate_sha256"]
        or sha(source) != marks["base_sha256"]
    ):
        raise ValueError("Landmark images do not match their recorded hashes")
    base = cv2.imread(str(source))
    edited = cv2.imread(str(candidate))
    if base.shape != edited.shape:
        raise ValueError("Base and candidate must use the same pixel dimensions")
    size = np.array(base.shape[1::-1])
    fit, reference = landmarks(marks["fit"], size)
    checks, check_reference = landmarks(marks["check"], size)
    fit_lines = marks.get("fit_y_lines", [])
    check_lines = marks.get("check_y_lines", [])
    line_pairs = fit_lines + check_lines
    point_names = [item["name"] for item in marks["fit"] + marks["check"]]
    names = point_names + [item["name"] for item in line_pairs]
    if len(set(names)) != len(names) or any(
        np.array_equal(p, q) for p in fit for q in checks
    ):
        raise ValueError("Use independent anatomical landmarks for checking")
    for line in line_pairs:
        values = np.array([line["candidate_y"], line["base_y"]], float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values >= size[1]).any()
        ):
            raise ValueError("Horizontal lines must lie inside the image")
    if {l["candidate_y"] for l in fit_lines} & {l["candidate_y"] for l in check_lines}:
        raise ValueError("A fitted horizontal line cannot also be an independent check")
    # x' = scale*x + dx and y' = scale*y + dy. Ground lines constrain y only.
    design, targets = [], []
    for (x, y), destination in zip(fit, reference):
        design.extend(([x, 1, 0], [y, 0, 1]))
        targets.extend(destination)
    for line in fit_lines:
        design.append([line["candidate_y"], 0, 1])
        targets.append(line["base_y"])
    solution, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    scale, offset = solution[0], solution[1:]
    if rank != 3 or scale <= 0:
        raise ValueError("Landmarks do not determine a positive scale and translation")
    transform = np.column_stack((scale * np.eye(2), offset))
    registered = cv2.warpAffine(
        edited, transform, tuple(size), borderMode=cv2.BORDER_REPLICATE
    )
    tolerance = float(marks["tolerance_px"])
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Set a finite positive registration tolerance in base pixels")
    residuals = np.linalg.norm(
        np.vstack((fit, checks)) * scale
        + offset
        - np.vstack((reference, check_reference)),
        axis=1,
    )
    line_errors = [
        abs(scale * line["candidate_y"] + offset[1] - line["base_y"])
        for line in line_pairs
    ]
    residuals = np.concatenate((residuals, line_errors))
    within = bool(np.all(residuals <= tolerance))
    args.out.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(args.out / "registered.png"), registered)
    cv2.imwrite(
        str(args.out / "overlay.png"), cv2.addWeighted(base, 0.5, registered, 0.5, 0)
    )
    report = dict(
        source_hash=receipt["source_hash"],
        view=receipt["view"],
        base_sha256=sha(source),
        candidate_sha256=sha(candidate),
        anchors_sha256=sha(args.anchors),
        transform=transform.tolist(),
        tolerance_px=tolerance,
        errors_px=dict(zip(names, residuals)),
        anchors_within_tolerance=within,
        status="needs_contour_review" if within else "registration_failed",
    )
    (args.out / "registration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())
