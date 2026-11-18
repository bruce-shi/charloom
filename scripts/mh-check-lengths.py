"""Compare bilateral vertical offsets, 3D landmark chains and garment tape paths on a calibrated guide and exported skin.

    python scripts/mh-check-lengths.py GUIDE POINTS.json MODEL.glb --out REPORT.json --tolerance-mm 4

POINTS records the guide's source_hash, front_view, source_to_working matrix,
body_node, triangle_sha256 and reference_px and skin bindings for each endpoint.
A centre binding has vertices and coefficients. A fingertip or sole binding may
use lowest_in_joints (joint names) and minimum_weight. Review the endpoints in
the neutral pose before reusing bindings on another export of the same topology.
Supply reference_xyz_m for each landmark to compare the 3D chains. For sleeve
and outer leg measurements, provide surface_paths with reference_xyz_m,
triangle_ids and barycentric coordinates in path order from proximal to distal.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from animation.surface import Surface


SEGMENTS = {
    "arm": ("shoulder", "wrist"),
    "upper_arm": ("shoulder", "elbow"),
    "forearm": ("elbow", "wrist"),
    "hand": ("wrist", "fingertip"),
    "torso": ("shoulder", "hip"),
    "leg": ("hip", "ankle"),
    "thigh": ("hip", "knee"),
    "shin": ("knee", "ankle"),
    "hip_to_sole": ("hip", "sole"),
}


def read_skin_point(surface, part, vertices, binding):
    if "lowest_in_joints" in binding:
        names = binding["lowest_in_joints"]
        joints = [surface.scene.joint(name) for name in names]
        weight = (part.weights * np.isin(part.bones[part.palette], joints)).sum(1)
        ids = np.unique(part.faces)
        ids = ids[weight[ids] >= binding["minimum_weight"]]
        if not len(ids):
            raise ValueError(f"No visible skin triangles weighted to {names}")
        return vertices[ids[np.argmin(vertices[ids, 1])]]
    ids = np.asarray(binding["vertices"], dtype=int)
    weights = np.asarray(binding["coefficients"], dtype=float)
    if (not len(ids) or ids.shape != weights.shape or ids.min() < 0
            or ids.max() >= len(vertices) or not np.isfinite(weights).all()
            or not np.isclose(weights.sum(), 1.0)):
        raise ValueError("Invalid skin landmark binding")
    return weights @ vertices[ids]


def distance_record(target_m, model_m):
    return {
        "target_mm": round(float(target_m) * 1000, 2),
        "model_mm": round(float(model_m) * 1000, 2),
        "error_mm": round(float(model_m - target_m) * 1000, 2),
    }


def path_length(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("A 3D path requires at least two finite XYZ points")
    return np.linalg.norm(np.diff(points, axis=0), axis=1).sum()


def measure(guide, points, model):
    annotation = json.loads((guide / "guidance.json").read_text())
    if annotation["source_hash"] != points["source_hash"]:
        raise ValueError("Measurements belong to a different scan")
    cameras = json.loads((guide / "views.json").read_text())["cameras"]
    cam = next(c for c in cameras if c["id"] == points["front_view"])
    frame = np.asarray(points["source_to_working"]).reshape(4, 4).T
    horizontal = frame[:3, :3] @ cam["right"]
    up = frame[:3, :3] @ cam["up"]
    if abs(up[1]) < 1e-6 or abs((frame[:3, :3] @ cam["forward"])[1]) > 1e-6:
        raise ValueError("A frontal orthographic view aligned with standing height is required")
    source = Surface(model)
    matches = [i for i, p in enumerate(source.parts) if source.scene.names[p.node] == points["body_node"]]
    if len(matches) != 1:
        raise ValueError("Select an unambiguous bare skin primitive")
    i = matches[0]
    part = source.parts[i]
    topology = hashlib.sha256(part.faces.astype("<i8").tobytes()).hexdigest()
    if topology != points["triangle_sha256"]:
        raise ValueError("Skin topology changed. Review the vertex bindings.")
    landmarks = {}
    spatial = {}
    for name, point in points["landmarks"].items():
        x, y = point["reference_px"]
        source_height = (x * horizontal[1] - y * up[1]) * cam["meters_per_pixel"]
        pos = read_skin_point(source, part, source.rest[i], point["skin"])
        landmarks[name] = (source_height, pos[1])
        if "reference_xyz_m" in point:
            spatial[name] = (point["reference_xyz_m"], pos)
    measurements = {}
    for side in ("l", "r"):
        for label, (a, b) in SEGMENTS.items():
            target, actual = np.array(landmarks[f"{a}_{side}"]) - landmarks[f"{b}_{side}"]
            measurements[f"vertical_{label}_{side}"] = distance_record(target, actual)
        for label, names in {
            "arm": ["shoulder", "elbow", "wrist"], "leg": ["hip", "knee", "ankle"],
            "torso": ["shoulder", "hip"], "hand": ["wrist", "fingertip"],
        }.items():
            keys = [f"{name}_{side}" for name in names]
            if all(k in spatial for k in keys):
                reference, model_path = zip(*(spatial[k] for k in keys))
                measurements[f"spatial_{label}_{side}"] = distance_record(path_length(reference), path_length(model_path))
    for label, trace in points.get("surface_paths", {}).items():
        triangles = np.asarray(trace["triangle_ids"], dtype=int)
        weights = np.asarray(trace["barycentric"], dtype=float)
        if (not len(triangles) or triangles.min() < 0 or triangles.max() >= len(part.faces)
                or weights.shape != (len(triangles), 3) or not np.isfinite(weights).all()
                or weights.min() < -1e-6 or not np.allclose(weights.sum(axis=1), 1.0)):
            raise ValueError(f"Invalid surface path {label}")
        triangles_xyz = source.rest[i][part.faces[triangles]]
        model_path = np.einsum("ij,ijk->ik", weights, triangles_xyz)
        measurements[f"surface_{label}"] = distance_record(path_length(trace["reference_xyz_m"]), path_length(model_path))
    return measurements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guide", type=Path)
    parser.add_argument("points", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance-mm", type=float, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0:
        parser.error("Choose a positive tolerance based on the image resolution and endpoint uncertainty")
    values = measure(args.guide, json.loads(args.points.read_text()), args.model)
    with args.model.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    failures = [k for k, v in values.items() if abs(v["error_mm"]) > args.tolerance_mm]
    args.out.write_text(json.dumps({
        "model_sha256": digest, "points_sha256": hashlib.sha256(args.points.read_bytes()).hexdigest(),
        "units": "millimetres; vertical offsets, spatial landmark chains, or skin-surface paths as named", "tolerance_mm": args.tolerance_mm,
        "measurements": values, "failed": failures,
    }, indent=2) + "\n")
    print(f"{len(values) - len(failures)}/{len(values)} measurements within tolerance")
    raise SystemExit(1 if failures else 0)
