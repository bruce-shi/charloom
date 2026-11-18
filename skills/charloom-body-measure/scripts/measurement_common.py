"""Shared IO and calibrated geometry for the body measurement commands."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from animation.surface import Surface
from transfer.guide import project
from transfer.scan import transform_points


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def unit(vector):
    vector = np.asarray(vector, float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm < 1e-12:
        raise ValueError("Expected a finite, nonzero direction")
    return vector / norm


def matrix(value):
    """Measurement specs use explicit row-major 4x4 matrices."""
    result = np.asarray(value, float)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("world_from_model/source_to_working must be a row-major 4x4")
    if not np.allclose(result[3], [0, 0, 0, 1]):
        raise ValueError("Expected an affine matrix")
    if abs(np.linalg.det(result[:3, :3])) < 1e-12:
        raise ValueError("Singular coordinate transform")
    return result


def resolve_path(base, name):
    path = Path(name)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def checked_file(base, record):
    path = resolve_path(base, record["file"])
    if digest(path) != record["sha256"]:
        raise ValueError(f"File changed: {path}")
    return path


def load_mesh(base, record):
    path = checked_file(base, record)
    surface = Surface(path)
    matches = [i for i, p in enumerate(surface.parts)
               if surface.scene.names[p.node] == record["node"]]
    slot = record.get("primitive", 0)
    if not isinstance(slot, int) or slot < 0 or slot >= len(matches):
        raise ValueError(f"Missing primitive {slot} on node {record['node']}")
    i = matches[slot]
    part = surface.parts[i]
    frame = matrix(record.get("world_from_model", np.eye(4)))
    if record.get("clip"):
        points = surface.evaluate(*surface.scene.sample(record["clip"], record["time"]))[i]
    else:
        points = surface.rest[i]
    vertices = transform_points(points, frame)
    topology = hashlib.sha256(part.faces.astype("<i8").tobytes()).hexdigest()
    if record.get("triangle_sha256", topology) != topology:
        raise ValueError("Triangle topology changed; review mesh selections")
    return surface, part, vertices, topology


def face_selection(part, surface, selection):
    """Explicit selections only; do not silently cut clothing or the adjacent torso."""
    methods = [k for k in ("triangle_ranges", "joints", "all_faces") if k in selection]
    if len(methods) != 1:
        raise ValueError("Choose exactly one face selection: triangle_ranges, joints, all_faces")
    if "triangle_ranges" in selection:
        ids = []
        for start, end in selection["triangle_ranges"]:
            if not (isinstance(start, int) and isinstance(end, int)
                    and 0 <= start < end <= len(part.faces)):
                raise ValueError("Invalid half-open triangle range")
            ids.extend(range(start, end))
        return np.unique(np.asarray(ids, int))
    if "joints" in selection:
        if part.weights is None:
            raise ValueError("Joint selection requires a skinned mesh")
        if not selection["joints"]:
            raise ValueError("Empty joint selection")
        joints = [surface.scene.joint(n) for n in selection["joints"]]
        threshold = selection["minimum_mean_weight"]
        if not 0 < threshold <= 1:
            raise ValueError("minimum_mean_weight must be in (0, 1]")
        weight = (part.weights * np.isin(part.bones[part.palette], joints)).sum(1)
        return np.flatnonzero(weight[part.faces].mean(1) > threshold)
    if selection["all_faces"] is not True:
        raise ValueError("all_faces must explicitly be true")
    return np.arange(len(part.faces))


def bound_point(vertices, faces, binding):
    """Surface barycentric or reviewed embedded anatomical centre."""
    if "triangle" in binding:
        tri = binding["triangle"]
        w = np.asarray(binding["barycentric"], float)
        if (not isinstance(tri, int) or not 0 <= tri < len(faces)
                or w.shape != (3,) or not np.isfinite(w).all()
                or w.min() < -1e-8 or not np.isclose(w.sum(), 1)):
            raise ValueError("Invalid triangle/barycentric binding")
        return w @ vertices[faces[tri]]
    ids = np.asarray(binding["vertices"], int)
    weights = np.asarray(binding["coefficients"], float)
    if (not len(ids) or ids.shape != weights.shape or ids.min() < 0
            or ids.max() >= len(vertices) or not np.isfinite(weights).all()
            or not np.isclose(weights.sum(), 1)):
        raise ValueError("Invalid embedded skin centre")
    return weights @ vertices[ids]


def project_working(cam, frame, points):
    return project(cam, transform_points(np.atleast_2d(points), np.linalg.inv(frame)))[:, :2]


def reference_context(base, spec):
    guide_path = resolve_path(base, spec["guide"])
    guide = read(guide_path / "guidance.json")
    views = read(guide_path / "views.json")
    if guide["source_hash"] != spec["source_hash"] or views["source_hash"] != spec["source_hash"]:
        raise ValueError("Guide belongs to a different source")
    return guide_path, guide, {c["id"]: c for c in views["cameras"]}, matrix(spec["source_to_working"])
