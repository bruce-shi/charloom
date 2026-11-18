"""Bind the textured scan surface to a calibrated MakeHuman scaffold using NumPy."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def body_collar(rest, faces):
    prior, first, weld = np.unique(np.round(rest, 6), axis=0, return_index=True, return_inverse=True)
    tri = weld[faces]
    crown = prior[:, 1].argmax()
    high = prior[:, 1] > prior[crown, 1] * .851
    edges = tri[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges = edges[high[edges].all(axis=1)]
    graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(prior), len(prior)))
    _, labels = connected_components(graph, directed=False)
    head = labels == labels[crown]
    keep = ~head[tri].all(axis=1)
    edges = tri[keep][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges = edges[head[edges].all(axis=1)]
    _, first_edge, uses = np.unique(np.sort(edges, axis=1), axis=0, return_index=True, return_counts=True)
    boundary = edges[first_edge[uses == 1]]
    if len(boundary) <= 20:
        raise ValueError("Missing anatomical neck boundary")
    successors = dict(boundary.tolist())
    if len(successors) != len(boundary):
        raise ValueError("Branch in the anatomical neck boundary")
    start = current = boundary[0, 0]
    ring = []
    while current in successors:
        ring.append(first[current])
        current = successors.pop(current)
    if current != start or successors:
        raise ValueError("Open or disconnected neck boundary")
    return keep, np.asarray(ring), weld


def smooth(value):
    t = np.clip(value, 0., 1.)
    return t * t * (3. - 2. * t)


def bind_face(positions, names, curves, eyes, bite_pitch=0.):
    x, y, z = positions.T
    blink = np.zeros_like(positions)
    for side, (center, radius) in eyes.items():
        xs, top, bottom, depth = curves[side]
        width = np.ptp(xs)
        upper, lower, front = (np.interp(x, xs, v) for v in (top, bottom, depth))
        dx = np.abs(x - np.clip(x, xs[0], xs[-1]))
        active = (z > front - .042) & (dx < width * .19)
        upper_lid = y >= (upper + lower) * .5
        distance = np.abs(y - np.where(upper_lid, upper, lower))
        influence = smooth(1. - distance / (width * .6)) * smooth(1. - dx / (width * .18))
        blink[:, 1] += active * (upper - lower) * np.where(upper_lid, -.75, .25) * influence
        disc = radius**2 - (x - center[0])**2 - (y + blink[:, 1] - center[1])**2
        inside = active & (disc > 0) & (distance < width * .8)
        clearance = center[2] + np.sqrt(np.maximum(0., disc)) + .003 - z
        blink[:, 2] = np.maximum(blink[:, 2], np.where(inside, clearance, 0.))

    xs, top, bottom, depth = curves["mouth"]
    width = np.ptp(xs)
    front = np.interp(x, xs, depth)
    distance = y - np.interp(x, xs, (top + bottom) * .5) + bite_pitch * (z - front)
    lateral = np.abs(x - xs.mean()) / (width * .5)
    active = (z > front - .024) & (np.abs(x - np.clip(x, xs[0], xs[-1])) < width * .19)
    fade = active * smooth((1.09 - lateral) / .19) * (1. - np.clip((front - z - .004) / .020, 0., 1.))
    lip = fade * smooth(1. - np.abs(distance) / (width * .24))
    jaw = (distance < 0) * (1. - lip) * fade * smooth(1. - np.abs(distance) / (width * .62)) * .66
    side = np.minimum(1., lateral)
    weights = np.column_stack((1. - lip - jaw, lip * (1. - side), lip * side, jaw))
    upper = distance >= 0
    left = x > xs.mean()
    bones = np.empty((len(x), 4), dtype=np.uint16)
    bones[:, 0], bones[:, 3] = names.index("head"), names.index("jaw")
    bones[:, 1] = np.where(upper, names.index("oris05"), names.index("oris01"))
    for mask, prefix in ((upper, "oris03"), (~upper, "oris07")):
        bones[mask, 2] = np.where(left[mask], names.index(prefix + ".L"), names.index(prefix + ".R"))
    return bones, weights, blink


def project_trace(run, report, vertices_mm, faces, tree, trace=None):
    config = json.loads((run / "config.json").read_text())
    guide = Path(config["guidance"]).parent
    if trace is None:
        trace = run / "mouth-trace.json"
        if not trace.is_file():
            trace = guide / "mouth-trace.json"
    trace = json.loads(Path(trace).read_text())
    views = json.loads((guide / "views.json").read_text())
    camera = next(c for c in views["cameras"] if c["id"] == trace["view"])
    image = guide / "views" / (camera["id"] + ".png")
    def digest(path):
        with open(path, "rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    if (views["source_hash"] != report["source_hash"] or digest(report["source"]) != trace["source_sha256"]
            or Path(trace["image"]).name != image.name or digest(image) != trace["image_sha256"]):
        raise ValueError("Lip trace, camera and original scan must match")
    transform = np.array(report["working_frame"]["source_to_working"]).reshape(4, 4).T
    basis = transform[:3, :3]
    pixels = np.asarray(trace["pixels"], dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 2 or not np.isfinite(pixels).all():
        raise ValueError("Expected at least two finite lip trace pixels")
    scale = camera["meters_per_pixel"]
    origin = np.asarray(camera["center"]) + (pixels[:, :1] - camera["width"] / 2) * scale * camera["right"]
    origin += (camera["height"] / 2 - pixels[:, 1:]) * scale * camera["up"]
    direction = basis @ camera["forward"]
    direction /= np.linalg.norm(direction)
    extent = np.linalg.norm(np.ptp(vertices_mm, axis=0))
    origin = (origin @ basis.T + transform[:3, 3]) * 1000 - direction * extent
    hits, distance, _ = tree.intersect_ray_first(vertices_mm, faces, origin, np.tile(direction, (len(origin), 1)))
    if np.any(hits < 0) or not np.isfinite(distance).all():
        raise ValueError("Lip trace misses the source surface")
    points = (origin + distance[:, None] * direction) / 1000
    points = points[np.argsort(points[:, 0])]
    if np.any(np.diff(points[:, 0]) <= 0):
        raise ValueError("Lip trace needs distinct horizontal positions")
    return points
