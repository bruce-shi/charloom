"""Calibrated orthographic scan views shared by annotation, segmentation and fitting."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit
from PIL import Image, ImageDraw


def camera(name, center, yaw, width, height, extent):
    theta = np.deg2rad(yaw)
    return dict(
        id=name,
        width=width,
        height=height,
        center=np.asarray(center).tolist(),
        right=[np.cos(theta), 0.0, -np.sin(theta)],
        up=[0.0, 1.0, 0.0],
        forward=[-np.sin(theta), 0.0, -np.cos(theta)],
        meters_per_pixel=float(extent / height),
    )


def cameras_for(scan, yaws=(0, 90, 180, 270)):
    low, high = scan.vertices.min(0), scan.vertices.max(0)
    size = high - low
    center = (low + high) / 2
    head = center.copy()
    head[1] = high[1] - size[1] * 0.085
    for name, at, height, extent in (
        ("body", center, 1024, size[1] * 1.08),
        ("head", head, 768, size[1] * 0.18),
    ):
        for yaw in yaws:
            cam = camera(f"{name}_y{int(yaw) % 360:03d}", at, yaw, 768, height, extent)
            if name == "body":
                cam["meters_per_pixel"] = max(
                    cam["meters_per_pixel"], 1.08 * (size @ np.abs(cam["right"])) / 768
                )
            yield cam


def project(cam, vertices):
    local = vertices - cam["center"]
    scale = cam["meters_per_pixel"]
    return np.column_stack(
        (
            local @ cam["right"] / scale + cam["width"] / 2,
            cam["height"] / 2 - local @ cam["up"] / scale,
            local @ cam["forward"],
        )
    )


@njit(cache=True)
def rasterize(projected, faces, width, height):
    ids = np.full((height, width), -1, dtype=np.int32)
    depths = np.full((height, width), np.inf, dtype=np.float32)
    weights = np.zeros((height, width, 3), dtype=np.float32)
    for i, tri in enumerate(faces):
        a, b, c = projected[tri[0]], projected[tri[1]], projected[tri[2]]
        denom = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(denom) < 1e-12:
            continue
        xmin = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        xmax = min(width - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        ymin = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        ymax = min(height - 1, int(np.ceil(max(a[1], b[1], c[1]))))
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                px, py = x + 0.5 - a[0], y + 0.5 - a[1]
                beta = (px * (c[1] - a[1]) - (c[0] - a[0]) * py) / denom
                gamma = ((b[0] - a[0]) * py - px * (b[1] - a[1])) / denom
                alpha = 1.0 - beta - gamma
                if min(alpha, beta, gamma) < -1e-9:
                    continue
                z = np.float32(a[2] * alpha + b[2] * beta + c[2] * gamma)
                if z < depths[y, x]:
                    ids[y, x], depths[y, x] = i, z
                    weights[y, x, 0], weights[y, x, 1], weights[y, x, 2] = (
                        alpha,
                        beta,
                        gamma,
                    )
    return ids, weights


def render(scan, cam, normals, folder):
    ids, bary = rasterize(
        project(cam, scan.vertices), scan.faces, cam["width"], cam["height"]
    )
    visible = ids >= 0
    faces = ids[visible]
    color = scan.colors(faces, bary[visible])
    normal = np.einsum("ij,ijk->ik", bary[visible], normals[scan.faces[faces]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-20)
    light = np.array([0.3, 0.5, 0.0]) - cam["forward"]
    shade = 0.45 + 0.55 * np.abs(normal @ (light / np.linalg.norm(light)))
    pixels = np.empty((*visible.shape, 3), dtype=np.uint8)
    pixels[:] = (96, 100, 108)
    pixels[visible] = (np.clip(color * shade[:, None], 0, 1) ** (1 / 1.1) * 255).astype(
        np.uint8
    )
    image = Image.fromarray(pixels)
    clean = folder / (cam["id"] + ".png")
    image.save(clean)
    drawing = ImageDraw.Draw(image)
    for x in range(0, image.width, 50):
        for y in range(0, image.height, 3 if x % 100 == 0 else 6):
            drawing.point((x, y), fill=(255, 255, 0))
        if x % 100 == 0:
            drawing.text((x + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(0, image.height, 50):
        for x in range(0, image.width, 3 if y % 100 == 0 else 6):
            drawing.point((x, y), fill=(255, 255, 0))
        if y % 100 == 0:
            drawing.text((2, y + 2), str(y), fill=(255, 255, 0))
    grid = clean.with_stem(cam["id"] + "-grid")
    image.save(grid)
    return dict(
        sha256=hashlib.sha256(clean.read_bytes()).hexdigest(),
        grid_sha256=hashlib.sha256(grid.read_bytes()).hexdigest(),
        rendered_unix=int(time.time()),
    )


def write_views(scan, out, cameras, views):
    folder = out / "views"
    folder.mkdir(exist_ok=True)
    # Weld shading normals across texture seams on the original mesh.
    _, weld = np.unique(np.round(scan.vertices, 7), axis=0, return_inverse=True)
    tri = scan.vertices[scan.faces]
    normals = np.zeros((weld.max() + 1, 3))
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for ids in scan.faces.T:
        np.add.at(normals, weld[ids], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
    normals = normals[weld]
    for cam in cameras:
        views["images"][cam["id"]] = render(scan, cam, normals, folder)
        views["cameras"] = [c for c in views["cameras"] if c["id"] != cam["id"]] + [cam]
    (out / "views.json").write_text(json.dumps(views, indent=2))


def prepare(scan, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    from shutil import copyfile

    root = Path(__file__).resolve().parents[2]
    copyfile(root / "docs/scan-annotation.md", out / "INSTRUCTIONS.md")
    copyfile(root / "schemas/guidance.json", out / "guidance.schema.json")
    views = dict(
        schema="charloom-mh-guidance-v1",
        source=str(scan.path),
        source_hash=scan.source_hash,
        cameras=[],
        images={},
    )
    write_views(scan, out, cameras_for(scan), views)
    names = ["nose_tip", "neck", "eye_l", "eye_r", "mouth_l", "mouth_r"]
    for side in ("l", "r"):
        names.extend(f"eye_{side}_{p}" for p in ("inner", "outer", "top", "bottom"))
        names.extend(
            f"{joint}_{side}"
            for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "palm")
        )
    guide = dict(
        schema=views["schema"],
        source_hash=scan.source_hash,
        front_view="body_y000",
        subject={"kind": "human", "sex": "unknown", "notes": ""},
        landmarks={n: [] for n in names},
        skin=[],
        garments=[],
        view_sha256={k: v["sha256"] for k, v in views["images"].items()},
    )
    (out / "guidance.template.json").write_text(json.dumps(guide, indent=2))
    return views


def load_annotations(scan, path):
    guide = json.loads(path.read_text())
    views = json.loads((path.parent / "views.json").read_text())
    if (
        scan.source_hash != guide["source_hash"]
        or scan.source_hash != views["source_hash"]
    ):
        raise ValueError("Guidance is bound to a different original scan")
    cams = {c["id"]: c for c in views["cameras"]}
    annotations = [o for items in guide.get("landmarks", {}).values() for o in items]
    annotations.extend(guide.get("skin", []))
    annotations.extend(guide.get("skin_strokes", []))
    for piece in guide.get("garments", []):
        annotations.extend(piece.get("regions", []))
        annotations.extend(piece.get("strokes", []))
    for name in {o["view"] for o in annotations}:
        if name not in cams:
            raise ValueError(f"Missing marked camera: {name}")
        digest = hashlib.sha256(
            (path.parent / "views" / (name + ".png")).read_bytes()
        ).hexdigest()
        pinned = guide.get("view_sha256", {}).get(name)
        recorded = views.get("images", {}).get(name, {}).get("sha256")
        if not (pinned or recorded):
            raise ValueError(f"Missing reviewed image hash for marked view: {name}")
        if (recorded and digest != recorded) or (pinned and pinned != digest):
            raise ValueError(f"Marked image changed: {name}")
    return guide, views, cams


def resolve(scan, guide, cams, matrix):
    """Triangulate joint marks and project visible surface marks into the working frame."""
    import igl
    from .scan import transform_points

    # Ray intersections are more stable in millimetres on densely tessellated scans.
    vertices = scan.vertices * 1000.0
    tree = igl.AABB()
    tree.init(vertices, scan.faces)
    span = np.linalg.norm(np.ptp(vertices, axis=0)) * 2
    landmarks, sections, outlines = {}, {}, []
    for name, marks in guide["landmarks"].items():
        if not marks:
            continue
        joint = name.startswith(
            (
                "neck",
                "shoulder_",
                "elbow_",
                "wrist_",
                "hip_",
                "knee_",
                "ankle_",
                "finger",
                "palm_",
            )
        )
        rays, weights = [], []
        for mark in marks:
            c = cams[mark["view"]]
            x, y = mark["px"]
            if not 0 <= x < c["width"] or not 0 <= y < c["height"]:
                raise ValueError(f"Landmark {name} outside {c['id']}")
            weight = mark.get("confidence", 1.0) * {
                "occluded": 0.5,
                "partial": 0.8,
            }.get(mark.get("visibility"), 1.0)
            if weight <= 0:
                continue
            o = np.array(c["center"]) + (x - c["width"] / 2) * c[
                "meters_per_pixel"
            ] * np.asarray(c["right"])
            o += (c["height"] / 2 - y) * c["meters_per_pixel"] * np.asarray(c["up"])
            rays.append((o, c["forward"]))
            weights.append(weight)
        if not rays:
            continue
        origin, direction = (np.asarray(v) for v in zip(*rays))
        projectors = np.eye(3) - direction[:, :, None] * direction[:, None, :]
        sums = np.einsum("i,ijk->jk", weights, projectors)
        if joint and np.linalg.cond(sums) < 100:
            point = np.linalg.solve(
                sums, np.einsum("i,ijk,ik->j", weights, projectors, origin)
            )
        else:
            front = origin * 1000 - span * direction
            hits, distances, _ = tree.intersect_ray_first(
                vertices, scan.faces, front, direction
            )
            valid = hits >= 0
            if not valid.any():
                if name.startswith("outline_"):
                    continue
                raise ValueError(f"Landmark {name} misses scan in marked views")
            points = (front[valid] + distances[valid, None] * direction[valid]) / 1000
            if joint:
                back = origin * 1000 + span * direction
                ids, reverse, _ = tree.intersect_ray_first(
                    vertices, scan.faces, back, -direction
                )
                if np.any(ids[valid] < 0):
                    raise ValueError(f"Cannot resolve joint depth for {name}")
                rear = (back[valid] - reverse[valid, None] * direction[valid]) / 1000
                points = (rear + points) / 2
            point = np.average(points, axis=0, weights=np.asarray(weights)[valid])
        world = transform_points(point, matrix).tolist()
        if name.startswith("outline_"):
            outlines.append({"name": name, "point": world})
        elif joint and len(rays) == 1:
            sections[name] = {"point": world}
        else:
            landmarks[name] = world
    return dict(
        landmarks=landmarks,
        joint_section_depth=sections,
        outlines=outlines,
        subject=guide["subject"],
    )


def refine(scan, path):
    guide, views, cams = load_annotations(scan, path)
    points = resolve(scan, guide, cams, np.eye(4))["landmarks"]
    left, right = (np.asarray(points[key]) for key in ("eye_l", "eye_r"))
    eyes = (left + right) / 2
    span = np.linalg.norm(left - right)
    mouths = [points[k] for k in ("mouth_l", "mouth_r") if k in points]
    center = (
        eyes + (np.mean(mouths, axis=0) if mouths else eyes - [0.0, span, 0.0])
    ) / 2
    front = float(guide["front_view"].removeprefix("body_y"))
    views_new = [
        camera("face_" + name, center, front + yaw, 768, 768, span * 3.2)
        for name, yaw in (("y000", 0), ("l35", 35), ("r35", -35))
    ]
    write_views(scan, path.parent, views_new, views)
