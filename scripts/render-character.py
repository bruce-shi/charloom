"""Render textured rest or motion samples from the exported GLB with a CPU z-buffer."""

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image

from animation.surface import Surface
from transfer.guide import camera, project, rasterize
from transfer.scan import Scan


def render_views(
    model,
    output,
    *,
    clip=None,
    times=(0.0,),
    selected=(),
    crop=None,
    clay=False,
    angles=None,
):
    surface = Surface(model)
    scene = surface.scene
    g = scene.glb
    parts = [
        (i, p)
        for i, p in enumerate(surface.parts)
        if not selected or any(scene.names[p.node].startswith(n) for n in selected)
    ]
    if not parts:
        raise ValueError(f"No visible primitives matching {selected}")
    mesh = object.__new__(Scan)
    mesh.glb = g
    mesh.materials = g.doc.get("materials", [])
    mesh.textures = []
    for img in g.doc.get("images", []):
        view = g.doc["bufferViews"][img["bufferView"]]
        start = view.get("byteOffset", 0)
        mesh.textures.append(
            np.asarray(
                Image.open(
                    io.BytesIO(g.blob[start : start + view["byteLength"]])
                ).convert("RGB")
            )
        )
    uv = []
    colors = []
    tri = []
    materials = []
    base = 0
    for _, p in parts:
        a = p.prim["attributes"]
        uv.append(
            g.array(a["TEXCOORD_0"])
            if "TEXCOORD_0" in a
            else np.zeros((len(p.positions), 2))
        )
        col = (
            g.array(a["COLOR_0"])[:, :3]
            if "COLOR_0" in a
            else np.ones_like(p.positions)
        )
        colors.append(col / np.iinfo(col.dtype).max if col.dtype.kind in "iu" else col)
        tri.append(p.faces + base)
        materials.extend([p.prim.get("material", -1)] * len(p.faces))
        base += len(p.positions)
    mesh.uv = np.vstack(uv)
    mesh.faces = np.vstack(tri)
    mesh.face_material = np.array(materials)
    colors = np.vstack(colors)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for t in times:
        all_points = surface.evaluate(*scene.sample(clip, t)) if clip else surface.rest
        points = np.vstack([all_points[i] for i, _ in parts])
        framing = (
            points
            if not crop
            else np.vstack(
                [
                    all_points[i]
                    for i, p in parts
                    if scene.names[p.node].startswith(crop)
                ]
            )
        )
        lo, hi = framing.min(0), framing.max(0)
        center = (lo + hi) * 0.5
        for side, angle in (angles or {"front": 0, "side": 90, "back": 180}).items():
            c = camera(
                side,
                center,
                angle,
                512,
                640,
                max(np.max((hi - lo)[[0, 2]]) * 1.28, (hi[1] - lo[1]) * 1.1),
            )
            ids, bary = rasterize(project(c, points), mesh.faces, 512, 640)
            mask = ids >= 0
            face = ids[mask]
            weights = bary[mask]
            f = mesh.faces[face]
            rgb = mesh.colors(face, weights) * np.einsum(
                "ij,ijk->ik", weights, colors[f]
            )
            if clay:
                rgb[:] = 0.7
            normal = np.cross(
                points[f[:, 1]] - points[f[:, 0]], points[f[:, 2]] - points[f[:, 0]]
            )
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
            light = np.array([0.3, 0.5, 0]) - c["forward"]
            rgb *= (0.5 + 0.5 * np.abs(normal @ (light / np.linalg.norm(light))))[
                :, None
            ]
            pixels = np.full((640, 512, 3), 105, np.uint8)
            pixels[mask] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            label = f"{clip}-{t:.2f}" if clip else "rest"
            tag = ("clay-" if clay else "") + (crop + "-" if crop else "")
            path = output / f"{model.parent.name}-{model.stem}-{tag}{side}-{label}.png"
            Image.fromarray(pixels).save(path)
            files.append(str(path))
    return files


def render_run(run, groups):
    output = run / "renders"
    files = []
    for tag, rel in (
        ("segment", "1-segment/segmentation.glb"),
        ("body", "3-skin/body.glb"),
        ("dressed", "4-garments/character.glb"),
    ):
        if tag in groups and (run / rel).is_file():
            files += render_views(run / rel, output)
    if "walk" in groups:
        model = run / "4-garments/character.glb"
        scene = Surface(model).scene
        if "walking" in scene.clips:
            _, end = scene.interval("walking")
            files += render_views(
                model, output, clip="walking", times=[0.25 * end, 0.60 * end]
            )
    if "chest" in groups:
        model = run / "3-skin/body.glb"
        for clay in (False, True):
            files += render_views(
                model,
                output,
                crop="underwear_punkduck_sport_bra",
                clay=clay,
                angles={"front": 0, "left": 60, "right": -60, "back": 180, "side": 90},
            )
    return files


if __name__ == "__main__":
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--only", default="segment,body,dressed,walk")
    a = p.parse_args()
    groups = set(a.only.split(","))
    extra = groups - {"segment", "body", "dressed", "walk", "chest"}
    if extra:
        p.error(f"Unsupported CPU render groups: {sorted(extra)}")
    files = render_run(a.run, groups)
    print(json.dumps(dict(images=len(files), files=files)))
