"""Render textured rest or motion samples from the exported GLB with a CPU z-buffer."""

import argparse
import io
from pathlib import Path

import numpy as np
from animation.surface import Surface
from PIL import Image
from transfer.guide import camera, project, rasterize
from transfer.scan import Scan


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    return np.where(
        rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055
    )


def shade(mesh, faces, bary, vertex_color, lighting, clay):
    if clay:
        return linear_to_srgb(srgb_to_linear(0.7) * lighting) * np.ones((1, 3))
    uv = np.einsum("ij,ijk->ik", bary, mesh.uv[mesh.faces[faces]])
    material_ids = mesh.face_material[faces]
    result = np.empty((len(faces), 3))
    for mid in np.unique(material_ids):
        selected = material_ids == mid
        material = mesh.materials[mid] if mid >= 0 else {}
        pbr = material.get("pbrMetallicRoughness", {})
        diffuse = vertex_color[selected] * pbr.get("baseColorFactor", [1, 1, 1, 1])[:3]
        if "baseColorTexture" in pbr:
            diffuse *= mesh.texture_rgb(pbr["baseColorTexture"], uv[selected])
        if "KHR_materials_unlit" in material.get("extensions", {}):
            result[selected] = diffuse
            continue
        emission = np.asarray(material.get("emissiveFactor", [0, 0, 0]))
        if np.any(emission) and "emissiveTexture" in material:
            emission = emission * mesh.texture_rgb(
                material["emissiveTexture"], uv[selected]
            )
        strength = (
            material.get("extensions", {})
            .get("KHR_materials_emissive_strength", {})
            .get("emissiveStrength", 1)
        )
        result[selected] = diffuse * lighting[selected] + emission * strength
    return linear_to_srgb(result)


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
            srgb_to_linear(
                np.asarray(
                    Image.open(
                        io.BytesIO(g.blob[start : start + view["byteLength"]])
                    ).convert("RGB"),
                    dtype=np.float32,
                )
                / 255
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
            vertex_color = np.einsum("ij,ijk->ik", weights, colors[f])
            normal = np.cross(
                points[f[:, 1]] - points[f[:, 0]], points[f[:, 2]] - points[f[:, 0]]
            )
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
            light = np.array([0.3, 0.5, 0]) - c["forward"]
            lighting = (0.5 + 0.5 * np.abs(normal @ (light / np.linalg.norm(light))))[
                :, None
            ]
            rgb = shade(mesh, face, weights, vertex_color, lighting, clay)
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
