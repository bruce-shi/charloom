"""Prepare source-coloured body and separate independently skinned garment meshes."""

import sys
from copy import deepcopy
from pathlib import Path

import igl
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "blender" / "charloom"))
from garment_binding import sleeve_weights

from .garment_seams import match_layer_seam
from .guide import rasterize
from .scaffold import bary_weights
from .scan import transforms


def bake_skin(g, body, scan, vertices, skin_ids, size):
    from roster.mesh_output import texture

    uv = g.array(body["attributes"]["TEXCOORD_0"])
    tri = g.array(body["indices"]).reshape(-1, 3)
    # In glTF texture coordinates (0,0) is the top left.
    projected = np.column_stack((uv * size, np.zeros(len(uv))))
    ids, bary = rasterize(projected, tri, size, size)
    mask = ids >= 0
    pixels = np.zeros((size, size, 3), dtype="uint8")
    skin = scan.faces[skin_ids]
    tree = igl.AABB()
    tree.init(vertices, skin)
    coords = np.einsum(
        "ijk,ij->ik",
        g.array(body["attributes"]["POSITION"])[tri[ids[mask]]],
        bary[mask],
    )
    _, near, closest = tree.squared_distance(vertices, skin, coords)
    closest_tri = skin[near]
    weights = igl.barycentric_coordinates(
        closest, *(vertices[closest_tri[:, i]] for i in range(3))
    )
    colors = scan.colors(skin_ids[near], weights)
    # The nearest skin sample on a hidden chest or leg can be a distant
    # knuckle, lip or collar. Extend the observed body albedo smoothly there.
    full_tree = igl.AABB()
    full_tree.init(vertices, scan.faces)
    _, raw_ids, _ = full_tree.squared_distance(vertices, scan.faces, coords)
    height = np.ptp(vertices[:, 1])
    distances = np.linalg.norm(coords - closest, axis=1)
    visible = np.isin(raw_ids, skin_ids)
    central = (coords[:, 1] > vertices[:, 1].min() + 0.12 * height) & (
        coords[:, 1] < vertices[:, 1].min() + 0.78 * height
    )
    trusted = visible & central & (distances < 0.025 * height)
    if np.count_nonzero(trusted) >= 64:
        neutral = np.median(colors[trusted], axis=0)
    else:
        # Fully dressed scans may expose only the head. Sample the reviewed
        # skin surface itself, weighted by triangle area, so the large hidden
        # body UV area cannot repeatedly vote for one collar or beard texel.
        corners = vertices[skin]
        areas = np.linalg.norm(np.cross(corners[:, 1]-corners[:, 0],
                                        corners[:, 2]-corners[:, 0]), axis=1)
        if areas.sum() <= 0:
            raise ValueError("Reviewed skin contains no measurable surface")
        samples = scan.colors(skin_ids, np.full((len(skin_ids), 3), 1 / 3))
        order = np.argsort(samples, axis=0)
        cumulative = np.cumsum(areas[order], axis=0)
        median_rows = np.argmax(cumulative >= areas.sum() / 2, axis=0)
        neutral = samples[order[median_rows, np.arange(3)], np.arange(3)]
    blend = np.clip(1.0 - distances / (0.03 * height), 0.0, 1.0) * visible
    colors = neutral + (colors - neutral) * blend[:, None]
    pixels[mask] = np.rint(colors * 255).clip(0, 255).astype("uint8")
    from scipy.ndimage import distance_transform_edt

    nearest = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    pixels[~mask] = pixels[tuple(nearest[:, ~mask])]
    body["material"] = texture(g, pixels, "Observed body skin")


def write_models(g, scan, vertices, names, owners, out, size, prior, guide, guide_dir):
    from .cloth import bind, rig_for

    subject = guide["subject"]
    pieces = {p["name"]: p for p in guide.get("garments", [])}
    library = guide_dir / guide.get("garment_preset_library", "garment-presets")
    chains = []
    from roster.mesh_output import add, finish, idx, material, normals

    body = next(m["primitives"][0] for m in g.doc["meshes"] if m.get("name") == "body")
    skin = np.flatnonzero(owners == 0)
    if not len(skin):
        raise ValueError("No skin triangles marked on source")
    bake_skin(g, body, scan, vertices, skin, size)
    from .basewear import add_basewear

    bare = deepcopy(g)
    bare_body = next(
        m["primitives"][0] for m in bare.doc["meshes"] if m.get("name") == "body"
    )
    basewear = add_basewear(bare, bare_body, prior, subject.get("sex", "unknown"))
    finish(bare, out / "3-skin/body.glb")
    del bare
    body_node = next(
        n for n in g.doc["nodes"] if n.get("name") == "body" and "skin" in n
    )
    v = g.array(body["attributes"]["POSITION"]).astype(float)
    tri = g.array(body["indices"]).reshape(-1, 3).astype(np.int64)
    tree = igl.AABB()
    tree.init(v, tri)
    joints = g.array(body["attributes"]["JOINTS_0"]).copy()
    weights = g.array(body["attributes"]["WEIGHTS_0"]).copy()
    skin_id = body_node["skin"]
    worlds, _ = transforms(g.doc)
    skeleton = g.doc["skins"][skin_id]["joints"]
    bone_names = [g.doc["nodes"][i]["name"] for i in skeleton]
    centers = worlds[skeleton, :3, 3]
    copied_materials = {}
    reports = []
    garments = list(enumerate(names[1:], 1))
    prepared = {}
    for owner, name in sorted(
        garments, key=lambda item: pieces[item[1]].get("layer", 0)
    ):
        face_ids = np.flatnonzero(owners == owner)
        if not len(face_ids):
            raise ValueError(f"No triangles assigned to garment {name}")
        rig, fabric = rig_for(pieces[name], library)
        # Bind all material islands together before splitting the glTF primitives.
        garment_ids, remap = np.unique(scan.faces[face_ids], return_inverse=True)
        garment_faces = remap.reshape(-1, 3)
        garment_points = vertices[garment_ids]
        squared, near, closest = tree.squared_distance(v, tri, garment_points)
        neighbors = tri[near]
        bary = igl.barycentric_coordinates(
            closest, *(v[neighbors[:, k]] for k in range(3))
        )
        garment_joints, garment_weights = bary_weights(joints, weights, neighbors, bary)
        refined = None
        if rig["mode"] == "body":
            refined = sleeve_weights(
                garment_points,
                garment_faces,
                garment_joints,
                garment_weights,
                np.sqrt(squared),
                bone_names,
                centers,
            )
            if refined is not None:
                garment_joints, garment_weights = refined
                for lower_name, lower in prepared.items():
                    if lower["rig"]["mode"] != "body" or pieces[lower_name].get(
                        "layer", 0
                    ) >= pieces[name].get("layer", 0):
                        continue
                    matched = match_layer_seam(
                        (
                            garment_points,
                            garment_faces,
                            garment_joints,
                            garment_weights,
                        ),
                        lower["mesh"],
                    )
                    if matched is not None:
                        garment_joints, garment_weights = matched
        prepared[name] = dict(
            ids=garment_ids,
            face_ids=face_ids,
            rig=rig,
            fabric=fabric,
            refined=refined is not None,
            mesh=(garment_points, garment_faces, garment_joints, garment_weights),
        )

    for _, name in garments:
        entry = prepared[name]
        garment_ids, face_ids = entry["ids"], entry["face_ids"]
        rig, fabric = entry["rig"], entry["fabric"]
        garment_points, garment_faces, garment_joints, garment_weights = entry["mesh"]
        primitives = []
        materials = scan.face_material[face_ids]
        for mat in np.unique(materials):
            rows = garment_faces[materials == mat]
            selection, inverse = np.unique(rows.ravel(), return_inverse=True)
            part_faces = inverse.reshape(-1, 3)
            points = garment_points[selection]
            ji, wt = garment_joints[selection], garment_weights[selection]
            if mat not in copied_materials:
                copied_materials[mat] = material(g, scan.glb, {"material": int(mat)})
            prim = dict(
                indices=idx(g, part_faces),
                material=copied_materials[mat],
                attributes={
                    "POSITION": add(g, points.astype("<f4"), "VEC3"),
                    "NORMAL": add(
                        g, normals(points, part_faces)[0].astype("<f4"), "VEC3"
                    ),
                    "TEXCOORD_0": add(
                        g, scan.uv[garment_ids[selection]].astype("<f4"), "VEC2"
                    ),
                    "JOINTS_0": add(g, ji, "VEC4", 5123),
                    "WEIGHTS_0": add(g, wt, "VEC4"),
                },
            )
            primitives.append(prim)
        chains.extend(bind(g, primitives, body_node["skin"], name, rig, fabric))
        node = {
            "name": name,
            "mesh": len(g.doc["meshes"]),
            "skin": body_node["skin"],
            "extras": {"charloom_part": "garment"},
        }
        g.doc["meshes"].append(
            {
                "name": name,
                "primitives": primitives,
                "extras": {"charloom_part": "garment"},
            }
        )
        g.doc["scenes"][g.doc.get("scene", 0)]["nodes"].append(len(g.doc["nodes"]))
        g.doc["nodes"].append(node)
        report = {
            "name": name,
            "triangles": len(face_ids),
            "rig": rig,
            "fabric": fabric,
        }
        if entry["refined"]:
            report["flared_sleeves"] = True
        reports.append(report)
    if chains:
        g.doc.setdefault("extras", {})["charloomClothChains"] = chains
    covered = covered_triangles(g, body, scan, vertices, names, owners)
    body["indices"] = idx(g, tri[~covered])
    finish(g, out / "4-garments/character.glb")
    return dict(
        garments=reports, hidden_body_triangles=int(covered.sum()), basewear=basewear
    )


def write_segmentation(scan, vertices, owners, out):
    from roster.mesh_output import add, finish, idx

    g = deepcopy(scan.glb)
    # Split only the vertices at label boundaries, preserving source triangle order and UVs.
    pair = scan.faces * (int(owners.max()) + 1) + owners[:, None]
    unique, inverse = np.unique(pair.ravel(), return_inverse=True)
    source, labels = np.divmod(unique, int(owners.max()) + 1)
    colors = np.where((labels == 0)[:, None], [0.95, 0.55, 0.35], [0.2, 0.45, 0.9])
    attrs = {
        "POSITION": add(g, vertices[source].astype("<f4"), "VEC3"),
        "COLOR_0": add(g, colors.astype("<f4"), "VEC3"),
        "TEXCOORD_0": add(g, scan.uv[source].astype("<f4"), "VEC2"),
    }
    g.doc.update(
        nodes=[{"name": "segmentation", "mesh": 0}],
        scenes=[{"nodes": [0]}],
        scene=0,
        meshes=[
            {
                "name": "segmentation",
                "primitives": [{"attributes": attrs, "indices": idx(g, inverse)}],
            }
        ],
    )
    finish(g, out / "1-segment/segmentation.glb")


def covered_triangles(g, body, scan, vertices, names, owners):
    """Remove skin inside source clothing, retaining a strip under open hems."""
    p = g.array(body["attributes"]["POSITION"]).astype(float)
    faces = g.array(body["indices"]).reshape(-1, 3).astype(np.int64)
    v, weld = np.unique(p, axis=0, return_inverse=True)
    tris = weld[faces]
    centers = v[tris].mean(1)
    tree = igl.AABB()
    tree.init(vertices, scan.faces)
    dist, near, closest = tree.squared_distance(vertices, scan.faces, centers)
    hair = {
        i
        for i, n in enumerate(names)
        if any(k in n.lower() for k in ("hair", "brow", "lash", "tail"))
    }
    labels = owners[near]
    cloth = (labels > 0) & (labels < len(names)) & ~np.isin(labels, list(hair))
    # Use both exposed-skin and clothing evidence. The nearest garment can
    # lie just inside the fitted body at a wrinkle, where a single outward
    # ray would retain the exact skin triangle that pokes through.
    height = np.ptp(p[:, 1])
    cloth &= dist < (height * 0.08) ** 2
    for i, name in enumerate(names):
        if not any(
            k in name.lower() for k in ("boot", "shoe", "slipper", "footwear", "clog")
        ):
            continue
        shoe_points = vertices[np.unique(scan.faces[owners == i])]
        if not len(shoe_points):
            continue
        ankle_joints = np.array(
            [
                n["name"].lower().startswith(("foot", "toe"))
                for n in (g.doc["nodes"][j] for j in g.doc["skins"][0]["joints"])
            ]
        )
        a = body["attributes"]
        strength = (
            (g.array(a["WEIGHTS_0"]) * ankle_joints[g.array(a["JOINTS_0"])])
            .sum(1)[faces]
            .mean(1)
        )
        for side in (1, -1):
            half = shoe_points[shoe_points[:, 0] * side > 0]
            if len(half):
                cloth |= (
                    (centers[:, 0] * side > 0)
                    & (strength > 0.5)
                    & (centers[:, 1] < half[:, 1].max() - 0.008 * height)
                )
    return cloth
