"""Evaluate the installed MakeHuman MHCLO basewear on the fitted hm08 surface."""

import os
from pathlib import Path
import igl
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .scaffold import bary_weights


def read_obj(path, body_only=False):
    positions, uv, polygons, group = [], [], [], ""
    for line in path.read_text().splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "v":
            positions.append([float(x) for x in f[1:4]])
        elif f[0] == "vt":
            uv.append([float(f[1]), 1.0 - float(f[2])])
        elif f[0] == "g":
            group = f[1]
        elif f[0] == "f" and (not body_only or group == "body"):
            corners = np.array(
                [[int(i) - 1 for i in token.split("/")[:2]] for token in f[1:]]
            )
            polygons.extend(corners[[0, j, j + 1]] for j in range(1, len(corners) - 1))
    return np.array(positions), np.array(uv), np.array(polygons)


def add_basewear(g, body, prior, sex):
    from roster.mesh_output import add, idx, normals, texture

    root = Path(__file__).resolve().parents[2]
    assets = Path(
        os.environ.get(
            "CHARLOOM_UNDERWEAR_ASSETS", root / "out/tools/mpfb/underwear-candidates"
        )
    )
    base = assets / "base.obj"
    if not base.is_file():
        base = next(
            iter(
                sorted(
                    (assets.parent / "source").glob("*/src/mpfb/data/3dobjs/base.obj")
                )
            ),
            None,
        )
    if base is None or not assets.is_dir():
        return {"available": False, "assets": []}
    raw, uv, base_faces = read_obj(base, True)
    a = body["attributes"]
    positions = g.array(a["POSITION"]).astype(float)
    tri = g.array(body["indices"]).reshape(-1, 3).astype(np.int64)
    rows = base_faces.reshape(-1, 2)
    dist, near = cKDTree(uv[rows[:, 1]]).query(g.array(a["TEXCOORD_0"]))
    if (dist > 2e-4).any():
        raise ValueError("Body UVs do not match the hm08 base for underwear")
    hm = rows[near, 0]
    fitted = np.zeros_like(raw)
    counts = np.bincount(hm, minlength=len(raw))
    np.add.at(fitted, hm, positions)
    fitted /= np.maximum(counts[:, None], 1)
    # The body helpers used by some clothes follow a barycentric point on
    # the anatomical surface, retaining their authored offset in decimeters.
    helper = counts == 0
    faces = base_faces[:, :, 0]
    tree = igl.AABB()
    tree.init(raw, faces)
    _, ids, close = tree.squared_distance(raw, faces, raw[helper])
    neighbor = faces[ids]
    bary = igl.barycentric_coordinates(close, *(raw[neighbor[:, i]] for i in range(3)))
    fitted[helper] = (
        np.einsum("ij,ijk->ik", bary, fitted[neighbor]) + (raw[helper] - close) * 0.1
    )
    bodytree = igl.AABB()
    bodytree.init(positions, tri)
    recommendations = (
        (
            ("underwear02", "punkduck_sport_bra"),
            ("underwear02", "punkduck_sport_briefs"),
        )
        if sex == "female"
        else (("pants03", "mindfront_male_swimming_trunks_03"),)
    )
    credits = []
    bnode = next(n for n in g.doc["nodes"] if n.get("name") == "body" and "skin" in n)
    verts_j, verts_w = g.array(a["JOINTS_0"]).copy(), g.array(a["WEIGHTS_0"]).copy()
    for pack, name in recommendations:
        folder = assets / pack / "clothes" / name
        source = folder / (name + ".mhclo")
        if not source.exists():
            raise FileNotFoundError(source)
        header = {}
        refs = []
        in_refs = False
        for line in source.read_text().splitlines():
            row = line.lstrip("# ").split()
            if not row:
                continue
            if in_refs and row[0].isdigit():
                numbers = list(map(float, row))
                refs.append(
                    numbers
                    if len(numbers) == 9
                    else [numbers[0], 0, 0, 1, 0, 0, 0, 0, 0]
                )
            else:
                in_refs = row[0] == "verts"
                header[row[0]] = " ".join(row[1:])
        if not header.get("license") or not header.get("author"):
            raise ValueError(f"Missing source attribution for {name}")
        _, cloth_uv, corners = read_obj(folder / header["obj_file"])
        scale = np.full(3, 0.1)
        for axis, key in enumerate(("x_scale", "y_scale", "z_scale")):
            if key in header:
                p, q, den = header[key].split()
                scale[axis] = abs(fitted[int(p), axis] - fitted[int(q), axis]) / float(
                    den
                )
        refs = np.array(refs)
        coords = (
            np.einsum("ij,ijk->ik", refs[:, 3:6], fitted[refs[:, :3].astype(int)])
            + refs[:, 6:] * scale
        )
        dist, ids, close = bodytree.squared_distance(positions, tri, coords)
        near_tri = tri[ids]
        n = np.cross(
            positions[near_tri[:, 1]] - positions[near_tri[:, 0]],
            positions[near_tri[:, 2]] - positions[near_tri[:, 0]],
        )
        n /= np.maximum(1e-20, np.linalg.norm(n, axis=1, keepdims=True))
        gap = ((coords - close) * n).sum(1)
        coords += np.maximum(0, 0.003 - gap)[:, None] * n
        blend = igl.barycentric_coordinates(
            close, *(positions[near_tri[:, k]] for k in range(3))
        )
        ji, wt = bary_weights(verts_j, verts_w, near_tri, blend)
        render, faces_flat = np.unique(
            corners.reshape(-1, 2), axis=0, return_inverse=True
        )
        v, tid = render.T
        faces = faces_flat.reshape(-1, 3)
        props = {}
        for line in (folder / header["material"]).read_text().splitlines():
            key, _, val = line.strip().partition(" ")
            props[key] = val
        mat = texture(
            g,
            np.asarray(Image.open(folder / props["diffuseTexture"]).convert("RGB")),
            name,
        )
        credit = {
            "asset": name,
            "author": header["author"],
            "license": header["license"],
        }
        g.doc["materials"][mat]["extras"] = credit
        attrs = {
            "POSITION": add(g, coords[v].astype("<f4"), "VEC3"),
            "TEXCOORD_0": add(g, cloth_uv[tid].astype("<f4"), "VEC2"),
            "NORMAL": add(g, normals(coords[v], faces)[0].astype("<f4"), "VEC3"),
            "JOINTS_0": add(g, ji[v], "VEC4", 5123),
            "WEIGHTS_0": add(g, wt[v], "VEC4"),
        }
        node_name = "underwear_" + name
        node = dict(
            name=node_name,
            skin=bnode["skin"],
            mesh=len(g.doc["meshes"]),
            extras={"charloom_part": "underwear"},
        )
        g.doc["meshes"].append(
            dict(
                name=node_name,
                primitives=[
                    dict(attributes=attrs, indices=idx(g, faces), material=mat)
                ],
            )
        )
        g.doc["scenes"][g.doc.get("scene", 0)]["nodes"].append(len(g.doc["nodes"]))
        g.doc["nodes"].append(node)
        credits.append(credit)
    return {"available": True, "assets": credits}
