"""Inputs and mesh helpers for the two reconstructed head routes."""

import hashlib, os, sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[2]

batch = Path(
    os.getenv("CHARLOOM_TRANSFER_DIR", root / "out/transfers")
)
dst = Path(os.getenv("CHARLOOM_HEAD_DIR", root / "out/characters"))
sys.path.insert(0, str(root / "scripts"))
from roster.mesh import Glb as G
from roster.mesh_output import add, finish, idx, material, normals, texture


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def place_eye_joint(g, names, side, center):
    skin = g.doc["skins"][0]
    joint = names.index("eye." + side.upper())
    node_id = skin["joints"][joint]
    node = g.doc["nodes"][node_id]
    inverse = g.array(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    world = np.linalg.inv(inverse[joint]).astype(float)
    parent = next(
        i
        for i, part in enumerate(g.doc["nodes"])
        if node_id in part.get("children", [])
    )
    parent_world = np.linalg.inv(inverse[skin["joints"].index(parent)])
    node["translation"] = (
        node["translation"]
        + np.linalg.solve(parent_world[:3, :3], center - world[:3, 3])
    ).tolist()
    world[:3, 3] = center
    inverse[joint] = np.linalg.inv(world)
    return joint


def prim(g, name):
    return next(m["primitives"][0] for m in g.doc["meshes"] if m.get("name") == name)


def attr(g, p, k):
    return g.array(p["attributes"][k])


def zipper(a, b):
    faces = []
    i = j = 0
    while i < len(a) or j < len(b):
        if (i + 1) * len(b) < (j + 1) * len(a):
            faces.append([a[i % len(a)], a[(i + 1) % len(a)], b[j % len(b)]])
            i += 1
        else:
            faces.append([a[i % len(a)], b[(j + 1) % len(b)], b[j % len(b)]])
            j += 1
    return faces
