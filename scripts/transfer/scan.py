"""Original static GLB geometry, materials and coordinate frames for scan transfer."""

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
from roster.mesh import Glb
from scipy.spatial.transform import Rotation


def sample_rgb(image, uv):
    """Bilinearly sample an RGB texture with repeating UV coordinates."""
    height, width = image.shape[:2]
    point = (uv % 1.0) * [width, height] - 0.5
    floor = np.floor(point).astype(int)
    frac = point - floor
    rgb = np.zeros((len(point), 3))
    for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        w = (frac[:, 0] if dx else 1 - frac[:, 0]) * (
            frac[:, 1] if dy else 1 - frac[:, 1]
        )
        rgb += (
            w[:, None]
            * image[(floor[:, 1] + dy) % height, (floor[:, 0] + dx) % width]
        )
    return rgb


def transforms(doc):
    local = []
    for node in doc["nodes"]:
        if "matrix" in node:
            matrix = np.array(node["matrix"], dtype=float).reshape(4, 4).T
        else:
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(
                node.get("rotation", [0, 0, 0, 1])
            ).as_matrix()
            matrix[:3, :3] *= node.get("scale", [1, 1, 1])
            matrix[:3, 3] = node.get("translation", [0, 0, 0])
        local.append(matrix)
    parents = {
        child: i
        for i, node in enumerate(doc["nodes"])
        for child in node.get("children", [])
    }
    worlds = {}

    def world(i):
        if i not in worlds:
            worlds[i] = world(parents[i]) @ local[i] if i in parents else local[i]
        return worlds[i]

    return np.array([world(i) for i in range(len(local))]), parents


def transform_points(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


class Scan:
    """Load every triangle primitive of the default scene, in glTF Y-up coordinates."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.glb = g = Glb(self.path)
        doc = g.doc
        self.materials = doc.get("materials", [])
        length = doc["buffers"][0]["byteLength"]
        with self.path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256")
        digest.update(length.to_bytes(8, "little"))
        digest.update(memoryview(g.blob)[:length])
        self.textures = []
        for i in doc.get("images", []):
            if "bufferView" not in i:
                raise ValueError(
                    "Pack image resources into the scan GLB before transfer"
                )
            view = doc["bufferViews"][i["bufferView"]]
            start = view.get("byteOffset", 0)
            self.textures.append(
                np.asarray(
                    Image.open(
                        io.BytesIO(g.blob[start : start + view["byteLength"]])
                    ).convert("RGB")
                )
            )
        self.source_hash = digest.hexdigest()
        worlds, _ = transforms(doc)
        nodes = doc["nodes"]
        visible = set()
        todo = list(doc["scenes"][doc.get("scene", 0)]["nodes"])
        while todo:
            i = todo.pop()
            visible.add(i)
            todo.extend(nodes[i].get("children", []))
        positions, uv, faces, materials = [], [], [], []
        offset = 0
        for i, node in enumerate(nodes):
            if i not in visible or "mesh" not in node:
                continue
            if "skin" in node:
                raise ValueError("Export a static scan before transfer")
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    raise ValueError("Expected triangle primitives")
                attributes = prim["attributes"]
                v = transform_points(g.array(attributes["POSITION"]), worlds[i])
                t = (
                    g.array(prim["indices"]).ravel()
                    if "indices" in prim
                    else np.arange(len(v))
                ).reshape(-1, 3)
                positions.append(v)
                uv.append(
                    g.array(attributes["TEXCOORD_0"])
                    if "TEXCOORD_0" in attributes
                    else np.zeros((len(v), 2))
                )
                faces.append(t.astype(np.int64) + offset)
                materials.append(np.full(len(t), prim.get("material", -1), dtype=int))
                offset += len(v)
        self.vertices = np.concatenate(positions)
        self.faces = np.concatenate(faces)
        self.uv = np.concatenate(uv)
        self.face_material = np.concatenate(materials)

    def texture_rgb(self, texture, uv):
        image = self.textures[self.glb.doc["textures"][texture["index"]]["source"]]
        return sample_rgb(image, uv)

    def colors(self, face_ids, barycentric):
        """Sample the original base color and factor with bilinear interpolation."""
        mids = self.face_material[face_ids]
        uv = np.einsum("ij,ijk->ik", barycentric, self.uv[self.faces[face_ids]])
        colors = np.full((len(face_ids), 3), 0.7)
        for mid in np.unique(mids):
            if mid < 0:
                continue
            use = mids == mid
            material = self.materials[mid].get("pbrMetallicRoughness", {})
            factor = np.asarray(material.get("baseColorFactor", [1, 1, 1, 1])[:3])
            tex = material.get("baseColorTexture")
            if tex is None:
                colors[use] = factor
                continue
            colors[use] = self.texture_rgb(tex, uv[use]) * factor / 255
        return colors

    def working_frame(self, prior, front_view):
        body = prior.array(prior.body()["attributes"]["POSITION"])
        yaw = -float(front_view.removeprefix("body_y"))
        rotate = Rotation.from_euler("y", yaw, degrees=True).as_matrix()
        points = self.vertices @ rotate.T
        low, high = points.min(0), points.max(0)
        height = np.ptp(body[:, 1])
        scale = height / (high[1] - low[1])
        center = (body.max(0) + body.min(0)) / 2
        center[1] = 0
        shift = -(low + high) / 2
        shift[1] = -low[1]
        result = np.eye(4)
        result[:3, :3] = scale * rotate
        result[:3, 3] = center + scale * shift
        return result
