"""Write embedded mesh attributes, indices, textures and materials into a GLB."""

import copy
import io
import numpy as np
from PIL import Image


def normals(v, t):
    q = v[t]
    f = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, t[:, k], f)
    return n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-20), f / np.maximum(
        np.linalg.norm(f, axis=1)[:, None], 1e-20
    )


def add(g, a, kind, ct=5126):
    a = np.ascontiguousarray(a)
    g.blob += b"\0" * (-len(g.blob) % 4)
    offset = len(g.blob)
    g.blob += a.tobytes()
    vi = len(g.doc["bufferViews"])
    g.doc["bufferViews"].append(
        {"buffer": 0, "byteOffset": offset, "byteLength": a.nbytes, "target": 34962}
    )
    if kind=="MAT4":
        g.doc["bufferViews"][vi].pop("target",None)
    acc = {"bufferView": vi, "componentType": ct, "count": len(a), "type": kind}
    if kind == "VEC3":
        acc.update(min=a.min(0).tolist(), max=a.max(0).tolist())
    i = len(g.doc["accessors"])
    g.doc["accessors"].append(acc)
    return i


def finish(g, p):
    g.doc["buffers"][0]["byteLength"] = len(g.blob)
    p.parent.mkdir(exist_ok=True, parents=True)
    g.write(p)


def idx(g, faces):
    i = add(g, np.asarray(faces, dtype="<u4").ravel(), "SCALAR", 5125)
    g.doc["bufferViews"][g.doc["accessors"][i]["bufferView"]]["target"] = 34963
    return i


def texture(g, img, name):
    f = io.BytesIO()
    Image.fromarray(img).save(f, format="PNG")
    data = f.getvalue()
    g.blob.extend(bytes(-len(g.blob) % 4))
    off = len(g.blob)
    g.blob.extend(data)
    vi = len(g.doc["bufferViews"])
    g.doc["bufferViews"].append(dict(buffer=0, byteOffset=off, byteLength=len(data)))
    ii = len(g.doc["images"])
    g.doc["images"].append(dict(bufferView=vi, mimeType="image/png", name=name))
    ti = len(g.doc["textures"])
    g.doc["textures"].append(dict(source=ii))
    mat = len(g.doc["materials"])
    g.doc["materials"].append(
        dict(
            name=name,
            doubleSided=True,
            pbrMetallicRoughness=dict(
                baseColorTexture=dict(index=ti), metallicFactor=0, roughnessFactor=0.92
            ),
        )
    )
    return mat


def material(g, s, sp):
    """Copy a material and its embedded texture dependencies into a derived GLB."""
    source_id = sp.get("material", -1)
    mat = (
        copy.deepcopy(s.doc["materials"][source_id])
        if source_id >= 0
        else {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}}
    )
    textures = {}

    def remap(fields):
        for key, info in fields.items():
            if not isinstance(info, dict):
                continue
            if not key.endswith("Texture") or "index" not in info:
                remap(info)
                continue
            old = info["index"]
            if old not in textures:
                tex = copy.deepcopy(s.doc["textures"][old])
                image = copy.deepcopy(s.doc["images"][tex["source"]])
                view = s.doc["bufferViews"][image["bufferView"]]
                start = view.get("byteOffset", 0)
                data = s.blob[start : start + view["byteLength"]]
                g.blob.extend(bytes(-len(g.blob) % 4))
                target = dict(buffer=0, byteOffset=len(g.blob), byteLength=len(data))
                image["bufferView"] = len(g.doc["bufferViews"])
                g.doc["bufferViews"].append(target)
                g.blob.extend(data)
                tex["source"] = len(g.doc.setdefault("images", []))
                g.doc["images"].append(image)
                if "sampler" in tex:
                    sampler = copy.deepcopy(s.doc["samplers"][tex["sampler"]])
                    tex["sampler"] = len(g.doc.setdefault("samplers", []))
                    g.doc["samplers"].append(sampler)
                textures[old] = len(g.doc.setdefault("textures", []))
                g.doc["textures"].append(tex)
            info["index"] = textures[old]

    remap(mat)
    i = len(g.doc.setdefault("materials", []))
    g.doc["materials"].append(mat)
    return i
