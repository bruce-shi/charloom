"""Review mesh interchange for Blender and the CV assembly tools."""

import json, struct
from pathlib import Path
import numpy as np


class Glb:
    def __init__(self, path):
        if __package__:
            from .gltf_buffers import chunks, pack
        else:
            from gltf_buffers import chunks, pack

        path = Path(path)
        raw = path.read_bytes()
        data, binary = chunks(raw) if raw.startswith(b"glTF") else (raw, b"")
        self.doc = json.loads(data)
        self.blob = pack(self.doc, path.parent, binary)

    def array(self, index):
        a = self.doc["accessors"][index]
        types = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
        dtype = np.dtype(types[a["componentType"]])
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        def view(info, count, size, kind):
            b = self.doc["bufferViews"][info["bufferView"]]
            return np.ndarray((count, size), dtype=kind, buffer=self.blob,
                              offset=b.get("byteOffset", 0) + info.get("byteOffset", 0),
                              strides=(b.get("byteStride", size * kind.itemsize), kind.itemsize))
        result = (view(a, a["count"], width, dtype) if "bufferView" in a else
                  np.zeros((a["count"], width), dtype=dtype))
        if "sparse" in a:
            sparse = a["sparse"]
            ids = sparse["indices"]
            rows = view(ids, sparse["count"], 1, np.dtype(types[ids["componentType"]])).ravel()
            result = result.copy()
            result[rows] = view(sparse["values"], sparse["count"], width, dtype)
        return result

    def body(self):
        return max(
            (p for m in self.doc["meshes"] for p in m["primitives"]),
            key=lambda p: self.doc["accessors"][p["attributes"]["POSITION"]]["count"],
        )

    def write(self, path):
        data = json.dumps(self.doc, separators=(",", ":")).encode()
        data += b" " * (-len(data) % 4)
        blob = bytes(self.blob) + b"\0" * (-len(self.blob) % 4)
        Path(path).write_bytes(
            struct.pack("<III", 0x46546C67, 2, 28 + len(data) + len(blob))
            + struct.pack("<II", len(data), 0x4E4F534A)
            + data
            + struct.pack("<II", len(blob), 0x004E4942)
            + blob
        )


def adjacency(t, n):
    from scipy import sparse

    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    row, col = np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]]
    a = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    degree = np.asarray(a.sum(1)).ravel()
    return a, sparse.eye(n, format="csr") - sparse.diags(1 / np.maximum(degree, 1)) @ a


def fold_edges(x, rest, t, head):
    edge = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    ids = np.tile(np.arange(len(t)), 3)
    edge = np.sort(edge, axis=1)
    order = np.lexsort(edge.T[::-1])
    edge, ids = edge[order], ids[order]
    start = np.r_[0, np.flatnonzero(np.any(edge[1:] != edge[:-1], axis=1)) + 1]
    count = np.diff(np.r_[start, len(edge)])
    pair = start[count == 2]
    pair = pair[head[edge[pair]].all(1)]
    a, b = ids[pair], ids[pair + 1]

    def bend(p):
        n = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-20)
        return np.arccos(np.clip((n[a] * n[b]).sum(1), -1, 1))

    bad = bend(x) - bend(rest) > 0.8
    return int(bad.sum())
