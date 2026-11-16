"""Embed the Gallery's sampled dialogue and expressions in an assembled GLB."""

import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

if __package__:
    from .mesh import Glb
else:
    from mesh import Glb


def write_clips(model, clips, *, extend=False):
    glb = Glb(model)
    doc = glb.doc
    glb.blob = glb.blob[: doc["buffers"][0]["byteLength"]]
    names = {clip["name"] for clip in clips}
    animations = [
        clip for clip in doc.get("animations", []) if clip.get("name") not in names
    ]

    def accessor(values, typ, time=False):
        data = np.ascontiguousarray(values, dtype="<f4")
        if data.size == 0 or not np.isfinite(data).all():
            raise ValueError("Invalid animation samples")
        glb.blob.extend(bytes(-len(glb.blob) % 4))
        view = {"buffer": 0, "byteOffset": len(glb.blob), "byteLength": data.nbytes}
        acc = {
            "bufferView": len(doc["bufferViews"]),
            "componentType": 5126,
            "count": len(data),
            "type": typ,
        }
        if time:
            if np.any(np.diff(data) <= 0):
                raise ValueError("Animation times must increase")
            acc.update(min=[float(data[0])], max=[float(data[-1])])
        doc["bufferViews"].append(view)
        doc["accessors"].append(acc)
        glb.blob.extend(data.tobytes())
        return len(doc["accessors"]) - 1

    for clip in clips:
        samplers, channels = [], []
        for track in [*clip["tracks"], *clip["morph_tracks"]]:
            morph = "node" in track
            node = track["node"] if morph else track["joint"]
            n = doc["nodes"][node]
            if "matrix" in n and not morph:
                from scipy.spatial.transform import Rotation

                mat = np.array(n.pop("matrix")).reshape(4, 4).T
                scale = np.linalg.norm(mat[:3, :3], axis=0)
                if np.linalg.det(mat[:3, :3]) < 0:
                    scale[0] *= -1
                n.update(
                    translation=mat[:3, 3].tolist(),
                    scale=scale.tolist(),
                    rotation=Rotation.from_matrix(mat[:3, :3] / scale)
                    .as_quat()
                    .tolist(),
                )
            prop = "weights" if morph else track["property"]
            values = np.asarray(track["values"])
            if len(values) != len(track["times"]):
                raise ValueError("Animation key count mismatch")
            if morph:
                mesh = doc["meshes"][doc["nodes"][node]["mesh"]]
                if any(
                    len(p["targets"]) != values.shape[1] for p in mesh["primitives"]
                ):
                    raise ValueError("Animation target count mismatch")
                values, typ = values.ravel(), "SCALAR"
            else:
                width = 4 if prop == "rotation" else 3
                values, typ = values[:, :width], "VEC" + str(width)
            channels.append(
                {"sampler": len(samplers), "target": {"node": node, "path": prop}}
            )
            samplers.append(
                {
                    "input": accessor(track["times"], "SCALAR", time=True),
                    "output": accessor(values, typ),
                    "interpolation": "LINEAR",
                }
            )
        if extend:
            existing = next(
                a for a in doc["animations"] if a.get("name") == clip["name"]
            )
            targets = {(c["target"]["node"], c["target"]["path"]) for c in channels}
            for c in existing["channels"]:
                if (c["target"]["node"], c["target"]["path"]) not in targets:
                    channels.append(dict(c, sampler=len(samplers)))
                    samplers.append(existing["samplers"][c["sampler"]])
        animations.append(
            {"name": clip["name"], "channels": channels, "samplers": samplers}
        )
    doc["animations"] = animations
    doc["buffers"][0]["byteLength"] = len(glb.blob)
    with NamedTemporaryFile(dir=model.parent, suffix=".glb", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        glb.write(path)
        os.replace(path, model)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    write_clips(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()))
