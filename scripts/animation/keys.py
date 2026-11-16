"""Canonicalize duplicate samples in imported GLBs when producing a new animation asset."""

import numpy as np

from roster.mesh import Glb


def deduplicate(model):
    glb = Glb(model)
    doc = glb.doc
    fixed = 0

    def add(values, original, inputs=False):
        values = np.ascontiguousarray(values, dtype="<f4")
        view = len(doc["bufferViews"])
        glb.blob.extend(bytes(-len(glb.blob) % 4))
        doc["bufferViews"].append(
            dict(buffer=0, byteOffset=len(glb.blob), byteLength=values.nbytes)
        )
        glb.blob.extend(values.tobytes())
        idx = len(doc["accessors"])
        acc = dict(
            bufferView=view,
            componentType=5126,
            count=len(values),
            type=original["type"],
        )
        if inputs:
            acc.update(min=[float(values.min())], max=[float(values.max())])
        doc["accessors"].append(acc)
        return idx

    for clip in doc.get("animations", []):
        for sampler in clip["samplers"]:
            times = glb.array(sampler["input"]).flatten()
            if np.any(np.diff(times) < 0):
                raise ValueError(f"Animation times decrease in {clip.get('name')}")
            keep = np.r_[np.diff(times) > 0, True]
            if keep.all():
                continue
            # Duplicate times sample the last key at that instant. Preserve the
            # same right-continuous evaluation used by the source reader.
            idx = sampler["output"]
            values = glb.array(idx).copy()
            out = values.reshape(len(times), -1, values.shape[1])[keep].reshape(
                -1, values.shape[1]
            )
            sampler["output"] = add(out, doc["accessors"][idx])
            sampler["input"] = add(
                times[keep], doc["accessors"][sampler["input"]], True
            )
            fixed += 1
    if fixed:
        doc["buffers"][0]["byteLength"] = len(glb.blob)
        glb.write(model)
    return fixed
