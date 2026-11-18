"""Restore a recorded walking action onto its original bind skeleton."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .mesh import Glb


def restore_walk(model, original, output):
    source, target = Glb(original), Glb(model)
    palettes = [g.doc["skins"][0] for g in (source, target)]
    joints, dest = (s["joints"] for s in palettes)
    if len(joints) != len(dest) or not np.array_equal(
            source.array(palettes[0]["inverseBindMatrices"]), target.array(palettes[1]["inverseBindMatrices"])):
        raise ValueError("Incompatible walking bind palette")
    maps = [{child: i for i, node in enumerate(g.doc["nodes"]) for child in node.get("children", [])}
            for g in (source, target)]
    names = set()
    mapping = dict(zip(joints, dest))
    for i, j in mapping.items():
        name = source.doc["nodes"][i]["name"]
        if name in names:
            raise ValueError(f"Ambiguous walking joint: {name}")
        names.add(name)
        while i is not None and j is not None:
            a, b = source.doc["nodes"][i], target.doc["nodes"][j]
            for key, default in (("name", ""), ("translation", [0, 0, 0]), ("rotation", [0, 0, 0, 1]),
                                 ("scale", [1, 1, 1]), ("matrix", None)):
                if a.get(key, default) != b.get(key, default):
                    raise ValueError(f"Walking bind transform differs for {name}")
            i, j = maps[0].get(i), maps[1].get(j)
        if i != j:
            raise ValueError(f"Walking hierarchy differs for {name}")
    clip = next((deepcopy(a) for a in source.doc.get("animations", []) if a.get("name") == "walking"), None)
    if clip is None:
        raise ValueError("Source has no walking clip")
    for channel in clip["channels"]:
        node = channel["target"]
        if node["path"] not in ("translation", "rotation", "scale") or node["node"] not in mapping:
            raise ValueError("Expected a skeletal walking channel")
        node["node"] = mapping[node["node"]]
    copied = {}
    for sampler in clip["samplers"]:
        for key in ("input", "output"):
            idx = sampler[key]
            if idx not in copied:
                acc = deepcopy(source.doc["accessors"][idx])
                if "sparse" in acc:
                    raise ValueError("Sparse walking samples require expansion")
                data = source.array(idx).tobytes()
                target.blob.extend(bytes(-len(target.blob) % 4))
                acc["bufferView"] = len(target.doc["bufferViews"])
                acc.pop("byteOffset", None)
                target.doc["bufferViews"].append({"buffer": 0, "byteOffset": len(target.blob), "byteLength": len(data)})
                target.blob.extend(data)
                copied[idx] = len(target.doc["accessors"])
                target.doc["accessors"].append(acc)
            sampler[key] = copied[idx]
    target.doc["animations"] = [a for a in target.doc.get("animations", []) if a.get("name") != "walking"] + [clip]
    extras = target.doc["asset"].setdefault("extras", {})
    extras.setdefault("charloom_experimental_preview", {})["neutral_only"] = False
    target.doc["buffers"][0]["byteLength"] = len(target.blob)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    target.write(output)
    return len(clip["channels"])
