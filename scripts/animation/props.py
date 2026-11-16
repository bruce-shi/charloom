"""Append embedded, static glTF props without rebuilding the character's skin or materials."""

from copy import deepcopy

from roster.mesh import Glb


def add_props(model, props, base):
    if not props:
        return
    out = Glb(model)
    dst = out.doc
    for prop in props:
        g = Glb(base / prop["path"])
        src = g.doc
        if src.get("skins") or src.get("animations") or src.get("extensionsRequired"):
            raise ValueError("Expected an embedded static glTF prop")
        if any(n.get("name") == prop["name"] for n in dst["nodes"]):
            raise ValueError(f"Duplicate prop name {prop['name']}")
        sections = (
            "bufferViews",
            "accessors",
            "images",
            "samplers",
            "textures",
            "materials",
            "meshes",
            "nodes",
            "cameras",
        )
        offset = {k: len(dst.setdefault(k, [])) for k in sections}
        data = deepcopy(src)
        out.blob.extend(bytes(-len(out.blob) % 4))
        for v in data.get("bufferViews", []):
            v["byteOffset"] = len(out.blob) + v.get("byteOffset", 0)
        out.blob.extend(g.blob)
        for acc in data.get("accessors", []):
            for obj in [acc, *acc.get("sparse", {}).values()]:
                if isinstance(obj, dict) and "bufferView" in obj:
                    obj["bufferView"] += offset["bufferViews"]
        for img in data.get("images", []):
            if "bufferView" not in img:
                raise ValueError("Pack prop images into the GLB")
            img["bufferView"] += offset["bufferViews"]
        for tex in data.get("textures", []):
            tex["source"] += offset["images"]
            if "sampler" in tex:
                tex["sampler"] += offset["samplers"]

        def adjust_material(fields):
            for key, val in fields.items():
                if isinstance(val, dict):
                    if key.endswith("Texture") and "index" in val:
                        val["index"] += offset["textures"]
                    else:
                        adjust_material(val)

        for mat in data.get("materials", []):
            adjust_material(mat)
        for mesh in data.get("meshes", []):
            for p in mesh["primitives"]:
                for values in [p["attributes"], *p.get("targets", [])]:
                    for k in values:
                        values[k] += offset["accessors"]
                if "indices" in p:
                    p["indices"] += offset["accessors"]
                if "material" in p:
                    p["material"] += offset["materials"]
        for node in data["nodes"]:
            for key, section in (("mesh", "meshes"), ("camera", "cameras")):
                if key in node:
                    node[key] += offset[section]
            if "children" in node:
                node["children"] = [i + offset["nodes"] for i in node["children"]]
        for k in sections:
            dst[k].extend(data.get(k, []))
        idx = len(dst["nodes"])
        children = [
            i + offset["nodes"] for i in data["scenes"][data.get("scene", 0)]["nodes"]
        ]
        dst["nodes"].append(
            dict(name=prop["name"], children=children, **prop.get("transform", {}))
        )
        dst["scenes"][dst.get("scene", 0)]["nodes"].append(idx)
    dst["buffers"][0]["byteLength"] = len(out.blob)
    out.write(model)
