"""Open the source assets and metadata of a saved Charloom project in an editable .blend."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

import bpy


def load(project, output):
    if output.exists() or output.suffix.lower() != ".blend":
        raise ValueError("Select a new .blend output path")
    state = json.loads(project.read_text())
    doc = state["document"]
    if doc.get("format") != "charloom-project" or doc.get("version") != 1:
        raise ValueError("Expected a Charloom v1 saved project")
    root = project.resolve().parents[2]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        tmp = Path(tmp)

        def materialize(folder, name, blob):
            if (
                not name
                or name != Path(name).name
                or name in (".", "..")
                or not re.fullmatch("[0-9a-f]{64}", blob)
            ):
                raise ValueError("Invalid project asset name or digest")
            data = (root / "blobs" / blob).read_bytes()
            if hashlib.sha256(data).hexdigest() != blob:
                raise ValueError(f"Saved asset digest mismatch: {name}")
            dest = folder / name
            with dest.open("xb") as f:
                f.write(data)
            return dest

        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene["charloom_project_id"] = state["id"]
        scene["charloom_project_revision"] = state["revision"]
        bpy.data.texts.new("project.json").write(
            json.dumps(state, ensure_ascii=False, indent=2)
        )
        changes = project.with_name("changes.jsonl")
        if changes.is_file():
            bpy.data.texts.new("changes.jsonl").write(changes.read_text())
        imported = []
        for i, res in enumerate(doc["resources"]):
            assets = state["assets"][res["id"]]
            d = tmp / str(i)
            d.mkdir()
            basename = Path(res["name"]).name
            f = materialize(d, basename, assets["blob"])
            for name, blob in assets.get("companions", {}).items():
                materialize(d, name, blob)
            before = set(scene.objects)
            fmt = res["format"]
            if fmt in ("glb", "gltf"):
                bpy.ops.import_scene.gltf(filepath=str(f))
            elif fmt == "fbx":
                bpy.ops.import_scene.fbx(filepath=str(f))
            elif fmt == "obj":
                bpy.ops.wm.obj_import(filepath=str(f))
            elif fmt == "geometry":
                bpy.data.texts.new(basename).write(f.read_text())
            elif res["role"] == "reference":
                pic = bpy.data.images.load(str(f))
                pic.pack()
            else:
                raise ValueError(f"Unsupported saved resource {basename} ({fmt})")
            collection = bpy.data.collections.new(basename)
            scene.collection.children.link(collection)
            for obj in set(scene.objects) - before:
                for group in list(obj.users_collection):
                    group.objects.unlink(obj)
                collection.objects.link(obj)
            collection["charloom_resource_id"] = res["id"]
            collection["charloom_role"] = res["role"]
            imported.append(basename)
        bpy.ops.file.pack_all()
        blend = tmp / "project.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True)
        blend.rename(output)
    return dict(file=str(output), revision=state["revision"], assets=imported)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("project", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    print(json.dumps(load(args.project, args.output)))
