"""Independently compare saved Blender surfaces with freshly imported GLBs."""
import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils.kdtree import KDTree

parser = argparse.ArgumentParser()
parser.add_argument("roots", type=Path, nargs="+")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def scene_surface():
    points = []
    triangles = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    widgets = {bone.custom_shape for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
               for bone in obj.pose.bones if bone.custom_shape is not None}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render or obj in widgets:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        used = {index for face in mesh.loop_triangles for index in face.vertices}
        points.extend(evaluated.matrix_world @ mesh.vertices[index].co for index in used)
        triangles += len(mesh.loop_triangles)
        evaluated.to_mesh_clear()
    return points, triangles


def directed_error(a, b):
    tree = KDTree(len(b))
    for index, point in enumerate(b):
        tree.insert(point, index)
    tree.balance()
    return max(tree.find(point)[2] for point in a)


for root in args.roots:
    root = root.resolve()
    records = []
    report = json.loads((root / "report.json").read_text())
    for preset in report["muscle_variants"]:
        folder = root / preset["preset"]
        for source_name, exported_name in [("editable.blend", "body.glb"), ("clothed.blend", "clothed.glb")]:
            bpy.ops.wm.open_mainfile(filepath=str(folder / source_name), use_scripts=False)
            expected, source_triangles = scene_surface()
            source_bones = sum(len(obj.data.bones) for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.gltf(filepath=str(folder / exported_name))
            actual, triangles = scene_surface()
            bones = sum(len(obj.data.bones) for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
            weighted = {obj.vertex_groups[group.group].name
                        for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.find_armature() is not None
                        for vertex in obj.data.vertices for group in vertex.groups if group.weight > 1e-8}
            fingers = sorted(name for name in weighted if name.startswith("finger"))
            assert len(fingers) == 30, (folder, exported_name, fingers)
            assert {"jaw", "eye.L", "eye.R"}.issubset(weighted), (folder, exported_name, weighted)
            assert any(name.startswith("tongue") for name in weighted)
            assert expected and actual
            forward = directed_error(expected, actual)
            backward = directed_error(actual, expected)
            assert max(forward, backward) < 1e-5, (folder, exported_name, forward, backward)
            assert source_triangles == triangles, (folder, exported_name, source_triangles, triangles)
            assert bones == source_bones == 163, (folder, bones, source_bones)
            records.append({"preset": preset["preset"], "export": exported_name, "source_surface_vertices": len(expected),
                            "export_surface_vertices": len(actual), "triangles": triangles, "bones": bones,
                            "weighted_bones": len(weighted), "weighted_finger_bones": fingers,
                            "weighted_face_bones": sorted(name for name in weighted if name.startswith(("eye.", "tongue")) or name == "jaw"),
                            "maximum_source_to_export_distance_m": forward, "maximum_export_to_source_distance_m": backward,
                            "passed": True})
    (root / "export-verification.json").write_text(json.dumps({"method": "fresh Blender GLB import vs evaluated saved scene; bidirectional surface vertex coverage and triangle count", "tolerance_m": 1e-5, "checks": records, "passed": True}, indent=2) + "\n")
    print("MPFB_EXPORT_CHECK=" + json.dumps({"root": str(root), "checks": len(records), "passed": True}))
