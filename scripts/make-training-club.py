"""Blender background fixture: independent, rigid training club in meter units.

Run: Blender --background --factory-startup --python scripts/make-training-club.py
The source scene and GLB are generated only inside fixtures/generated/.
"""
from pathlib import Path
import bpy
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures/generated"
OUT.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)


def material(name, color, metal=0.0, roughness=0.5):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1)
    result.use_nodes = True
    node = result.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1)
    node.inputs["Metallic"].default_value = metal
    node.inputs["Roughness"].default_value = roughness
    return result


wood = material("Warm ash wood", (0.33, 0.16, 0.065), 0, 0.55)
leather = material("Dark grip wrap", (0.055, 0.032, 0.021), 0, 0.8)
bronze = material("Bronze binding", (0.38, 0.22, 0.08), 0.7, 0.34)


def cylinder(name, bottom, top, radius1, radius2, mat, vertices=24):
    bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=radius1, radius2=radius2,
        depth=top-bottom, location=(0,0,(bottom+top)/2))
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    bevel = obj.modifiers.new("Soft edges", "BEVEL")
    bevel.width = 0.004
    bevel.segments = 2
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    for face in obj.data.polygons:
        face.use_smooth = True
    return obj


cylinder("Club wood", -0.12, 0.95, 0.024, 0.073, wood)
cylinder("Handle leather", -0.09, 0.30, 0.028, 0.040, leather)
for index in range(13):
    z = -0.08 + index*0.029
    radius = 0.029 + (z+0.09)/0.39*0.012
    cylinder(f"Grip wrap {index+1:02}", z, z+0.008, radius, radius, leather)
cylinder("Pommel", -0.135, -0.10, 0.034, 0.034, bronze)
cylinder("Head lower band", 0.49, 0.525, 0.056, 0.058, bronze)
cylinder("Head upper band", 0.84, 0.875, 0.070, 0.071, bronze)
cylinder("Head cap", 0.94, 0.96, 0.073, 0.071, bronze)
bpy.context.scene.unit_settings.system = "METRIC"
bpy.ops.object.select_all(action="SELECT")
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "training-club.blend"))
bpy.ops.export_scene.gltf(filepath=str(OUT / "training-club.glb"), export_format="GLB", use_selection=True,
    export_yup=True, export_animations=False, export_cameras=False, export_lights=False)
path = OUT / "training-club.glb"
(OUT / "training-club.provenance.json").write_text(json.dumps({
    "source":"Authored for the Charloom integration example in Blender; no third-party geometry",
    "script":"scripts/make-training-club.py", "units":"meters", "shaft_axis":"+Y",
    "origin":"inside handle, near primary grip", "sha256":hashlib.sha256(path.read_bytes()).hexdigest()
},indent=2)+"\n")
