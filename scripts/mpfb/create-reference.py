"""Use MPFB's own human, macro-target and rig services in a private Blender process."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import math
import sys
import time
import addon_utils
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("--gender", choices=["male", "female"], default="male")
parser.add_argument("--muscles", type=float, nargs="+", default=[.50, 1.0])
parser.add_argument("--weight", type=float, default=.50)
parser.add_argument("--height", type=float, default=.50)
parser.add_argument("--age", type=float, default=.50)
parser.add_argument("--muscle-proxy", action="store_true", help="Use the official muscular alternative topology")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
for value in [*args.muscles, args.weight, args.height, args.age]:
    if not math.isfinite(value) or not 0 <= value <= 1:
        parser.error("MakeHuman macro parameters must be finite values between 0 and 1")
if len({round(value * 100) for value in args.muscles}) != len(args.muscles):
    parser.error("Muscle presets must have distinct integer percentages")
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
revision = json.loads((ROOT / "out/tools/mpfb/revision.json").read_text())["sha"]
source = ROOT / "out/tools/mpfb/source" / ("mpfb2-" + revision) / "src/mpfb"
repository = ROOT / "out/tools/mpfb/runtime/repository"
repository.mkdir(parents=True, exist_ok=True)
package_dir = repository / "mpfb"
if not package_dir.exists():
    package_dir.symlink_to(source, target_is_directory=True)
bpy.context.preferences.extensions.repos.new(name="Charloom MPFB", module="charloom_mpfb", custom_directory=str(repository))
bpy.utils.refresh_script_paths()
package = "bl_ext.charloom_mpfb.mpfb"
addon_utils.enable(package, default_set=True, persistent=False)
module = importlib.import_module(package)
assert module.MPFB_CONTEXTUAL_INFORMATION, "MPFB failed to register"
HumanService = importlib.import_module(package + ".services.humanservice").HumanService
TargetService = importlib.import_module(package + ".services.targetservice").TargetService
ObjectService = importlib.import_module(package + ".services.objectservice").ObjectService
ExportService = importlib.import_module(package + ".services.exportservice").ExportService
LocationService = importlib.import_module(package + ".services.locationservice").LocationService
AssetService = importlib.import_module(package + ".services.assetservice").AssetService
HumanObjectProperties = importlib.import_module(package + ".entities.objectproperties").HumanObjectProperties
asset_pack = ROOT / "out/tools/mpfb/makehuman_system_assets_cc0.zip"
if not AssetService.system_assets_pack_is_installed():
    error = AssetService.fix_and_extract_asset_pack_zip(str(asset_pack), LocationService.get_user_data())
    assert error is None, error
    AssetService.rescan_pack_metadata()
records = []
body = None
original_objects = set()
equipped = []
face_refit_check = None


def snapshot():
    return {obj.name: [v.co.copy() for v in obj.data.vertices] for _, obj in equipped}


def displacements(before):
    changes = {}
    for kind, obj in equipped:
        assert len(before[obj.name]) == len(obj.data.vertices)
        changes[kind] = {"vertices": len(obj.data.vertices), "max_vertex_displacement_m": max((v.co - a).length for v, a in zip(obj.data.vertices, before[obj.name]))}
    return changes


def point_camera(camera, height=.90, scale=2.05, angle=0):
    target = Vector((0, 0, height))
    camera.location = target + Vector((4 * math.sin(angle), -4 * math.cos(angle), .02))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = scale


def export_character(folder, filename, include_clothes):
    export_root = ExportService.create_character_copy(body, name_suffix="_export")
    export_body = ObjectService.find_object_of_type_amongst_nearest_relatives(export_root, "Basemesh")
    export_proxy = ObjectService.find_object_of_type_amongst_nearest_relatives(export_root, "Proxymeshes")
    # GLB is a snapshot; preserve macro controls in the editable .blend instead.
    # Baking first avoids the export service's small-delta cutoff when rebuilding keys.
    ObjectService.activate_blender_object(export_body, deselect_all=True)
    TargetService.bake_targets(export_body)
    if not include_clothes:
        # MPFB's bake service applies delete masks even if disabled in the viewport.
        # Remove them on the disposable export copy so the naked GLB stays complete.
        for obj in [export_body, export_proxy]:
            if obj is not None:
                for modifier in list(obj.modifiers):
                    if modifier.type == "MASK" and modifier.name.startswith("Delete."):
                        obj.modifiers.remove(modifier)
    ExportService.bake_modifiers_remove_helpers(export_body, bake_masks=True, bake_subdiv=False, remove_helpers=True)
    bpy.ops.object.select_all(action="DESELECT")
    export_root.select_set(True)
    for child in ObjectService.get_list_of_children(export_root):
        if export_proxy is not None and child == export_body:
            continue
        if include_clothes or not ObjectService.object_is(child, "Clothes"):
            child.hide_set(False)
            child.select_set(True)
    bpy.context.view_layer.objects.active = export_root
    bpy.ops.export_scene.gltf(filepath=str(folder / filename), export_format="GLB", use_selection=True, export_animations=False)
    for obj in [*ObjectService.get_list_of_children(export_root), export_root]:
        bpy.data.objects.remove(obj, do_unlink=True)

for muscle in args.muscles:
    for obj in list(bpy.context.scene.objects):
        if obj not in original_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    macro = TargetService.get_default_macro_info_dict()
    macro.update(gender=1. if args.gender == "male" else 0., age=args.age, muscle=muscle, weight=args.weight, height=args.height, proportions=.50)
    macro["race"] = {"asian": 1., "caucasian": 0., "african": 0.}
    tick = time.perf_counter()
    changes = {}
    if body is None:
        info = {"phenotype": macro, "rig": "default", "skin_mhmat": "", "skin_material_type": "NONE", "proxy": ""}
        if args.muscle_proxy:
            info["proxy"] = "male_muscle_13290.proxy" if args.gender == "male" else "female_muscle_13442.proxy"
        settings = HumanService.get_default_deserialization_settings()
        settings.update(subdiv_levels=0, load_clothes=False)
        body = HumanService.deserialize_from_dict(info, settings)
        rig = body.find_armature()
        assert rig is not None
        proxy = ObjectService.find_object_of_type_amongst_nearest_relatives(body, "Proxymeshes")
        if proxy is not None:
            equipped.append(("Proxymeshes", proxy))
        for subdir, filename, kind in [
            ("eyes", "low-poly.mhclo", "Eyes"),
            ("eyebrows", "eyebrow001.mhclo", "Eyebrows"),
            ("eyelashes", "eyelashes01.mhclo", "Eyelashes"),
            ("teeth", "teeth_base.mhclo", "Teeth"),
            ("tongue", "tongue01.mhclo", "Tongue"),
            ("clothes", "male_casualsuit04.mhclo" if args.gender == "male" else "female_casualsuit01.mhclo", "Clothes"),
        ]:
            asset_path = AssetService.find_asset_absolute_path(filename, asset_subdir=subdir)
            assert asset_path is not None, filename
            obj = HumanService.add_mhclo_asset(asset_path, body, asset_type=kind, material_type="GAMEENGINE", subdiv_levels=0)
            equipped.append((kind, obj))
        skin = AssetService.find_asset_absolute_path(f"young_asian_{args.gender}.mhmat", asset_subdir="skins")
        assert skin is not None
        HumanService.set_character_skin(skin, body, skin_type="GAMEENGINE")
        before = snapshot()
        bone_before = {b.name: b.head_local.copy() for b in rig.data.bones}
        target_path = source / "data/targets/head/head-scale-horiz-incr.target.gz"
        head_target = TargetService.load_target(body, str(target_path), weight=.5)
        HumanService.refit(body)
        face_changes = displacements(before)
        for kind in ["Eyes", "Teeth", "Tongue"]:
            assert face_changes[kind]["max_vertex_displacement_m"] > 1e-5, (kind, face_changes)
        head_target.value = 0
        HumanService.refit(body)
        restored = displacements(before)
        assert max(d["max_vertex_displacement_m"] for d in restored.values()) < 1e-5, restored
        restored_bones = max((b.head_local - bone_before[b.name]).length for b in rig.data.bones)
        assert restored_bones < 1e-5, restored_bones
        face_refit_check = {"official_target": "head/head-scale-horiz-incr", "temporary_value": .5, "asset_changes": face_changes, "maximum_restore_error_m": max(d["max_vertex_displacement_m"] for d in restored.values()), "rig_restore_error_m": restored_bones}
        original_objects = set(bpy.context.scene.objects)
    else:
        before = snapshot()
        before_bones = {b.name: b.head_local.copy() for b in rig.data.bones}
        HumanObjectProperties.set_value("muscle", muscle, entity_reference=body)
        TargetService.reapply_macro_details(body)
        HumanService.refit(body)
        changes.update(displacements(before))
        changes["rig"] = {"max_joint_displacement_m": max((b.head_local - before_bones[b.name]).length for b in rig.data.bones)}
    body.name = f"MakeHuman_{args.gender}_muscle_{muscle:.2f}"
    prefix = f"{args.gender}-muscle-{round(muscle * 100):03}"
    folder = out / prefix
    folder.mkdir()
    weighted = {body.vertex_groups[g.group].name for v in body.data.vertices for g in v.groups if g.weight > 1e-8}
    deform = {bone.name for bone in rig.data.bones if bone.use_deform}
    record = {"preset": prefix, "macro_parameters": TargetService.get_macro_info_dict_from_basemesh(body), "mpfb_version": list(module.VERSION), "mpfb_revision": revision, "geometry_backend": "official MPFB HumanService and TargetService", "rig_backend": "official MPFB default rig and weights", "rig_bones": len(rig.data.bones), "weighted_deform_bones": len(deform & weighted), "shape_keys": len(body.data.shape_keys.key_blocks), "mpfb_user_data": LocationService.get_user_data(), "generation_seconds": time.perf_counter()-tick, "equipped_assets": [{"type": kind, "object": obj.name} for kind, obj in equipped], "live_muscle_refit_changes": changes, "asset_pack_sha256": hashlib.sha256(asset_pack.read_bytes()).hexdigest(), "mesh_to_meshy_fitted": False, "user_approved": False}
    record.update(body_topology="official muscle proxy" if proxy is not None else "standard MakeHuman basemesh", face_width_refit_check=face_refit_check)
    clothes = [obj for kind, obj in equipped if kind == "Clothes"]
    clothing_masks = [m for obj in [body, proxy] if obj is not None for m in obj.modifiers if m.type == "MASK" and m.name.startswith("Delete.")]
    for obj in clothes:
        obj.hide_render = True
        obj.hide_set(True)
    for modifier in clothing_masks:
        modifier.show_render = modifier.show_viewport = False
    export_character(folder, "body.glb", include_clothes=False)
    record["glb_sha256"] = hashlib.sha256((folder / "body.glb").read_bytes()).hexdigest()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 896
    scene.render.resolution_percentage = 100
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (.58, .58, .58)
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (.18, .18, .18)
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    camera = bpy.data.objects.new("reference-camera", bpy.data.cameras.new("reference-camera"))
    scene.collection.objects.link(camera)
    camera.data.type = "ORTHO"
    scene.camera = camera
    for region, height, scale in [("full", .90, 2.05), ("torso", 1.36, .9)]:
        for view, angle in [("front", 0), ("side", math.pi/2), ("back", math.pi), ("opposite-side", -math.pi/2)]:
            point_camera(camera, height, scale, angle)
            scene.render.filepath = str(folder / f"{region}-{view}.png")
            bpy.ops.render.render(write_still=True)
    point_camera(camera)
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "editable.blend"))
    for obj in clothes:
        obj.hide_render = False
        obj.hide_set(False)
    for modifier in clothing_masks:
        modifier.show_render = modifier.show_viewport = True
    export_character(folder, "clothed.glb", include_clothes=True)
    for view, angle in [("front", 0), ("side", math.pi/2), ("back", math.pi)]:
        point_camera(camera, angle=angle)
        scene.render.filepath = str(folder / f"clothed-{view}.png")
        bpy.ops.render.render(write_still=True)
    point_camera(camera)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "clothed.blend"))
    (folder / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    records.append(record)
(out / "report.json").write_text(json.dumps({"workflow": "direct MPFB parameterized human generation", "muscle_variants": records, "source_inputs_and_live_blender_scene_modified": False, "mesh_to_meshy_fitting_complete": False}, indent=2) + "\n")
