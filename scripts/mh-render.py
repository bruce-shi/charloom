"""Headless review renders for charloom mh-transfer outputs.

Usage:
  Blender -b --factory-startup --python scripts/mh-render.py -- OUT_DIR GLB [GLB...]
Options after the GLBs: --frames 0.0,0.5 (fraction of the first action), --color (vertex colours)
Writes OUT_DIR/<glb-stem>-<view>-<frame>.png for front, side and back views.
"""
import bpy, sys, os, math
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
files, frames, vcol, crop, only, overlay, eevee, hide = [], [None], False, None, None, False, False, []
clip = None
i = 1
while i < len(argv):
    if argv[i] == "--clip":
        clip = argv[i+1]; i += 2
    elif argv[i] == "--frames":
        frames = [float(x) for x in argv[i + 1].split(",")]; i += 2
    elif argv[i] == "--hide":
        hide = argv[i + 1].split(","); i += 2
    elif argv[i] == "--eevee":
        eevee = True; i += 1
    elif argv[i] == "--overlay":
        overlay = True; i += 1
    elif argv[i] == "--only":
        only = argv[i + 1].split(","); i += 2
    elif argv[i] == "--crop":
        crop = [float(x) for x in argv[i + 1].split(",")]; i += 2
    elif argv[i] == "--color":
        vcol = True; i += 1
    else:
        files.append(argv[i]); i += 1
os.makedirs(out, exist_ok=True)

def setup(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for p in (files if overlay else [path]):
        bpy.ops.import_scene.gltf(filepath=p)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    if eevee:
        # Material-accurate preview: honours alpha (eyelash cards, hair cards).
        engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
        sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        sun.data.energy = 3.0
        sun.rotation_euler = (math.radians(50), 0, math.radians(20))
        scene.collection.objects.link(sun)
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "VERTEX" if vcol else "TEXTURE"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 1024
    scene.render.film_transparent = False
    world = bpy.data.worlds.new("w"); scene.world = world
    world.color = (0.35, 0.37, 0.4)
    if eevee:
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.37, 0.4, 1)
        world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    return scene

def bounds():
    lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
    dg = bpy.context.evaluated_depsgraph_get()
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH" or not ob.visible_get() or len(ob.data.polygons) < 64:
            continue
        ev = ob.evaluated_get(dg)
        for v in ev.to_mesh().vertices:
            p = ev.matrix_world @ v.co
            lo = Vector(map(min, lo, p)); hi = Vector(map(max, hi, p))
        ev.to_mesh_clear()
    return lo, hi

for path in (files[:1] if overlay else files):
    scene = setup(path)
    for ob in scene.objects:
        if ob.type == "MESH" and any(ob.name.startswith(n) for n in hide):
            ob.hide_render = True
    if only:
        for ob in list(scene.objects):
            if ob.type == "MESH" and not any(ob.name.startswith(n) for n in only):
                bpy.data.objects.remove(ob)
    stem = os.path.splitext(os.path.basename(path))[0] + ("-" + "+".join(only) if only else "") + ("-eevee" if eevee else "") + ("-no-" + "+".join(hide) if hide else "")
    tag = os.path.basename(os.path.dirname(path))
    action = bpy.data.actions[0] if bpy.data.actions else None
    if clip:
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"blender"))
        from charloom import api
        rigs=[ob for ob in scene.objects if ob.type=="ARMATURE"]
        if len(rigs)!=1:
            raise ValueError("A named clip requires one armature")
        api.preview(rigs[0],clip)
        action=rigs[0].animation_data.action
        stem += "-"+clip
    for f in frames:
        for ob in scene.objects:
            if ob.type == "ARMATURE":
                ob.data.pose_position = "REST" if f is None else "POSE"
        if f is not None and action:
            start, end = action.frame_range
            scene.frame_set(int(start + (end - start) * f))
        lo, hi = bounds()
        if crop:
            h = hi.z - lo.z
            lo.z, hi.z = lo.z + h * crop[0], lo.z + h * crop[1]
            w = (hi.z - lo.z) * 0.75
            mid = (lo.x + hi.x) / 2
            lo.x, hi.x = mid - w / 2, mid + w / 2
        center = (lo + hi) / 2
        size = max(hi.z - lo.z, (hi.x - lo.x) * 1024 / 768) * 1.08
        cam_data = bpy.data.cameras.new("c"); cam_data.type = "ORTHO"; cam_data.ortho_scale = size
        cam = bpy.data.objects.new("c", cam_data); scene.collection.objects.link(cam); scene.camera = cam
        for view, angle in [("front", 0), ("side", 90), ("back", 180)]:
            a = math.radians(angle)
            # glTF +Z (front) becomes Blender -Y.
            d = Vector((math.sin(a), -math.cos(a), 0))
            cam.location = center + d * 5
            cam.rotation_euler = (math.radians(90), 0, a)
            label = ("rest" if f is None else f"{f:.2f}") + ("-crop" if crop else "")
            scene.render.filepath = os.path.join(out, f"{tag}-{stem}-{view}-{label}.png")
            bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)
