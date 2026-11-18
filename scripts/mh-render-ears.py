"""Ear close-ups of one mh-transfer output, both sides, next to the scan.

Usage:
  Blender -b --factory-startup --python scripts/mh-render-ears.py -- RUN_DIR [RUN_DIR...]

Writes RUN_DIR/renders/ears-{l,r}.png: one row per side, the same orthographic
camera for every column:
  scan (texture only) | scan (shape) | body (texture only) | body (shape) |
  character (lit, with hair)
The scan is the run's source model (report.json `source`) moved into the
working frame (`working_frame.source_to_working`), so a misplaced pinna and a
second ear painted next to it show side by side.
"""
import bpy, sys, os, math, json
import numpy as np
from mathutils import Matrix, Vector

runs = sys.argv[sys.argv.index("--") + 1:]
TILE = 448


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_into(name, path, matrix=None):
    scene = bpy.context.scene
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    layer = bpy.context.view_layer
    layer.active_layer_collection = layer.layer_collection.children[name]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    if matrix is not None:
        for ob in set(bpy.data.objects) - before:
            if ob.parent is None:
                ob.matrix_world = matrix @ ob.matrix_world
    return col


def mesh_points(col, pred=lambda ob: True):
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for ob in col.all_objects:
        if ob.type != "MESH" or not pred(ob):
            continue
        ev = ob.evaluated_get(dg)
        m = ev.to_mesh()
        pts += [ev.matrix_world @ v.co for v in m.vertices]
        ev.to_mesh_clear()
    return pts


def render_run(run):
    clear()
    scene = bpy.context.scene
    report = json.load(open(os.path.join(run, "report.json")))
    cols = {}
    for key, rel in [("body", "3-skin/body.glb"), ("character", "4-garments/character.glb")]:
        p = os.path.join(run, rel)
        if os.path.exists(p):
            cols[key] = import_into(key, p)
    src = report.get("source")
    wf = report.get("working_frame", {}).get("source_to_working")
    if src and wf and os.path.exists(src):
        m = Matrix([wf[0:4], wf[4:8], wf[8:12], wf[12:16]]).transposed()  # column-major
        c = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))  # glTF Y-up -> Blender Z-up
        cols["scan"] = import_into("scan", src, c @ m @ c.inverted())
    if "body" not in cols:
        return
    for ob in cols["body"].all_objects:
        if ob.type == "ARMATURE":
            ob.data.pose_position = "REST"
    if "character" in cols:
        for ob in cols["character"].all_objects:
            if ob.type == "ARMATURE":
                ob.data.pose_position = "REST"
    bpy.context.view_layer.update()
    eyes = mesh_points(cols["body"], lambda ob: ob.data.name.startswith("eyes"))
    body = mesh_points(cols["body"], lambda ob: not ob.data.name.startswith("eyes") and len(ob.data.vertices) > 5000)
    if not eyes or not body:
        return
    ez = sum(p.z for p in eyes) / len(eyes)
    ey = sum(p.y for p in eyes) / len(eyes)
    span = max(p.x for p in eyes) - min(p.x for p in eyes)
    unit = span / 0.095  # eye span of a ~1.75 m adult
    mid = sum(p.x for p in eyes) / len(eyes)
    band = [p for p in body if ez - 0.075 * unit < p.z < ez + 0.02 * unit and p.y > ey]
    world = bpy.data.worlds.new("w")
    scene.world = world
    world.color = (0.35, 0.37, 0.4)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = scene.render.resolution_y = TILE
    scene.display.shading.color_type = "TEXTURE"
    out = os.path.join(run, "renders")
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, ".ear-tile.png")
    for side, sign in (("l", 1), ("r", -1)):
        pts = [p for p in band if (p.x - mid) * sign > 0]
        if not pts:
            continue
        xmax = max((p.x - mid) * sign for p in pts)
        ear = [p for p in pts if (p.x - mid) * sign > xmax - 0.012 * unit]
        centre = sum(ear, Vector()) / len(ear)
        cam_data = bpy.data.cameras.new("c")
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = 0.13 * unit
        cam = bpy.data.objects.new("c", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
        a = math.radians(90 * sign)
        cam.location = centre + Vector((math.sin(a), -math.cos(a), 0)) * 2
        cam.rotation_euler = (math.radians(90), 0, a)
        cam_data.clip_end = 4
        tiles = []
        for key, light, colour in (("scan", "FLAT", "TEXTURE"), ("scan", "STUDIO", "SINGLE"), ("body", "FLAT", "TEXTURE"),
                                   ("body", "STUDIO", "SINGLE"), ("character", "STUDIO", "TEXTURE")):
            if key not in cols:
                tiles.append(np.full((TILE, TILE, 4), 0.2, np.float32))
                continue
            for k, c in cols.items():
                c.hide_render = k != key
            scene.display.shading.light = light
            scene.display.shading.color_type = colour
            scene.render.filepath = tmp
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(tmp)
            tiles.append(np.array(img.pixels[:], np.float32).reshape(TILE, TILE, 4))
            bpy.data.images.remove(img)
        row = np.concatenate(tiles, axis=1)
        grid = bpy.data.images.new(f"ears-{side}", row.shape[1], row.shape[0])
        grid.pixels = row.ravel()
        grid.filepath_raw = os.path.join(out, f"ears-{side}.png")
        grid.file_format = "PNG"
        grid.save()
        bpy.data.objects.remove(cam)
    if os.path.exists(tmp):
        os.remove(tmp)
    print("ears:", out, flush=True)


for run in runs:
    render_run(run)
