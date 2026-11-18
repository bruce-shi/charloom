"""All review renders of one mh-transfer output in a single Blender session.

Usage:
  Blender -b --factory-startup --python scripts/mh-render-review.py -- RUN_DIR

Each GLB is imported once into its own collection; the passes switch
collections, engine, crop and frames. Writes RUN_DIR/renders/ with the same
names as separate scripts/mh-render.py calls:
  1-segment-segmentation-{front,side,back}-rest.png        (skin/hair/garment labels)
  1-segment-pieces-*-rest.png + 1-segment-pieces-legend.json (one colour per piece)
  3-skin-body-*-rest.png, 4-garments-character-*-rest.png  (Workbench)
  *-eevee-*-rest-crop.png                                   (EEVEE head close-ups)
  eyes-{open,blink,scan}-{front,three-quarter}.png           (EEVEE eye close-ups, lids posed; scan alongside)
  4-garments-character-*-{0.25,0.60}.png                    (walk frames)
  hand-{scan,posed,body}-{l,r}-{front,side}.png             (hand close-ups: scan
                                          labels, posed MakeHuman before the fit, fitted body)
  neck-{segmentation,scan,pieces,character}-{front,quarter,side,back}-{rest,0.25,0.60}.png
                                          (neckline close-ups: labels, scan texture, pieces, dressed)

  ... -- RUN_DIR --neck-only [OUT_DIR]   only the neckline close-ups (into OUT_DIR)
"""
import bpy, sys, os, math, time, json
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
run = args[0]
neck_only = "--neck-only" in args
out = os.path.join(run, "renders")
if neck_only and len(args) > args.index("--neck-only") + 1:
    out = args[args.index("--neck-only") + 1]
os.makedirs(out, exist_ok=True)
started = time.time()

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
layer = bpy.context.view_layer
GLBS = {
    "segmentation": "1-segment/segmentation.glb",
    "body": "3-skin/body.glb",
    "character": "4-garments/character.glb",
    "posed": "1-segment/posed-prior.glb",
}
if neck_only:
    GLBS = {k: GLBS[k] for k in ("segmentation", "character")}
collections = {}
for key, rel in GLBS.items():
    path = os.path.join(run, rel)
    if not os.path.exists(path):
        continue
    col = bpy.data.collections.new(key)
    scene.collection.children.link(col)
    layer.active_layer_collection = layer.layer_collection.children[key]
    bpy.ops.import_scene.gltf(filepath=path)
    collections[key] = (col, os.path.basename(os.path.dirname(path)), os.path.splitext(os.path.basename(path))[0])
# Every garment/hair piece in its own flat colour, to review the split.
PALETTE = [(0.9, 0.16, 0.16), (0.16, 0.55, 1.0), (0.16, 0.78, 0.27), (1.0, 0.63, 0.0), (0.75, 0.24, 1.0),
           (0.0, 0.86, 0.86), (0.94, 0.9, 0.16), (0.59, 0.35, 0.12), (1.0, 0.24, 0.67), (0.08, 0.43, 0.16),
           (0.55, 0.55, 0.55), (0.3, 0.2, 0.7)]
piece_dir = os.path.join(run, "1-segment", "pieces")
legend = {}
if os.path.isdir(piece_dir):
    col = bpy.data.collections.new("pieces")
    scene.collection.children.link(col)
    layer.active_layer_collection = layer.layer_collection.children["pieces"]
    for k, f in enumerate(sorted(x for x in os.listdir(piece_dir) if x.endswith(".glb"))):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.join(piece_dir, f))
        c = PALETTE[k % len(PALETTE)]
        legend[os.path.splitext(f)[0]] = "#%02x%02x%02x" % tuple(int(x * 255) for x in c)
        for ob in set(bpy.data.objects) - before:
            ob.color = (*c, 1.0)
    collections["pieces"] = (col, "1-segment", "pieces")
    with open(os.path.join(out, "1-segment-pieces-legend.json"), "w") as fh:
        json.dump(legend, fh, indent=1)
# The textured source scan in the working frame (eye close-up comparison).
try:
    rep = json.load(open(os.path.join(run, "report.json")))
    src, m = rep["source"], rep["working_frame"]["source_to_working"]
except (OSError, KeyError, ValueError):
    src = None
if src and os.path.exists(src):
    col = bpy.data.collections.new("scan")
    scene.collection.children.link(col)
    layer.active_layer_collection = layer.layer_collection.children["scan"]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=src)
    # glTF column-major source_to_working, conjugated into Blender's Z-up.
    M = Matrix([[m[c * 4 + r] for c in range(4)] for r in range(4)])
    C = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    T = C @ M @ C.inverted()
    for ob in set(bpy.data.objects) - before:
        if ob.parent is None:
            ob.matrix_world = T @ ob.matrix_world
    collections["scan"] = (col, "0-source", "scan")
print(f"imported in {time.time() - started:.1f}s", flush=True)

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(50), 0, math.radians(20))
scene.collection.objects.link(sun)
scene.display.shading.light = "STUDIO"
scene.render.resolution_x = 768
scene.render.resolution_y = 1024
scene.render.film_transparent = False
world = bpy.data.worlds.new("w")
scene.world = world
world.color = (0.35, 0.37, 0.4)
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.37, 0.4, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
EEVEE = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"


def show(key):
    for k in collections:
        layer.layer_collection.children[k].exclude = k != key


def bounds():
    lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
    dg = bpy.context.evaluated_depsgraph_get()
    for ob in scene.objects:
        if ob.type != "MESH" or not ob.visible_get() or len(ob.data.polygons) < 64:
            continue
        ev = ob.evaluated_get(dg)
        for v in ev.to_mesh().vertices:
            p = ev.matrix_world @ v.co
            lo = Vector(map(min, lo, p)); hi = Vector(map(max, hi, p))
        ev.to_mesh_clear()
    return lo, hi


def render(key, eevee=False, vcol=False, crop=None, frames=(None,), object_color=False):
    if key not in collections:
        return
    show(key)
    col, tag, stem = collections[key]
    stem += "-eevee" if eevee else ""
    scene.render.engine = EEVEE if eevee else "BLENDER_WORKBENCH"
    scene.display.shading.color_type = "OBJECT" if object_color else "VERTEX" if vcol else "TEXTURE"
    arms = [ob for ob in col.all_objects if ob.type == "ARMATURE"]
    action = next((a.animation_data.action for a in arms if a.animation_data and a.animation_data.action), None)
    for f in frames:
        for a in arms:
            a.data.pose_position = "REST" if f is None else "POSE"
        if f is not None and action:
            start, end = action.frame_range
            scene.frame_set(int(start + (end - start) * f))
        else:
            scene.frame_set(scene.frame_current)
        lo, hi = bounds()
        eyes = [ob for ob in col.all_objects if ob.type == "MESH" and ob.data.name.startswith("eyes")]
        heads = [ob for ob in col.all_objects if ob.type == "MESH" and ob.data.name.startswith("scan_head")]
        if crop and eyes:
            # Frame the face on the eyeballs (brow to chin), whatever the
            # head size or hairstyle.
            dg = bpy.context.evaluated_depsgraph_get()
            ev = eyes[0].evaluated_get(dg)
            pts = [ev.matrix_world @ v.co for v in ev.to_mesh().vertices]
            ev.to_mesh_clear()
            ex = [p.x for p in pts]; ez = [p.z for p in pts]
            span = max(ex) - min(ex)
            cz = (max(ez) + min(ez)) / 2 - 0.3 * span
            mid = (max(ex) + min(ex)) / 2
            lo.z, hi.z = cz - 1.3 * span, cz + 1.3 * span
            lo.x, hi.x = mid - 0.975 * span, mid + 0.975 * span
        elif crop and heads:
            # Head route keep_scan: no MakeHuman eyes; frame the scan head.
            dg = bpy.context.evaluated_depsgraph_get()
            pts = []
            for ob in heads:
                ev = ob.evaluated_get(dg)
                pts += [ev.matrix_world @ v.co for v in ev.to_mesh().vertices]
                ev.to_mesh_clear()
            zs = [p.z for p in pts]; xs = [p.x for p in pts]
            h = max(zs) - min(zs)
            mid = (max(xs) + min(xs)) / 2
            lo.z, hi.z = min(zs) - 0.05 * h, max(zs) + 0.1 * h
            lo.x, hi.x = mid - 0.55 * h, mid + 0.55 * h
        elif crop:
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
            cam.location = center + Vector((math.sin(a), -math.cos(a), 0)) * 5
            cam.rotation_euler = (math.radians(90), 0, a)
            label = ("rest" if f is None else f"{f:.2f}") + ("-crop" if crop else "")
            scene.render.filepath = os.path.join(out, f"{tag}-{stem}-{view}-{label}.png")
            bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)


def neck_render():
    """Neckline close-ups, framed on the character rig's neck bones (posed in
    the walk frames): scan labels, pieces and the dressed character."""
    if "character" not in collections:
        return
    arms = [ob for ob in collections["character"][0].all_objects if ob.type == "ARMATURE"]
    arm = next((a for a in arms if any(b.name.startswith("neck") for b in a.data.bones)), None)
    if arm is None:
        return
    action = next((a.animation_data.action for a in arms if a.animation_data and a.animation_data.action), None)
    scene.render.resolution_x = scene.render.resolution_y = 768
    for f in (None, 0.25, 0.6):
        for a in arms:
            a.data.pose_position = "REST" if f is None else "POSE"
        if f is not None and action:
            start, end = action.frame_range
            scene.frame_set(int(start + (end - start) * f))
        bpy.context.view_layer.update()
        necks = [b for b in arm.pose.bones if b.name.startswith("neck")]
        pts = [arm.matrix_world @ p for b in necks for p in (b.head, b.tail)]
        lo = Vector(map(min, *pts)); hi = Vector(map(max, *pts))
        length = (hi - lo).length
        # The neckline sits around the base of the neck chain.
        center = (lo + hi) / 2 - Vector((0, 0, 0.45 * (hi.z - lo.z)))
        size = 2.0 * length
        # "scan": the segmentation mesh with the scan texture (where the
        # collar really is), when segmentation.glb carries it.
        keys = [("segmentation", "segmentation", True, False), ("segmentation", "scan", False, False),
                ("pieces", "pieces", False, True), ("character", "character", False, False)]
        for key, name, vcol, object_color in keys:
            if key not in collections or (f is not None and key != "character"):
                continue
            if name == "scan" and not any(
                    ob.type == "MESH" and ob.active_material and ob.active_material.node_tree
                    and any(nd.type == "TEX_IMAGE" for nd in ob.active_material.node_tree.nodes)
                    for ob in collections[key][0].all_objects):
                continue
            show(key)
            scene.render.engine = "BLENDER_WORKBENCH"
            scene.display.shading.color_type = "OBJECT" if object_color else "VERTEX" if vcol else "TEXTURE"
            cam_data = bpy.data.cameras.new("c"); cam_data.type = "ORTHO"; cam_data.ortho_scale = size
            cam = bpy.data.objects.new("c", cam_data); scene.collection.objects.link(cam); scene.camera = cam
            for view, angle in [("front", 0), ("quarter", 40), ("side", 90), ("back", 180)]:
                a = math.radians(angle)
                cam.location = center + Vector((math.sin(a), -math.cos(a), 0)) * 5
                cam.rotation_euler = (math.radians(90), 0, a)
                label = "rest" if f is None else f"{f:.2f}"
                scene.render.filepath = os.path.join(out, f"neck-{name}-{view}-{label}.png")
                bpy.ops.render.render(write_still=True)
            bpy.data.objects.remove(cam)
    for a in arms:
        a.data.pose_position = "REST"


if neck_only:
    neck_render()
    print(f"neck renders done in {time.time() - started:.1f}s", flush=True)
    sys.exit(0)
render("segmentation", vcol=True)
render("pieces", object_color=True)
render("character")
render("body")
# Face close-ups in EEVEE: honours alpha on eyelash and hair cards.
render("character", eevee=True, crop=(0.84, 1.0))
render("body", eevee=True, crop=(0.84, 1.0))
render("character", frames=(0.25, 0.6))
neck_render()


def render_eyes(key, blink={"orbicularis03.L": -0.5, "orbicularis03.R": -0.5}):
    """EEVEE close-up of both eyes (lashes, lids), front and three-quarter,
    open and with the lids rotated closed by `blink` radians per bone
    ({"orbicularis03.L": .., ...}): eyes-{open,blink}-{front,three-quarter}.png."""
    if key not in collections:
        return
    show(key)
    col = collections[key][0]
    eyes = [ob for ob in col.all_objects if ob.type == "MESH" and ob.data.name.startswith("eyes")]
    if not eyes:
        return
    arms = [ob for ob in col.all_objects if ob.type == "ARMATURE"]
    actions = {a: a.animation_data.action for a in arms if a.animation_data}
    for a in actions:
        a.animation_data.action = None
    scene.render.engine = EEVEE
    scene.render.resolution_x, scene.render.resolution_y = 1024, 512

    def pose(angle_rad):
        for a in arms:
            a.data.pose_position = "POSE"
            for pb in a.pose.bones:
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = ((angle_rad or {}).get(pb.name, 0), 0, 0)
                pb.location = (0, 0, 0)
                pb.scale = (1, 1, 1)
        bpy.context.view_layer.update()

    pose(None)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = eyes[0].evaluated_get(dg)
    pts = [ev.matrix_world @ v.co for v in ev.to_mesh().vertices]
    ev.to_mesh_clear()
    lo = Vector(map(min, *pts)); hi = Vector(map(max, *pts))
    center = (lo + hi) / 2
    span = hi.x - lo.x
    for state, angle_rad in [("open", None), ("blink", blink)]:
        pose(angle_rad)
        cam_data = bpy.data.cameras.new("c"); cam_data.type = "ORTHO"; cam_data.ortho_scale = span * 1.3
        cam = bpy.data.objects.new("c", cam_data); scene.collection.objects.link(cam); scene.camera = cam
        for view, deg in [("front", 0), ("three-quarter", 35)]:
            r = math.radians(deg)
            cam.location = center + Vector((math.sin(r), -math.cos(r), 0)) * 5
            cam.rotation_euler = (math.radians(90), 0, r)
            scene.render.filepath = os.path.join(out, f"eyes-{state}-{view}.png")
            bpy.ops.render.render(write_still=True)
            if state == "open" and "scan" in collections:
                # The textured scan from the same camera, for comparison.
                show("scan")
                scene.render.filepath = os.path.join(out, f"eyes-scan-{view}.png")
                bpy.ops.render.render(write_still=True)
                show(key)
        bpy.data.objects.remove(cam)
    for a in arms:
        for pb in a.pose.bones:
            pb.rotation_euler = (0, 0, 0)
        a.data.pose_position = "REST"
    for a, act in actions.items():
        a.animation_data.action = act
    scene.render.resolution_x, scene.render.resolution_y = 768, 1024


# The lid turns that close this character's eyes, per lid and side
# (report.json stage2_lid_closing_rad); MakeHuman's own eye closes with the
# upper lid at about -0.5 rad.
try:
    _close = json.load(open(os.path.join(run, "report.json")))["stage2_lid_closing_rad"]
except (OSError, KeyError, ValueError):
    _close = {}
render_eyes("character", {
    f"{bone}.{s}": _close.get(lid, {}).get(s.lower(), default)
    for lid, bone, default in [("upper", "orbicularis03", -0.5), ("lower", "orbicularis04", 0.0)]
    for s in "LR"
})


def hand_box(side):
    """Rest-pose bounds of one hand's bones (wrist and below) on the body rig."""
    if "body" not in collections:
        return None
    arms = [ob for ob in collections["body"][0].all_objects if ob.type == "ARMATURE"]
    wrist = next((a.data.bones.get(f"wrist.{side}") for a in arms if a.data.bones.get(f"wrist.{side}")), None)
    if wrist is None:
        return None
    arm = next(a for a in arms if a.data.bones.get(f"wrist.{side}"))
    pts = [arm.matrix_world @ p for b in [wrist, *wrist.children_recursive] for p in (b.head_local, b.tail_local)]
    lo = Vector(map(min, *pts)); hi = Vector(map(max, *pts))
    pad = max(hi - lo) * 0.25
    return lo - Vector((pad,) * 3), hi + Vector((pad,) * 3)


# Hand close-ups: the scan (labels) and the fitted body from the same camera.
scene.render.resolution_x = scene.render.resolution_y = 768
for side in ("L", "R"):
    box = hand_box(side)
    if box is None:
        continue
    lo, hi = box
    center = (lo + hi) / 2
    for key, name, vcol in [("segmentation", "scan", True), ("posed", "posed", False), ("body", "body", False)]:
        if key not in collections:
            continue
        show(key)
        for a in [ob for ob in collections[key][0].all_objects if ob.type == "ARMATURE"]:
            a.data.pose_position = "REST"
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.color_type = "VERTEX" if vcol else "TEXTURE"
        cam_data = bpy.data.cameras.new("c"); cam_data.type = "ORTHO"; cam_data.ortho_scale = max(hi - lo)
        cam = bpy.data.objects.new("c", cam_data); scene.collection.objects.link(cam); scene.camera = cam
        # Side view from outside the hand (the character's left is +X).
        for view, angle in [("front", 0), ("side", 90 if side == "L" else -90)]:
            a = math.radians(angle)
            cam.location = center + Vector((math.sin(a), -math.cos(a), 0)) * 5
            cam.rotation_euler = (math.radians(90), 0, a)
            scene.render.filepath = os.path.join(out, f"hand-{name}-{side.lower()}-{view}.png")
            bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)
print(f"review renders done in {time.time() - started:.1f}s", flush=True)
