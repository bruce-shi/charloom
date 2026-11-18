"""Scriptable character assembly and scene operations in Blender's Python runtime."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import bpy
import numpy as np

from .garment_binding import sleeve_weights
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform


class Character(NamedTuple):
    rig: bpy.types.Object
    body: bpy.types.Object | None


def _worker(checkout, python, script, args, log=None):
    root = Path(
        checkout or os.getenv("CHARLOOM_ROOT") or Path(__file__).resolve().parents[2]
    ).resolve()
    interpreter = Path(python) if python else root / "out/tools/cv/.venv/bin/python"
    worker = root / "scripts" / script
    for file in (interpreter, worker):
        if not file.is_file():
            raise FileNotFoundError(file)
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.setdefault("BLENDER", bpy.app.binary_path)
    return subprocess.Popen(
        [str(interpreter.absolute()), str(worker), *map(str, args)],
        cwd=root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _wait(process):
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, process.args)


def prepare_guide(source, output, *, checkout=None, python=None):
    """Render calibrated views and an annotation template for an original scan."""
    output = Path(output).resolve()
    _wait(
        _worker(
            checkout, python, "scan-guide.py", [Path(source).resolve(), "--out", output]
        )
    )
    return output / "guidance.template.json"


def transfer_scan(
    source,
    prior,
    guidance,
    output,
    *,
    motion=None,
    mouth_trace=None,
    segment_review=None,
    segment_evidence=None,
    review_out=None,
    checkout=None,
    python=None,
):
    """Create a calibrated body and wardrobe from the scan and reviewed guide."""
    output = Path(output).resolve()
    args = [
        Path(source).resolve(),
        "--prior",
        Path(prior).resolve(),
        "--guidance",
        Path(guidance).resolve(),
        "--out",
        output,
    ]
    for flag, path in (
        ("--motion", motion),
        ("--mouth-trace", mouth_trace),
        ("--segment-review", segment_review),
        ("--segment-evidence", segment_evidence),
        ("--review-out", review_out),
    ):
        if path is not None:
            args.extend([flag, Path(path).resolve()])
    _wait(_worker(checkout, python, "scan-transfer.py", args))
    return output


def start_build(
    transfer,
    output,
    *,
    checkout=None,
    config=None,
    python=None,
    bun=None,
    log=None,
    prior=None,
    guidance=None,
    motion=None,
    mouth_trace=None,
    name=None,
    segment_review=None,
    segment_evidence=None,
    head_binding=None,
):
    """Start the shared worker; return its Popen handle for a non-blocking Blender panel."""
    transfer, output = Path(transfer).resolve(), Path(output).resolve()
    bun = (
        Path(bun) if bun else Path(shutil.which("bun") or Path.home() / ".bun/bin/bun")
    )
    if not bun.is_file():
        raise FileNotFoundError(bun)
    if not prior and not (transfer / "0-guide/resolved.json").is_file():
        raise FileNotFoundError(transfer / "0-guide/resolved.json")
    args = [transfer, output, "--bun", bun.absolute()]
    if prior or guidance:
        if not (prior and guidance and transfer.is_file()):
            raise ValueError(
                "Raw input requires a source GLB, prior and reviewed guide"
            )
        for flag, value in (
            ("--prior", prior),
            ("--guidance", guidance),
            ("--motion", motion),
            ("--mouth-trace", mouth_trace),
            ("--segment-review", segment_review),
            ("--segment-evidence", segment_evidence),
        ):
            if value:
                args.extend([flag, Path(value).resolve()])
        if name:
            args.extend(["--name", name])
    if config:
        args.extend(["--config", Path(config).resolve()])
    if head_binding:
        args.extend(["--head-binding", head_binding])
    return _worker(checkout, python, "roster/build_blender_character.py", args, log)


def build_character(transfer, output, **options):
    """Bind the original head and verify bare and dressed GLBs. Return the completed directory."""
    _wait(start_build(transfer, output, **options))
    return Path(output).resolve()


def retarget_animation(
    source, target, output, *, clip=None, config=None, checkout=None, python=None
):
    """Retarget a GLB/FBX clip to a rigged GLB and return the new file and its joint mapping report."""
    output = Path(output).resolve()
    args = ["retarget", Path(source).resolve(), Path(target).resolve(), "--out", output]
    if clip:
        args.extend(["--clip", clip])
    if config:
        args.extend(["--config", Path(config).resolve()])
    _wait(_worker(checkout, python, "animate.py", args))
    return json.loads((output / "report.json").read_text())


def build_animation(recipe, output, *, checkout=None, python=None):
    """Bake a local animation recipe into a new GLB and return sampled constraint errors."""
    output = Path(output).resolve()
    _wait(
        _worker(
            checkout,
            python,
            "animate.py",
            ["build", Path(recipe).resolve(), "--out", output],
        )
    )
    return json.loads((output / "report.json").read_text())


def solve_pose(
    model,
    effectors,
    output,
    *,
    clip=None,
    time=0.0,
    config=None,
    checkout=None,
    python=None,
):
    """Solve a sampled pose in glTF world coordinates and return residuals for each target."""
    output = Path(output).resolve()
    args = [
        "solve",
        Path(model).resolve(),
        "--effectors",
        Path(effectors).resolve(),
        "--time",
        str(time),
        "--out",
        output,
    ]
    if clip:
        args.extend(["--clip", clip])
    if config:
        args.extend(["--config", Path(config).resolve()])
    _wait(_worker(checkout, python, "animate.py", args))
    return json.loads((output / "report.json").read_text())


def character_objects(scene, rig):
    if rig is None or rig.name not in scene.objects:
        raise ValueError("Choose the character armature")
    family = set(rig.children_recursive) | {rig}
    return [
        obj
        for obj in scene.objects
        if obj in family or (obj.type == "MESH" and obj.find_armature() == rig)
    ]


def import_character(path):
    """Import one rigged GLB and return its armature and optional body mesh."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = set(bpy.data.objects)
    # The importer records the Blender-space local rest matrix before applying
    # the first action, but normally exposes it only with its optional UI panel.
    temporary_rest_slot = not hasattr(bpy.types.Object, "gltf2_animation_rest")
    if temporary_rest_slot:
        bpy.types.Object.gltf2_animation_rest = bpy.props.FloatVectorProperty(
            size=(4, 4), subtype="MATRIX"
        )
    try:
        if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(path)):
            raise ValueError(f"Could not import {path.name}")
        created = set(bpy.data.objects) - before
        for obj in created:
            if obj.animation_data and obj.is_property_set("gltf2_animation_rest"):
                paths = animated_key_paths(obj.animation_data)
                rest = obj.gltf2_animation_rest
                location, rotation, scale = rest.decompose()
                defaults = {"location": location, "rotation_quaternion": rotation, "scale": scale}
                if obj.rotation_mode not in {"AXIS_ANGLE", "QUATERNION"}:
                    defaults["rotation_euler"] = rotation.to_euler(obj.rotation_mode)
                if obj.rotation_mode == "AXIS_ANGLE":
                    axis, angle = rotation.to_axis_angle()
                    defaults["rotation_axis_angle"] = (angle, *axis)
                animated = {key: list(value) for key, value in defaults.items() if key in paths}
                if animated:
                    obj["_charloom_object_action_rest"] = animated
            if temporary_rest_slot:
                obj.property_unset("gltf2_animation_rest")
    finally:
        if temporary_rest_slot:
            del bpy.types.Object.gltf2_animation_rest
    rigs = [obj for obj in created if obj.type == "ARMATURE"]
    if len(rigs) != 1:
        for obj in created:
            bpy.data.objects.remove(obj, do_unlink=True)
        raise ValueError(f"Expected one character armature, found {len(rigs)}")
    meshes = [
        obj for obj in created if obj.type == "MESH" and obj.find_armature() == rigs[0]
    ]
    body = next((obj for obj in meshes if obj.name.partition(".")[0] == "body"), None)
    return Character(rigs[0], body)


def fit_surface(body, scan, group):
    if (
        not body
        or not scan
        or body == scan
        or body.type != "MESH"
        or scan.type != "MESH"
    ):
        raise ValueError("Choose a body mesh and a separate aligned skin surface")
    if not group or body.vertex_groups.get(group) is None:
        raise ValueError("Choose the vertex group for the exposed body surface")
    if body.data.users != 1 or (
        body.data.shape_keys and not body.data.shape_keys.use_relative
    ):
        raise ValueError("Choose a single-user body mesh with relative shape keys")
    reference = scan.evaluated_get(bpy.context.evaluated_depsgraph_get())
    if not reference.data.polygons:
        raise ValueError("The skin reference has no surface")
    surface = BVHTree.FromPolygons(
        [reference.matrix_world @ v.co for v in reference.data.vertices],
        [tuple(p.vertices) for p in reference.data.polygons],
    )
    inverse = body.matrix_world.inverted()
    group_index = body.vertex_groups[group].index
    keys = body.data.shape_keys
    basis = keys.reference_key.data if keys else body.data.vertices
    coordinates = []
    for vertex, base in zip(body.data.vertices, basis):
        weight = next((g.weight for g in vertex.groups if g.group == group_index), 0.0)
        point = base.co.copy()
        if weight > 0:
            nearest, _, _, _ = surface.find_nearest(body.matrix_world @ point)
            point = point.lerp(inverse @ nearest, weight)
        coordinates.extend(point)
    if keys is None:
        body.shape_key_add(name="Basis", from_mix=False)
    keys = body.data.shape_keys
    name = "Charloom · Exposed skin"
    fitted = keys.key_blocks.get(name) or body.shape_key_add(name=name, from_mix=False)
    fitted.relative_key = keys.reference_key
    fitted.data.foreach_set("co", coordinates)
    fitted.value = 1.0
    body.data.update()


def bind_clothing(body, rig, clothing):
    """Bind explicitly supplied aligned garment objects using the body rest mesh."""
    context = bpy.context
    clothing = list(clothing)
    if body is None or rig is None or body.find_armature() != rig:
        raise ValueError("Choose the skinned body and its armature")
    if not clothing:
        raise ValueError("Select the aligned clothing mesh in Object Mode")
    deform = {bone.name for bone in rig.data.bones if bone.use_deform}
    for obj in clothing:
        if obj == body or obj.type != "MESH" or obj.data.users != 1:
            raise ValueError("Each garment must have its own mesh data")
        if obj.find_armature() not in (None, rig):
            raise ValueError(f"{obj.name} is already bound to a different armature")

    deform_names = sorted(deform)
    bone_ids = {name: i for i, name in enumerate(deform_names)}
    centers = np.asarray(
        [rig.matrix_world @ rig.data.bones[name].head_local for name in deform_names]
    )
    # Sample the evaluated body in rest pose, including its fitted surface.
    pose = rig.data.pose_position
    rig.data.pose_position = "REST"
    context.view_layer.update()
    evaluated = body.evaluated_get(context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    assignments = []
    try:
        mesh.calc_loop_triangles()
        triangles = [tuple(tri.vertices) for tri in mesh.loop_triangles]
        coords = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        if not triangles:
            raise ValueError("The body has no surface triangles")
        tree = BVHTree.FromPolygons(coords, triangles, all_triangles=True)
        groups = {g.index: g.name for g in evaluated.vertex_groups if g.name in deform}
        axes = (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))
        for obj in clothing:
            rows = []
            vertices = [obj.matrix_world @ v.co for v in obj.data.vertices]
            distances = []
            for vertex, position in zip(obj.data.vertices, vertices):
                point, _, index, distance = tree.find_nearest(position)
                distances.append(distance)
                ids = triangles[index]
                blend = barycentric_transform(point, *(coords[i] for i in ids), *axes)
                weights = {}
                for i, fraction in zip(ids, blend):
                    for g in mesh.vertices[i].groups:
                        if g.group in groups and fraction > 0 and g.weight > 0:
                            bone = groups[g.group]
                            weights[bone] = weights.get(bone, 0.0) + fraction * g.weight
                row = sorted(weights.items(), key=lambda item: -item[1])[:4]
                total = sum(w for _, w in row)
                if not total:
                    raise ValueError(
                        f"No body skin weights under {obj.name} vertex {vertex.index}"
                    )
                rows.append([(name, weight / total) for name, weight in row])
            obj.data.calc_loop_triangles()
            faces = np.asarray([tuple(f.vertices) for f in obj.data.loop_triangles])
            joints = np.zeros((len(rows), 4), dtype=np.int32)
            values = np.zeros((len(rows), 4))
            for i, row in enumerate(rows):
                for k, (name, weight) in enumerate(row):
                    joints[i, k] = bone_ids[name]
                    values[i, k] = weight
            refined = sleeve_weights(
                np.asarray(vertices),
                faces,
                joints,
                values,
                np.asarray(distances),
                deform_names,
                centers,
            )
            if refined is not None:
                rows = [
                    [(deform_names[j], w) for j, w in zip(js, ws) if w > 0]
                    for js, ws in zip(*refined)
                ]
            assignments.append((obj, rows))
    finally:
        evaluated.to_mesh_clear()
        rig.data.pose_position = pose
        context.view_layer.update()

    for obj, rows in assignments:
        # Keep custom vertex groups used by cloth simulation or modifiers.
        for group in list(obj.vertex_groups):
            if group.name in deform:
                obj.vertex_groups.remove(group)
        targets = {}
        for index, row in enumerate(rows):
            for name, weight in row:
                if name not in targets:
                    targets[name] = obj.vertex_groups.new(name=name)
                targets[name].add([index], weight, "REPLACE")
        mod = next(
            (m for m in obj.modifiers if m.type == "ARMATURE" and m.object == rig), None
        )
        if mod is None:
            mod = obj.modifiers.new("Charloom · Armature", "ARMATURE")
            mod.object = rig
        world = obj.matrix_world.copy()
        obj.parent = rig
        obj.matrix_world = world
        obj["charloom_part"] = "garment"


def owners(scene, rig):
    for obj in character_objects(scene, rig):
        if obj.animation_data:
            yield obj
        keys = getattr(obj.data, "shape_keys", None)
        if keys and keys.animation_data:
            yield keys


def animated_key_paths(data):
    paths = set()
    for track in data.nla_tracks:
        for strip in track.strips:
            if not strip.action or not strip.action_slot:
                continue
            for layer in strip.action.layers:
                for action_strip in layer.strips:
                    bag = action_strip.channelbag(strip.action_slot)
                    if bag:
                        paths.update(curve.data_path for curve in bag.fcurves)
    return paths


def preview(rig, clip):
    """Select a named clip on the timeline; use scene.frame_set() to sample it."""
    context = bpy.context
    matches = []
    for owner in owners(context.scene, rig):
        data = owner.animation_data
        strip = next(
            (
                track.strips[0]
                for track in data.nla_tracks
                if track.name == clip and len(track.strips)
            ),
            None,
        )
        matches.append((owner, strip))
    if not any(strip for _, strip in matches):
        raise ValueError(f"The character has no {clip} NLA track")
    end = max(strip.action_frame_end for _, strip in matches if strip)
    for owner, strip in matches:
        data = owner.animation_data
        data.use_nla = False
        data.action = strip.action if strip else None
        if strip and strip.action_slot:
            data.action_slot = strip.action_slot
        if isinstance(owner, bpy.types.Object):
            for key, value in owner.get("_charloom_object_action_rest", {}).items():
                setattr(owner, key, value)
        if owner == rig:
            for bone in rig.pose.bones:
                bone.matrix_basis = Matrix.Identity(4)
        if isinstance(owner, bpy.types.Key):
            paths = animated_key_paths(data)
            for key in owner.key_blocks[1:]:
                # Keep authored body proportions and the exposed-skin correction.
                if key.path_from_id("value") in paths:
                    key.value = 0.0
    scene = context.scene
    scene.frame_start, scene.frame_end = 0, max(1, round(end))
    scene.frame_set(0)


def export_character(rig, path):
    """Export only this character and its actions as a GLB."""
    context = bpy.context
    path = Path(path).resolve()
    if path.suffix.lower() != ".glb":
        raise ValueError("Expected a .glb export path")
    path.parent.mkdir(parents=True, exist_ok=True)
    objects = character_objects(context.scene, rig)
    selected, active = list(context.selected_objects), context.view_layer.objects.active
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        context.view_layer.objects.active = rig
        result = bpy.ops.export_scene.gltf(
            filepath=str(path),
            export_format="GLB",
            use_selection=True,
            export_extras=True,
            export_animations=True,
            export_animation_mode="ACTIONS",
            export_merge_animation="NLA_TRACK",
            export_anim_single_armature=False,
            export_force_sampling=False,
            export_morph=True,
        )
        if "FINISHED" not in result:
            raise ValueError("GLB export did not finish")
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected:
            obj.select_set(True)
        context.view_layer.objects.active = active
