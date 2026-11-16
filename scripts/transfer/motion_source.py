"""Sample one imported action in glTF world coordinates using Blender's FBX/glTF readers."""

import bpy
import numpy as np
from mathutils import Matrix
from pathlib import Path
import math
import sys

source, output, clip, fps, *trim = sys.argv[sys.argv.index("--") + 1 :]
fps = float(fps)
bpy.ops.wm.read_factory_settings(use_empty=True)
if Path(source).suffix.lower() == ".fbx":
    bpy.ops.import_scene.fbx(filepath=source, anim_offset=0)
else:
    bpy.ops.import_scene.gltf(filepath=source)
rigs = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
if len(rigs) != 1:
    raise ValueError(f"Expected one motion armature; found {len(rigs)}")
rig = rigs[0]
data = rig.animation_data
actions = [
    (t.name, s.action, s.action_slot)
    for t in data.nla_tracks
    for s in t.strips
    if s.action
]
if data.action:
    actions.append((data.action.name, data.action, data.action_slot))
for name, action, slot in actions:
    if not clip or clip in {name, action.name, *action.name.split("|")}:
        break
else:
    raise ValueError(f"Missing {clip!r}; available actions: {[a[0] for a in actions]}")
data.use_nla = False
data.action = action
if slot:
    data.action_slot = slot
scene = bpy.context.scene
native = scene.render.fps / scene.render.fps_base
begin, end = action.frame_range
seconds = (end - begin) / native
if fps <= 0 or not math.isfinite(fps):
    raise ValueError("Sampling rate must be finite and positive")
# Compare in glTF accessor precision so the final frame cannot round to a duplicate.
start = float(trim[0]) if trim and trim[0] else 0.
stop = float(trim[1]) if len(trim)>1 and trim[1] else seconds
if not 0<=start<=stop<=seconds+1e-6:
    raise ValueError("Motion trim exceeds the source action")
times = np.unique(np.r_[np.arange(start, stop, 1/fps), stop].astype(np.float32))
bones = list(rig.data.bones)
ys = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
rest = np.array([ys @ rig.matrix_world @ bone.matrix_local for bone in bones])
poses = []
for time in times:
    frame = begin + float(time) * native
    scene.frame_set(math.floor(frame), subframe=frame - math.floor(frame))
    poses.append(
        np.array([ys @ rig.matrix_world @ rig.pose.bones[b.name].matrix for b in bones])
    )
np.savez(
    output,
    names=[b.name for b in bones],
    rest=rest,
    poses=poses,
    times=times-start,
    clips=[n for n, _, _ in actions],
    clip=action.name,
)
