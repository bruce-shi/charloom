---
name: charloom-scan-volume
description: 用 Charloom 原扫描中独立的躯干和四肢壳体构建 scan-volume 裸模。适用于已确认五个可分离皮肤壳体的角色。
---

# Scan-volume body

Use this opt-in assembly with an observed volume head from the same scan. The
builder combines the torso and limbs into one solid and joins it to the head
along their actual intersection. The output keeps the skeleton, separate
garments and measured hand articulation. The skin has scan-derived topology
and carries no MakeHuman body-size morphs.

## Select the surfaces

Start from the completed calibrated transfer directory for this character. Its
`report.json` identifies the source, prior and working frame.
`0-guide/resolved.json` supplies the face marks and measured joint sections.
The assembler reads both staged GLBs and the matching `mouth-trace.json`.
It also reads `body-selection.json` when a corrected body fit was selected.
Use the static, single-mesh, single-primitive textured scan used by the
transfer. A `raw_head` file or a matching roster recipe that supplies a
prepared head volume selects an incompatible head input.

On the **original scan**, isolate the torso and each complete limb. The
executing agent determines ownership from the surfaces, their attachments
and the calibrated views. Count connected face components after welding
coincident positions across atlas seams. Record one **zero-based triangle
index in the original primitive's index order** per shell. The five indices
must resolve to five distinct components. A single continuous whole-body
mesh does not meet this builder's input contract. If ownership is ambiguous,
inspect the source triangles before assigning a seed.

Key the config by character name. Under `body_volume`, supply the transfer's `source_hash` and a `skin_seeds` object with one reviewed triangle index for each of `torso`, `arm_l`, `arm_r`, `leg_l` and `leg_r`. Where the proximal shells end under clothing and leave a measured gap, `pelvis_fill: proximal_hull` bridges the interval. Inspect that inferred surface against the garment and bare silhouette. The body must form one connected solid with a positive-volume overlap with the head.

## Assemble and verify

From the checkout, set `PY` to the Python interpreter with the project CV dependencies. Set `transfer`, `result` and `config` to the current calibrated transfer directory, a fresh output directory and the reviewed configuration file.

```sh
"$PY" scripts/roster/build_blender_character.py "$transfer" "$result" --config "$config"
"$PY" scripts/animate.py validate "$result/3-skin/body.glb"
"$PY" scripts/animate.py validate "$result/4-garments/character.glb"
```

In `$result/report.json`, confirm that `body_volume.surfaces` contains the
five reviewed shells and `neck_skin_fit.method` is `shared_boolean_boundary`.
The general `route` value `free_topology` also occurs for a volume head on a
template body. `body_volume.source_crossings` describes the input shells;
measure the exported skin separately.

Compare source, fitted body and volume body from the same cameras at the
neck, shoulders, wrists and feet. Play the exported walk through a full
cycle. Check paired neck vertices for separation **and** nearby nonadjacent
faces for intersections. Measure local edge stretch at the wrists and
inspect the fingers and toes at peak flexion. A hand without supported
digit measurements follows the wrist. Check which digits actually move,
the blink and dialogue, and clothing clearance in both exports. Record
remaining defects separately from structural validation and Gallery adoption.
