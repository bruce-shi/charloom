# Replicating the walking of existing characters

English · [简体中文](walk-replay.zh-CN.md)

From the last correct walkout of the character modeling, records include the original scan, the source GLB or FBX with actions, specific segments, actual skinned skeleton, binding pose, initial binding script, and post-build action corrections. For re-running, corresponding input files, scripts, and parameters are required; the final GLB should be kept for comparison and should be provided separately to the reviewer. The same character appearance and action names cannot identify the production process.

## Drive motion with the source rig

1. Read the original skeleton with dynamic actions, parent-child hierarchy, local rotation axes, and skinning weights. Preserve the time of the source clip, interpolation, and rotation keyframes. Record the matrix from the scan to the character workspace, as well as the target positions of each joint within that space.
2. Adapt the original skeleton using the position of the target joints, while preserving the static orientation of the bones. Calculate the local static translations for each joint using the new parent-child world matrix. If the original clip has local translation tracks, add the difference between the new and old local static translations to the value of the corresponding keyframes. Retain the time, rotation, and scale tracks. Recalculate the inverse bind matrices.
3. Map the scanned vertices back to the static shape of the action source to find neighborhood weights, and write them back to the original scanned surface in the workspace. The reproduction of existing binding scripts follows the same spatial transformation, joint targets, neighborhoods, weight mixture, and local corrections. Independent verification of the neutral pose, original UV, skinning, and gait after binding.
4. When using MakeHuman for the body surface later, the body will be fitted to the character workspace. The skinned joints from the template will be mapped to the existing walking skeleton. The correspondence between the left and right sides, back, shoulders, arms, fingers, and toes will be processed, and the bind transforms of each component will be checked. The original head, body, and clothing will share the same adapted skeleton. The presence of MakeHuman body topology in the file does not imply an actual driving skeleton.
5. Apply necessary local action corrections according to the original construction records. For example, if the swing arm has been adjusted to avoid interference, its rotation trajectory should be restored, and gaps should be checked on the final body and clothing. If the static pose of the skeleton has changed, re-derive the affected trajectories. Compare files from corresponding stages to avoid considering intermediate versions as the final action.

### Establish the character binding once

Existing characters use their fixed skeleton files directly. When re-establishing the same binding from a static scan, prepare a single-surface static scan and a single-surface skeleton GLB with skin weights and original segments. Run the following command in the root directory of the repository:

```sh
"$PY" skills/charloom/scripts/bind_fixed_rig.py "$SCAN" "$RIG_SOURCE" "$BIND_CONFIG" "$FIXED_RIG"
```

In `BIND_CONFIG`, set `scan_sha256` and `rig_sha256` to the two input files’ SHA-256 digests. Give the scan-world to rig-world transform in metres as a row-major 4×4 matrix in `scan_to_rig_world`. List calibrated 3D targets by bone name in `joint_positions_m`. Use `source_clip`, `output_clip` and `mesh_name` to identify the clip and meshes. Reuse `rbf_smoothing`, `neighbors`, `distance_floor_m` and `distance_power` from the original binding record. The optional `height_weight_ramps` records the calibrated transition in weights by height using `joint` and `range_m`. If the original build corrected the arm swing, use `rotation_track_edits` to list `times` and the quaternion `values` at the same key times for each bone. Keep the provenance of these character-specific parameters.

After completing the bare model and clothing geometry, as well as the head and facial expression keys, the two layers will be bound together in one `FIXED_RIG`:

```sh
"$PY" skills/charloom/scripts/attach_fixed_rig.py "$BARE_LAYER" "$FIXED_RIG" "$LAYER_CONFIG" "$FINAL_BARE"
"$PY" skills/charloom/scripts/attach_fixed_rig.py "$DRESSED_LAYER" "$FIXED_RIG" "$LAYER_CONFIG" "$FINAL_DRESSED"
```

`LAYER_CONFIG.layer_to_rig_world` represents the 4×4 matrix from the layer's workspace to the fixed skeleton's world space; it can be synthesized from the calibrated transformations scanned to the side spaces.`skin_joint_map` maps each template bone with a weight within the layer to the fixed bone names. When multiple thin bones correspond to the same target, the weights are merged. Facial movements are configured according to the segment names in `expressions`, with each segment retaining its shape keys. Rigid movements such as head and neck use the correspondence "fixed bone names → input pose bone names". If the character relies on independent eye, jaw, or finger bones for control, these joints should be included in the initial skeleton and a clear mapping should be defined. After binding, re-measure the neck seams, limbs, static surfaces, and deformations during movement.

## Template skeleton carrying actions

If the original build retargeted motion, reuse the original source and clip, template skeleton, source-to-target joint map, reference chains, sample rate and subsequent corrections. Verify the options supported by the entry point. `scan-transfer.py --motion` selects the default clip and uses built-in reference-chain alignment. If the build used a custom configuration, replay it through its original retargeting entry point. For a new character use the motion specified for the task.

## Check the tracks actually written to the GLB

Execute from the root directory of the repository in the CV Python environment. The comparison files should be taken from the same stage being compared.

```sh
"$PY" skills/charloom/scripts/compare_walk.py "$WALK_BASELINE" "$RESULT_GLB" --source-clip "$BASELINE_CLIP" --result-clip "$RESULT_CLIP"
```

This script compares channels, keyframe time, values, and interpolation based on node names and attributes. It also compares animation nodes and their parent-child relationships, as well as the static transforms of ancestors. If there are differences, it returns 1 and lists the specific channels. The expected local corrections must match the original build records item by item. Steps that maintain the track should be fully matched. If the exporter uses resampling, it compares world-space joints with heel and forefoot trajectories at common time points. It checks stride, support transition, swing arm, and clothing contact. The final export includes additional information such as stage height. The only additional item expected is the `translation` for the specified container. In the comparison report, the channels, hierarchy, and static transforms of the original skeleton should all be identical. The shoe sole is rechecked with [Walking grounding](walking-ground.md), and the takeoff time is preserved during running.

## Add motion to the existing character rig

For each animation segment, record the source, original clip, required joints, and the mapping of inputs to the character. Use the previous version of the character's GLB as the target, and the same configuration applies to both naked and dressed versions of the same action. Add new clips using the unique `name`. If there is a segment with the same name, the redirect entry will update it.

```sh
"$PY" scripts/animate.py map "$ANIMATION_SOURCE" "$FINAL_DRESSED" --config "$CLIP_CONFIG"
"$PY" scripts/animate.py retarget "$ANIMATION_SOURCE" "$FINAL_DRESSED" --clip "$SOURCE_CLIP" --config "$CLIP_CONFIG" --out "$DRESSED_NEXT"
"$PY" scripts/animate.py retarget "$ANIMATION_SOURCE" "$FINAL_BARE" --clip "$SOURCE_CLIP" --config "$CLIP_CONFIG" --out "$BARE_NEXT"
```

The key for `joint_map` represents the fixed character bone names, with the value being the source bone name. It checks the anatomical order of the spinal chain segments; the necessary joints are listed in `target_joint_scope` and the corresponding `min_target_coverage` is set. When a scene requires a circular motion on the same spot, `in_place_axes` can specify the world axis for the cycle of net displacement, preserving the relative swing during that axis cycle and the movement of unassigned axes. When scene displacement is needed, the original root motion should be retained.

This entry will verify the original character's skeleton hierarchy, static transformation, inverse binding matrix, mesh joint indices, and weights. Each layer is recorded in the `report.json.fixed_target_rig` section as the skeleton and skinning hash. It will also check the original actions, add new segments during playback, and switch between naked and dressed states in the same frame. The Blender API's `retarget_animation` section uses the same entry.

### Secondary bone chains and motion corrections

1. Compare the target rig, existing tracks and garment weights. Identify joints driven by the new motion, attachments that follow their parents, and hair, skirt or cape chains needing independent motion. Record the treatment of each chain. Coverage of the mapped set measures only the selected body joints.
2. When there are no corresponding secondary bone tracks for the new action, first check the existing secondary motion scripts or constraints of the character. Using key points like the head and pelvis as references, follow and swing back along the free segments baked from the new action. Verify that the movement is stable near the key points, that the end of the movement indeed swings in the coordinate system of the key points, and that the loop is continuous. The delay and swing range are determined based on the length, posture, and clothing gaps of the character.
3. If an intersection occurs only in the new clip, locate the time and adjust the relevant upper-arm or secondary bone rotation in that clip. Write the correction in the current parent bone’s local space. On copies of both exports, call `roster.write_face_clips.write_clips(model, [clip], extend=True)` to update selected channels. The `clip` record includes the existing clip’s `name`, `tracks` and `morph_tracks`. Each bone track needs the layer-specific `joint` node index, `property`, `times` in seconds and local `values`. Resolve the indices by bone name for each layer. Store local rotation quaternions in x, y, z, w order.
4. On each layer in the final export, use the above `compare_walk.py` to compare each segment with the old walk, expressions, and spoken parts by name. Also, review the newly added bone tracks for both naked and dressed scenes. List the individual added ground tracks in the notes. Save the action sources and the hashes of the two-layer input, the mapping, the secondary action scripts, and the ground parameters. It is necessary to reproduce the entire set of action steps in the new directory and verify the final file.

### Check local intersections during motion

In the root directory of the repository, list the triangular primitives of the final GLB clothing using CV Python. Create paired selection areas based on actual proximity of the parts, such as a pair of free hair bundles for the cuffs, hands for the body, and skirt edges for the thighs. Avoid fixed attachment points and seams that are originally connected.

```sh
"$PY" skills/charloom/scripts/check_motion_contacts.py "$FINAL_DRESSED" --inventory
"$PY" skills/charloom/scripts/check_motion_contacts.py "$FINAL_DRESSED" --config "$CONTACT_PAIRS" --clip "$RESULT_CLIP" --samples "$SAMPLES" > "$REVIEW/clip-contacts.json"
```

`CONTACT_PAIRS` is JSON. In `regions`, map each selection name to a record such as `{"mesh": 0, "geometry_hash": "...", "faces": [12, 13]}`. Substitute the primitive index from the inventory, its geometry hash and triangle IDs reviewed on that primitive. Use `"faces": "all"` for a whole piece. In `pairs`, list pairs of selections to compare, for example `[["free_hair", "sleeve"], ["hand", "jacket"]]`. Two separate face selections may belong to one primitive.

`SAMPLES` represents the number of uniformly sampled points, which are selected based on the duration of the action and the movement speed. The original segments and the new segments are compared using the same selection area. Sampling is densified when there is rapid change in contact. The report includes the number of static intersections, the number of pairs of intersecting surfaces at different times, as well as the time and example surface numbers where there are the most intersections. The report separately records the intersections in the original and the new actions; the sampling results only cover the selected surfaces and times. Whether geometric intersections are visible is determined according to [Gallery - Reviewing the Same Frame](../../charloom-gallery-review/SKILL.md).
