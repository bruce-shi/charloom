# Replayable reshaping of the complete chest and local limbs

English · [简体中文](section-refit.zh-CN.md)

If the front silhouette matches but the side view looks flat, render the full template, the landmark-only deformation and the contour-adjusted body in profile. Compare thickness at the same sites. If depth collapses during landmark fitting, check that the left and right landmarks are distinguishable in front view, that the source and target cameras agree, and that depth observations constrain joint centres. Set `subject.template_warp` to `frontal_depth_preserving` in this guide and rerun the transfer if appropriate. This mode interpolates 3D displacements from the template’s frontal XY positions, preserving thickness between front and rear points at the same XY. Then adjust the measured sections. Use the default `spatial` for side-on or overlapping poses or when the front and rear surfaces must move independently. Check `stage2_rig.template_warp` in `report.json` and remeasure the joints, contours and sections.

`scan-transfer.py` First, use the marker to fit the skeleton and the complete MakeHuman surface, and then use the registration contour to constrain the visible projection. The first output still requires individual measurements. If the chest is flattened, or if there are double peaks or the shoulders and arms are pulled together, restore the original complete template as follows; do not repeatedly trace the unilateral contours on the already bent chest.

## Prepare source maps for cross-sections

1. Save the bare model of this round, the original template, and all measurement targets. Select the body primitive corresponding to the original template; the fitted body must retain the vertex order of the template, and the faces can be a subset of the original face set. The re-topologized body requires a separate correspondence adjustment, and this script cannot be used.
2. The torso should include heights corresponding to the waist, lower ribs, chest peak, shoulders, neck root, and upper neck. The positions should be recorded in the original template, reference front, and reference side, first addressing the inconsistency in anatomical heights between the two views. The sectional target should include both the sides and the center, not just the width. Do not force the chest peak from the original template to be superimposed with another height from the reference.
3. The limbs are articulated around the axes of shoulder-elbow, elbow-wrist, hip-knee, and knee-ankle. Record the actual standing position from the cut at the same anatomical positions as the template, preserving the shape with respect to the target width and depth, as well as the central mapping. The length of the inner side of the upper arm is constrained by the path from the armpit to the inner elbow; the length of the shoulder-elbow chain cannot be substituted.
4. Create a one-dimensional `blend.npy` for each operation, with one value per template vertex. Zero protects the vertex, one applies the full correction, and intermediate values blend smoothly. Start from reviewed bone-weight regions, adjust them as Blender vertex groups and export the weights. Protect the front chest, neck seam, palms and other limbs. Weights for all operations on a vertex must sum to at most one. Selecting a whole ring by height can pull the rib cage into the upper-arm correction.

## Configuration and Execution

```sh
"$PY" skills/charloom-body-fit/scripts/refit_sections.py "$FIT_CONFIG" --out "$BODY_REFIT"
```

The JSON top layer `mesh` is the raw template to be edited, and `native_mesh` is the complete template; the formats for both are shown in[Body input](../../charloom-body-measure/references/script-inputs.md#2-record-meshes-and-face-selections). The original template can be explicitly transformed using `world_from_model`, while the raw template only allows the use of the original form without additional transformations.

`operations` Each item includes:

- `name`、`blend:{file,sha256}`。
- `native_origin_m`, `target_origin_m`; `native_basis`, `target_basis` are 3×3 **column vectors** `[axis,width_direction,depth_direction]`. They must be orthogonal and right-handed. The limbs use their own local axes; the torso uses an axis along the height.
- `native_region`: Only for the selection of the original template surface for the cut section, with the same format as the homologue `regions`.
- `stations`: Arranged in anatomical order, each item includes `native_axis_m`, `target_axis_m` (axial distance from the origin in meters), `target_bounds_m:[[min_width,min_depth],[max_width,max_depth]]`, and `evidence` (source file type, position, measured or estimated value). The invisible depth should include the estimation basis, which cannot be left blank.
- `native_seed_m` Can be specified in the station for points within the original cross-section.`native_bounds_m` Automatically calculated from the real triangular closure; if provided, it will be verified to match the actual cross-section. When the shoulder cut opening is set, a station should be placed at a closed and corresponding position; the maximum range of the opening segment cannot be considered as a complete cross-section.

The native axial stations must cover all vertices with nonzero blend weights. Values outside that range raise an error. The script maps heights monotonically in anatomical order and maps the native cross-section laterally using paired boundaries. It applies all operations to one complete skin, preserving vertex order, UVs, rig and other objects. Use `displacement_fairing` for optional Laplacian smoothing of displacement on the welded surface (default 0). Remeasure after smoothing because dimensions can change.

The outputs are `body.glb` and `refit.json`. The latter records source digests, actual template sections, the number of modified vertices and maximum displacement. The result still needs remeasurement. Check dense contours, outer-arm, outer-leg and inner-upper-arm paths, local sections and hand and foot lengths. Inspect untextured front, side and oblique views, then play the complete motion.

## Bring the same assembly that was passed in the physical test

When the body is being shaped before the original head assembly, use the modified version as the bare model for the transfer. Update the clothing export for the same skeleton using the following commands. First, copy the entire stage directory to the new revision directory, keeping the old version. After writing the new file name, replace the corresponding file in the new directory.

```sh
"$PY" skills/charloom-body-fit/scripts/copy_body.py "$BODY_REFIT/body.glb" "$TRANSFER/3-skin/body.glb" --out "$REVISION/body.glb"
"$PY" skills/charloom-body-fit/scripts/copy_body.py "$BODY_REFIT/body.glb" "$TRANSFER/4-garments/character.glb" --out "$REVISION/character.glb"
```

The script verifies bone names, rest transformations, vertex order, and triangle correspondence. It preserves the face collection, materials, other objects, and actions of the target file. The revised skin and weight transformations are transferred to the target coordinates. The bare model retains the complete lower body; the dressed version can retain its own occluded cut surfaces. Do not copy the dressed cut surfaces back to the bare model.

Clothing also needs to be adjusted according to the final physical examination and re-fitting process; this step ensures that the body does not revert to the old state quietly in the future. If the shape is changed due to changes in the skeleton or topology, the transfer/head/clothing correspondence needs to be redone, and forced transplantation is not allowed.

## Arm lifting test for fixed chest

```sh
"$PY" skills/charloom-body-fit/scripts/check_pose_regions.py "$POSE_CONFIG" --out "$POSE_REVIEW"
"$PY" scripts/animate.py validate "$POSE_REVIEW/pose-check.glb"
```

Configure `mesh` using the same hash-bound model record. Set `reference_bone` to the actual rib-cage bone. In `regions`, identify the medial chest, the outer pectoral transition to the anterior axilla, and the upper-arm transition separately. For each, enter reviewed body vertex IDs in `vertices` and selection images and reasoning in `evidence`. A patch at the sternum misses the outer chest and anterior axilla.

`poses` Each item corresponds to `name`, `bone`, `axis_world:[x,y,z]`, and `degrees`. For example, in GLB Y-up coordinates, the left and right arms are lifted outward around the world Z-axis, with the angle starting from 48°. The symbols are determined based on the actual skeleton and posture, and the bone names are derived from the actual data. `clips` refers to the complete action to be sampled, while `samples_per_clip` defaults to 33, indicating an increase in sampling near the problematic frames.

For each region the report gives maximum and 95th-percentile displacement in rib-cage coordinates, edge-length growth, maximum stretch ratio and the worst edge. A diagnostic GLB retains the original clips and adds an arm-raise clip from rest at 0 s to the specified pose at 1 s. Open this diagnostic separately in Blender or the Gallery and inspect the armpit from an oblique view. Deliver the actual animation file separately.

If the chest muscles are dragged away when lifting the arms, check the impact on the upper arm/shoulder bones in that area. Versioned weight fixes have been provided for `scripts/skin.py`:

```sh
"$PY" scripts/skin.py start "$BODY" --config "$SKIN_CONFIG" --out "$SKIN_SESSION"
"$PY" scripts/skin.py step "$SKIN_SESSION" --plan "$SKIN_PLAN"
"$PY" scripts/skin.py inspect "$SKIN_SESSION"
```

The plan requires `expected_round` and `diagnosis`. Set `strategy` to `exclude_influences` to remove a verified incorrect influence, `relax_strain` to smooth a local transition, or `weld_seams` to match weights across a real seam. Limit `regions` with `mask:{mesh,geometry_hash,vertices}`. Specify excluded joints in each region’s `exclude_joints` and protect joints with the top-level `protect_joints`. Use `max_weight_change_l1` to bound changes and retain a gradual transition at the front of the axilla. Read the round file actually accepted by the session. Remeasure the body, raise the arm and check the clothing. Weight editing changes motion deformation; rest-shape dimensions require surface editing.

## Check and correct wrist and palm orientation

In the profile view of the same pose, the wrist center and the same fingertips are marked, preserving the working coordinate direction and the straight-line distance from the wrist to the fingertips. Compare the profile view of the bare hand with that of the gloves. The vertical palm side view of another pose is only used to measure the abdominal edge. If the error is due to the bending of individual fingers, the posture of each finger segment needs to be checked; only when the entire hand is facing straight or there is a vertical dimension deviation, the following script can be used.

After reshaping and before rebinding the walk, create `hand-rest.json`. Set `body_node` and `body_positions_sha256` (SHA-256 of the full body POSITION attribute as contiguous little-endian float32). In `evidence`, record the measurement file’s `file` and `sha256`. For each entry in `hands`, give the actual wrist bone in `wrist`, the fingertip vertex on the bare model in `tip_vertex`, and the 3D wrist-to-tip vector from the same-pose reference in `wrist_to_tip_m:[dx,dy,dz]`. The script changes orientation and length along the fingertip axis within the wrist subtree, blending into the forearm by skin weight. Inspect thumb and palm rotation separately because two endpoints leave twist around the axis undetermined.

```sh
"$PY" skills/charloom-body-fit/scripts/fit_hand_rest.py "$TRANSFER/3-skin/body.glb" "$HAND_CONFIG" --out "$NEXT/3-skin/body.glb" --rebuild-clip walking
"$PY" skills/charloom-body-fit/scripts/fit_hand_rest.py "$TRANSFER/4-garments/character.glb" "$HAND_CONFIG" --out "$NEXT/4-garments/character.glb" --rebuild-clip walking
```

The two inputs use the same body vertices and skeleton. The input does not include the omission of the `--rebuild-clip walking` when there is no walking. If this parameter is specified, the new file will remove the old actions that need to be re-targeted; the original actions and mappings are re-targeted to two new GLBs, followed by the original head assembly. Check the static positions of the two-layer wrist bones and finger bones, as well as the skinning. Re-measure the vectors from the wrist to the fingertips, the width of the hand, the height difference from the shoulder to the fingertips, and the openings of the sleeves. For the clothing version, hide the body separately, and determine whether the light-colored fingertips come from the glove material or are exposed.
