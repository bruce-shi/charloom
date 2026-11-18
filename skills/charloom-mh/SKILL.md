---
name: charloom-mh
description: "Build an animated, dressable MakeHuman character from a scan. Follow the complete Charloom workflow or find the stage to repeat for an existing character."
---

# Build an animated character with interchangeable clothing from a scan

English · [简体中文](SKILL.zh-CN.md)

Follow the stages in order. Each linked skill gives the procedure and the evidence needed to complete its stage. For a new character, start at step 0. For a repair, identify the file actually loaded in the Gallery, its source scan and the problem frame, then rerun the affected stages and downstream checks. Reuse earlier evidence only when the source, camera, pose and scale match.

Calibrate the original scan, register references for occluded anatomy, measure the body, fit the MakeHuman body and remeasure. MakeHuman supplies the body topology. Preserve the scan’s head geometry, facial surfaces, UVs and textures. Bind lids and lips directly on that surface and join the head to the body. At the first rigging, record the bone names, hierarchy, rest transforms, bind matrices and skin weights. Check for a registered binding first and reuse it when present. To add motion, begin with the registered bare and dressed exports. Map the new clip to their rig, including character-specific chains for hair, garments and props. [Keep the rig fixed when adding motion](../charloom/references/walk-replay.md#add-motion-to-the-existing-character-rig)

## 0. Confirm inputs, deliverables and environment

Install CV Python, Bun, and Blender according to [Environment and Input Preparation](../charloom/references/setup.md). Point `PY` towards this CV environment and export `CHARLOOM_CV_PYTHON="$PY"`, ensuring that the Python used by Bun has the same set of dependencies.

1. Locate the task’s original scan GLB and record its provenance, absolute path, SHA-256, mesh structure and materials. Check for skinning, animation, multiple primitives or texture atlases. When preparing a static, merged or re-atlased copy, retain the original and record the conversion. Bind the guide to the exact static file used by the pipeline.
2. Confirm which outputs are needed: bare and dressed models, an editable Blender scene, motion clips and a Gallery entry. A full character build preserves the original head and includes body measurements before and after fitting. Include facial motion, equipment and clothing as requested for the character.
3. Generate a suitable MakeHuman template and record the body parameters and path. To reproduce an existing walk, use [Step Action Tracing and Re-run](../charloom/references/walk-replay.md) to identify the original motion, rig, binding entry point and local keyframe corrections. Before delegating, compare the supplied inputs with the original build records. For a new character, select its designated rig before choosing motion material. The motion file’s rig describes the source clip. The template’s dimensions are only starting values for fitting.
4. Set up output directories for this guide, measurement, transfer, build, and review. Keep the stage input and report, and use a new directory for the revised build to avoid overwriting the version being compared.

Use absolute paths. Point `SCAN`, `PRIOR` and `MOTION` to the input files, and `GUIDE`, `MEASURE`, `PARTS`, `TRANSFER`, `BUILD_PARENT` and `BLENDER_OUT` to this run’s directories. The last component of `TRANSFER` is `NAME`. Use fresh output directories. You can create `MEASURE` in advance, leaving each command’s output subdirectory absent. Run the remaining commands from the repository root.

**Completion Requirements:** The actual original scan, template, actions, and environment must be readable; the body, head, and clothing routes to be followed are clearly defined. If the original scan is missing, the project will stop at the input preparation stage. There is no need to use a template or synthesized character to replace the original scan for completion verification.

## 1. Calibrate the original scan and identify surface structures

Follow [Scanning Calibration](../charloom-scan-guide/SKILL.md). Render front, side and rear views and detail views as needed. Mark shoulders, elbows, wrists, palms, hips, knees, ankles, neck base and head using the character’s own left and right. Verify the camera, pixel scale and view direction.

The original text requires separate checks for connection components and textures: distinguishing facial skin, open eye patches, real independent eyeballs, eyelashes, eyebrows, hair, and oral surface fragments. On the calibration map, mark the actual lip seams and the boundaries of the upper and lower eyelids. The open eye patches cannot be treated as spherical eyeballs.

```sh
"$PY" scripts/scan-guide.py "$SCAN" --out "$GUIDE"
cp "$GUIDE/guidance.template.json" "$GUIDE/guidance.json"
```

After editing the new guide's tags based on the scan calibration skill, refining the camera settings, and saving the lip lines, proceed with the following:

```sh
"$PY" scripts/scan-guide.py "$SCAN" --check "$GUIDE/guidance.json"
```

Execute the command `--check` to visually inspect the marker placement. This command checks the hash and parses the annotations, but it does not represent the actual body measurement, nor can it replace the inspection of the marker overlay diagram and the separation diagram of parts.

**Deliverables:** A `guidance.json` that matches the source hash, camera data, a diagram of the markers, as well as the lips and eye markers required for the original head construction. The file SHA-256 and the `source_hash` generated by the tool are kept separate and not mixed together.

**Completion check.** Verify that each landmark lies on its anatomical feature. Distinguish visible skin, the outer garment surface and occluded anatomy. Correct the calibration if the view, side or source hash disagrees.

## 2. Register measurement references for occluded anatomy

Follow [Body measurement](../charloom-body-measure/SKILL.md). Measure exposed skin on the original scan. For a bare body hidden by clothing or armour this route requires a registered bodysuit reference. Reuse a valid reference, or generate and register one with the same character, pose and camera. Image-service dependencies apply when generating or editing a reference. See [Image service preparation](../charloom/references/setup.md#image-service-requirements).

Follow [Generate, register and mark measurement references](../charloom-body-measure/references/measurement-plates.md) to prepare front and side references. For the torso profile, place the palm vertically against the outer thigh, fingertips down, with the lower abdomen’s front edge fully visible. If occluded, generate a separate view in that pose with the same camera. After registration, update only visible torso observations.

Check the correspondence between the head, hands, feet, orientation, anatomical height, and contours for each view. Read the acceptance status, scale, displacement, anchor points, and coverage area of `views/bodysuit.json`; also examine the actual images. Do not approve just because `accepted` is considered true. When the local proportions of hands, feet, or the body are changed by the generation diagram, correct the reference data for that area or exclude unreliable data.

```sh
"$PY" scripts/mh-bodysuit.py "$GUIDE" --edit
"$PY" scripts/mh-suit-contours.py "$GUIDE"
```

First, read the original calibration map and generate a reference for the tight-fitting garment's size; check the generated reference, actual left/right reference, and dense endpoints before proceeding to the second step. If there is an effective cache with the same source and camera, reuse the cache to avoid duplicate submission of the paid task.

Extract the contours from the checked references, save the selected `suit_contours.json` or ensemble files, and generate receipts. The automatically sampled contours, line width, and scale are only the input for the mass unit. The next step is to establish the target values and measurement paths for each part.

**Completion check.** The required front and side views are registered and cover the body regions being estimated. Mark occluded anatomy inferred from generated references as estimates. Resolve missing or mismatched references before fitting the bare body.

## 3. Measure and save target dimensions before fitting

[Measurement points, paths, and export measurement methods](../charloom-body-measure/references/surface-measurements.md). The targets for this stage are derived from the original scan and the aligned reference. They must be saved before the first body fit, and cannot be inferred from the fit results.

Please fill in the input according to the format [Body script input](../charloom-body-measure/references/script-inputs.md). Run the "coordinate" and "densities" commands first:

```sh
"$PY" skills/charloom-body-measure/scripts/working_frame.py "$SCAN" --prior "$PRIOR" --guide "$GUIDE" --out "$MEASURE/frame.json"
"$PY" skills/charloom-body-measure/scripts/outline_points.py "$GUIDE" --frame "$MEASURE/frame.json" --out "$MEASURE/outline"
```

Then, using the visibly skinned face confirmed by the segmentation, the aligned human mask, and the dual-view anatomical center, create the `tape-spec.json`:

```sh
"$PY" skills/charloom-body-measure/scripts/surface_tape.py "$MEASURE/tape-spec.json" --out "$MEASURE/tape-reference"
```

To obtain the skin segmenting range, perform step 4 first, as it does not depend on body fitting. The path specifications should cover the bilateral outer arms, outer legs, and the six paths inside the inner side of the armpit. Check the segment length and the actual overlay. The cross-section targets use the image projection of the same local three-dimensional axes, defined to remain in `sections-spec.json`. First, output the target values and the endpoint map, and then compare with the actual grid cross-section in step 5.

```sh
"$PY" skills/charloom-body-measure/scripts/mesh_sections.py "$MEASURE/sections-spec.json" --out "$MEASURE/sections-reference"
```

### Fixed dimensions and sources of evidence

Record scan units, camera pixel scale, the source-to-working transform and the scale reference. The current `Scan.working_frame(prior, front_view)` scales the scan bounding box to the template body height. Compute this before transfer and convert measurements again if the template or scale changes. Check whether headwear or shoe soles affect the bounding box. Without a known real height, report millimetres in working-model space.

Each target item includes a name, left/right side, pose, measurement point or path, value, unit, reference image/three-dimensional surface, and error range. Evidence is categorized into: actual three-dimensional skin measurement from original scans, actual measurement of outer clothing surfaces, two-dimensional projection measurements from reference images, and hidden body estimation based on registration views. The outer shell of armor should not be considered as bare skin; individual views cannot claim to have a complete three-dimensional surface path.

### Required components

| Parts | Targets to be recorded |
| --- | --- |
| Left and right arms | Surface path and segment length from shoulder tip to outer elbow to outer wrist; length of the inner arm along the inner side of the armpit's lower edge to the inner side of the elbow, measured separately on both sides, without using the length of the shoulder and elbow bones as a substitute; the chain length of the shoulder-elbow-wrist joint is listed separately; length from the wrist to the visible fingertips, vertical distance from the shoulder to the fingertips, width and direction of the palm. |
| Left and right legs and feet | Surface paths and segmented lengths for the hip, knee, and ankle; additional list of hip-knee-ankle joint chain lengths; height from hip to the sole of the foot, foot length, foot width, and orientation. |
| Trunk and Neck | Length from shoulder to hip, shoulder width, the position of the cross-section of the chest, lower ribs, abdomen, waist, hips, and neck root, center, width, and depth; the gap between the left and right arms at the same height. |
| Local section of the limbs | Cross-sectional positions, centers, width, and depth of the left and right arms, elbows, forearms, wrists, thighs, knees, calves, and ankles. The cross-section is perpendicular to the axis of the limb and should not be confused with the horizontal width of the image. |

Bind reproducible 3D paths or sections to exposed scan skin. For covered anatomy, estimate from the front and side references, recording correspondences and depth assumptions. Distinguish surface arc length, joint-chain length, straight-line distance and vertical drop. Mark insufficient evidence as unmeasured. To cover a missing item, obtain another view or record an estimate with its evidence.

**Deliverables:** Measurement sheets, machine-readable point/path/slice data, annotation diagrams, coordinate conversion instructions. Run [Body script input](../charloom-body-measure/references/script-inputs.md), `working_frame.py`, `outline_points.py`, `surface_tape.py`, `mesh_sections.py`; retain the original dense contours without compressing them into a few joints. The values should be able to be recalculated from the original evidence table; each required part has measured or clearly annotated estimates and error ranges.

**Conditions for entry into the fitting process:** The above data have been checked item by item. The target corresponds to the original image, and the units are consistent. Any missing items have been resolved. The generation of reference images, extraction of contours, or marking of joints does not constitute completion of this step. If the task has been delegated, the agent must first submit these files and actual values. The main agent will review the evidence before approving the fitting, and oral progress cannot replace the necessary checks.

## 4. Review part ownership on the original scan

Before exporting the body and clothing, view the original scan components and small fragments using the [Clothing sub-pieces](../charloom-garment-separate/SKILL.md) isolation mode. Bind the faces of the face, neck, original hand fragments, hair, eyebrows, internal fragments, clothing, and individual equipment to the `source_hash`. Planar or bonded parts are subdivided according to the actual material and geometric boundaries.

```sh
"$PY" skills/charloom-garment-separate/scripts/review_parts.py "$SCAN" --guide "$GUIDE" --out "$PARTS"
```

Please check the side-by-side comparison in the `components.png` and component directory, and write the review decisions in a separate `decisions.json` document. Empty actions in the template cannot be directly submitted. For recompilation commands and mixed component details, refer to the component-specific skills.

**Deliverables:** Subdivided evidence, checked `decisions` and corresponding original triangular area ranges. The decisions will be passed to transfer using `--segment-evidence` and `--segment-review`. The command requires a review and rerunning when making a conditional judgment. All unmasked pieces should not be automatically placed in the clothing layer.

## 5. Fit the body and remeasure against the targets from step 3

Use [Fit the MakeHuman body](../charloom-body-fit/SKILL.md) with the scan, template, guide, selected contours, target measurements and segmentation decisions. Verify which rig drives the motion as in step 0. If the original build retains the source walk rig, use [Drive motion with the source rig](../charloom/references/walk-replay.md#drive-motion-with-the-source-rig) to establish the fixed rig once. Omit `--motion` for a static transfer. Remeasure the body and assemble the original head and garments, then attach both layers to the fixed rig with `attach_fixed_rig.py` and remeasure. The transfer command below applies when the MakeHuman template rig drives the motion.

```sh
"$PY" scripts/scan-transfer.py "$SCAN" --prior "$PRIOR" --guidance "$GUIDE/guidance.json" --out "$TRANSFER" --motion "$MOTION" --mouth-trace "$GUIDE/mouth-trace.json" --segment-evidence "$PARTS/components.json" --segment-review "$PARTS/decisions.json"
```

If side-view thickness collapses after landmark deformation, inspect the deformation field using [Landmark deformation and cross-section refitting](../charloom-body-fit/references/section-refit.md). When the pose has distinct frontal landmarks, select the mode that preserves depth.

After the first export, create hash bindings for the actual bare model according to the measurement skill of `body-model.json` and `sections-model.json`. Select the real body primitive and body parts, and then execute:

```sh
"$PY" skills/charloom-body-measure/scripts/outline_points.py "$GUIDE" --frame "$MEASURE/frame.json" --model-spec "$MEASURE/body-model.json" --out "$MEASURE/outline-export"
"$PY" skills/charloom-body-measure/scripts/surface_tape.py "$MEASURE/tape-spec.json" --model-spec "$MEASURE/body-model.json" --out "$MEASURE/tape-export"
"$PY" skills/charloom-body-measure/scripts/mesh_sections.py "$MEASURE/sections-spec.json" --model-spec "$MEASURE/sections-model.json" --out "$MEASURE/sections-export"
```

`body-model.json.regions` should cover both dense contour groups and six path names. When shape correction is needed, follow[Complete body and local cross-section shaping](../charloom-body-fit/references/section-refit.md) to`refit_sections.py`. Use the complete template with the same topology to restore the chest and adjust the local limbs. Then, use`copy_body.py` to bring the body into the current round of bare model and clothing documents. Use a new directory for each round and recalculate the corresponding document summaries.

Understanding the actual capabilities of the current program:`fit_template` Performing spatial deformation on templates and skeletons using anatomical markers;`fit_body` Constraining the body with the contours of the registration view, and maintaining the initial shape through smoothing terms. They do not automatically read the complete measurement table from step 3, nor do they automatically prevent body part loss. Shapes that are not fully constrained are still affected by templates and regularization terms. The first export must be trimmed and re-measured according to body part targets, and cannot be considered completed.

1. Check the source hash, template, working coordinates, `stage2_body_fit.contours` and `registered_views` in `report.json`. Confirm this run used the intended inputs. The implementation prefers an existing ensemble file, so verify that an older contour set did not take precedence.
2. Rebind the corresponding measurement points and surface paths on the actual `3-skin/body.glb`, and perform the re-measurement according to the same definition as in step 3. Use `mh-check-lengths.py` to check the bilateral length it supports; the cross-section and hip gap are calculated using the corresponding method. Record the target value, actual value, signed error, and the pre-selected tolerance.
3. Overlay the front and side profiles according to the original calibration camera, and examine the diagonal sides. Check the chest and abdominal thickness, continuity of the shoulder and armpit area, neck base, wrist, fingertips, and sole for each part. The median error of the overall body contour should not be used to replace individual dimensions or surface quality.
4. Use a textureless material to check the geometric concave folds, and use the original material to check skin color and texture. If the armor color is incorrect or off by a significant margin in terms of naked body proportions, neck and shoulder fractures, or hand and foot ratios, all need to be corrected to the corresponding input or shape; re-measure the affected items after each correction. Do not let errors be allowed to accumulate to cover up deviations.
5. Play the entire walking and shoulder, elbow, knee-related movements to check the surface length, cross-section, and underarm deformation. Consistent bone lengths do not mean consistent skin sizes.

**Completion Criteria:** The naked target dimensions must be within an explainable error range. Local contours and geometric checks must be passed. There should be no unresolved deformations in the movement. The body must have been completed before subsequent assembly can be considered a successful delivery, but it cannot be recorded as completed.

**Condition substitution:** Only when the original scan indeed has separable trunk and limbs, with five skin shells, and the task is applicable, will the process proceed to [Scanning Shell Assembly](../charloom-scan-volume/SKILL.md). First, check the shell evidence; this branch still requires measurement and export for verification. The selection of the body's layout does not change the requirement to retain the original scan head.

## 6. Attach the original head and bind the facial features

Connect the original head with the tested and approved body, and then bind the original scanning surface using [Mandible and Neck](../charloom-body-fit/references/neck-and-jaw.md) and [Face and Speech](../charloom-face-dialogue/SKILL.md)/[Build commands](../charloom-face-dialogue/references/face-build.md).

By default use `bind_scan_head.py` or `api.build_character(..., head_binding="scan-surface")`. Check `route=scan_surface_binding`, the source hash and surface correspondence error against the scan. Retain the neutral visible face, UVs and texture. The split lip seam, eyelids and oral lining should follow their actual boundaries. Select the `anatomical` route to reconstruct eyeballs or an anatomical face only when the task requires it.

This round of transfer retains the original source materials required for the transfer `mouth-trace.json`. Set the directory and execute:

```sh
export CHARLOOM_TRANSFER_DIR="$(dirname "$TRANSFER")"
export CHARLOOM_HEAD_DIR="$BUILD_PARENT"
export CHARLOOM_CV_PYTHON="$PY"
NAME="$(basename "$TRANSFER")"
"$PY" scripts/roster/bind_scan_head.py "$NAME" --config "$CONFIG"
bun scripts/roster/embed-clips.ts "$NAME"
bun scripts/roster/verify-face.ts "$NAME"
```

`CONFIG` Specify the original facial, eye, and retained face regions based on facial skills, with `NAME` as the key; this is omitted when there is no additional segmentation configuration.`--config` is omitted. The output consists of `$BUILD_PARENT/$NAME/3-skin/body.glb` and `4-garments/character.glb`, and also generates a report of facial movements corresponding to the original surface.

When facial movements are required, incorporate and verify them. Check in both bare model and dressed versions: neutral face, complete blinking, eye gaze changes, M/B/P closed lips, F/V contact, maximum opening of the mouth, and head rotation. Record the duration of movements, close-up/side views, and geometric results. Check for closed lips with any signs of lip penetration, eyelid folds, exposed intraoral areas, and neck seams. Do not rely on individual numerical values to replace the visual inspection.

**Completion Conditions:** The correspondence between the actual execution entrance and the original head has been verified; the required facial features and the connection between the head and neck have also been checked.

## 7. Fit clothing and equipment and repair textures

Follow [Clothing and equipment adaptation](../charloom-garment-fit/SKILL.md) and adjust sizing and binding against the final body. For guards, measure proximal, middle and opening sections separately. Use the appropriate attachment for rigid pieces and the intended motion for cloth and free panels. Check cuffs, gloves, hair, collars and props for stray triangles and wrong bone assignments.

The names, hierarchy, `rig`, and `fabric` of the clothing's sub-parts are written in the `garments` section of the guide. The first round of reading is performed by transfer; the exact fields and re-binding points can be found in [Clothing execution methods](../charloom-garment-fit/references/build-and-check.md). When the body is later modified, local adjustments are made to the final assembly. The new output is saved, and no re-fitting is required for the initial fit that covers the already verified body.

View the same frame in the bare model, pure clothing, and dressed versions. Ensure that there are no remnants of old fingers, nails, or skin fragments on the clothing layers. Record the worst-interrupted frames for static, complete gaits, arm lifts, and bending postures, and correct them. The complete bare model must retain the underlying body structure; the cut parts from the dressed file cannot be used as a bare model.

Changing the appearance is only performed when required by the task; in cases of texture dislocation, black surfaces, or missing textures, it is handled according to [Clothing texture restoration](../charloom-garment-texture/SKILL.md), with priority given to retaining the visible original scanned surface and UVs. The range for retilement and service receipts must be traceable.

**Completion Requirements:** Segments, correct wearing size and movement; changes in clothing caused by body adjustments have been reviewed. Any unresolved issues with interlacing or texture remain clearly listed.

## 8. Export the editable scene and final motion files

Import the checked build according to [Blender construction and actions](../charloom/SKILL.md), save the editable scene, and export the GLB required for the task. If the integration build generates a new body or head, verify the actual inputs and outputs used. Re-run the verification for the reconstructed parts. Do not default to retaining manual corrections.

When reproducing a walk, run [Walkway comparison](../charloom/references/walk-replay.md#check-the-tracks-actually-written-to-the-glb) on both bare and dressed exports. Account for differences from the original build in each track and in the rest skeleton. Check palm orientation and sole height in the original pose. For floating or penetrating soles, follow [Walking grounding](../charloom/references/walking-ground.md) on the actual dressed motion and write the same root translation to both layers. Use the corrected build directory in the commands below.

Completed GLB direct import, avoiding re-fitting:

```sh
"$BLENDER" -b --factory-startup --python-exit-code 1 --python scripts/blender_character.py -- "$BUILD_PARENT/$NAME/4-garments/character.glb" "$BLENDER_OUT" --clip walking --export-glb
"$PY" scripts/animate.py validate "$BUILD_PARENT/$NAME/3-skin/body.glb"
"$PY" scripts/animate.py validate "$BUILD_PARENT/$NAME/4-garments/character.glb"
"$PY" scripts/animate.py validate "$BLENDER_OUT/character.glb"
```

Perform `scripts/animate.py validate` for the final bare model, dressed version, and additional Blender exports. Verify the action names, duration, skeleton, skinning, facial morph, and materials. Save the source files, build configuration, report, and final file hash.

**Completion Conditions:** The final file must be consistent with the version being tested; the editable scenes must be open, and the required actions must actually be playable. The successful exit of the command only proves that the execution is complete, and does not replace the verification of the entity, actions, and visual checks.

## 9. Verify the delivery in the Gallery

First, preview the registration in[Export and Gallery Verification](../charloom-gallery-review/SKILL.md) mode, and then synchronize it to the current local Gallery. Open the actual character URL and check if the requested GLB is the file that was just verified.

```sh
python3 scripts/mh-ide-sync.py "$BUILD_PARENT" --dry-run
python3 scripts/mh-ide-sync.py "$BUILD_PARENT"
(cd viewer && bun run dev)
```

Use the actual entries and URLs from the synchronized output. Reuse the running viewer if necessary. If the GLB file is modified in Blender, register this file in the entries selected for this round and verify the source. Do not allow the Gallery to load early builds.

During the paused non-zero action time, keep the camera and actions unchanged. Switch between bare model, pure clothing, and dressed versions. Verify the time and maintain the integrity of the body beneath the clothing. Continue to check the cutaways during playback. Perform actual playback of walking, facial expressions, and actions required for the task, reviewing from the front, side, back, and close-up of the problem areas. Retain the URL, file hash, action duration, and screenshots.

**Delivery Records:** List the target quantities and retesting results, actual body/original head route, bare model and dressed GLB, Blender scene, actual loading of the gallery, as well as any remaining issues. Internal checks, runtime verification, and user visual approval are each recorded; when the user has not yet reviewed, it is not marked as approved by the user.

## Interruption, revisions, and verification after delegation

Proceed to the last stage with evidence. Modifying the scan will render the hash binding data invalid; modifying the camera or scale requires recalculating the alignment and body proportions; changing the body topology requires re-binding and remeasuring the path; modifying the body shape requires reprocessing the affected clothing and action checks. Test cases, temporary directories, failure logs, and role-specific parameters are stored in the task artifacts and are not included in the general skill.

When using a subagent, the main agent must verify the following: the original input and hash, measurement evidence before fitting, build entry and configuration, and the final load file. Phase reports should include verifiable documents and actual results. If it is found that dependencies were skipped, input was replaced, or different head routes were used, the affected steps must be corrected and redone before proceeding downstream.
