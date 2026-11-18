# Runnable Body Commands and Inputs

English · [简体中文](script-inputs.zh-CN.md)

Run from the root directory of the repository. `PY` is the Python environment for the CV setup. All JSON paths are resolved relative to the directory where the JSON file is located; absolute paths are recommended. The millimeters are in the working model scale. All new output directories must not exist.

## 1. Establish coordinates and dense contours before fitting

```sh
"$PY" skills/charloom-body-measure/scripts/working_frame.py "$SCAN" --prior "$PRIOR" --guide "$GUIDE" --out "$MEASURE/frame.json"
"$PY" skills/charloom-body-measure/scripts/outline_points.py "$GUIDE" --frame "$MEASURE/frame.json" --out "$MEASURE/outline"
```

The first item only reads the template height and calculates the `Scan.working_frame`, without fitting to a human body. `frame.json.source_to_working` represents **4×4 row priority**; the flat matrix in the transfer report is column priority, and `outline_points.py` can also directly read this report. Do not mix the two representations directly.

The second reading of the calibrated `views/bodysuit.json.marks`. The endpoints named `outline_trunkNN_w1/w2`, `outline_arm_lNN_w1/w2`, and `outline_leg_rNN_d1/d2` are paired together; `w` represents the frontal width projection, and `d` represents the lateral depth projection. The actual names are based on the original records. `mh-bodysuit.py` takes stations along the contour at a spacing of approximately 10 pixels; it does not simply mark a single point for each part. The total number of points is determined by the image size and visibility range. Retain the entire batch of dense points. Review `points-VIEW.png`, remove any handles, accessories, or background when the ends of the torso are identified, and then recalculate. The output `outline-targets.json` contains information such as the pair of pixels, the calibrated projection millimeters, and the source hash; it does not claim the width of the oblique projection as the true cross-sectional diameter.

Use `--model-spec MODEL.json` to project the final skin onto the same camera and compare each pair. The `regions` in MODEL must cover all `group` in the output, such as `trunk`, `arm_l`, `arm_r`, `leg_l`, `leg_r`; do not let the torso cover the arm measurements.

## 2. Record meshes and face selections

Each `mesh` / `source_mesh` represents an object:

- `file`, `sha256`: Actual GLB and file summary.
- `node`: Node names in GLB, not the Blender display names or mesh indices.
- `primitive`: The serial number of the triangular primitive for nodes with the same name, default is 0.
- `world_from_model`: Optional 4×4 row-padding transformation. The original scan should fill in the matrix for `frame.json`; the export in work coordinates is usually omitted.
- `triangle_sha256` is an optional digest of the original triangles encoded as `<i8` bytes. It prevents reuse of a selection after topology changes.
- `clip`, `time`: Optional actions and seconds; if omitted, it means "rest". The target and export must be in the same pose.

Each value in `regions` can only be chosen once: `{"triangle_ranges":[[first_face,end_face]]}` (left-closed and right-open), or `{"joints":["bone_name"],"minimum_mean_weight":threshold}`, or the confirmed `{"all_faces":true}`. The local arm section should be isolated from the arm surface; the chest should not be included in the arm when the whole body is cut.

The model verification records also include `tolerance_mm` and `landmark_bindings`. Each central point is bound as `{"triangle":face_id,"barycentric":[a,b,c]}`, or `{"vertices":[vertex_id...],"coefficients":[coefficient...]}`. The former is used for surface points, with non-negative coefficients and a sum of 1; the latter is used for anatomical centers buried within the skin, with coefficients that sum to 1 and can be negative. The bindings are derived from reviewed corresponding positions, and the center of the old skeleton should not be directly used as the measurement for the new skin. `mh-check-lengths.py` uses another point table format, as mentioned in the script header, and does not directly accept this record.

## 3. Measure surface paths on both sides

```sh
"$PY" skills/charloom-body-measure/scripts/surface_tape.py "$MEASURE/tape-spec.json" --out "$MEASURE/tape-reference"
"$PY" skills/charloom-body-measure/scripts/surface_tape.py "$MEASURE/tape-spec.json" --model-spec "$MEASURE/body-model.json" --out "$MEASURE/tape-export"
```

`tape-spec.json` Field:

- Complete character settings `scope:"full_body"`, the script forces double-sided outer arms, outer legs, and inner arms under the armpits; local repairs can be set `selected_paths`, and the report still lists all missing parts of the body.

- `guide`, `source_hash`, `source_to_working`: This guide's table of contents, tool source summary, 4×4 matrix.
- `source_mesh`: The actual original scan record mentioned above; it uses the same matrix as `world_from_model`.
- `reviewed_skin_triangle_ranges`: The visible area of the original scanned skin, based on individual review; armor, gloves, and trouser legs should not be included.
- `paths`: Four definitions for outer arms and outer legs, plus two definitions for inner arms under the 6th section. Each item includes `name` (such as `arm_l`), `kind` (`arm` or `leg`), `side` (`l`/`r`), `joints` (for arms: `shoulder_l,elbow_l,wrist_l`, for legs: `hip_l,knee_l,ankle_l`, same for the right side), `view`, `reference_mask`, `max_radius_mm`, and optionally `step_mm` (default value: 1). The search radius must be sufficient to hit the corresponding limb without crossing adjacent parts of the body.
- `reference_mask`: `file`, `sha256`, `kind:"binary"`. The white areas represent the reviewed human regions, and their size must match the calibrated camera. It is also supported to use `kind:"figure"` and `background_rgb`, `threshold`, but only for reference images that have been checked and show effective background color separation. Do not automatically consider the entire photo as a nude image when the non-white areas are present.

The centers of each joint are calculated from at least two non-parallel views from the `guidance.json` perspective; centers that are not visible must be marked as occluded and their sources estimated. The script first attempts to find the true ray intersections along the original scan, and only uses those intersections that pass the skin review; occluded areas are constrained by the same three-dimensional ray on the aligned human mask, and are saved as estimates. It is forbidden to falsify the surface by linearly filling the Z-values of lateral joints with a series of positive points.

The default is 45 intervals per segment, giving 91 samples over two segments. These are numerical samples along the path. The path follows the outer shoulder and the rear outer elbow. Use `recipe` to override `bend_start,bend_span,elbow_width,shoulder_width,elbow_back,shoulder_up`. Inspect the overlay after any change and verify the anatomical route. Save the raw hits and lightly smoothed curve with the shoulder/hip, elbow/knee and wrist/ankle anchors fixed. Report the face and barycentric coordinates or reference pixel per point, segment lengths, total surface length and joint-chain length.

For the export, `body-model.json` uses the common `mesh`, `landmark_bindings`, `regions` keyed by the four path names, and the predetermined `tolerance_mm`. Check segment errors as well as the total. Opposite upper- and lower-segment errors can cancel.

## 4. Record section targets and measure the exported mesh

No model save target provided:

```sh
"$PY" skills/charloom-body-measure/scripts/mesh_sections.py "$MEASURE/sections-spec.json" --out "$MEASURE/sections-reference"
```

Start the fitting process after checking the objectives and endpoint plots. Export the validation:

```sh
"$PY" skills/charloom-body-measure/scripts/mesh_sections.py "$MEASURE/sections-spec.json" --model-spec "$MEASURE/sections-model.json" --out "$MEASURE/sections-export"
```

The specifications include the same `guide/source_hash/source_to_working` and `sections` arrays. Each item:

- `name` matches a key in the model’s `regions`. Give two landmark names for `axis`, for example `shoulder_l,elbow_l`.
- `width_hint` Specifies the width axis orientation; `frontal_normal` Default to `[0,0,1]`. The actual width axis is the cross product of the limb axis and the normal vector, perpendicular to both the depth axis and the width axis.
- `fractions` is the position along the section of the model, with the range `[0,1]`. It covers the deltoid muscle, the middle part of the upper arm, the area near the elbow, as well as the forearm, wrist, thigh, knee, calf, and ankle; the density increases at the transition points. Do not consider fixed station counts as a valid coverage.
- `references.width` and `references.depth` contain `view`, `mask`, `max_radius_mm`, and `step_mm` respectively; the mask uses the above records. The script projects along the same three-dimensional cross-sectional axis, and no additional perpendicular line is drawn in the side view.
- Fill in `{"unavailable":"describe the occlusion"}` for directions that are not visible, and `incomplete` for the output. The estimated hidden depth that needs to be fitted is noted separately, and it cannot be calculated using ellipses as observations.
- `loop_seed_mm` Optional, indicates the reviewed internal points within the width/depth plane of the section; default axis. If the axis falls outside the skin, the actual selection points of the section must be considered, and the largest closed loop cannot be automatically selected.

Intersect the actual mesh triangles with the section plane. Accept only a unique closed loop containing the designated seed. Report its 3D coordinates, contributing faces, width, depth, offset from the axis centre, reference boundaries, endpoint errors on both sides and overlays. For an open section, torso contamination, multiple loops or a boundary beyond the image, review the face selection and evidence first.

## 5. Measure the hands, feet and torso and repeat on export

Mark visible fingertips, heels, toes and sole contacts in the guide. Add paired endpoints for palm and foot width and thickness. Record endpoints and the camera for each value. Use step 1 for dense torso width and depth. Match the anatomical heights of chest, waist, shoulder and neck between front and side views before using the profile depths. To measure arm-to-waist clearance, section both at the same height. See [surface measurement definitions](surface-measurements.md). Use `mh-check-lengths.py` on joint chains, fingertips and surface paths. Remeasure sections with step 4. Keep source targets and export measurements in separate files.

## 6. Measure each inner upper arm from the axilla

Outside the four outer arm leg paths, add `upperarm_inner_l` and `upperarm_inner_r`. Use the same `surface_tape.py` command for the outer arms, but define it as `kind:"upperarm_inner"`, `joints:["shoulder_l","elbow_l"]`, and `endpoints:["axilla_l","elbow_inner_l"]` (change to `r` on the right). Also fill in `endpoint_tolerance_mm`, based on the camera and marker resolution.

`axilla` Marks the actual starting point below the lower edge of the armpit, on the inner side of the skin of the upper arm; `elbow_inner` Marks the corresponding end point on the inner side of the elbow. These points are not located at the center of the shoulder bone, at the bottom of the sleeve, or at the cuff. Both points require non-parallel views for verification or skin binding. The invisible armpit must have a clear estimation basis or be marked as not measured. A fixed number cannot be deducted from the shoulder.

The script positions the cross-sections at both ends along the shoulder-elbow axis, finds the intersection points of the skin with the inner side of the upper arm, and retains the entire curve along the body. If the endpoints of the resulting rays do not fall within the tolerance for the marked armpit/elbow inner side, the process fails directly. First, correct the selected surface or point. The exported record requires the addition of these two surface endpoint bindings, corresponding to the `regions`. Output the independent inner edge length, the straight-line distance between the two ends, and the endpoint deviation. This cannot be used together with the shoulder tip-external elbow section. Similar checks should be performed on the outer arm and outer leg, and the final GLB should be rechecked.

## 7. Bind reviewed landmarks to the exported skin

```sh
"$PY" skills/charloom-body-measure/scripts/bind_landmarks.py "$MEASURE/binding-input.json" --out "$MEASURE/body-model.json"
```

At the top level, supply `mesh`, `regions`, `tolerance_mm` and a `points` dictionary. For each point record `xyz_m` (working coordinates located on the selected export), `evidence` (selection image or template correspondence and reasoning), `region` (explicit local face selection) and `kind`.

- Use `kind:"surface"` for observable surface points such as the axilla, inner elbow, fingertip and sole. Set `max_snap_mm`. The script finds the nearest point on the selected faces and records its triangle index and barycentric coordinates. A distance above the limit fails the check.
- `kind:"embedded"` is used for the anatomical center within the skin, with weighted affine binding based on local vertices; default value: `neighbors:90`, `softness_m:0.008`. It is necessary to select the local face of this limb; failure occurs if the neighborhood degenerates and the three-dimensional center cannot be represented. The output includes the original position, the reconstructed position, and the residual.

When using the anatomical center corresponding to the original template, first verify the vertex order and the correspondence with the current body. Then, maintain the binding as it is during the skin re-measurement. It is prohibited to directly copy the target coordinates before fitting them as the measurement points after fitting, and then use affine coefficients to forcibly reconstruct the target. This only proves that the coefficients can derive the target, but does not prove that the body dimensions are correct. Each exported point must have independent positioning evidence on the current skin.

Near the shoulder or hip, a limb naturally joins the torso. When a unique closed loop is unavailable, export remeasurement records `no_unique_closed_section` for that station, retains the others and returns a nonzero exit code. Review the selected faces and seed. If anatomy prevents a separate loop, inspect actual mesh hits, triangles and errors in the positive and negative width and depth directions under `directed_endpoints_only`. List any missed directions. Use valid hits individually for reshaping. Circumference requires a closed loop.
