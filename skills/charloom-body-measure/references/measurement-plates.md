# Generate a digital model from the original scan.

English · [简体中文](measurement-plates.zh-CN.md)

Run in the root directory of the repository. Let `PY` be the Python of the project's CV environment, `SCAN` be the original scan GLB, `GUIDE` be the calibration directory of this scan, and `MEASURE` be the volume product directory. The source image uses the orthographic view rendered by the calibration camera. Preserve the pixel coordinates of each image, `views.json` and the source hash, to facilitate the projection of endpoints back to the same working coordinate system.

## 1. Prepare calibrated base images

Follow [Scanning Calibration](../../charloom-scan-guide/SKILL.md) and render the original scan with `scan-guide.py "$SCAN" --out "$GUIDE"`. Mark the front direction and joints, then run `scan-guide.py "$SCAN" --check "$GUIDE/guidance.json"`. Read `front_view` and choose the calibrated side camera with a yaw 90° beyond the front camera. Check the entire head, feet and ground line. For an existing matching guide, resume at these checks.

## 2. Generate a same-pose bodysuit reference

First, prepare ImageMagick, genmedia, and fal as per [Image service requirements](../../charloom/references/setup.md#image-service-requirements). The following commands maintain the original arm pose, used for joint, arm length, and camera calibration.

```sh
"$PY" scripts/mh-bodysuit.py "$GUIDE" --edit --back
"$PY" scripts/mh-suit-contours.py "$GUIDE"
```

The first command splices the frontal, side, and dorsal views of the same character horizontally, and uses fal's GPT Image 2.5 Sunburst edit to generate a PNG image with dimensions 2560 × 1136, with medium quality. When only the frontal and side views are used, the dimensions become 2048 × 1360, omitting the `--back`. The actual prompt can be found in [The SHEET and PROMPT of the script](../../../scripts/mh-bodysuit.py). The same character and camera are used, with the armor replaced with a green bodysuit. If the original shoulder, elbow, and wrist positions are allowed, the basic side view shows the palms lying vertically on the outside of the thighs, with the fingertips pointing downward. If the entire arm needs to be moved to expose the abdominal edge, the same pose input is retained, and the next step will generate a lateral view of the torso. The `views/bodysuit_sheet-2k.png`, native outputs of each view, and `bodysuit_<view>-reg.png` are preserved. The receipt is recorded in the `views/bodysuit.json` section of the `sheet`, and the cache is located in the `views/measurement-cache/`. When the `--edit` command is executed with the same parameters, the cache is reused or the original request is restored.

Compare the calibration map with the registration map. Check the face, wrist, foot, orientation, overall scale, and local contours; examine `checks.scale`, `anchors`, `side_feet_px`, and `accepted`. The second command generates the full-body contour from the registration map. Observe which part each endpoint actually falls on in each row. If the fingers, wrist, or forearm cover the abdomen, retain the same pose map for arm measurement, and remove the covered trunk endpoints from the fitted input.

## 3. Prepare a torso profile with the palm against the thigh

A side view of a full-body character is used. The lower palm is placed vertically on the outside of the thigh, with the wrist at the top and the fingertips pointing downward. In the side projection, the fingers should be within the contour of the thigh, and the front edge of the abdomen should be continuously visible. The torso, pelvis, head, legs, feet, shooting direction, proportions, and ground lines should be consistent with those in the aligned side view. When the same pose image meets the requirements, it can be used directly to measure the torso.

```sh
PROFILE="$MEASURE/profile-hand-at-thigh"
"$PY" skills/charloom-body-measure/scripts/torso_profile.py "$GUIDE" --out "$PROFILE"
"$PY" skills/charloom-body-measure/scripts/torso_profile.py "$GUIDE" --out "$PROFILE" --generate
```

The first command writes only `plan.json`. Check the input images, source hash, prompt and parameters there. The second submits or resumes the same cached request through genmedia. The native medium-quality PNG `profile-native.png` is 1536 × 2048. The image `profile-candidate.png` uses the base camera’s 768 × 1024 canvas. File `generation.json` records the source image hash, generation parameters and service receipt. Check the whole character and finger positions before registering the image.

## 4. Register the additional profile to the same camera

In the candidate map and base registration side map, select at least two points with the same name that are sufficiently far apart for proportional scaling and translation. Also select at least two different positions for independent verification. Prioritize including the still recognizable posterior, knee, or gluteal margins to avoid only local inspections. Record the arrays for `anchors.json`, `base_sha256` and `candidate_sha256` from the two maps, the acceptable `tolerance_px`, `fit`, and `check`. Specify the specific anatomical point names as `name` and two sets of actual pixel values as `candidate_px: [x,y]` and `base_px: [x,y]`. Determine the tolerance based on the visibility of the measuring points and the conversion between millimeters/pixels for the calibration camera. Palms, forearms, and the generated abdominal margin should not be used as registration anchors.

```sh
"$PY" skills/charloom-body-measure/scripts/register_profile.py "$PROFILE" "$PROFILE/anchors.json" --out "$PROFILE/registration"
```

Inspect the residual for every point in `registration.json`. Enlarge `overlay.png` and compare the head, back, buttocks, legs and feet. Once the anchors agree, inspect the abdominal contour row by row for discontinuities or a hand or cuff included by mistake. Exclude localized deformation separately. Keep both the original image and the overlay.

## 5. Render the measurement overlays

Follow [Body script input](script-inputs.md) to generate `frame.json`. Run `outline_points.py` to extract dense point pairs from the base image and write `points-<view>.png` for each view. For the abdomen in profile, return to the same calibrated row in the reviewed `registration/registered.png`. Mark the actual front and back silhouette pixels. Record the site, row, endpoints, source image hash and edge uncertainty. Retain the old endpoint on the hand alongside its correction. Mark hidden outlines as bodysuit-reference estimates. Use the reviewed torso-view mask for section depth and the original same-pose view for arm paths. An image with a changed arm pose cannot replace the full-body fit input. Record boundaries still hidden as unmeasured.

Mark the row endpoints with [profile_rows.py](../scripts/profile_rows.py). Assign a pixel to each measurement row inside the torso, and save it with `stations.json`. Write `registered_sha256`, `anterior` (fill `left` for the front edge of the body on the left, and `right` for the right side) and `stations` arrays in the top layer. For each item, write the part `name`, the full pixel coordinates `inside_px: [x,y]`, the boolean value `anterior_visible` and `posterior_visible`, as well as the basis for interpretation and uncertainty range. During initial review, you can fill both ends with "false" to first look at the diagram, and then confirm row by row. When selecting endpoints, distinguish between the outer edge of the forearm, the outer edge of the palm, and the outer edge of the torso.

```sh
"$PY" skills/charloom-body-measure/scripts/profile_rows.py "$GUIDE" "$PROFILE/registration" "$PROFILE/stations.json" --frame "$MEASURE/frame.json" --out "$PROFILE/rows"
```

`profile-points.png` Use blue dots to mark visible areas, and red dots to mark unvisible areas. Lines that are visible on both ends are connected by yellow lines, and the output projection thickness is indicated.`profile-rows.json` Preserve the intersection points of the original and supplementary diagrams, as well as visibility and working scale. Output to the new directory during the overlay revision process, and keep the initial inspection records.

## The arm in the original posture and the distal leg

The vertical palm side illustration is specifically for the torso. If the sleeves and chest abdomen are connected in the same pose illustration, or if the legs overlap on the side, you can use [limb_views.py](../scripts/limb_views.py) to combine the front and side views into one image. First, confirm that both opposite side cameras have available PNG images from the complete calibration catalog. Reuse the already aligned front and side close-fitting images, and use the opposite side original scan view for the far side. Green sleeves and gloves are distinguished from the blue torso, legs, and socks, making it easier to identify the edges of different parts of the same person's pose.

```sh
LIMBS="$MEASURE/limb-views"
"$PY" skills/charloom-body-measure/scripts/limb_views.py "$GUIDE" --out "$LIMBS" --generate
```

Retain the markings like `sheet-input.png`, 2560 × 1136 as `sheet-native.png`, as well as the parameter receipts. Each `body_yNNN/` sub-directory contains candidate images and sources of the same size as the corresponding camera. Fill in the anchor points for this view, and pass them to `register_profile.py` as per step 4. Isolate the aligned sleeves as a dedicated mask for the arms. Confirm the correspondence between the sleeves and the glove edges for the corresponding limbs. Local axes, postures, and ipsilateral joints can be compared. The legs and feet are also identified as left or right according to the actual camera's proximal anatomy. The generated set of images can only be used for hiding the estimated human body; any unverified local directions should be left blank.

After replacing shoes and boots with thin socks, the toe tips and feet often do not correspond individually. If two images do share the same horizontal ground, an anchor file can include arrays of `fit_y_lines` or `check_y_lines`. Each item should include `name`, `candidate_y`, and `base_y`. Read the row coordinates from the actual grounding points of the two images; the ground constraint only participates in the calculation of the vertical coordinate. Use the markers that are not involved in the fit to separately check the horizontal direction, orientation, and local contours.
