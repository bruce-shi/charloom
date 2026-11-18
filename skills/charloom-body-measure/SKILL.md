---
name: charloom-body-measure
description: "Measure a scanned character’s proportions, surface paths on both arms and legs, and limb cross-sections. Use a registered bodysuit reference for occluded anatomy."
---

# Measure in the character’s calibrated coordinate frame

English · [简体中文](SKILL.zh-CN.md)

This step must be completed before body fitting. Save the per-part targets, coordinate scale, evidence, and error range by pressing [Step 3 of the main process](../charloom-mh/SKILL.md#3-measure-and-save-target-dimensions-before-fitting); generating reference maps and extracting contours are only preparatory steps and cannot be counted as completed body work.

Follow [Generate, register and mark measurement references](references/measurement-plates.md) to prepare front and side references. For a torso profile, place the palm vertically against the outer thigh with fingertips down and the front edge of the lower abdomen visible. If the abdomen is occluded, generate a separate view in that pose with the same camera. After registration, update only visible torso observations.

1. Measure exposed skin directly on the original scan. For areas covered by clothing or armour, use a bodysuit reference showing the same character, pose and camera, and identify the regions estimated from it. Check for registered images and their receipts first.
2. For the armor-covering process, the effective registration reference must be entered; reusable verified references can be used, and they must be generated if missing. Before generation, check ImageMagick, image service credentials, and execution permissions using [Preparation for the environment](../charloom/references/setup.md#image-service-requirements). Point `PY` to the Python script that depends on the installed project from the repository root directory, and point `GUIDE` to the current calibration directory. Run `"$PY" scripts/mh-bodysuit.py "$GUIDE" --edit --back`. Compare the input to check the face, hands, feet, side orientation, and height at the same anatomical position. Verify the `accepted`, scale, registration anchors, and measurement coverage area in the `views/bodysuit.json`.
3. For one registered reference set, extract contours using `"$PY" scripts/mh-suit-contours.py "$GUIDE"`. Use `"$PY" scripts/mh-suit-ensemble.py "$GUIDE" --edit --runs 3` only when repeated observations are needed to resolve contour ambiguity. Compare the samples and inspect `views/suit_contours.ensemble.json`. Keep the selected contour file and its receipt.
4. Measure the three-dimensional surface paths along the shoulder tips, outer elbows, outer wrists, and lateral hips, outer knees, and outer ankles on each side. Also record the joint chain length, vertical drop, and the length from the wrist to the visible fingertips. The girth or thickness is recorded along the local limb axis in vertical cross-sections, with the position, center, width, and depth being noted. Static and motion frames are recorded separately.
5. Review the exported skin using the same measurement path. Use [Measurement points, paths, and export validation commands](references/surface-measurements.md) when needing to reproduce the length or hip-to-hip gap errors.

Distinguish between visible skin's 3D measurements, clothing shell measurements, 2D projections, and hidden naked estimates. Unobservable depths and paths should not be mistaken for measured values; retain the basis for estimates and error margins. Items with insufficient evidence should be filled in. When there is no basis in real-world dimensions, record the millimeters of the working model and its conversion sources. Do not use the fitted body to infer the target before the fitting.

## Commands

First, calculate the working coordinate using [Script input and step-by-step commands](references/script-inputs.md), and then output the dense contour, bilateral outer arm/outer leg paths, and the bilateral armpit inner side path. The `upperarm_inner` in `surface_tape.py` must be marked with separate `axilla_l/r` and `elbow_inner_l/r` surfaces; measure from the lower edge of the armpit to the inner side of the elbow, and do not use the length from the shoulder joint to the elbow joint as the total length. Finally, perform a real-tripolygon section and export the binding verification.
