---
name: charloom-scan-guide
description: "Calibrate cameras, body landmarks, part boundaries and the lip seam on a Charloom scan. Produce a guide bound to the source GLB hash."
---

# Calibrate the original scan

English · [简体中文](SKILL.zh-CN.md)

Start with the original textured GLB. The observed-head builder uses static triangles and a single UV atlas. From the repository root, point `PY` to Python with the project CV dependencies, and `SCAN` and `GUIDE` to the scan and guide directory. Verify any existing guide against the current scan.

1. Run `"$PY" scripts/scan-guide.py "$SCAN" --out "$GUIDE"` when creating a new one, and copy the newly generated `guidance.template.json` as `guidance.json`.
2. In the calibration view, mark the left and right sides of the anatomy, shoulders, elbows, wrists, hips, knees, ankles, neck base, eyes, and mouths. Highlight bare skin, hair, and clothing and accessories that move independently. Smaller areas can be enlarged using `--zoom "$GUIDE/guidance.json" --view VIEW --box X0 Y0 X1 Y1 --name NAME`. [Marked fields](../../docs/scan-annotation.md)
3. For the original head route, use `--refine "$GUIDE/guidance.json"` to refine the facial camera, mark the eye sockets, and use `--trace "$GUIDE/guidance.json" --view VIEW --pixels POINTS.json --output "$GUIDE/mouth-trace.json"` to outline the visible lip lines. These parameters are passed to the previous step's scan-guide command, and the original scan is retained as a positional parameter.
4. Execute `"$PY" scripts/scan-guide.py "$SCAN" --check "$GUIDE/guidance.json"`. The materials to be passed to the downstream include the original scan hash, cameras, coordinate transformation, verified markers, as well as the stitches required for this route.

## Number of markers and their uses:

Basic verification requires 26 named markers: `neck,nose_tip`, both sides' respective `eye,mouth,shoulder,elbow,wrist,hip,knee,ankle`, and the respective `inner,outer,top,bottom` for the eyes. The template also lists the `palm_l,palm_r`; when completing the entire body, the palms should also be marked. These markers only define anatomical positions and cannot replace dense contours and volume along the body.

For full measurements, mark `axilla_l/r` and `elbow_inner_l/r` on each side. These mark the lower axillary crease at the start of the inner upper arm and the inner elbow endpoint. Store them separately from shoulder and elbow joint centres. Mark paired endpoints for visible fingertips, heel, toe, sole, and palm/foot width and thickness. Name them for this character and retain pixel or surface evidence. Use at least two nonparallel views for body and inner-arm measurements. Explain estimates for occluded points. Sample dense contours according to visibility and spacing along the body.

In the facial model, in addition to the four corners of the eye slits, three intermediate points are marked along each visible upper and lower eyelid (a total of 12 points between the eyes). The names of these points are listed in the annotation fields. The lips are marked with ordered pixel lines, with the output being `mouth-trace.json`. These markings correspond to the original scan and the actual camera hash. Before fitting, the measurement graph is output using [Body script](../charloom-body-measure/references/script-inputs.md). It is important to check the original pixel positions rather than just the read-only values.
