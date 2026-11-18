---
name: charloom-body-fit
description: "Fit a MakeHuman body to a Charloom scan. Correct the rib cage, shoulders, armpits, upper arms, wrists and neck, then check shape and deformation in motion."
---

# Fit and inspect the bare body

English · [简体中文](SKILL.zh-CN.md)

Start with the character’s original scan, calibrated guide, MakeHuman template and selected contours. Fit the entire rib cage in one working frame, then adjust cross-sections along each arm’s own axis. For an existing character, trace the export loaded in the Gallery back to the most recent intact bare-body fit and its build record.

Before fitting, complete [Pre-fit measurements](../charloom-mh/SKILL.md#3-measure-and-save-target-dimensions-before-fitting). Check target values, paths, sections, sources and scale. The transfer reads landmarks and contours directly. The caller must verify the measurement evidence and use the target table to remeasure and correct the first export.

- Check the `transfer/report.json` with the `stage2_body_fit`. The `source_hash`, `contours`, and `registered_views` should correspond to this guide, the selected contour, and the calibrated view. If there are errors in the corresponding old ensemble or side faces, correct the input and re-perform the transfer.
- Overlay the skin derived from the front, side, and oblique sides, and verify the transition between the chest peak, lateral ribs, in front of the armpit, upper arm, and wrist. When lifting the arm on one side, fix the chest, and compare the displacement of the inner side of the chest, the side length of the in-front grid, and the continuous section from the deltoid to the biceps. [Chest, shoulder and arm section views and arm lifting examination](references/chest-and-arms.md)
- For the neck root step, chin gap, or skin color boundary, check the geometry, normal, texture mapping, and head-neck weight respectively. Review the seams when turning the head. [Jaw and neck connection](references/neck-and-jaw.md)

The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "The translated text is: "

## Choose the rig that drives the walk

When it is necessary to retain the source walking skeleton, map the fitted body and weights to the skeleton using [Original walking binding process](../charloom/references/walk-replay.md). The following commands are specifically used for migration driven by the template skeleton.

Set `SCAN`, `PRIOR`, `GUIDE`, `TRANSFER`, and `MOTION` in the root directory of the repository for this file or directory. TRANSFER uses a new directory, with the sub-directory name serving as the key for subsequent character configuration.

```sh
"$PY" scripts/scan-transfer.py "$SCAN" --prior "$PRIOR" --guidance "$GUIDE/guidance.json" --out "$TRANSFER" --motion "$MOTION" --mouth-trace "$GUIDE/mouth-trace.json"
```

During the review of component requirements, read the candidate surface file output by the command, complete the "decisions" section as per [Parted process](../charloom-garment-separate/SKILL.md), and then try again. The results include `0-guide/resolved.json`, `3-skin/body.glb`, `4-garments/character.glb`, and `report.json`, which are used for subsequent head assembly.

## Reshape the complete surface and update the exports

When the dimensions or chest and shoulder shapes are off, run [Original template section mapping](references/section-refit.md) and `refit_sections.py`. Use evidence-based anatomical positions and regional mix weights to restore a complete chest and adjust local limbs. Repeat the same definition for six body paths (including both bilateral axillary and medial arms), dense width, true section, and hands and feet. After using `copy_body.py`, update the compatible naked/made-up model file, and re-check the clothing; both commands retain the old output.

When the length from the wrist to the fingertips is close, but the hand still tends towards the front or back, check the 3D orientation with [Palm orientation re-measurement and correction](references/section-refit.md#check-and-correct-wrist-and-palm-orientation) and re-bind the actual movement.
