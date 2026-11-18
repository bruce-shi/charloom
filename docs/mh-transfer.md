# Transfer a scan onto a character rig

English · [简体中文](mh-transfer.zh-CN.md)

Begin with a static textured scan, a MakeHuman body template and a calibrated guide for the same scan. Record which skeleton will drive the character. For a character that already has a bound rig, follow the [fixed-rig motion workflow](../skills/charloom/references/walk-replay.md). For a new scan, complete [annotation](scan-annotation.md), [measurement and fitting](mh-body-fitting.md) and [garment separation](../skills/charloom-garment-separate/SKILL.md).

Run the template-rig transfer from the checkout root. Set the variables to your own inputs and use a fresh output directory.

```sh
"$CHARLOOM_CV_PYTHON" scripts/scan-transfer.py "$SCAN" --prior "$PRIOR" \
  --guidance "$GUIDE/guidance.json" --mouth-trace "$GUIDE/mouth-trace.json" \
  --motion "$MOTION" --out "$TRANSFER"
```

Inspect `0-guide/resolved.json` for the landmarks used, `1-segment/review.json` for the segmentation and `report.json` for the fit. When the command requests a review of ambiguous triangles, resolve the candidate ranges on the original scan and supply the matching decisions and evidence. The bare body is `3-skin/body.glb`. The dressed version is `4-garments/character.glb`.

Complete the [original-head assembly](../skills/charloom-face-dialogue/SKILL.md), then measure and play back both final exports. The [end-to-end skill](../skills/charloom-mh/SKILL.md) specifies the inputs and completion checks at each stage.
