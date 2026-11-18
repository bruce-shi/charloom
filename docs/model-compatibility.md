# Evaluate a new character model

English · [简体中文](model-compatibility.zh-CN.md)

Record the original mesh, textures, authored rig, motion sources and redistribution terms for the model under test. When preprocessing is necessary, retain the original and attach the scan guide to the exact file consumed by the transfer. [Scan calibration](../skills/charloom-scan-guide/SKILL.md)

Compare the original, bare export and dressed export with matched cameras and lighting. Use the original rig playing the same motion as a control when available. For stylized anatomy, inspect each limb section and the neck transition. For a realistic face, compare the neutral profile, eye shape and lip line with the original before sampling dialogue and blink. [Body fitting](mh-body-fitting.md) · [Face binding](../skills/charloom-face-dialogue/SKILL.md)

Report the mesh and stage behind each metric. Record sampled times, units and the surfaces compared. A zero gap between paired neck vertices can coexist with nearby crossing triangles. A structurally valid GLB can still show misplaced teeth, a visible collar seam or incomplete eyelid closure. Inspect the delivered geometry through the full motion in the [Gallery](../skills/charloom-gallery-review/SKILL.md) and record remaining defects.
