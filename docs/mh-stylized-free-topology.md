# Assemble and review a stylized character

English · [简体中文](mh-stylized-free-topology.zh-CN.md)

Inspect the original textured head from the front, side and three-quarter views. Identify each eye, eyelid, lash, oral surface and piece of facial hair. Preserve the original head surface and UVs around the openings. Bind the teeth and tongue inside the measured oral cavity and check the fitted neck against the body and collar. [Face assembly and dialogue](../skills/charloom-face-dialogue/SKILL.md)

Choose the body route from the source geometry. The usual route fits a [MakeHuman template](mh-body-fitting.md). A scan with a separate torso and four complete limb shells can use the [scan-volume body](../skills/charloom-scan-volume/SKILL.md). Record the source hash and the reviewed shell seeds before assembly.

For a single character, the [Blender build command](../README.md) accepts a raw scan with reviewed inputs, or an existing transfer directory. To rebuild named characters from calibrated transfers, supply your own roster JSON with a `characters` array of named entries. Under the transfers directory, each character has its own subdirectory with the matching name.

```sh
"$CHARLOOM_CV_PYTHON" scripts/roster/rebuild.py "$NAME" \
  --roster "$ROSTER_JSON" --transfers "$TRANSFERS" --out "$NEW_OUTPUT"
```

In both bare and dressed versions, check lip contact for M/B/P, the lower lip against the upper teeth for F/V, eyelid closure, gaze and the jaw-to-neck outline. During a complete walk, inspect the hands, the moving neck seam and clearance between clothing and skin. Use the [Gallery review skill](../skills/charloom-gallery-review/SKILL.md) to verify which files and frames the viewer actually loads.
