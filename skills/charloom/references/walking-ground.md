# Check walking contact at the boot soles

English · [简体中文](walking-ground.zh-CN.md)

First, distinguish the continuous support during walking from the takeoff phase of running from the frontal view. In world coordinates, check the lowest point of the gait when the foot is at rest, and differentiate between barefoot, insole space, heel, and sole. When converting thick boots to thin socks, the sole may inherit the ground conditions of the original boot; record the source and deviation of this item. If the reference only provides the heel height, it is not possible to directly draw the barefoot sole to that height.

When the redirected walk is clearly done with one foot or both feet floating, use `scripts/ground-walk.py` after checking the original head, clothing, and action configuration. First, check the nodes where the soles belong in the actual clothing GLB, the action names, and the common root bone of all layers. Configure the JSON for this export:

- `reference` Refers to the clothing GLB with actual contact surfaces.`variants` Represents the object that connects the expected output relative path to the input GLB path. It requires filling in two files: the naked and clothing files, and also including a reference.
- `contact_meshes` is a list of node names that have been isolated for review and are truly grounded.`clip` and`root` use action names and node names that control the overall position from the target file.
- `ground_y` represents the ground height in the target world, measured in meters.`sample_hz` is the initial sampling frequency.`tolerance_mm` is the ground clearance limit. This value is determined based on the original animation frame rate, shoe style, and actual scene conditions. Continuous support walking uses `walking`; running with takeoff uses `motion_kind=running`, which only lifts the sole of the shoe that is below the ground, while retaining the part above the ground. When it is necessary to preserve the old bone trajectory with the same value progression, `root` should be directed towards the actual common container above the character's skeleton, and a separate stage height track should be added to this container.

```sh
"$PY" scripts/ground-walk.py "$GROUND_CONFIG" --out "$GROUNDED"
```

Inspect `ground-walk.json` for the original contact-height range, corrected extrema, export hashes and checks on non-root animation channels. The script remeasures contact between keyframes and writes a root translation shared by both layers. Inspect weight transfer, sliding and knees over the whole gait. Keep the same world position when switching between bare feet and footwear. Load the corrected files in Blender and the Gallery and verify their motion and both hashes.

If the lowest point reaches the floor but only the toes make contact at every step, correct the motion mapping before grounding. Compare the support heel, ball and swing foot at matching times in source and export. Measure the rest direction from ankle to mapped child on both rigs. One may use a short foot arch and the other connect directly to a distal toe. With `align_reference_chains` enabled this difference can introduce a constant pitch offset. If confirmed, add the actual left and right foot bone names to `preserve_reference_joints` in the retarget config. Keep the target foot’s rest orientation and apply the source rotation changes. Rebuild both layers with the original motion, inspect the feet at stride extremes and support transitions, then remeasure root motion and soles.
