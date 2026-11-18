---
name: charloom-garment-fit
description: "Fit clothing, gauntlets, armour and props to a Charloom character. Correct intersections, oversized cuffs, attachment bones and garment dimensions."
---

# Fit the wardrobe to the final body

English · [简体中文](SKILL.zh-CN.md)

**Forearm Protector.** Take the distances from the elbow to the wrist for each side, and cut the armor, current armor, bare arm, and inner sleeve at the same position as the wrist. Register the center, width, and depth of the cut section, ensuring it covers the opening near the elbow, the middle section, and the wrist area. Check the taper, flanges, and the overlap with the gloves from the front and side views. After attaching the protector to the center, determine the dimensions based on the original shape and the space required by the actual skin.

**Collisions.** Position the hit triangular faces on the visible outer shell, inward-folded edges, internal closures, or inner sleeves. First, check the binding and axis offset. If only the hidden closure presses into the forearm, make local adjustments to the openings and inner walls, while retaining the visible outer edge, UV, and shell outline. Compare the original and the modified width and depth in the same section, and then retest the posture where the collisions occur.

**Binding.** The rigid parts, including the cuff, fasteners, and distal fixation points, are attached to the same hanging point, with configurations marked as `rig.mode=rigid` and `rig.bone`. The gloves are adapted to the final palm and each finger. The undergarments deform smoothly with the body; the cape, skirt hem, and belt ends are bound to the fixed edges and secondary bone chains. The scabbard, sword blade, and hilt are checked against the actual hanging points of the equipment. Unfold each layer to check for any missing surface attachments or misaligned bones.

**Outfit (optional).** When swapping outfits between two characters using the corresponding scaffolds, point `PY` to the Python script in the project's CV environment, and run `"$PY" scripts/swap-outfit.py DONOR RECIPIENT OUTPUT.glb`. The reviewed skin exclusions and binding areas can be passed in through the `--fit-review` associated with the donor's hash binding. Keep the dimensions of the handheld props.

**Review.** Check the skin and clothing, inner and outer layers of clothing, as well as the wrist area during static, full gait, arm lifting, and related joint bends. Rigid parts should maintain their shape in the local coordinates of the hanging points; flexible panels should have seamless seams and texture continuity. Record the worst frame and the actual interlacing surface.

For a detailed guide to fields, Blender object inventory, commands for local transformations and rebinding, as well as interleaved configurations, see [Build and Action Review](references/build-and-check.md). Use `fit_garments.py` to save the changed dimensions/hang points as new GLB files and editable scenes, and then measure the actual cross-sections and the entire sequence of actions.
