---
name: charloom-garment-separate
description: "Separate skin, hair, garments and accessories from a scan. Resolve fingers left in clothing, floating nails and hair fused to a collar."
---

# Assign original scan triangles to parts

English · [简体中文](SKILL.zh-CN.md)

1. Replicate the same pose and perspective using the original scan, bare model, pure clothing, and dressed versions. Recheck after hiding the entire item; when the same object has multiple materials, isolate each layer in the editing mode. Determine based on connections, depth, anatomical positions, and textures.
2. Bind the confirmed original triangular faces or component IDs to this `source_hash`. When a component needs to be determined, label it with the component ID in the `decisions` of the review JSON, using `skin`, `internal`, `brow`, or the component name in the `piece`. Replay in the new output directory using `--segment-evidence`, `--segment-review`.
3. When there are fragments at hand, check the left and right cuffs to the fingertips. If the novice mode is hidden, the old nail and skin texture fragments still remain in the pure clothing layer. Remove the areas that do not belong to the original hand. Check the repaired empty cuffs, glove edges, and the outlines of both hands.
4. When the character has nail polish, sample the original nail color and write it to the UV of the current finger's nail bed or the bound nails. The MakeHuman hand can be used with a nail mask that matches the current UV. Verify the mask fall points for each finger, switch between bare model, pure clothing, and dressed versions in the same frame, and play the wrist movement to ensure that the nail color always matches the corresponding fingertip.

Where hair shares faces with the collar, separate them along visible material boundaries. Inspect the back of the hair strands and the empty collar. [Processing of coplanar collar, skin, and hair](references/collar-and-hair.md)

## Pre-fitting individual components

```sh
"$PY" skills/charloom-garment-separate/scripts/review_parts.py "$SCAN" --guide "$GUIDE" --out "$PARTS"
```

Outputs include `components.json` (original triangle ranges and digest per component), `components.png`, front, side and rear views of each piece, and `decisions.template.json`. Copy the template to `decisions.json` and set `action` for each item to `skin`, `internal`, `brow`, or `piece` with a `name` that matches an existing garment in the guide. A blank entry is undecided.

Mixed components should not be categorized together: first, use skin/clothing polygons or brushes to subdivide the areas in the main side view of the guide, then re-divide the parts. If it is necessary to define the boundaries of sub-faces, save the new candidates using the semi-open range of these sub-faces in the evidence. The ID should be a SHA-256 hash of the ordered face number array `<u8` bytes, and the decision should be written with the same ID. The ranges must not overlap. The entire connected area containing hair and collars should not be marked as hair.

Replay and check in the new directory:

```sh
"$PY" skills/charloom-garment-separate/scripts/review_parts.py "$SCAN" --guide "$GUIDE" --evidence "$PARTS/components.json" --review "$PARTS/decisions.json" --out "$PARTS_REPLAY"
```

Outputs are `owners.npy`, `review.json` and `skin-regions.json`. The `triangle_ranges` in the last file are candidate visible-skin ranges for measurements. Exclude hidden faces and keep brows separate. Pass the same evidence and decisions to `scan-transfer.py --segment-evidence ... --segment-review ...`. Recompute correspondences if the source mesh changes. If a garment is renamed, check the guide and decisions agree.
