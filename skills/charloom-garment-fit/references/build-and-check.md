# From garment separation to motion review

English · [简体中文](build-and-check.zh-CN.md)

## First Bind

In `guidance.json.garments`, an independent moving object uses one `name`, which is consistent with the `piece.name` in the sub-part determination. Fill in the `regions` multi-view polygons according to the annotation format; check the front and back sides with the actual drawing. Clothing that is connected but moves differently should be named separately.

Each item can specify `layer` (increasing from inner to outer layers), `rig` and `fabric`.

- Prosthetic fabric: `rig:{mode:"body"}`. Adjust the smoothness weight of the final skin, and apply a local weight correction for the cuffs.
- Rigid armor, equipment: `rig:{mode:"rigid",bone:"attachment_bone"}`. Bone names are derived from actual data; template bones and redirected bone names are different and should not be used interchangeably.
- Skirts, belts, and cloaks: `mode` is taken as `skirt,ribbon,panel,cape,veil`. Specify `bone`, and use `direction` and `width_axis` to determine the direction of the hanging elements and panels; `skirt.cut` is either `flared,bell,pleated` or `{"panels":integer}`. The hanging points are fixed, and secondary chains are created along the length.
- `fabric` refers to the existing material name in the script (e.g., `woven,leather,silk`), or `{"delay_s":value,"gain":value,"max_angle_rad":value}`. Select motion parameters that match the fabric, and review them throughout the entire action.

The `write_models` in `scan-transfer.py` is bound by layer. The bare model is saved with the complete body first, and then the clothing occlusion is processed. The `rig_for`, `bind`, and `bake` in `transfer/cloth.py` implement these fields. The rigidity/panel configuration of this guide will affect the first build; existing local repairs to the final body are done using the following explicit object operations to avoid rerunning the early body fitting.

## Final body adaptation

First, list the actual objects and the attachment points:

```sh
"$BLENDER" -b --factory-startup --python-exit-code 1 --python skills/charloom-garment-fit/scripts/fit_garments.py -- "$DRESSED" --inventory "$GARMENT_INVENTORY"
```

Read the inventory report and isolate the corresponding objects in Blender. Cut the protective gear and the final body part at the proximal, mid-proximal, and distal ends along the lateral limb axis; record the center, width, depth, and inner and outer layer gaps respectively. Check each part such as the cuffs, flanges, and hidden seals. Do not oversize the entire protective gear just because a small piece of the inner wall is interspersed.

Create an edit JSON: The top level `source_sha256` must match the inventory summary, with `pieces` being the operation array. Each item `objects` should be the exact object name from the inventory; `bind` can be taken from `preserve`, `body`, or `rigid`. The rigid rebind specifies the actual `bone`. If size adjustment is required, also fill in `bone`, 4×4 column-major affine `matrix`, and `measurement_evidence` (section file and specific positions). The matrix plays a role in **Blender's skeletal local meter units**; the inventory provides `bone.matrix_local`, do not directly use GLB's Y-up as Blender coordinates. Use the original/target center to calculate translation, and the aspect ratio to determine the corresponding local scaling. Do not guess the protective gear size based on the full object bounding box.

The default matrix applies to the entire item; for local inner walls or cuffs, use `vertices:{object_name:[vertex_id...]}` for explicit selection. The transition between local selections should be smooth, with checks for normal and UV stretching. Use `bind:"body"` to apply the current body transmission weight, and `rigid` to assign the deformable bone weights of all selected objects to the same pivot point; other objects that already meet the criteria will not be included in the configuration. Secondary chains such as cloaks already have their own links, and use `preserve` instead of `body` to preserve their freedom of movement.

```sh
"$BLENDER" -b --factory-startup --python-exit-code 1 --python skills/charloom-garment-fit/scripts/fit_garments.py -- "$DRESSED" --config "$GARMENT_EDITS" --out "$GARMENT_OUTPUT"
"$PY" scripts/animate.py validate "$GARMENT_OUTPUT/character.glb"
```

Outputs include editable `character.blend`, `character.glb` and `garments.json` with provenance. The script rejects a direct transform when the garment has morph targets. Inspect them and apply the corresponding edit in Blender. Check collisions and review the garment after the edit.

## Complete actions and interlaced sequences

```sh
"$PY" scripts/skin.py analyze "$DRESSED" > "$REVIEW/skin-analysis.json"
"$PY" scripts/skin.py contacts "$DRESSED" --config "$CONTACT_CONFIG" > "$REVIEW/contacts.json"
"$PY" scripts/render-character.py "$BUILD" --only body,dressed,walk,chest
```

The contacts configuration is set to `regions:[{mesh:primitive_index,geometry_hash:digest,vertices:[vertex_id...]}]`, plus `clip` and `times` (the number of seconds in the actual complete segment). Use `Surface(FILE).parts`'s `hash` and node names to verify the primitive; this is a geometric hash that differs from the one used in the body script, and does not use file SHA as a substitute. The results are distributed across the rest and additional elements for each frame; the original distribution also needs to be checked, and cannot be considered zero if it increases. Changes in seams and edge lengths are reported in the analyze report. Currently, the checks only cover horizontal intersections, and cannot replace visual inspections for close proximity, coplanar layers, and visual checks.

Review the static pose, full walk, arm lifting, elbow bending, and related movements. Locate the worst frame, specific bilateral triangular surfaces, and recalculate after modifying the binding/local contours; the sectional dimensions must still match the original shape. At the same time, switch between bare model, pure clothing, and dressed versions, and check for old fingers, nails, collar, and hair residue surfaces. After approval, register the final dressed file along with the complete bare model in the Gallery.
