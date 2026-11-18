# Original texture projection, occlusion patch generation, and write-back

English · [简体中文](texture-commands.zh-CN.md)

Suitable for collars around the neck, near cylindrical protective gear. Ordinary flat fabrics maintain their continuous UV properties; do not force a change in the overall layout using cylindrical scripts.

1. Isolate the target material from the original scan, and record the original triangular surface range. Define the center of the cylinder, the work coordinate transformation, and the up and down heights; first generate a cylinder diagram that only contains the actual observed area:

```sh
"$PY" skills/charloom-garment-texture/scripts/cylinder_chart.py "$CHART_CONFIG" --out "$CHART_OUT"
```

Configuration fields: `source`, `source_sha256`, 4×4 row priority `source_to_working`, `center_m`, original plane `triangle_ranges`, `size:[width,height]`, `height_range_m:[lower,upper]`. Optional `height_radius_slope` defines the expansion height `h=y-k*r`, used for collars that fall diagonally on the shoulders; default 0. `subdivisions` default 2, interpolates UVs for the original triangle, and performs subdivision projection. The edges of the left and right images are joined at the back, with an angle of `atan2(x-center.x,z-center.z)`.

Output `observed.png` (no observation in transparent areas), `coverage.png`, `source-pixels.npz` (centroid coordinates of each pixel source surface and effective pixel original surface), and `chart.json`. The observation diagram retains the source color; hair or skin occlusion should be explicitly excluded from the segmentation and supplementary mask, and should not be projected as if they were clothing material.

2. Mark the areas actually occluded. Refer to visible examples of the same weave, seams, piping or chain pattern. When inpainting is needed, use the fal image-edit workflow selected for the task with explicit 2K, medium quality and a cached receipt. Specify the atlas position, scale, left/right seams and visible decorations. Fill only missing fabric. Keep the original, mask, edited image, parameters and receipt. Projection and visible-texture work can proceed without authorizing a paid image request.

3. In Blender, create continuous UVs for the panels that need to be replaced: cut along the back seam of the collar, unify the u-values along the circumferential direction, and unify the v-values from the lower edge to the opening; ensure that the scale matches the actual supplementary image. Retain the UVs of the original triangulated surfaces; use the new atlas only for new panels or clearly defined supplementary areas. The subdivided original surfaces should inherit UVs through centroid interpolation; prevent each new triangle from being assigned to a discontinuous small color patch. Set the texture island padding, and check the empty back side of the collar and the top edge.

4. Inspect and save the final continuous atlas, then write the clearly selected primitives:

```sh
"$PY" skills/charloom-garment-texture/scripts/apply_atlas.py "$ATLAS_CONFIG" --out "$TEXTURED_GLB"
"$PY" scripts/animate.py validate "$TEXTURED_GLB"
```

Configure `model:{file,sha256}`, `atlas:{file,sha256}`, `primitives:[{node:exact_node_name,primitive:index,uv_evidence:reviewed_uv_image_path_and_notes}]`, optional `material_name`. This command retains the existing UV/geometry/weights and other material settings. It only clones the material used for the selected faces and replaces the base-color atlas. UVs are not automatically re-expanded, and the overdraw area is not incorrectly applied to the entire original garment.

5. Use the camera that first showed the defect. Inspect pattern scale, rear seams, folded edges, black faces and gaps exposing an adjacent layer. Play the walk and shoulder/neck motion. Compare against the visible texture in the original scan. Identify which regions retain observed detail and which were filled under occlusion. Record the final GLB and atlas digests, then switch Gallery layers at the same frame.
