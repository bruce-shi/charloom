---
name: charloom-garment-texture
description: "Restore garment patterns, stretched UVs, dark triangles and missing texture in covered regions while preserving visible scan UVs and materials."
---

# Fixing clothing textures

English · [简体中文](SKILL.zh-CN.md)

1. Compare the original scan from the same perspective with the pure clothing and dressed versions. Cut to a neutral material, separating the outer layer, inner lining, and adjacent hair. First, determine the actual sources of missing faces, exposed backfaces, interlayer intersections, UV misalignment, or missing textures.
2. The visible chains, edges, buttons, and seams in the original scan are represented using the original triangular faces and their UV interpolations. Adjacent layers are handled with continuous local deformation and compatible weights, ensuring that the scale and seams of the pattern remain consistent between static and dynamic states.
3. **Optional local image editing.** When hair and other elements cover the original material, or when there is a lack of available pixels, use adjacent similar fabrics and seams as references to fill in the missing areas within the masking mask. Re-bake the result into continuous UV maps and retain the edge fill. Save the source projection, mask, replacement image, and the actual covered area.
4. From the perspective of the original problem, compare the back of the open collar with the shoulders during movement. Contrast the repeated textures and edges, and note which parts are from scanning, and which parts are supplementary. [Panel, UV and Mapping Baking Check](references/panel-uv.md)

The original scan projection and reviewed atlas for the cylindrical collar/protective device are reused[Projection and atlas commands](references/texture-commands.md).`cylinder_chart.py` The original plane and centroid coordinates are preserved`apply_atlas.py`. Only the explicitly selected primitives have new atlases created, while the geometry, UV mapping, and other material parameters remain unchanged. The new atlases are first generated in Blender with continuous UV mapping; if there is no corresponding UV mapping, the image cannot be directly applied.
