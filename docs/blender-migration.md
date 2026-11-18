# Blender authoring tools

English · [简体中文](blender-migration.zh-CN.md)

Use the Blender extension to edit the scene and the local Gallery to inspect exported results. Install the extension and its Python environment from the [project README](../README.md).

| Task | Entry point | What to inspect |
| --- | --- | --- |
| Calibrate and fit a scan | `scripts/scan-guide.py`, `scripts/scan-transfer.py` | Scan and camera hashes, landmarks, registered body contours and garment ownership |
| Bind the scanned head | `scripts/roster/build_blender_character.py` | Mouth trace, eyelids, dental arches and the neck seam on both exports |
| Change body shape or garment weights | `blender/charloom/api.py` | Shape keys in Blender, weights around joints, covered body triangles and garment clearance |
| Retarget or layer animation | `scripts/animate.py` | Rest transforms, joint mapping, foot contacts, morph tracks and prop constraints |
| Inspect skin deformation | `scripts/skin.py` | Sampled edge strain, surface intersections and separation at seams |
| Open a saved workspace | `scripts/blender_project.py` | Imported collections, packed textures and the original project data |
| Review a delivery | [Gallery review skill](../skills/charloom-gallery-review/SKILL.md) | Loaded GLB, selected layer, camera, motion and playback time |

Start with the [scan workflow](mh-transfer.md) for a new character or [workspace import](local-workspace.md) for a saved project.
