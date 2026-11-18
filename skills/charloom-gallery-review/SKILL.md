---
name: charloom-gallery-review
description: "Verify the final GLBs, Gallery entry, model layers and animation. Inspect the same frame after a repair and before character delivery."
---

# Verify the delivered character in the Gallery

English · [简体中文](SKILL.zh-CN.md)

1. Determine the original scan, output path, and hash from the build report. Check the final `build/3-skin/body.glb`, `build/4-garments/character.glb`, or the actual alternative exported this time. Point `PY` to the Python of the project CV environment, and use `"$PY" scripts/animate.py validate FILE.glb` to verify the file.
2. Match the review to the edit. Check armour over the full gait, arm raises and turns. For the face, check blink, speech and head turns. Inspect the jaw, hair strands and collar from the front, side, rear and below at an angle. Reproduce the original defect with the same camera, pose and lighting, then capture the repaired view and complementary angles. For new motion, play a complete cycle on the dressed layer. Watch arm swings, turns and hair swinging back. If [Part Intersection Check](../charloom/references/walk-replay.md#check-local-intersections-during-motion) finds a crossing, go to the reported time, inspect the dressed surface from the outside and behind, then switch layers to locate the triangles. Report the geometry and its visible extent separately. For pixel counts, specify camera, resolution and sampled frame.
3. To update the local Gallery, register the selected exports in the project’s entry format. Preview with `python3 scripts/mh-ide-sync.py BUILD_PARENT --dry-run` from the repository root, then omit `--dry-run` to sync. Use the existing local Gallery or start it following [Project README](../../README.md).
4. Open the actual character URL. Check the source pane against the original scan coordinates and compare the result pane’s requested GLB with the verified file hash. Load the complete bare-body export for the bare layer. Use the assembled GLB for dressed and garments-only views, filtering by part in the latter. Pause at one frame and switch all three layers to check the clip, time and completeness of the skin under clothing. Repeat while playing. Play the requested clips and check the skeleton overlay when applicable.
5. Document the export process, perspectives, action times, measurements, and any remaining issues. Note down the file checks, the actual loading of the gallery, as well as any visual feedback provided by the user.
