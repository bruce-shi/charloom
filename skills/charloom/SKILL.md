---
name: charloom
description: "Use Blender Python to assemble and edit scanned characters, retarget FBX or GLB clips, solve poses, bake animation recipes, inspect skin deformation and browse published character results. Use charloom-mh for calibrated scan measurements."
---

# Charloom in Blender

English · [简体中文](SKILL.zh-CN.md)

For a new environment, follow [environment and input setup](references/setup.md).

Run your Python inside Blender with `blender -b --factory-startup --python-exit-code 1 --python your_script.py`. Import `charloom.api` from the checkout's `blender/` directory. The same functions power the Charloom panel. For installation, prerequisites, an agent script, and sample commands see the [project README](../../README.md).

- For an existing character, add each new motion to its existing bound GLB with `api.retarget_animation`. Keep the target rig and skin fixed. If the character has hair, cloth or prop bones, follow [adding a clip and reviewing secondary motion](references/walk-replay.md#add-motion-to-the-existing-character-rig) to account for those chains and check the dressed result through the motion cycle.
- For a new character, use the [scan workflow](../charloom-mh/SKILL.md). `api.prepare_guide` renders a scan. `api.transfer_scan` fits a MakeHuman scaffold and separates the reviewed garments. `api.build_character(..., head_binding="scan-surface")` attaches the observed head and binds its original facial surface. Give each operation its own new output directory.
- `rig, body = api.import_character(path)` imports a rigged GLB. `api.fit_surface(body, scan, group)` writes a shape key on exposed skin. `api.bind_clothing(body, rig, objects)` transfers up to four body influences per vertex to aligned fitted fabric. For rigid equipment or loose panels, follow the [per-piece rig workflow](../charloom-garment-fit/SKILL.md) before retargeting or previewing an auto-rigged export.
- To reproduce an existing walk, identify its original rig, binding, source clip and subsequent track edits before fitting or retargeting. Follow [walk reconstruction and track comparison](references/walk-replay.md).
- `api.retarget_animation(source, target, output, clip=..., config=...)`, `api.solve_pose(model, effectors, output, clip=..., time=...)` and `api.build_animation(recipe, output)` return reports and an exported file. Review joint coverage and effector residuals. Retargeting a T-pose source onto a character with lowered arms needs reference-chain alignment.
- `api.preview(rig, clip)` selects the rig and morph actions. Use `bpy.context.scene.frame_set(frame)` for samples. Save the editable scene using `bpy.ops.wm.save_as_mainfile` and export with `api.export_character(rig, path)`.
- In the CV environment, use `scripts/animate.py inspect|sample|validate` on a GLB. Use `scripts/skin.py analyze|contacts` for skin and seam measurements, or `start|step|inspect` for versioned repair. Repair plans carry the required round and a diagnosis. Inspect both diagnostic numbers and actual playback.
- Run `python3 scripts/mh-ide-sync.py BUILD_PARENT --dry-run` to inspect registration and drop `--dry-run` when registering results. Use `cd viewer && bun run dev` to browse Gallery assets.

When walking in the world space, either above or below the ground, see [Check walking contacts at the shoe soles](references/walking-ground.md). Repeat the same translation for clothing and naked bodies.
