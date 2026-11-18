# Bind expressions on the original scan surface

English · [简体中文](face-build.zh-CN.md)

Use the original static scan, its calibrated guide, lip trace and reviewed source-triangle assignments. The MakeHuman template supplies the fitted body and skeleton. The scan supplies the visible head, facial features, UVs and texture.

## Inspect and mark the source

Render the face from the front and oblique views. Inspect its connected components separately. Distinguish painted eyes, open textured eye patches, closed eyeballs, eyelashes and internal mouth fragments. Trace the actual closed lip seam and both eyelid margins on the calibrated source images. Keep the file SHA-256 in the lip trace and the pipeline `source_hash` in guide and transfer records. These are separate hashes produced by the current tools.

For explicit assignments, the configuration JSON maps the transfer directory's basename to its settings. Put the transfer report's `source_hash` under `face_atlas.source_hash`. The `face_atlas.parts` entries are reviewed half-open source triangle ranges.

- `skin` selects the connected facial skin and neck surface.
- `face_overlay` retains open textured patches that belong to the visible face. They receive the same surface expressions.
- `eye_l` and `eye_r` identify actual separate eyeballs. Confirm their geometry and pivots before using these assignments.
- Separate eyebrows use the segmentation review's `brow` assignment. Review obsolete internal fragments with `internal` so they are removed from garment exports.

For a painted or closed scanned face, use the scan-surface entry point below. It subdivides the observed facial triangles, splits the traced lip seam, preserves the neutral surface and UV interpolation, and adds facial morph targets. Its oral lining follows the actual split lip edges. The head attachment reuses the calibrated neck assembly.

## Build and verify

From the checkout, set `PY` to the CV interpreter and install the viewer's locked Bun dependencies. Set `CHARLOOM_TRANSFER_DIR` to the parent of the completed transfer, `CHARLOOM_HEAD_DIR` to a fresh output parent, and `NAME` to the transfer basename. The transfer must contain its matching mouth trace and embedded `walking` clip.

```sh
export CHARLOOM_TRANSFER_DIR CHARLOOM_HEAD_DIR
export CHARLOOM_CV_PYTHON="$PY"
"$PY" scripts/roster/bind_scan_head.py "$NAME" --config "$CONFIG"
bun scripts/roster/embed-clips.ts "$NAME"
bun scripts/roster/verify-face.ts "$NAME"
```

Omit `--config` when no explicit source-part assignments are needed. For the integrated Blender workflow, pass `head_binding="scan-surface"` to `api.build_character`. The worker CLI accepts `--head-binding scan-surface`.

Check `report.json` for `route=scan_surface_binding` and the original source path and hash. Check `controls.json` for the measured neutral facial-surface distance to the original scan. `animated-checks.json` covers both final bare and dressed GLBs.

Play both exports in Gallery. Inspect eyelids at rest, through glances and at full blink. Check the full lip seam on M/B/P, the lower lip at F/V, the oral lining at maximum opening, and the neck through head turns. Record source correspondence, geometric checks and actual visual observations. Use the same clip time and camera when comparing layers.

The anatomical facial assembly remains available through `volume_head.py` and `head_binding="anatomical"` for explicitly selected reconstruction work, including fitted eye globes and dental components. Choose that route from the source structure and task requirements before building.

If the neck-cutting report indicates that the default cut plane is below the selected head surface, first confirm that the isolation map for `face_atlas.parts.skin` contains a real neck skin, and then measure the horizontal cross-section with a complete neck ring and below the chin in the actual working coordinates. Specify this height using `plane_y_m` in the same character configuration, and check the generated head-neck junction and head rotation frames. The neutral surface deviation check is only performed on the facial triangles from the original scan; the artificial neck bridge and seal still move with the skull movement. For eye blinking, also check the close-up view of closed eyes, eyebrow shape, and the actual eye slit in the exported action. The closed lips and F/V are reviewed frame by frame in the speaking segments.

If the high collar clothing is positioned at the back of the neck and at the lower part of the chin, the horizontal cut will bring the fabric onto the bare skin. Isolate the original head shell, and mark a working space point on each side of the neck and at the back of the neck along the actual skin exposure boundary. In the character configuration, use `neck_plane_points_m` to write three sets of `[x,y,z]` meter coordinates, and use `plane_y_m` to specify the reference height that preserves the original head shell at the neck. The script determines the cut line based on these three points, preserving the original head surface above, fixing the boundaries at both ends, and smoothing the transition from the neck to the shoulders. The lower seam is chosen from the bare skin color map. Use the bare side and solid color materials to examine the entire neck, and then compare the collar, chin, and back of the head in the same frame. Record the actual selected points and the reference data.

**Loose collar version**. In the front turn or eye-closing frames, hide `body`, `fitted_head`, `FaceSurface`, and the actual collar in sequence, ensuring the skin tone jaggedness is covered by the clothing. If the original scan shows a complete neckband, which shares a recognizable geometric boundary with the loose collar, and the clothing covers the lower template neck surface, add `dressed_neck_from_scan: true` in the character item with the configured three-point cut. The loose collar version will retrieve the original scan neckband and its UVs along the cut lines, verify the edge points of the original collar, and use the existing positions and weights of the collar to connect to the skull. For the naked model, use the smoothed neckline from the previous step. Check the shared edge count and the number of original neckband surfaces in the report, and recheck in front eye-closing, side walking, turning, and at maximum opening. When the collar is low, off-shoulder, or the neck boundary is incomplete, it is necessary to first determine the actual visible skin area.

When a high collar and other garments cover every body triangle, the dressed layer may omit that invisible mesh. Deliver the complete body in the bare GLB and editable scene. A dressed export using the original neck band records `dressedNeck` in `controls.json`. The verifier `verify-face.ts` samples corresponding points on the visible original head, face, neck band and collar during motion. Read `neckMaxMm` for each layer, then inspect the dressed close-up.
