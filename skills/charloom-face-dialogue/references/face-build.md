# Bind expressions on the original scan surface

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

若切颈报告显示默认切面在已选原头表面下方，先确认 `face_atlas.parts.skin` 的隔离图含真正的颈皮肤，再在实际工作坐标中量出有完整颈环、且低于嘴角的水平截面。在同一角色配置中用 `plane_y_m` 指定该高度，检查生成的头颈接缝和转头帧。中性表面偏差检查只对来自原扫描的表情三角面执行；人工颈桥与封口仍随头骨运动。眨眼同时查看闭眼近景、眉形与实际导出动作中的眼裂，闭唇与 F/V 则在说话片段中逐帧复查。

若高领衣物在后颈高、下巴处低，水平切面会把衣料带到裸身上。隔离原头壳，沿实际露肤边界在左右颈侧和后颈各标一个工作空间点。在角色配置里用 `neck_plane_points_m` 写三组 `[x,y,z]` 米坐标，用 `plane_y_m` 指定保留颈部原始壳体的参考高度。脚本按这三个点定切面，保留上方原头表面，固定两端边界，平滑模板颈部到肩的过渡。下方接缝从裸身肤色图取色。用裸身正侧背和纯色材质查看整个颈部，再在同一转头帧对比衣领、下巴与后脑勺。记录实际选点和原图依据。

**穿衣版的高领**。在同一正面转头或闭眼帧，依次隐藏 `body`、`fitted_head`、`FaceSurface` 和实际衣领，定位覆盖衣料的肤色锯齿。若原扫描的露肤颈带完整、与高领共用可辨认的几何边界，且衣服覆盖了下方模板颈面，可在已配置三点切面的角色项中加 `dressed_neck_from_scan: true`。穿衣版会沿切面取回原扫描颈带及其 UV，核对原衣领边点，使用衣领已有的位置和权重衔接到头骨。裸身仍使用前一步的光顺颈部。检查报告的共享边点数及原颈带面数，并在正面闭眼、侧面行走、转头与最大张口处复查。低领、露肩或颈部边界不完整时，需要先确定真正可见的皮肤范围。

穿衣身体被高领和其他衣物完全遮住时，可以省略这一层没有可见三角面的身体 mesh；另行交付含完整身体的裸身 GLB 和可编辑场景。启用原颈带的穿衣版会在 `controls.json` 中记录 `dressedNeck`，由 `verify-face.ts` 在实际动作中核对可见的原头、面部、颈带与衣领对应点。读取两层各自的 `neckMaxMm`，再检查穿衣近景。
