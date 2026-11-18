---
name: charloom-body-fit
description: "拟合 Charloom MakeHuman 裸模，修复胸廓、肩腋、上臂、手腕或颈部轮廓与动作变形。"
---

# 拟合并复查裸模

[English](SKILL.md) · 简体中文

从当前角色的原扫描、标定 guide、MakeHuman 模板和已选轮廓开始。在同一工作坐标系拟合完整胸廓，再沿左右臂各自的轴调整局部截面。对已有角色，从 Gallery 对应的导出追溯最近一次形状完整的裸模与构建收据。

开始前必须完成[主流程的拟合前量体](../charloom-mh/SKILL.zh-CN.md#3-先量体并保存目标再开始拟合)，检查目标值、路径、截面、来源及尺度。当前 transfer 只直接使用标记和轮廓，不读取完整量体表，也不强制检查量体是否完成；调用方必须检查前置证据，并以目标表逐项复测和修正首次导出。

- 核对 `transfer/report.json` 的 `stage2_body_fit`。其中 `source_hash`、`contours`、`registered_views` 应对应本次 guide、选用的轮廓和已标定视图。出现旧 ensemble 或侧面对应错误时，修正输入并重做 transfer。
- 在正面、侧面、斜侧面叠加导出的皮肤，核对胸峰、侧肋、腋前、上臂与腕掌过渡。单侧抬臂时固定胸廓，比较胸内侧位移、腋前网格边长及三角肌到二头肌的连续截面。[胸廓、肩臂截面与抬臂检查](references/chest-and-arms.zh-CN.md)
- 对颈根台阶、下巴缝隙或肤色分界，分别检查几何、法线、贴图及头颈权重，复查转头时的接缝。[下颌与颈部接合](references/neck-and-jaw.zh-CN.md)

输出是能与原扫描和量体图逐项对照的裸模。修改肩臂后，在最终裸模和穿衣版复测受影响的长度、截面和腋下运动。

## 按实际步行骨架选择绑定入口

需要保留源步行骨架时，按[原步行绑定流程](../charloom/references/walk-replay.zh-CN.md)把拟合好的身体和权重映射到该骨架。以下命令专用于模板骨架驱动的迁移。

从仓库根目录设置 `SCAN`、`PRIOR`、`GUIDE`、`TRANSFER` 和 `MOTION` 为本次文件或目录。TRANSFER 使用新目录，末级目录名作为后续角色配置的键。

```sh
"$PY" scripts/scan-transfer.py "$SCAN" --prior "$PRIOR" --guidance "$GUIDE/guidance.json" --out "$TRANSFER" --motion "$MOTION" --mouth-trace "$GUIDE/mouth-trace.json"
```

分件要求复核时，读取命令输出的候选面文件，按[分件流程](../charloom-garment-separate/SKILL.zh-CN.md)补齐 decisions 后重试。结果包含 `0-guide/resolved.json`、`3-skin/body.glb`、`4-garments/character.glb` 和 `report.json`，供后续原头装配使用。

## 完整表面修形与带入下游

尺寸或胸肩形状超差时，按[原模板截面映射](references/section-refit.zh-CN.md)运行 `refit_sections.py`，用有证据的解剖站位和区域混合权重恢复完整胸廓并调整局部肢体。按同一定义重测六条沿体路径（包含双侧腋下大臂内侧）、密集宽深、真实截面和手脚。通过后用 `copy_body.py` 更新兼容的裸模/穿衣文件，再重查服装；两个命令均保留旧输出。

腕到指尖的长度接近而手仍偏向前后时，按[腕掌朝向复测与修正](references/section-refit.zh-CN.md#核对并修正腕掌朝向)检查三维方向，并重新绑定实际动作。
