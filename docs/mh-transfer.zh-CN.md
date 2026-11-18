# 将扫描角色迁移到目标骨架

[English](mh-transfer.md) · 简体中文

准备静态带贴图的扫描、MakeHuman 身体模板，以及与该扫描匹配的标定 guide。记录驱动角色的骨架。已有绑定的角色按[固定装备动作工作流程](../skills/charloom/references/walk-replay.zh-CN.md)追加动作。新扫描先完成[注释](scan-annotation.zh-CN.md)、[测量与拟合](mh-body-fitting.zh-CN.md)和[服装分离](../skills/charloom-garment-separate/SKILL.zh-CN.md)。

从仓库根目录执行模板骨架驱动的迁移命令。将变量设为自己的输入文件，并使用新的输出目录。

```sh
"$CHARLOOM_CV_PYTHON" scripts/scan-transfer.py "$SCAN" --prior "$PRIOR" \
  --guidance "$GUIDE/guidance.json" --mouth-trace "$GUIDE/mouth-trace.json" \
  --motion "$MOTION" --out "$TRANSFER"
```

在 `0-guide/resolved.json` 核对实际使用的标记，在 `1-segment/review.json` 核对分件结果，在 `report.json` 核对拟合情况。若工具要求确认归属不明的三角面，回到原扫描检查候选范围，再传入匹配的决定和证据。裸身模型为 `3-skin/body.glb`，穿衣模型为 `4-garments/character.glb`。

完成[原始头部组件](../skills/charloom-face-dialogue/SKILL.zh-CN.md)后，复测并播放两份最终导出。各阶段的输入和完成条件见[端到端技能](../skills/charloom-mh/SKILL.zh-CN.md)。
