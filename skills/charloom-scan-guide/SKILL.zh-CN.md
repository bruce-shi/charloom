---
name: charloom-scan-guide
description: "标定 Charloom 原始扫描的相机、人体标记、分件边界和唇线，生成与源 GLB 哈希匹配的 guide。"
---

# 标定原始扫描

[English](SKILL.md) · 简体中文

输入为原始带贴图 GLB。原头构建器使用静态三角网格和单张 UV 图集。从仓库根目录将 `PY` 指向已安装项目 CV 依赖的 Python，并将 `SCAN`、`GUIDE` 指向本次扫描与标定目录。如果已有 guide，先验证它是否属于当前扫描。

1. 新建时运行 `"$PY" scripts/scan-guide.py "$SCAN" --out "$GUIDE"`，把新生成的 `guidance.template.json` 复制为 `guidance.json`。
2. 在标定视图标出解剖左右侧、肩肘腕掌、髋膝踝、颈根、眼和嘴。标清裸露皮肤、头发及独立运动的衣物和配件。细小区域可用 `--zoom "$GUIDE/guidance.json" --view VIEW --box X0 Y0 X1 Y1 --name NAME` 放大。[标注字段](../../docs/scan-annotation.zh-CN.md)
3. 原头路线使用 `--refine "$GUIDE/guidance.json"` 细化面部相机，标出眼裂，用 `--trace "$GUIDE/guidance.json" --view VIEW --pixels POINTS.json --output "$GUIDE/mouth-trace.json"` 描出可见唇缝。这些参数均传给上一步的 scan-guide 命令，保留原扫描作为位置参数。
4. 执行 `"$PY" scripts/scan-guide.py "$SCAN" --check "$GUIDE/guidance.json"`。交给下游的资料包括原扫描哈希、相机、坐标变换、已核对的标记以及本路线需要的口缝线。

## 标记数量与用途

基础校验要求 26 个命名标记：`neck,nose_tip`，双侧各自的 `eye,mouth,shoulder,elbow,wrist,hip,knee,ankle`，以及双眼各自 `inner,outer,top,bottom`。模板另外列出 `palm_l,palm_r`；做完整身体时两掌也要标。它们只确定解剖位置，不能代替密集轮廓和沿体量体。

完整量体另标双侧 `axilla_l/r`、`elbow_inner_l/r`，分别为腋窝下缘的大臂内侧起点和肘内侧终点；与肩、肘关节中心分开保存。可见指尖、足跟、足尖、足底以及手掌/脚部宽厚需有成对端点；按当前角色添加命名，不填没有像素或面绑定证据的尺寸。身体和内臂量体至少有两个非平行视图，遮挡处注明估计依据。密集轮廓点数随可见范围与采样间距产生，不规定全角色仅测少数截面。

面部除眼裂四角，还沿可见上下眼睑各标三个中间点（两眼共 12 个），名称见标注字段；唇缝用有序像素折线，输出 `mouth-trace.json`。这些标记保持与原扫描和实际相机哈希对应。拟合前用[量体脚本](../charloom-body-measure/references/script-inputs.zh-CN.md)输出测量图，检查原像素落点而非只读数值。
