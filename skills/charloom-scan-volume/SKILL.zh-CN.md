---
name: charloom-scan-volume
description: "用 Charloom 原扫描中独立的躯干和四肢壳体构建 scan-volume 裸模。适用于已确认五个可分离皮肤壳体的角色。"
---

# 扫描体积主体

[English](SKILL.md) · 简体中文

使用此可选装配体，并采用同一扫描得到的观察体积头部。构建者将躯干和四肢合并为一个整体，并沿它们的实际交点与头部相连。输出结果保留了骨骼、独立的躯干部分以及可测量的手部关节结构。皮肤具有来自扫描的拓扑结构，并且没有使用MakeHuman的体型变形功能。

## 选择表面

从该角色的已完成的校准传输目录开始。其`report.json`标识了源、先前和工作框架。`0-guide/resolved.json`提供了面部标记和测量的关节部分。组装工具会读取两个阶段的GLB以及匹配的`mouth-trace.json`。当选择了正确的身体尺寸时，也会读取`body-selection.json`。使用传输过程中使用的静态、单网格、单一原语纹理扫描。一个`raw_head`文件或匹配的列表配方，它提供了准备好的头部体积，可以选择不兼容的头部输入。

在**原始扫描图**中，分离出躯干和每个完整的肢体。执行代理根据表面、它们的附件以及校准后的视图来确定所有权。在图库接缝处焊接后，计算连接的面组件数量。每个外壳对应一个**原始原语索引顺序中的零基三角形索引**。这五个索引必须对应五个不同的组件。单个连续的全身网格不符合此构建器的输入要求。如果所有权不明确，请在分配种子之前检查原始三角形。

按字符名称为配置项赋值。在`body_volume`下，提供转移的`source_hash`和一个`skin_seeds`对象，每个`torso`、`arm_l`、`arm_r`、`leg_l`和`leg_r`对象都使用一个已审核的三角形索引。当近端外壳在衣物下方并留下一个测量出的间隙时，`pelvis_fill: proximal_hull`桥梁将填补该间隙。检查推断出的表面与服装和裸露的轮廓是否匹配。身体必须与头部形成一个具有正体积的重叠部分，形成一个连续的整体。

## 组装和验证

在检查器中，将 `PY` 设置为与项目 CV 依赖项一起使用的 Python 解释器。将 `transfer`、`result` 和 `config` 分别设置为当前已校准的传输目录、新的输出目录以及已审核的配置文件。

```sh
"$PY" scripts/roster/build_blender_character.py "$transfer" "$result" --config "$config"
"$PY" scripts/animate.py validate "$result/3-skin/body.glb"
"$PY" scripts/animate.py validate "$result/4-garments/character.glb"
```

在 `$result/report.json` 中确认 `body_volume.surfaces` 包含已复核的五个壳体，且 `neck_skin_fit.method` 为 `shared_boolean_boundary`。模板身体接体积头时也可能出现 `route` 值 `free_topology`。`body_volume.source_crossings` 描述输入壳体的相交情况，导出的皮肤需另外测量。

比较来自相同摄像机的源模型、拟合身体模型和体积身体模型，在颈部、肩部、手腕和脚处进行测量。播放导出的行走过程，使其完成一个完整的周期。检查配对的颈部顶点是否分离，以及附近的非相邻面是否存在交点。测量手腕处的局部边缘拉伸情况，并检查在最大屈曲状态下的手指和脚趾。没有支撑手指的手部测量数据跟随手腕。检查哪些手指会实际移动，眨眼和对话，以及两种导出结果中的服装间隙。将剩余缺陷与结构验证和画廊采用情况分开记录。
