# 原纹理投影、遮挡补图与写回

[English](texture-commands.md) · 简体中文

适用于绕颈的衣领、近圆筒形护具。普通平面布料保持其既有连续 UV；不要为了使用圆筒脚本强行改变整件布局。

1. 隔离原扫描中的目标材料，记录原三角面范围。设圆柱中心、工作坐标变换、上下高度；先生成仅含实际观察的圆筒图：

```sh
"$PY" skills/charloom-garment-texture/scripts/cylinder_chart.py "$CHART_CONFIG" --out "$CHART_OUT"
```

配置字段：`source`、`source_sha256`、4×4 行优先 `source_to_working`、`center_m`、原面 `triangle_ranges`、`size:[宽,高]`、`height_range_m:[下,上]`。可选 `height_radius_slope` 定义展开高度 `h=y-k*r`，用于斜落在肩上的衣领；默认 0。`subdivisions` 默认 2，以原三角形插值 UV 后细分投影。左右图边在背面接缝相接，角度为 `atan2(x-center.x,z-center.z)`。

输出 `observed.png`（透明处无观察）、`coverage.png`、`source-pixels.npz`（每像素源面及有效像素原面重心坐标）和 `chart.json`。观察图保留源色；头发或皮肤遮挡应在分件与补图 mask 中明确排除，不能把它们当衣领材质继续投射。

2. 把可见的同种布纹、边缝、滚边和金属链纹作为参考，标出实际被遮挡的区域。需要补绘时按任务已选定的 fal 图像编辑流程执行，使用显式 2K、medium 和缓存收据。提示词保留同一展开图的位置、尺度、左右接缝及可见装饰；只补缺失布料。保留编辑前图、mask、补图、参数与请求收据。没有服务授权时不提交付费任务，仍可完成投影与已观察纹理部分。

3. 在 Blender 为确实需要替换的面板建立连续 UV：沿衣领背缝切开，环向统一 u，沿下缘到开口统一 v；检查比例与实际补图一致。保留原扫描可见三角面的 UV，只有新面板或明确补绘区用新图集。细分出的原面以重心插值继承 UV，不能让每个新三角形各自投到一个不连续的小色块。设置纹理岛 padding，检查空衣领背面与顶边。

4. 目检并保存最终连续图集，再写入明确选定的 primitive：

```sh
"$PY" skills/charloom-garment-texture/scripts/apply_atlas.py "$ATLAS_CONFIG" --out "$TEXTURED_GLB"
"$PY" scripts/animate.py validate "$TEXTURED_GLB"
```

配置 `model:{file,sha256}`、`atlas:{file,sha256}`，`primitives:[{node:精确节点名,primitive:序号,uv_evidence:已核对的UV图路径与说明}]`，可选 `material_name`。该命令保留已有 UV/几何/权重与其他材质设置，只克隆选定面使用的材质并更换 base-color 图集；不自动重展开 UV，不把补绘图区错误地应用于整件原服装。

5. 在原问题相机检查图案尺度、背缝、翻边、黑面和邻层露底，播放行走与肩颈动作。对照原扫描可见纹理；注明哪些是观察保留、哪些是遮挡补绘。完成后记录最终 GLB 和 atlas 摘要，再进入 Gallery 同帧切层。
