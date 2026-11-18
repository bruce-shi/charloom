# 可运行量体命令与输入

[English](script-inputs.md) · 简体中文

从仓库根目录运行。`PY` 是 CV 环境 Python。所有 JSON 路径相对于该 JSON 所在目录解析；推荐绝对路径。毫米均为工作模型尺度。所有新输出目录必须不存在。

## 1. 拟合前固定坐标和密集轮廓

```sh
"$PY" skills/charloom-body-measure/scripts/working_frame.py "$SCAN" --prior "$PRIOR" --guide "$GUIDE" --out "$MEASURE/frame.json"
"$PY" skills/charloom-body-measure/scripts/outline_points.py "$GUIDE" --frame "$MEASURE/frame.json" --out "$MEASURE/outline"
```

第一条只读取模板高度并计算 `Scan.working_frame`，不拟合人体。`frame.json.source_to_working` 是 **4×4 行优先**；transfer 报告的扁平矩阵是列优先，`outline_points.py` 也能直接读取该报告。不得把两种表示直接混用。

第二条读取已接受配准的 `views/bodysuit.json.marks`。命名为 `outline_trunkNN_w1/w2`、`outline_arm_lNN_w1/w2`、`outline_leg_rNN_d1/d2` 的端点两两配对；`w` 是正面宽度投影，`d` 是侧面深度投影。实际名称以原记录为准。`mh-bodysuit.py` 沿轮廓按约 10 像素间距取站，不是每部位只标一个点；图像尺寸和可见范围决定总点数。保留整批密集点，查看 `points-VIEW.png`，修掉把手臂、配件或背景当躯干的端点，再重算。输出 `outline-targets.json` 含每对像素、标定投影毫米、来源哈希；它不把斜截投影宽度宣称为真实截面直径。

可用 `--model-spec MODEL.json` 将最终皮肤投影到同一相机并逐对比较。MODEL 的 `regions` 必须覆盖输出里所有 `group`，如 `trunk`、`arm_l`、`arm_r`、`leg_l`、`leg_r`；别让躯干遮住手臂测量。

## 2. 通用模型与面选择记录

每个 `mesh` / `source_mesh` 是一个对象：

- `file`、`sha256`：实际 GLB 及文件摘要。
- `node`：GLB 中的节点名称，不是 Blender 显示名称或 mesh 索引。
- `primitive`：同名节点的三角面 primitive 序号，默认 0。
- `world_from_model`：可选 4×4 行优先变换。原扫描要填 `frame.json` 的矩阵；工作坐标中的导出通常省略。
- `triangle_sha256`：可选原三角面 `<i8` 字节摘要；用于阻止拓扑改变后复用选面。
- `clip`、`time`：可选动作与秒数；省略则读 rest。目标与导出必须同姿势。

`regions` 的每个值恰好选一种：`{"triangle_ranges":[[起始面,结束面]]}`（左闭右开），或 `{"joints":["实际骨名"],"minimum_mean_weight":阈值}`，或经确认的 `{"all_faces":true}`。局部手臂截面要隔离臂面；不能默认切全身后把胸廓也算入手臂。

模型复测记录还包含 `tolerance_mm` 和 `landmark_bindings`。每个中心绑定为 `{"triangle":面号,"barycentric":[a,b,c]}`，或 `{"vertices":[顶点号...],"coefficients":[系数...]}`。前者用于表面点，系数非负且和为 1；后者用于埋在皮肤内部的解剖中心，系数和为 1，可为负。绑定从已审阅的对应位置得到，不能把旧骨架中心直接当成新皮肤测量。`mh-check-lengths.py` 使用另一份点表格式，见脚本头部，不直接接受本记录。

## 3. 双侧沿体路径

```sh
"$PY" skills/charloom-body-measure/scripts/surface_tape.py "$MEASURE/tape-spec.json" --out "$MEASURE/tape-reference"
"$PY" skills/charloom-body-measure/scripts/surface_tape.py "$MEASURE/tape-spec.json" --model-spec "$MEASURE/body-model.json" --out "$MEASURE/tape-export"
```

`tape-spec.json` 字段：

- 完整角色设置 `scope:"full_body"`，脚本强制双侧外臂、外腿、腋下内臂六种覆盖齐全；局部修复可设 `selected_paths`，报告仍列完整身体缺项。

- `guide`、`source_hash`、`source_to_working`：本次 guide 目录、工具源摘要、4×4 矩阵。
- `source_mesh`：上述实际原扫描记录；其 `world_from_model` 使用同一矩阵。
- `reviewed_skin_triangle_ranges`：原扫描真正可见皮肤的面范围，来自分件复核；衣甲、手套、裤腿不得加入。
- `paths`：外臂、外腿四个定义，加第 6 节的两个腋下内臂定义。每项含 `name`（`arm_l` 等）、`kind`（`arm` 或 `leg`）、`side`（`l`/`r`）、`joints`（臂为 `shoulder_l,elbow_l,wrist_l`，腿为 `hip_l,knee_l,ankle_l`，右侧同理）、`view`、`reference_mask`、`max_radius_mm`，以及可选 `step_mm`（默认 1）。搜索半径要足以命中对应肢体而不跨到邻近人体部分。
- `reference_mask`：`file`、`sha256`、`kind:"binary"`，白色为已复核的人体区域，大小必须等于标定相机。也支持 `kind:"figure"` 加 `background_rgb`、`threshold`，但只能用于已检查背景颜色分离有效的参考图。不要把整张照片非白色区域自动当裸身。

各关节中心从 `guidance.json` 的至少两个非平行视图解算；观察不到的中心必须标注遮挡和估计来源。脚本先沿原扫描求真正射线命中，只有命中经审阅皮肤才采用；遮挡区改用同一条三维射线在已配准人体 mask 上的边界约束，保存为估计。禁止用一串正面点再线性填侧面关节 Z 值来伪造表面。

默认每段 45 个间隔，每条两段共 91 个采样点；这是数值采样密度，不是要求人工标 91 个关节。路径绕过肩端和肘外后缘；`recipe` 可覆盖 `bend_start,bend_span,elbow_width,shoulder_width,elbow_back,shoulder_up`，修改后须查看叠图确认仍沿同一解剖路径。保存原命中点和轻度平滑的量体曲线，肩/髋、肘/膝、腕/踝端点保持不动。输出逐点面号/重心坐标或参考像素、分段长度、总长及关节链长。

导出 `body-model.json` 使用通用 `mesh`、`landmark_bindings`、按四条路径命名的 `regions`、预先确定的 `tolerance_mm`。逐段误差和总长均需检查；总长相同不能抵消上下段相反偏差。

## 4. 拟合前截面目标与导出真实截面

先不提供模型保存目标：

```sh
"$PY" skills/charloom-body-measure/scripts/mesh_sections.py "$MEASURE/sections-spec.json" --out "$MEASURE/sections-reference"
```

核对目标和端点图之后开始拟合。导出复测：

```sh
"$PY" skills/charloom-body-measure/scripts/mesh_sections.py "$MEASURE/sections-spec.json" --model-spec "$MEASURE/sections-model.json" --out "$MEASURE/sections-export"
```

规格包含同一 `guide/source_hash/source_to_working` 和 `sections` 数组。每项：

- `name` 对应模型 `regions` 键；`axis` 为两个标记名，如 `shoulder_l,elbow_l`。
- `width_hint` 指定宽轴朝向；`frontal_normal` 默认 `[0,0,1]`。实际宽轴为肢体轴与正面法线的叉乘，深轴与两者垂直。
- `fractions` 是沿该段的截面位置，范围 `[0,1]`。覆盖三角肌、上臂中段、肘附近，以及前臂、腕、大腿、膝、小腿和踝；过渡变化处加密。不要把固定站数当成覆盖合格。
- `references.width` 和 `references.depth` 分别含 `view`、`mask`、`max_radius_mm`、`step_mm`；mask 使用上述记录。脚本投影的是同一条三维截面轴，不在侧视图另画一条不对应的垂线。
- 看不到的方向填 `{"unavailable":"具体遮挡原因"}`，输出为 `incomplete`。需要拟合的隐藏深度另记有来源的估计，不能用椭圆反算充当观测。
- `loop_seed_mm` 可选，表示截面局部宽/深平面内的已审阅内部点；默认轴心。轴心落在皮肤外时必须看实际截面选点，不能自动挑最大闭环。

真实网格通过三角面与平面求交；仅接受包含指定 seed 的唯一闭环。输出闭环三维坐标、参与的面、宽深、轴中心偏移、参考边界、两侧端点误差及叠图。开口、混入躯干、多闭环或边界超出图像时先修正选面与证据。

## 5. 手脚、躯干与复测

在 guide 添加可见指尖、脚跟、足尖、足底，以及腕掌和脚宽厚的成对端点。每个值保留端点与相机；不填写没有点证据的固定像素数。躯干密集宽深用第 1 步；胸、腰、肩、颈的同名高度分别在正侧图标出，先校正解剖高度再使用深度。臂腰间隙在相同高度分别切躯干和手臂，见 [surface-measurements.md](surface-measurements.zh-CN.md)。最终 `mh-check-lengths.py` 检查关节链、指尖与表面路径；截面仍由第 4 步复测。目标数据与导出数据分别保存。

## 6. 腋下大臂内侧长度，双侧独立

在四条外侧臂腿路径之外，增加 `upperarm_inner_l`、`upperarm_inner_r`。与外臂用同一个 `surface_tape.py` 命令，但定义为 `kind:"upperarm_inner"`，`joints:["shoulder_l","elbow_l"]`，`endpoints:["axilla_l","elbow_inner_l"]`（右侧改 `r`）。另填 `endpoint_tolerance_mm`，按相机与标点分辨率选定。

`axilla` 标在腋窝下缘进入大臂内侧皮肤的实际起点，`elbow_inner` 标在肘内侧的对应终点；不是肩骨中心、衣袖底或袖口。两点均需非平行视图观察或可复核的皮肤绑定。看不见的腋窝必须明确估计依据或标为未测，不能从肩膀减去一个固定数。

脚本沿肩肘轴定位两端的截面，向大臂内侧求皮肤交点，保留整条沿体曲线。若所得射线端点不在已标记腋窝/肘内侧容差内，直接失败，先改正选面或标点。导出记录需多加这两个表面端点绑定、相应 `regions`。输出独立内侧沿体长度、两端直线距离和端点偏差，不能与肩尖—外肘那一段混用。与外臂、外腿同样检查图像并复测最终 GLB。

## 7. 把已审阅位置绑定到导出皮肤

```sh
"$PY" skills/charloom-body-measure/scripts/bind_landmarks.py "$MEASURE/binding-input.json" --out "$MEASURE/body-model.json"
```

输入顶层为 `mesh`、`regions`、`tolerance_mm`，再加 `points` 字典。每个点填：`xyz_m`（在所选导出上实际定位的工作坐标）、`evidence`（定位图/原模板对应及理由）、`region`（明确的局部面选择）、`kind`。

- `kind:"surface"` 用于腋窝、肘内侧、指尖和足底等真实表面点，必须给 `max_snap_mm`。脚本在选定面上求最近点并保存三角面与重心坐标，超出该距离就失败。
- `kind:"embedded"` 用于皮肤内的解剖中心，按局部顶点做加权仿射绑定；默认 `neighbors:90`、`softness_m:0.008`。必须选该肢体的局部面，邻域退化无法表示三维中心时失败。输出保存原位置、重建位置和残差。

使用原模板对应的解剖中心时先核对顶点顺序及当前身体对应，随后保持绑定随皮肤复测。禁止把拟合前目标坐标直接复制为拟合后测点，再用仿射系数强行重建该目标；那只能证明系数解得出目标，不能证明身体尺寸正确。每个导出点都要有当前皮肤上的独立定位证据。

截面靠近肩关节或髋部时，肢体会自然并入躯干。导出复测会为缺乏唯一闭环的测站记录 `no_unique_closed_section`，保留其余测站结果，并以非零退出码提示缺项。检查选区和种子点；如果解剖结构确实无法独立成环，核对该站 `directed_endpoints_only` 下宽、深正负方向的真实网格命中、三角面与误差，列明未命中的方向。命中的端点可单独参与修形。不可用这组端点计算或报告闭环周长。
