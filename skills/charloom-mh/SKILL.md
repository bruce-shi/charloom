---
name: charloom-mh
description: Charloom 扫描角色制作的主流程和技能索引。从原扫描构建可换装、可动画的 MakeHuman 角色，或定位已有角色需要重做的步骤。
---

# 从原扫描制作可换装、可动画角色

主流程规定依赖顺序和每步完成条件，小 skill 提供该步的操作细节。新角色从第 0 步执行；局部修复先确认 Gallery 实际加载的文件、源扫描和问题帧，再重做受影响步骤及下游验证。已有资料只有在源文件、相机、姿势、尺度与本次输入一致时才能复用。

默认身体路线是：原扫描标定 → 遮挡区配准参考 → 身体量体 → MakeHuman 身体拟合及复测。头部路线是保留原扫描头、五官表面、UV 和贴图，在原表面绑定眼睑和嘴唇，再与身体接合。MakeHuman 提供身体拓扑。角色第一次绑定时确定一套固定骨架，记录骨名、层级、静止姿态、绑定矩阵和网格权重。先查角色已登记的绑定；已有绑定的角色始终沿用它。已有角色只追加动作时，从登记的裸身和穿衣成品开始。新动作映射到原骨架，并核对发束、衣摆或道具等角色专属骨链。[固定骨架与追加动作](../charloom/references/walk-replay.md#在同一角色骨架上增加动作)。

## 0. 确认输入、交付范围和环境

按[环境与输入准备](../charloom/references/setup.md)安装 CV Python、Bun、Blender；将 `PY` 指向该 CV 环境，并导出 `CHARLOOM_CV_PYTHON="$PY"`，使 Bun 调用的 Python 使用同一组依赖。

1. 找到任务指定的原始扫描 GLB，记录来源、绝对路径、文件 SHA-256、网格与材质结构。检查是否带蒙皮、动画、多 primitive 或多个贴图图集。需要预处理时保留原文件，记录静态化、合并或图集转换；guide 绑定实际送入流水线的静态文件。
2. 确认所需交付：裸模、穿衣版、可编辑 Blender 场景、动作以及是否登记 Gallery。完整角色制作包含原头保留、身体量体与复测；面部动作、装备和服装按角色及任务要求执行。
3. 生成适合角色的 MakeHuman 模板，记录所选体型参数与模板路径。复现已有步行时，按[步行动作溯源与复跑](../charloom/references/walk-replay.md)确认原动作、骨架、绑定入口和局部关键帧修正。在分派任务前核对提供的输入与原构建收据。新角色先选项目指定的角色骨架，再选动作素材；动作文件中的骨架只负责解释该动作。模板体型是初值，不能作为角色已经测量的证据。
4. 为本次 guide、测量、transfer、build 和 review 设置输出目录。保留阶段输入与报告，修订构建使用新目录，避免覆盖正在对比的版本。

设置本轮绝对路径变量：`SCAN`、`PRIOR`、`MOTION` 指向输入文件，`GUIDE`、`MEASURE`、`PARTS`、`TRANSFER`、`BUILD_PARENT`、`BLENDER_OUT` 指向本轮目录。`TRANSFER` 末级名称作为 `NAME`；输出目录不要重复使用。`MEASURE` 可以先创建，其各命令输出子目录保持不存在。后续命令均从仓库根目录执行。

**完成条件：** 实际原扫描、模板、动作和环境可读取；明确本次要走的身体、原头和服装路线。缺原扫描就停在输入准备，不用模板或合成角色替代原扫描完成交付验证。

## 1. 标定原扫描并识别表面结构

使用[扫描标定](../charloom-scan-guide/SKILL.md)。生成正、侧、后视图和必要的特写；在角色自身的坐标系区分解剖左右，标出肩、肘、腕、掌、髋、膝、踝、颈根及头部标记。核对镜头、像素尺度和视图朝向。

原头需要单独检查连接组件及贴图：区分面部皮肤、开放眼部薄片、真实独立眼球、睫毛、眉毛、头发和口腔碎面。在标定图上描实际唇缝和上下眼睑边界。开放眼部薄片不能直接作为球形眼球处理。

```sh
"$PY" scripts/scan-guide.py "$SCAN" --out "$GUIDE"
cp "$GUIDE/guidance.template.json" "$GUIDE/guidance.json"
```

按扫描标定 skill 编辑新 guide 的标记，细化相机并保存唇线后，再执行：

```sh
"$PY" scripts/scan-guide.py "$SCAN" --check "$GUIDE/guidance.json"
```

执行 guide 的 `--check`，并目检标记落点。此命令检查哈希并解析标注，不代表身体已经量体，也不能替代标记叠图和分件隔离图的检查。

**交付物：** 与源哈希匹配的 `guidance.json`、相机数据、标记证据图，以及原头构建所需的唇线和眼部标记。文件 SHA-256 与工具生成的 `source_hash` 分别保留，不混用。

**完成条件：** 标点确实落在对应解剖部位；可见皮肤、衣物外表面和遮挡部位已区分。视图、左右侧或源哈希不一致时先修正标定。

## 2. 为遮挡身体取得配准量体参考

使用[身体量体](../charloom-body-measure/SKILL.md)。可见皮肤直接测原扫描；衣甲遮挡了要制作的裸身时，配准的紧身衣参考是本路线的必需输入。已有有效参考可复用；没有就按该 skill 生成同一角色、姿势和相机的参考，再做配准。图片服务只在需要生成或编辑参考时使用，其依赖见[图片服务准备](../charloom/references/setup.md#图片服务依赖)。

按[生成、配准与标记量体图](../charloom-body-measure/references/measurement-plates.md)制作正侧参考。侧面量躯干时，手掌竖直贴大腿外侧，指尖朝下，小腹前缘完整可见。遮挡时用同一相机另生成此姿势的侧图，配准后只更新可见的躯干测点。

逐视图检查头、手、脚、朝向、解剖高度及轮廓对应关系。读取 `views/bodysuit.json` 的接受状态、尺度、位移、锚点和覆盖区域；同时检查实际图像，不能仅凭 `accepted` 为真就通过。局部手脚或身体比例被生成图改变时，修正该区域的参考或排除不可靠数据。

```sh
"$PY" scripts/mh-bodysuit.py "$GUIDE" --edit
"$PY" scripts/mh-suit-contours.py "$GUIDE"
```

第一条读取原标定图并生成、配准紧身衣量体参考；检查生成收据、实际正侧参考和密集端点后才运行第二条。已有相同源和相机的有效缓存时复用缓存，不重复提交付费任务。

从通过检查的参考提取轮廓，保存选用的 `suit_contours.json` 或 ensemble 文件及生成收据。自动采样的轮廓、行宽和比例只是量体输入，下一步仍须建立逐部位的目标数值与测量路径。

**完成条件：** 所需正侧视图已配准，覆盖本次要估计的身体区域；生成参考中的隐蔽人体形状明确标为估计。参考缺失或失配时，不进入裸模拟合。

## 3. 先量体并保存目标，再开始拟合

执行[测点、路径和导出复测方法](../charloom-body-measure/references/surface-measurements.md)。这一阶段的目标来自原扫描和已配准参考，必须在第一次身体拟合之前保存，不能从拟合结果反推目标。

具体输入按[量体脚本输入](../charloom-body-measure/references/script-inputs.md)填写。先运行坐标和密集点命令：

```sh
"$PY" skills/charloom-body-measure/scripts/working_frame.py "$SCAN" --prior "$PRIOR" --guide "$GUIDE" --out "$MEASURE/frame.json"
"$PY" skills/charloom-body-measure/scripts/outline_points.py "$GUIDE" --frame "$MEASURE/frame.json" --out "$MEASURE/outline"
```

然后用经分件确认的可见皮肤面、已配准人体 mask 和双视图解剖中心构成 `tape-spec.json`，运行：

```sh
"$PY" skills/charloom-body-measure/scripts/surface_tape.py "$MEASURE/tape-spec.json" --out "$MEASURE/tape-reference"
```

需先取得皮肤分件范围时，先执行第 4 步，它不依赖身体拟合。路径规格应覆盖双侧外臂、外腿及腋下大臂内侧六条路径；检查分段长度和实际叠图。截面目标使用相同局部三维轴的图像投影，定义留在 `sections-spec.json`，先输出目标数值与端点图，再在第 5 步与真实网格截面比较。

```sh
"$PY" skills/charloom-body-measure/scripts/mesh_sections.py "$MEASURE/sections-spec.json" --out "$MEASURE/sections-reference"
```

### 固定尺度和证据来源

记录原扫描单位、相机像素尺度、源坐标到工作坐标的矩阵及尺度来源。当前 `Scan.working_frame(prior, front_view)` 按模板身体高度缩放扫描包围盒，可在运行 transfer 前单独计算；换模板或变更缩放后必须重新换算测量值。检查头饰、鞋底等是否改变包围盒高度。没有已知现实身高时，毫米表示工作模型的尺度，不冒称真实人体的物理尺寸。

每个目标项保留名称、左右侧、姿态、测点或路径、数值、单位、来源图/三角面及误差范围。证据分为：原扫描可见皮肤三维实测、衣物外表面实测、参考图二维投影测量、基于配准视图的隐藏身体估计。衣甲外壳不能记成裸身皮肤；单个视图不能宣称获得了完整三维表面路径。

### 必需的量体项目

| 部位 | 需要记录的目标 |
| --- | --- |
| 左右臂 | 肩尖—外肘—外腕的表面路径及分段长度；**腋窝下缘—肘内侧的大臂内侧沿体长度**单独测，左右各一条，不用肩肘骨长代替；肩肘腕关节链长另列；腕到可见指尖长度、肩到指尖垂直落差、手掌宽厚和方向。 |
| 左右腿与脚 | 侧髋—外膝—外踝的表面路径及分段长度；髋膝踝关节链长另列；髋到脚底高度、脚长、脚宽及朝向。 |
| 躯干和颈部 | 肩到髋的长度，肩宽，胸、下肋、腹、腰、髋和颈根的截面位置、中心、宽与深；同高度的左右臂腰间隙。 |
| 肢体局部截面 | 左右上臂、肘、前臂、腕、大腿、膝、小腿、踝的截面位置、中心、宽与深。截面垂直局部肢体轴，不能用画面水平宽度冒充。 |

在原扫描可见皮肤上绑定可重现的三维路径或截面；对遮挡裸身，从正侧参考建立估计，记录对应点和深度假设。区分沿表面的弧长、关节折线长、直线距离和垂直落差。没有足够证据的项标为未测，不能填入模板值伪装成测量。补充视图或建立有依据的估计后，才算覆盖该项。

**交付物：** 测量表、机器可读的点/路径/截面数据、标注图、坐标换算说明。按[量体脚本输入](../charloom-body-measure/references/script-inputs.md)运行 `working_frame.py`、`outline_points.py`、`surface_tape.py`、`mesh_sections.py`；保留原密集轮廓，不压缩成少数关节。应能从原始证据重算表中的值；每个必需部位均有实测或明确标注的估计及误差范围。

**进入拟合的条件：** 已逐项检查上述数据，目标与原图对应，单位一致，缺项已解决。生成了参考图、提取了轮廓或标好了关节，都不等于完成此步骤。若任务已委派，执行 agent 先提交这些文件及实际数值，主 agent 查阅证据后再放行拟合，不以口头进度替代检查。

## 4. 复核原扫描分件归属

在身体和衣物导出前按[服装分件](../charloom-garment-separate/SKILL.md)隔离查看原扫描组件及细小面片。将脸、颈、原手碎片、头发、眉毛、内部碎面、衣服和独立装备的归属绑定到 `source_hash`。共面或粘连部分按实际材质与几何边界细分。

```sh
"$PY" skills/charloom-garment-separate/scripts/review_parts.py "$SCAN" --guide "$GUIDE" --out "$PARTS"
```

逐个检查 `components.png` 和组件目录的正侧后图，将复核决定写到独立的 `decisions.json`；模板中的空 action 不能直接提交。复放命令及混合组件细分见分件 skill。

**交付物：** 分件证据、经检查的 `decisions` 及对应原三角面范围。使用 `--segment-evidence`、`--segment-review` 将决定传给 transfer。命令要求补充分件判断时完成复核再重跑，不能把所有未明面片默认塞进衣服层。

## 5. 拟合身体，并用第 3 步目标复测

使用[MakeHuman 裸模拟合](../charloom-body-fit/SKILL.md)，输入原扫描、模板、guide、已选轮廓、量体目标及分件决定。先按第 0 步确认动作由哪套骨架驱动。原构建保留步行源骨架时，执行[源骨架承载动作](../charloom/references/walk-replay.md#源骨架承载动作)的步骤一次性建立固定骨架。静态 transfer 省略 `--motion`，复测身体后完成原头与衣物装配，再用 `attach_fixed_rig.py` 将两层接到固定骨架，执行最终复测。下面的 transfer 命令用于由 MakeHuman 模板骨架承载动作的路线。

```sh
"$PY" scripts/scan-transfer.py "$SCAN" --prior "$PRIOR" --guidance "$GUIDE/guidance.json" --out "$TRANSFER" --motion "$MOTION" --mouth-trace "$GUIDE/mouth-trace.json" --segment-evidence "$PARTS/components.json" --segment-review "$PARTS/decisions.json"
```

若标点变形后侧面厚度明显塌缩，按[标点变形与截面修形](../charloom-body-fit/references/section-refit.md)核查变形场，针对正面标点可区分的姿势选择保留深度的模式。

首次导出后按测量 skill 为实际裸模建立哈希绑定的 `body-model.json`、`sections-model.json`，选择真实身体 primitive 和部位面，再执行：

```sh
"$PY" skills/charloom-body-measure/scripts/outline_points.py "$GUIDE" --frame "$MEASURE/frame.json" --model-spec "$MEASURE/body-model.json" --out "$MEASURE/outline-export"
"$PY" skills/charloom-body-measure/scripts/surface_tape.py "$MEASURE/tape-spec.json" --model-spec "$MEASURE/body-model.json" --out "$MEASURE/tape-export"
"$PY" skills/charloom-body-measure/scripts/mesh_sections.py "$MEASURE/sections-spec.json" --model-spec "$MEASURE/sections-model.json" --out "$MEASURE/sections-export"
```

`body-model.json.regions` 要同时覆盖密集轮廓 group 和六条路径名称。需要修形时，按[完整身体及局部截面修形](../charloom-body-fit/references/section-refit.md)执行 `refit_sections.py`，用同拓扑的完整模板恢复胸廓、调整局部肢体，再用 `copy_body.py` 将通过复测的身体带入本轮裸模和穿衣文件。每轮使用新目录并重算对应文件摘要。

理解当前程序的实际能力：`fit_template` 用解剖标记对模板和骨架作空间变形；`fit_body` 用配准视图的轮廓约束身体，并通过平滑项保留初始形状。它们不自动读取第 3 步完整测量表，也不自动阻止量体缺失。未被充分约束的形状仍受模板和正则项影响，首次导出必须按量体目标修整和复测，不能当作完成品。

1. 检查 `report.json` 的源哈希、模板、工作坐标、`stage2_body_fit.contours` 和 `registered_views`，确认实际使用了本次输入。当前代码优先读取存在的 ensemble 文件，必须确认没有误用旧轮廓。
2. 在实际 `3-skin/body.glb` 上重绑对应测点和表面路径，按第 3 步相同定义复测。用 `mh-check-lengths.py` 检查它支持的双侧长度；截面和臂腰间隙另按量体方法计算。记录目标、实际值、带符号误差与预先选定的容差。
3. 按原标定相机叠加正面、侧面轮廓，并查看斜侧面。逐部位核对胸腹厚度、肩腋连续性、颈根、腕掌、指尖和脚掌。全身轮廓误差的中位数不能代替局部尺寸或表面质量。
4. 用无贴图材质检查几何凹折，再用原材质检查肤色与纹理。衣甲颜色投到了裸身、颈肩断层、手脚比例错误或超差，都要回到对应输入或形状修正；每次修正后复测受影响项目。不能放宽容差来掩盖偏差。
5. 播放完整行走及肩、肘、膝等相关动作，检查表面长度、截面和腋下变形。骨骼长度一致不代表皮肤尺寸一致。

**完成条件：** 裸身目标尺寸在可解释的误差范围内，局部轮廓和几何检查通过，运动中没有尚未解决的明显变形。身体还未通过时，后续装配仅能作为诊断，不能记为完成交付。

**条件替代：** 只有原扫描确实存在可分离的躯干和四肢五个皮肤壳体，且任务适用时，才进入[扫描壳体装配](../charloom-scan-volume/SKILL.md)。先检查壳体证据；该分支仍需量体和导出复测。身体路线的选择不改变保留原扫描头的要求。

## 6. 接合原头并绑定五官

按[下颌与颈部](../charloom-body-fit/references/neck-and-jaw.md)接合原头与已通过复测的身体，再按[面部与说话](../charloom-face-dialogue/SKILL.md)和[构建命令](../charloom-face-dialogue/references/face-build.md)绑定原扫描表面。

默认使用 `bind_scan_head.py` 或 `api.build_character(..., head_binding="scan-surface")`。核对报告中的 `route=scan_surface_binding`、源哈希与原扫描对应误差。中性脸的可见表面、UV、贴图应保留；唇缝分割、眼睑和口腔内壁随实际边缘运动。重建眼球或解剖脸的 `anatomical` 路线仅在任务明确需要时选用。

本轮 transfer 保留原头所需源资料和 `mouth-trace.json`。设置目录并执行：

```sh
export CHARLOOM_TRANSFER_DIR="$(dirname "$TRANSFER")"
export CHARLOOM_HEAD_DIR="$BUILD_PARENT"
export CHARLOOM_CV_PYTHON="$PY"
NAME="$(basename "$TRANSFER")"
"$PY" scripts/roster/bind_scan_head.py "$NAME" --config "$CONFIG"
bun scripts/roster/embed-clips.ts "$NAME"
bun scripts/roster/verify-face.ts "$NAME"
```

`CONFIG` 按面部 skill 指定原面部、眼部和保留面片的源面范围，以 `NAME` 为键；无额外分件配置时才省略 `--config`。输出为 `$BUILD_PARENT/$NAME/3-skin/body.glb` 和 `4-garments/character.glb`，同时生成原表面对应与面部动作报告。

需要面部动作时嵌入并验证动作，在裸模与穿衣版都检查：中性脸、完整眨眼、视线变化、M/B/P 闭唇、F/V 接触、最大张口和转头。保存动作时间、近景/侧视图及几何结果。检查闭唇同时是否有唇部挤穿、眼睑翻折、口内面片外露和颈缝，不以单个数值通过代替外观检查。

**完成条件：** 已核验实际执行入口与原头对应关系；所需五官动作和头颈接合均通过检查。

## 7. 适配衣物、装备与必要纹理

按[衣料与装备适配](../charloom-garment-fit/SKILL.md)，以最终身体为基准调整尺寸和绑定。护具分别量近端、中段和开口截面；刚性件用对应挂点，布料和自由面板按各自运动方式绑定。检查袖口、手套、头发、衣领和道具是否存在遗落面片或错误骨骼归属。

衣物的分件名称、层次、`rig` 和 `fabric` 写在 guide 的 `garments` 项中，首轮由 transfer 读取；其确切字段与重绑入口见[服装执行方法](../charloom-garment-fit/references/build-and-check.md)。身体后来发生修形时，对最终装配做局部适配，保存新输出，不重新跑首次拟合覆盖已验身体。

在裸模、纯衣服、穿衣版间查看同一帧，确认衣服层不残留旧手指、指甲或皮肤碎片。对静止、完整步态、抬臂和弯曲姿势记录最差穿插帧并修正。完整裸模必须保留衣下身体，不能直接把穿衣文件中裁过的身体当作裸模。

换装仅在任务要求时执行；纹理错位、黑面或缺纹理时按[服装纹理修复](../charloom-garment-texture/SKILL.md)处理，优先保留可见原扫描表面与 UV。补绘范围和服务收据需可追溯。

**完成条件：** 分件、佩戴尺寸和运动正确；身体调整引起的衣物变化已复查，未通过的穿插或纹理问题仍明确列出。

## 8. 导出可编辑场景和最终动作文件

按[Blender 构建与动作](../charloom/SKILL.md)导入已检查的构建，保存可编辑场景并导出任务所需 GLB。若集成构建重新生成了身体或头部，核对它实际使用的输入和输出，重跑被重建部分的复测，不能默认保留手工修正。

复现步行时，在裸模和穿衣版分别执行[步行轨道比较](../charloom/references/walk-replay.md#检查实际写入的动作)，对照原构建逐项解释轨道与静止骨架差异，再检查原姿态的腕掌朝向和脚底高度。走路出现鞋底穿地或漂浮时，按[步行接地](../charloom/references/walking-ground.md)测量并修正实际穿衣动作，把同一根平移写入两层，以下命令使用修正后的构建目录。

已完成的 GLB 直接导入，避免重新拟合：

```sh
"$BLENDER" -b --factory-startup --python-exit-code 1 --python scripts/blender_character.py -- "$BUILD_PARENT/$NAME/4-garments/character.glb" "$BLENDER_OUT" --clip walking --export-glb
"$PY" scripts/animate.py validate "$BUILD_PARENT/$NAME/3-skin/body.glb"
"$PY" scripts/animate.py validate "$BUILD_PARENT/$NAME/4-garments/character.glb"
"$PY" scripts/animate.py validate "$BLENDER_OUT/character.glb"
```

对最终裸模、穿衣版以及额外 Blender 导出分别执行 `scripts/animate.py validate`，核对动作名称、时长、骨架、蒙皮、面部 morph 与材质。保存源文件、构建配置、报告及最终文件哈希。

**完成条件：** 最终文件与被测版本一致；可编辑场景可打开，所需动作实际可播放。命令成功退出只证明执行完成，不代替量体、动作和视觉检查。

## 9. 在 Gallery 核对实际交付

按[导出与 Gallery 验证](../charloom-gallery-review/SKILL.md)先预览登记，再同步到本次本地 Gallery。打开实际角色 URL，检查请求的 GLB 是否为刚验证的文件。

```sh
python3 scripts/mh-ide-sync.py "$BUILD_PARENT" --dry-run
python3 scripts/mh-ide-sync.py "$BUILD_PARENT"
(cd viewer && bun run dev)
```

使用同步输出的实际条目和 URL，必要时复用正在运行的 viewer。若最后在 Blender 修改了 GLB，先把该文件登记到本轮明确选定的条目并核对来源，不能仍让 Gallery 加载早期 build。

在暂停的非零动作时间，保持相机和动作不变，切换裸模、纯衣服、穿衣版，核对时间保持和衣下身体完整性；继续检查播放中切层。实际播放行走、面部及任务所需动作，从正、侧、后和问题部位近景复查。保留 URL、文件哈希、动作时间和截图。

**交付记录：** 列出量体目标与复测结果、实际身体/原头路线、裸模与穿衣 GLB、Blender 场景、Gallery 实际加载情况，以及仍存在的问题。内部检查、运行时验证和用户视觉认可分别记录；用户尚未审阅时，不标为用户通过。

## 中断、修订与委派后的核验

以最后一个有证据通过的阶段继续。修改扫描会使哈希绑定资料失效；修改相机或尺度要重算配准与量体；改变身体拓扑要重绑复测路径；修改身体形状要重做受影响的衣物和动作检查。测试案例、临时目录、失败日志与角色专用参数保存在任务产物中，不写进通用 skill。

任务采用 subagent 时，主 agent 必须核对实际执行路径：原始输入及哈希、拟合前测量证据、构建入口与配置、最终加载文件。阶段汇报要给可查的文件和实际结果。发现跳过依赖、替换输入或采用不同头部路线，先纠正并重做受影响步骤，再继续下游。
