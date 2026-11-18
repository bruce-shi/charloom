# 完整胸廓和局部肢体的可重放修形

[English](section-refit.md) · 简体中文

首次导出出现正面吻合、侧面呈薄片时，分别渲染完整模板、仅标点变形、加轮廓变形的裸模侧视图，比较相同部位的前后厚度。若在标点变形阶段已变薄，检查前视图中双侧标点可区分，输入和目标相机方向一致，深度标记主要约束各关节中心。此时可在本次 guide 设置 `subject.template_warp` 为 `frontal_depth_preserving` 后重新运行 transfer。这一模式用模板的正面 XY 位置插值三轴位移，同一 XY 处的前后表面保留原模板厚度，再按真实截面目标修形。侧身、重叠或需要前后独立变形的姿势使用默认 `spatial`。从 `report.json` 的 `stage2_rig.template_warp` 核对执行模式，逐项复测关节、轮廓和截面。

`scan-transfer.py` 先用标记拟合骨架和完整 MakeHuman 表面，再用配准轮廓约束可见投影。首次输出仍需逐部位复测。若胸廓被压扁、出现双峰或肩臂被一起拉歪，按以下方式从原完整模板恢复；不在已经折坏的胸口反复追单侧轮廓。

## 准备有来源的截面映射

1. 保存本轮裸模、原模板和全部测量目标。选原模板对应的 body primitive；拟合后的身体必须保留该模板顶点顺序，面可以是原面集合的子集。经过重拓扑的身体需要另做对应关系，不能使用本脚本。
2. 躯干选腰、下肋、胸峰、肩、颈根及颈上端等同名高度。在原模板、参考正面和参考侧面分别记录位置，先解决两视图解剖高度不一致。截面目标包含两侧边界与中心，不能只有宽度。不要把原模板胸峰与参考的另一高度强行叠加。
3. 肢体以各自肩肘、肘腕、髋膝、膝踝为轴。记录实际截面站位；从模板相同解剖位置切面保留其形状，以目标宽深和中心映射。上臂内侧长度另用腋窝—肘内侧路径约束；肩肘链长不能代替。
4. 给每个操作建立一维 `blend.npy`：长度等于模板顶点数，0 表示保护，1 表示使用该操作结果，中间值平滑过渡。可从已审阅的骨骼权重区域开始，在 Blender 顶点组里检查并调整，再导出权重；保留前胸、颈缝、手掌和无关肢体保护区。各操作同顶点权重之和不得超过 1。不能仅按高度选一整圈，把胸廓也归入上臂。

## 配置与执行

```sh
"$PY" skills/charloom-body-fit/scripts/refit_sections.py "$FIT_CONFIG" --out "$BODY_REFIT"
```

JSON 顶层 `mesh` 是待修裸模，`native_mesh` 是完整模板；两者格式见[量体输入](../../charloom-body-measure/references/script-inputs.zh-CN.md#2-通用模型与面选择记录)。原模板可用 `world_from_model` 显式变换，待修裸模只允许未经额外变换的 rest。

`operations` 每项包含：

- `name`、`blend:{file,sha256}`。
- `native_origin_m`、`target_origin_m`；`native_basis`、`target_basis` 为 3×3 **列向量** `[轴,宽方向,深方向]`，必须正交且右手。肢体用自己的局部轴；躯干用沿身高的轴。
- `native_region`：只供切截面的原模板面选择，格式同量体 `regions`。
- `stations`：按解剖顺序排列，每项含 `native_axis_m`、`target_axis_m`（距各自 origin 的轴向米数），`target_bounds_m:[[宽下界,深下界],[宽上界,深上界]]`，以及 `evidence`（目标来自哪份量体文件、站位、实测或估计）。不可见深度要记录估计依据，不能默填。
- `native_seed_m` 可在 station 中指定原截面内部点。`native_bounds_m` 自动由真实三角面闭环计算，若提供则核验与实际切面相符。肩顶切面开口时，在闭合且可对应的位置设站；不能拿开口片段的最大范围当完整截面。

原模板轴向站位覆盖所有非零 blend 顶点，超出范围会报错。脚本按解剖顺序做单调高度映射，横向按成对边界映射原截面形状；全部操作写到同一完整皮肤，保留顶点顺序、UV、骨架和其他物体。`displacement_fairing` 可对焊接表面的位移做拉普拉斯平滑，默认 0；启用后必须再次复测，因为平滑会改变尺寸。

输出 `body.glb`、`refit.json`，后者包含源摘要、采用的实际模板截面、改动顶点数和最大位移。状态始终为待复测；这不是自动验收器。依次跑密集轮廓、外臂/外腿/腋下内臂路径、局部截面、手脚长度；查看无贴图正侧斜视，再检查整段动作。

## 将通过复测的身体带入同一装配

身体在完成原头装配前修形时，将修改版作为该轮 transfer 的裸模，并用下列命令更新同一骨架的穿衣导出。先复制整个阶段目录到新修订目录，保留旧版；用新文件名写出后，再在新目录内替换对应文件。

```sh
"$PY" skills/charloom-body-fit/scripts/copy_body.py "$BODY_REFIT/body.glb" "$TRANSFER/3-skin/body.glb" --out "$REVISION/body.glb"
"$PY" skills/charloom-body-fit/scripts/copy_body.py "$BODY_REFIT/body.glb" "$TRANSFER/4-garments/character.glb" --out "$REVISION/character.glb"
```

脚本核验骨名、rest 变换、顶点顺序和三角面对应，保留目标文件的面集合、材质、其他物体与动作，把修后皮肤和权重变换进目标坐标。裸模仍保留完整衣下身体；穿衣版可保留自己的遮挡裁面。不要把穿衣裁面复制回裸模。

衣物还需按最终身体检查和重新适配；这一步只保证身体没有在下游悄悄恢复旧版。修形若改了骨架或拓扑，重做 transfer/头部/服装对应，不能强行移植。

## 固定胸廓的抬臂检查

```sh
"$PY" skills/charloom-body-fit/scripts/check_pose_regions.py "$POSE_CONFIG" --out "$POSE_REVIEW"
"$PY" scripts/animate.py validate "$POSE_REVIEW/pose-check.glb"
```

配置 `mesh` 使用相同哈希模型记录，`reference_bone` 指实际胸廓骨。`regions` 分别保存胸内侧、外侧胸肌到腋前、大臂过渡区域：每项 `vertices` 为已审阅的身体顶点 ID，`evidence` 为选区图与说明。不要只选胸骨中心代替外胸和腋前。

`poses` 每项是 `name`、`bone`、`axis_world:[x,y,z]`、`degrees`。例如在 GLB Y-up 坐标中，左右上臂分别绕世界 Z 轴向外抬起，角度可从 48° 开始；符号根据实际骨架和姿态确认，骨名从实际导出读取。`clips` 指要采样的完整动作，`samples_per_clip` 默认 33，增加问题帧附近采样。

输出每个区域在胸廓局部坐标中的最大/95分位位移、边长增长、最大伸长比及最差边，另写带独立抬臂动作的诊断 GLB。原动作保留，诊断片段 0 秒是 rest，1 秒为指定抬臂；把此文件单独导入 Blender 或 Gallery 复查斜侧腋下。该文件只作诊断，不替代最终动作文件。

若 rest 正常而抬臂时胸肉被拖走，检查该区域的上臂/肩骨影响。已有 `scripts/skin.py` 提供版本化权重修复：

```sh
"$PY" scripts/skin.py start "$BODY" --config "$SKIN_CONFIG" --out "$SKIN_SESSION"
"$PY" scripts/skin.py step "$SKIN_SESSION" --plan "$SKIN_PLAN"
"$PY" scripts/skin.py inspect "$SKIN_SESSION"
```

plan 需 `expected_round` 与 `diagnosis`，`strategy` 可选 `exclude_influences` 去除已证实错误的骨影响，或 `relax_strain` 平滑局部过渡，或 `weld_seams` 对齐真实接缝权重。`regions` 使用 `mask:{mesh,geometry_hash,vertices}` 限定选区；排除项是该 region 的 `exclude_joints`，保留项可用顶层 `protect_joints`。`max_weight_change_l1` 限制改动，不能把胸廓全部硬绑后忽略腋前过渡。读取 session 实际采用的 round 文件，重新量体、抬臂和检查服装；权重修复不是 rest 身体尺寸修形器。

## 核对并修正腕掌朝向

在同姿正侧图中标腕心和同一指尖，保存工作坐标方向和腕到指尖的直线距离。对比裸手与原手套的正侧近景。另姿势的竖掌侧图仅用来量腹缘。如果误差来自手指单独弯曲，要检查各指段的姿态；只有整只手的静止朝向或纵向尺寸偏差才使用下列脚本。

在身体修形完成、重新绑定行走之前建立 `hand-rest.json`。填写 `body_node` 和 `body_positions_sha256`（完整身体 POSITION 属性按小端 float32 连续排列的 SHA-256），在 `evidence` 中给出测量文件的 `file` 和 `sha256`。`hands` 每项填实际腕骨名 `wrist`、裸身上的指尖顶点号 `tip_vertex`，以及同姿参考中从腕到该指尖的三维向量 `wrist_to_tip_m:[dx,dy,dz]`。脚本只在腕部骨骼子树内调整朝向和沿指端轴的长度，按蒙皮权重过渡到前臂。检查拇指和掌心旋转；两端点不能决定绕轴的扭转。

```sh
"$PY" skills/charloom-body-fit/scripts/fit_hand_rest.py "$TRANSFER/3-skin/body.glb" "$HAND_CONFIG" --out "$NEXT/3-skin/body.glb" --rebuild-clip walking
"$PY" skills/charloom-body-fit/scripts/fit_hand_rest.py "$TRANSFER/4-garments/character.glb" "$HAND_CONFIG" --out "$NEXT/4-garments/character.glb" --rebuild-clip walking
```

两个输入使用同一套身体顶点和骨架。输入尚无行走时省略 `--rebuild-clip walking`。若指定该参数，新文件会去掉需要重新定向的旧动作；从原动作和原映射对两份新 GLB 重新 retarget，再进行原头装配。检查两层腕骨、指骨静止位置与蒙皮，复测腕到指尖的向量、手宽厚、肩到指尖落差和衣袖开口。对穿衣版单独隐藏身体，判别浅色指尖来自手套材质还是露肤。
