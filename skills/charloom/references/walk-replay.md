# 复现已有角色的步行

从角色最近一次走路正确的导出倒查，记录原扫描、带动作的源 GLB 或 FBX、具体片段、实际蒙皮骨架、绑定姿态、初次绑定脚本和构建后的动作修正。复跑需要对应的输入文件、脚本与参数；只把最终 GLB 留作对照时，应单独交给复核方。相同角色外观和动作名称不能识别制作路线。

## 源骨架承载动作

1. 读取带步行动作的原始骨架、父子层级、局部旋转轴和蒙皮权重。保留源片段的时间、插值及旋转关键帧。记录 scan 到角色工作空间的矩阵，以及每个关节在该空间内的目标位置。
2. 用目标关节位置适配原骨架，同时保留骨骼静止朝向。由新的父子世界矩阵求各关节的局部静止平移。原片段若带局部平移轨，将新旧局部静止平移的差加到相应关键帧的值上。保留时间、旋转及缩放轨。重算 inverse bind matrices。
3. 将扫描顶点映射回动作源的静止体型以查找邻域权重，写回工作空间中的原扫描表面。已有绑定脚本的复现沿用同一空间变换、关节目标、邻域、权重混合和局部修正。独立核对绑定后中性姿态、原 UV、蒙皮和步态。
4. 后续采用 MakeHuman 身体表面时，将身体拟合到角色工作空间，把模板上的蒙皮关节映射到这套已有步行骨架。处理左右侧、脊背、肩臂、手指和脚趾的对应关系，并核对各部件的 bind transform。让原头、身体和衣物共用适配后的骨架。文件里出现 MakeHuman 身体拓扑，不能据此推断实际驱动骨架。
5. 按原构建记录施加必要的局部动作修正。例如已调整过摆臂避让时，应恢复其旋转轨并在最终身体和衣物上复查间隙。若改了骨架静止姿态，重新推导受影响的轨道。用对应阶段的文件比较，避免将中间版本当成最终动作。

### 一次性建立角色绑定

已有角色直接使用其固定骨架文件。要从静态扫描重新建立同一绑定时，准备单表面静态扫描和带皮肤权重、原始片段的单表面骨架 GLB。在仓库根目录运行：

```sh
"$PY" skills/charloom/scripts/bind_fixed_rig.py "$SCAN" "$RIG_SOURCE" "$BIND_CONFIG" "$FIXED_RIG"
```

`BIND_CONFIG` 的 `scan_sha256`、`rig_sha256` 是两个输入文件的 SHA-256。`scan_to_rig_world` 为从扫描世界坐标到角色世界坐标的 4×4 行矩阵，长度单位米。`joint_positions_m` 以骨名列出标定后的三维目标位置。`source_clip`、`output_clip`、`mesh_name` 指定动作和网格名称。复用原绑定记录中的 `rbf_smoothing`、`neighbors`、`distance_floor_m`、`distance_power`。可选 `height_weight_ramps` 以 `joint`、`range_m` 记录已校准的高度权重过渡。若原构建修正过局部摆臂，可用 `rotation_track_edits` 按骨名列出同一键时刻的 `times` 和四元数 `values`。保留这些角色专属参数的来源。

完成裸模和穿衣几何、装配原头和表情形态键后，将两层绑定到同一个 `FIXED_RIG`：

```sh
"$PY" skills/charloom/scripts/attach_fixed_rig.py "$BARE_LAYER" "$FIXED_RIG" "$LAYER_CONFIG" "$FINAL_BARE"
"$PY" skills/charloom/scripts/attach_fixed_rig.py "$DRESSED_LAYER" "$FIXED_RIG" "$LAYER_CONFIG" "$FINAL_DRESSED"
```

`LAYER_CONFIG.layer_to_rig_world` 为层的工作空间到固定骨架世界空间的 4×4 行矩阵；可由扫描到两侧空间的已标定变换合成。`skin_joint_map` 将层内每个有权重的模板骨名对应到固定骨名。多根细骨对应同一目标时合并权重。面部动作由 `expressions` 按片段名配置，每个片段保留形态键，头颈等刚性动作使用“固定骨名 → 输入姿态骨名”的对应关系。如果角色依赖独立的眼、颌或指骨控制，首次选骨架时应包含这些关节并明确配置映射。绑定后复测颈缝、手脚、静止表面及运动中的变形。

## 模板骨架承载动作

原构建如采用重新定向，则使用原动作和片段、同一模板骨架、源到目标骨名映射、参考链设置、采样率与后续修正。检查运行入口实际支持的参数。`scan-transfer.py --motion` 选择默认片段并使用内置的参考链对齐；有单独配置的构建应从其原重定向入口复跑。新角色选择动作素材时按任务指定的输入执行。

## 检查实际写入的动作

在 CV Python 环境从仓库根目录执行。对照文件应取自要比较的同一阶段。

```sh
"$PY" skills/charloom/scripts/compare_walk.py "$WALK_BASELINE" "$RESULT_GLB" --source-clip "$BASELINE_CLIP" --result-clip "$RESULT_CLIP"
```

此脚本按节点名和属性比较通道、关键帧时间、值和插值，同时比较动画节点及祖先的静止变换和父子关系。有差异返回 1，列出具体通道；预期的局部修正须与原构建记录逐条相符。在保持轨道的步骤应得到完全匹配。若导出器重采样，另在共同时间点比较世界空间关节与足跟、前掌轨迹，检查步幅、支撑转换、摆臂与衣物接触。最终导出如另加舞台高度，预期唯一新增项为指定容器的 `translation`。比较报告中原骨骼的通道、层级和静止变换都应相同。用[步行接地](walking-ground.md)复测靴底，跑步保留腾空。

## 在同一角色骨架上增加动作

为每段动画记录来源、原片段、所需关节和输入到角色的映射。使用角色的上一版成品 GLB 作目标，同一动作的配置同时应用到裸身及穿衣版。用唯一的 `name` 添加新片段。已有同名片段时，重定向入口会更新它。

```sh
"$PY" scripts/animate.py map "$ANIMATION_SOURCE" "$FINAL_DRESSED" --config "$CLIP_CONFIG"
"$PY" scripts/animate.py retarget "$ANIMATION_SOURCE" "$FINAL_DRESSED" --clip "$SOURCE_CLIP" --config "$CLIP_CONFIG" --out "$DRESSED_NEXT"
"$PY" scripts/animate.py retarget "$ANIMATION_SOURCE" "$FINAL_BARE" --clip "$SOURCE_CLIP" --config "$CLIP_CONFIG" --out "$BARE_NEXT"
```

`joint_map` 的键为固定角色骨名，值为源骨名。对脊柱链逐段核对解剖顺序；实际需要的关节列入 `target_joint_scope` 并设置相应的 `min_target_coverage`。舞台上需要原地循环时，`in_place_axes` 可指定要扣除周期净位移的世界轴，保留该轴周期内的相对摆动和未指定轴的运动。需要场景位移时保留原根运动。

此入口会核对原角色的骨架层级、静止变换、逆绑定矩阵、网格关节索引和权重。每层在 `report.json.fixed_target_rig` 中记录骨架与蒙皮哈希。核对原动作、播放新增片段，并在同一帧切换裸身和穿衣。Blender API 的 `retarget_animation` 使用同一入口。

### 次级骨链与动作修正

1. 对照目标骨架、已有动作轨道和服装权重，标出本次由动作源驱动的关节、依靠父骨随动的挂点，以及需要独立摆动的发束、裙摆或披风链。分别记录各链的处理方式。映射范围内的覆盖率只说明所选身体关节是否映射成功。
2. 新动作没有对应的次级骨轨道时，先查角色原有的次级运动脚本或约束。以头、骨盆等挂点为参考，沿新动作烘焙自由段的跟随和回摆。核对挂点附近是否稳定、末端在挂点坐标系中是否确实摆动、循环首尾是否连续。延迟和摆幅按本角色的长度、姿态及服装间隙确定。
3. 若仅新动作发生穿插，定位到具体时刻，先调整该片段相应的上臂或次级骨旋转。变换到当前父骨的局部空间后写回。在各层输出副本上可调用 `roster.write_face_clips.write_clips(model, [clip], extend=True)` 更新指定通道。`clip` 包含已有片段的 `name`、`tracks` 和 `morph_tracks`。每条骨轨包含本层的 `joint` 节点索引、`property`、以秒计的 `times` 和局部 `values`。按骨名分别解析两层索引；局部旋转四元数顺序为 x、y、z、w。
4. 在每层最终导出上，用上面的 `compare_walk.py` 按实际片段名逐一对照旧走路、表情和说话等所有旧片段，并复查裸身和穿衣共享的新增骨轨。在记录中列出单独添加的接地轨道。保存动作源及两层输入的哈希、映射、次级动作脚本和接地参数，需要复现时在新目录运行整套动作步骤并核对最终文件。

### 动作中的局部穿插

在仓库根目录，用 CV Python 列出最终穿衣 GLB 的三角 primitive。按实际贴近的部位建立成对选区，例如自由发束对袖口、手对衣身、裙边对大腿。避开固定挂点和本来就连在一起的缝边。

```sh
"$PY" skills/charloom/scripts/check_motion_contacts.py "$FINAL_DRESSED" --inventory
"$PY" skills/charloom/scripts/check_motion_contacts.py "$FINAL_DRESSED" --config "$CONTACT_PAIRS" --clip "$RESULT_CLIP" --samples "$SAMPLES" > "$REVIEW/clip-contacts.json"
```

`CONTACT_PAIRS` 是 JSON。`regions` 将自定义选区名映射到 `{"mesh": 0, "geometry_hash": "...", "faces": [12, 13]}` 这样的记录。替换为清单中的 primitive 序号、对应几何哈希和在该 primitive 上核对过的三角面序号。整件使用 `"faces": "all"`。`pairs` 列出要对照的选区名对，例如 `[["free_hair", "sleeve"], ["hand", "jacket"]]`。同一 primitive 内可以选取两组分离的面。

`SAMPLES` 为含首尾的均匀采样数，按动作时长和运动速度选取。用相同选区复测原有片段与新增片段，在接触变化快时加密采样。报告包含静止相交数、各时刻的相交面配对数，以及交叉最多时的时间和示例面号。分别记录原有交叉和新动作中的交叉；采样结果仅覆盖所选面和时刻。几何相交是否在外表可见，按 [Gallery 同帧复查](../../charloom-gallery-review/SKILL.md) 判断。
