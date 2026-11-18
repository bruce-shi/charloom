# 服装从分件到最终运动检查

## 首次绑定

在 `guidance.json.garments` 中，一件独立运动物体使用一个 `name`，与分件决定中的 `piece.name` 一致。按标注格式填写 `regions` 多视图多边形；用实际图检查前后面。相接但运动不同的衣物分开命名。

每项可含 `layer`（由内向外递增）、`rig` 和 `fabric`：

- 贴身布料：`rig:{mode:"body"}`，转移最终皮肤的平滑权重，并执行袖口局部权重修正。
- 刚性护甲、装备：`rig:{mode:"rigid",bone:"实际挂点骨名"}`。骨名从实际导出读取；模板骨与重定向后的骨名不同，不能混用。
- 裙、带、披风：`mode` 取 `skirt,ribbon,panel,cape,veil`。指定 `bone`，可用 `direction`、`width_axis` 定悬垂与面板方向；`skirt.cut` 为 `flared,bell,pleated` 或 `{"panels":整数}`。挂点处固定，沿长度产生次级链。
- `fabric` 为脚本已有材质名（如 `woven,leather,silk`），或 `{"delay_s":值,"gain":值,"max_angle_rad":值}`。选择与面料相符的运动参数，再在整段动作中复查。

`scan-transfer.py` 的 `write_models` 按 layer 绑定，裸模先保存完整身体，再处理穿衣版遮挡。`transfer/cloth.py` 的 `rig_for`、`bind` 和 `bake` 实现这些字段。改 guide 的刚性/面板配置会影响首次构建；已有最终身体的局部修复采用下面的明确对象操作，避免重跑早期身体拟合。

## 对最终身体适配

先列出实际对象与挂点：

```sh
"$BLENDER" -b --factory-startup --python-exit-code 1 --python skills/charloom-garment-fit/scripts/fit_garments.py -- "$DRESSED" --inventory "$GARMENT_INVENTORY"
```

读取库存报告并在 Blender 隔离相应物体。沿该侧肢体轴在近端、中段、远端切原护具和最终身体；分别记中心、宽、深、内外层间隙。袖口、翻边、隐藏封口各自检查，不能因为内壁一小片穿插就放大整个护具。

建立编辑 JSON：顶层 `source_sha256` 必须等于库存摘要，`pieces` 为操作数组。每项 `objects` 填库存里的精确对象名；`bind` 取 `preserve`、`body` 或 `rigid`。刚性重绑指定真实 `bone`。需要尺寸调整时还填 `bone`、4×4 行优先仿射 `matrix`、`measurement_evidence`（切面文件与具体站位）。矩阵在 **Blender 骨骼局部米制坐标**中作用；库存提供 `bone.matrix_local`，不要将 GLB 的 Y-up 直接当 Blender 坐标。用原/目标中心求平移、宽深比求对应局部缩放，不凭全物体包围盒猜护具尺寸。

默认矩阵作用于整件；局部内壁或袖口用 `vertices:{对象名:[顶点号...]}` 明确选择。局部选择边缘应平滑过渡，复查法线和 UV 拉伸。`bind:"body"` 使用当前 body 重传权重，`rigid` 把所有选定对象的可变形骨权重归到同一挂点；已经合格的其他物体不加入配置。披风等已有次级链采用 `preserve`，不要用 `body` 抹掉其自由运动。

```sh
"$BLENDER" -b --factory-startup --python-exit-code 1 --python skills/charloom-garment-fit/scripts/fit_garments.py -- "$DRESSED" --config "$GARMENT_EDITS" --out "$GARMENT_OUTPUT"
"$PY" scripts/animate.py validate "$GARMENT_OUTPUT/character.glb"
```

输出可编辑 `character.blend`、`character.glb` 和带来源的 `garments.json`。已有服装 morph 时脚本拒绝直接变换，先核对形变目标后做相应 Blender 编辑。此命令不自动计算穿插修复，也不表示衣物已经验收。

## 完整动作和穿插

```sh
"$PY" scripts/skin.py analyze "$DRESSED" > "$REVIEW/skin-analysis.json"
"$PY" scripts/skin.py contacts "$DRESSED" --config "$CONTACT_CONFIG" > "$REVIEW/contacts.json"
"$PY" scripts/render-character.py "$BUILD" --only body,dressed,walk,chest
```

contacts 配置是 `regions:[{mesh:primitive序号,geometry_hash:摘要,vertices:[顶点号...]}]`，加 `clip`、`times`（实际完整片段上的秒数）。用 `Surface(FILE).parts` 的 `hash` 和节点名核对 primitive；这是与量体脚本不同的几何哈希，不用文件 SHA 代替。结果分别给 rest 穿插和各帧新增穿插；原有穿插也要检查，不能只看新增为零。接缝和边长变化由 analyze 报告，当前 contacts 只检横向相交，不能替代近距离、共面层和视觉检查。

复查静止、完整行走、抬臂、弯肘及涉及的动作。定位最差帧、具体双方三角面，修正绑定/局部轮廓后重算；剖面尺寸与原外形仍需匹配。在同一时间切裸模、纯衣服、穿衣版，检查旧手指、指甲、衣领与头发残面。通过后将最终穿衣文件连同完整裸模登记到 Gallery。
