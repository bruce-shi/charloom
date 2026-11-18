# 从原扫描生成量体图

在仓库根目录运行。设 `PY` 为项目 CV 环境的 Python，`SCAN` 为原扫描 GLB，`GUIDE` 为这个扫描的标定目录，`MEASURE` 为量体产物目录。源图使用标定相机渲染的正交视图。保留各图的像素坐标、`views.json` 和源哈希，方便把端点投回同一工作坐标。

## 1. 取得标定底图

按[扫描标定](../../charloom-scan-guide/SKILL.md)用 `scan-guide.py "$SCAN" --out "$GUIDE"` 渲染原扫描。完成正面朝向与关节标注，运行 `scan-guide.py "$SCAN" --check "$GUIDE/guidance.json"`。读 `front_view`，从标定相机选择其偏航角加 90° 的侧图。检查头脚完整和地面线。若已有匹配的 guide，从这一步的检查继续。

## 2. 生成同姿紧身衣参考

先按[图片服务依赖](../../charloom/references/setup.md#图片服务依赖)准备 ImageMagick、genmedia 和 fal。以下命令保留原始手臂姿势，用于关节、臂长和相机配准。

```sh
"$PY" scripts/mh-bodysuit.py "$GUIDE" --edit --back
"$PY" scripts/mh-suit-contours.py "$GUIDE"
```

第一条命令把同一角色的正、侧、背标定图横向拼接，通过 fal 的 GPT Image 2.5 Sunburst edit 生成一张 2560 × 1136 的 PNG，质量为 medium。只要正、侧两图时省略 `--back`，尺寸为 2048 × 1360。实际提示词见[脚本的 SHEET 和 PROMPT](../../../scripts/mh-bodysuit.py)。使用同一角色与相机，衣甲替换为贴身绿色连体服。原肩、肘、腕位置允许时，基础侧图就让手掌竖直贴大腿外侧、指尖向下。若要移动整条手臂才能让出腹缘，则保留同姿输入，按下一步另做躯干侧图。保留 `views/bodysuit_sheet-2k.png`、各视图原生输出和 `bodysuit_<view>-reg.png`。收据记录在 `views/bodysuit.json` 的 `sheet`，缓存位于 `views/measurement-cache/`。相同参数重新执行 `--edit` 时复用缓存或恢复原请求。

叠看标定底图和注册图。核对脸、腕、脚底、朝向、整体尺度和局部轮廓；检查 `checks.scale`、`anchors`、`side_feet_px` 和 `accepted`。第二条命令从配准图生成全身轮廓。查看每一行两个端点实际落在哪个部位。手指、腕或前臂挡住腹部时，保留同姿图用于手臂测量，把被挡的躯干端点从拟合输入中移除。

## 3. 生成手掌竖贴大腿的侧面量体图

采用完整人物的侧视图。近侧手掌竖直贴在大腿外侧，腕在上、指尖朝下。侧面投影中，手指应落在大腿轮廓内，小腹前缘应连续可见。躯干、骨盆、头、腿、脚、拍摄方向、比例和地面线与已配准侧图一致。同姿图已经满足要求时可直接用它量躯干。

```sh
PROFILE="$MEASURE/profile-hand-at-thigh"
"$PY" skills/charloom-body-measure/scripts/torso_profile.py "$GUIDE" --out "$PROFILE"
"$PY" skills/charloom-body-measure/scripts/torso_profile.py "$GUIDE" --out "$PROFILE" --generate
```

第一条只写 `plan.json`，可先核对输入图、源哈希、提示词和参数。第二条通过 genmedia 提交或恢复同一个缓存请求。输出 `profile-native.png` 为 1536 × 2048、medium 的原生 PNG；`profile-candidate.png` 回到基础相机的 768 × 1024 画布；`generation.json` 记录源图哈希、生成参数和服务收据。检查整个人物和手指落点，再对图配准。

## 4. 把侧面补图配回同一相机

在候选图和基础配准侧图中，以左上为原点分别点取共同可见的稳定标志。在头、足等相距足够远的位置选择至少两个同名点拟合等比缩放和平移，另选至少两个不同位置作独立检查。优先加入仍可辨认的背、膝或臀后缘，避免只在同一个局部检查。记录 `anchors.json`，含两张图的 `base_sha256` 和 `candidate_sha256`、可接受的 `tolerance_px`、`fit` 与 `check` 两个数组。每项写具体解剖点名 `name` 和两组实际像素 `candidate_px: [x,y]`、`base_px: [x,y]`。按测点可辨度和标定相机的毫米/像素换算确定容差。手掌、前臂和生成补出的腹缘不能作配准锚点。

```sh
"$PY" skills/charloom-body-measure/scripts/register_profile.py "$PROFILE" "$PROFILE/anchors.json" --out "$PROFILE/registration"
```

检查 `registration.json` 内每个点的残差，放大 `overlay.png` 对照头、背、臀、腿、脚的位置。锚点合格后再看腹部逐行轮廓是否通顺、有没有手掌或袖口误入。出现局部形变时单独排除该区。保留原图和重合图。

## 5. 生成带测点的图

按[量体脚本输入](script-inputs.md)先生成 `frame.json`。运行 `outline_points.py` 提取基础图的密集点对，并输出每个视图的 `points-<view>.png`。对侧面腹部，在已复核的 `registration/registered.png` 上回到同一标定行，标出腹前和背后两个实际边界像素；逐行保存部位、行号、两个端点、来源图哈希及边缘不确定范围。把原来落在手掌上的端点和修订端点分别留在审核记录中。隐藏轮廓标为紧身衣参考估计。做截面时，将躯干专用视图的已复核 mask 作为对应深度方向的输入，手臂路径继续读取同姿图。改变了手臂姿态的整张轮廓不能覆盖原全身拟合输入，仍看不见的边界记为缺测。

用 [profile_rows.py](../scripts/profile_rows.py) 绘制逐行端点。为每个测量行在躯干内部点一个像素，保存 `stations.json`。顶层写 `registered_sha256`、`anterior`（图中身体前缘在左填 `left`，在右填 `right`）和 `stations` 数组。每项写部位 `name`、整像素坐标 `inside_px: [x,y]`、布尔值 `anterior_visible` 和 `posterior_visible`，以及判读依据和不确定范围。初检时两端都可填 false，以便先看图，再逐行确认。选择端点时区分前臂外缘、手掌外缘和躯干外缘。

```sh
"$PY" skills/charloom-body-measure/scripts/profile_rows.py "$GUIDE" "$PROFILE/registration" "$PROFILE/stations.json" --frame "$MEASURE/frame.json" --out "$PROFILE/rows"
```

`profile-points.png` 以青点标记确认可见端、红点标记未确认端。两端都可见的行才连接黄线，输出投影厚度。`profile-rows.json` 保存原图和补图两个轮廓交点、可见性及工作尺度。根据叠图修订时输出到新目录，保留初检记录。

## 原姿势下的手臂与远侧腿

竖掌侧图专供躯干。若同姿图里袖子和胸腹连成一片，或两条腿在侧面重叠，可用 [limb_views.py](../scripts/limb_views.py) 将正面和两侧的观测合并为一张图。先从完整标定目录确认两个相反的侧面相机都有源 PNG。命令复用已配准的正面和近侧紧身衣图，远侧读取同角色的相反侧原扫描视图。绿色袖子和手套与蓝色躯干、腿和袜子区分开，便于在同一人物姿态上辨认部位边缘。

```sh
LIMBS="$MEASURE/limb-views"
"$PY" skills/charloom-body-measure/scripts/limb_views.py "$GUIDE" --out "$LIMBS" --generate
```

保留 `sheet-input.png`、2560 × 1136 的 `sheet-native.png` 与参数收据。每个 `body_yNNN/` 子目录含与相应相机同尺寸的候选图和来源。分别填写该视图的锚点，按第 4 步传给 `register_profile.py`。将配准后的袖子隔离为手臂专用 mask，逐段确认袖子与手套边缘对应肢体，局部轴、姿势及同侧关节可比。腿和脚也按实际相机近侧识别解剖左右。生成图里的套装表面只可用于隐藏人体估计，证据不足的局部方向仍留空。

鞋靴换成薄袜后，靴尖和脚趾常常无法一一对应。若两幅图确实共用水平地面，锚点文件可加 `fit_y_lines` 或 `check_y_lines` 数组，每项给出 `name`、`candidate_y`、`base_y`。从两图实际接地处读取行坐标；地面约束只参与纵坐标计算。用未参与拟合的标志单独检查水平方向、朝向及局部轮廓。
