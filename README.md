# Charloom

Charloom 把带贴图的扫描模型做成可以在 Blender 编辑的角色。身体可用 MakeHuman 拟合，角色初次绑定时确定驱动骨架；追加动作映射到该角色已有的骨架。头部沿用原扫描几何和贴图，口缝和眼裂根据原图标注绑定。衣物保持独立蒙皮，行走和说话可以导出为 GLB 动作。

## Blender 插件

从仓库根目录安装几何处理环境和 Gallery 依赖。下列命令使用 Python 3.12。Blender 的内置 Python 执行场景操作，单独的 CV 环境执行几何拟合。

```sh
uv venv --python 3.12 out/tools/cv/.venv
uv pip install --python out/tools/cv/.venv/bin/python -r scripts/requirements-blender.txt
(cd viewer && bun install --frozen-lockfile)
blender -b --factory-startup -c extension build --source-dir blender/charloom --output-dir /tmp
```

在 Blender 的扩展管理器选择 **Install from Disk** 安装 ZIP，在插件偏好设置中填写仓库路径。打开 **3D 视图 → N 面板 → Charloom**，可以载入原始扫描和标注、已有标定迁移目录或成品 GLB。面板支持贴合可见皮肤、给已对齐衣物转移权重、切换时间线动作和导出 GLB。macOS 上可将命令中的 `blender` 替换为 `/Applications/Blender.app/Contents/MacOS/Blender`。

## 从原始扫描开始

原头绑定使用静态三角网格和单张 UV 图集。输入包括扫描 GLB、MakeHuman 身体模板、与源模型哈希匹配的标注、口缝线和行走 FBX 或 GLB。模板和动作的位置由调用方指定。[模板准备](docs/mpfb-prior.md)和[标注流程](skills/charloom-mh/SKILL.md)说明如何生成这些输入。已有图像测量数据可以复用。

下列命令展示构建入口。正式构建前，须完成主流程中的配准参考、逐部位量体和分件复核；当前构建命令不自动检查量体表是否完整。构建后仍须在实际导出上按同一测量定义复测。

```sh
out/tools/cv/.venv/bin/python scripts/scan-guide.py scan.glb --out out/guide
# 在 guidance.json 中标点、确认衣物分区，再制作嘴眼特写。
out/tools/cv/.venv/bin/python scripts/scan-guide.py scan.glb --refine out/guide/guidance.json
out/tools/cv/.venv/bin/python scripts/scan-guide.py scan.glb --check out/guide/guidance.json

blender -b --factory-startup --python-exit-code 1 --python scripts/blender_character.py -- \
  scan.glb out/new-character --name character \
  --prior prior.glb --guidance out/guide/guidance.json \
  --mouth-trace out/guide/mouth-trace.json --motion walking.fbx \
  --head-binding scan-surface --clip face_talk --export-glb
```

新目录里有 `character.blend`、可选的 `character.glb`、标定中间产物 `transfer/`、原始导出 `build/3-skin/body.glb` 和 `build/4-garments/character.glb`，以及 `build/animated-checks.json`。已有标定迁移目录可直接作为第一个参数，成品 GLB 也可直接载入。如果 MakeHuman 基础内衣资源已安装，裸模导出会将适用的 MHCLO 衣物装配到拟合后的身体上，授权信息写入材质。

## Agent 的 Python 接口

```python
import sys
from pathlib import Path
import bpy

checkout = Path.cwd()  # Run from the repository root.
sys.path.insert(0, str(checkout / "blender"))
from charloom import api

# transfer 是包含 0-guide/resolved.json 的标定目录，output 是新的输出路径。
built = api.build_character(transfer, output, checkout=checkout, head_binding="scan-surface")
rig, body = api.import_character(built / "4-garments/character.glb")
api.preview(rig, "face_talk")
bpy.context.scene.frame_set(18)
api.export_character(rig, built / "edited.glb")
```

原始扫描可用 `api.prepare_guide`、`api.transfer_scan`，也可以直接传给 `api.build_character` 并指定 `prior`、`guidance`、`motion` 和 `mouth_trace`。手动修整身体用 `api.fit_surface`，绑定衣物用 `api.bind_clothing`。已有角色追加动作见[固定骨架与动作导入](skills/charloom/references/walk-replay.md#在同一角色骨架上增加动作)。一般动画工具可用 `api.retarget_animation`、`api.solve_pose` 和 `api.build_animation`，或在 CV 环境中直接执行脚本。

```sh
out/tools/cv/.venv/bin/python scripts/animate.py build recipes/myth-explicit-foot-lock.json --out out/foot-lock
out/tools/cv/.venv/bin/python scripts/animate.py validate out/foot-lock/character.glb
out/tools/cv/.venv/bin/python scripts/skin.py analyze character.glb
```

## 只读 Gallery

在 `viewer` 下运行 `bun run dev`，打开终端打印的本地会话链接。Gallery 可对比原扫描、裸模、穿衣版和播放导出的动画。生产预览使用 `bun run build` 和 `bun run preview`。用 `python3 scripts/mh-ide-sync.py BUILD_PARENT --dry-run` 预览登记结果，再去掉 `--dry-run` 执行。旧项目的资产和标注可以[导入 Blender](docs/local-workspace.md)。

## Skills 和资产

从原扫描制作角色，读 [charloom-mh](skills/charloom-mh/SKILL.md)。在 Blender 中编辑已有角色或追加动作，读 [charloom](skills/charloom/SKILL.md)。两个入口会按步骤链接到量体、身体拟合、服装、面部和 Gallery 验证等专项 skill。每个目录的 `SKILL.md` 是入口，配套脚本从仓库根目录运行。

[资产目录约定](assets/README.md)说明共享素材和本地角色输入的存放位置。
