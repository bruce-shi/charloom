# Charloom

[English](README.md) · 简体中文 · [文档目录](docs/README.zh-CN.md)

Charloom 把带贴图的扫描模型做成可在 Blender 中编辑的角色。你可以拟合 MakeHuman 身体，绑定原扫描头部和衣物，并在角色已有骨架上追加动作。Blender 插件提供编辑面板，配套的 agent skills 指导标定、拟合、动画和验收。

## 从公库安装

以下命令适用于 macOS 或 Linux，需要 Git、uv、Bun 和 Blender 4.4 或更高版本。插件和 skills 运行时都会调用仓库中的脚本，请将仓库克隆到准备长期保留的位置。以下代码块在同一个 shell 中依次执行。

```sh
git clone https://github.com/bruce-shi/charloom.git
cd charloom
export CHARLOOM_ROOT="$(pwd -P)"
export BLENDER="${BLENDER:-blender}"
if [ "$(uname -s)" = Darwin ] && [ "$BLENDER" = blender ] && ! command -v blender >/dev/null; then
  export BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
fi
"$BLENDER" --version
uv venv --python 3.12 out/tools/cv/.venv
export CHARLOOM_CV_PYTHON="$CHARLOOM_ROOT/out/tools/cv/.venv/bin/python"
uv pip install --python "$CHARLOOM_CV_PYTHON" -r scripts/requirements-blender.txt
(cd viewer && bun install --frozen-lockfile)
```

### 1. 安装并启用 Blender 插件

先关闭其他 Blender 会话，再写入用户偏好。以下命令会生成 ZIP、安装到 Blender 默认的用户扩展仓库，并保存工程目录、Python 和 Bun 路径。装有多个 Blender 版本时，请将 `BLENDER` 指向实际编辑角色的版本。

```sh
mkdir -p out/extensions
"$BLENDER" -b -c extension build --source-dir blender/charloom \
  --output-filepath "$CHARLOOM_ROOT/out/extensions/charloom.zip"
"$BLENDER" -b -c extension install-file --repo user_default --enable \
  "$CHARLOOM_ROOT/out/extensions/charloom.zip"
"$BLENDER" -b --python-exit-code 1 --python-expr '
import bpy, os, shutil
prefs = bpy.context.preferences.addons["bl_ext.user_default.charloom"].preferences
prefs.checkout = os.environ["CHARLOOM_ROOT"]
prefs.python = os.environ["CHARLOOM_CV_PYTHON"]
prefs.bun = shutil.which("bun")
bpy.ops.wm.save_userpref()
'
```

另起一个 Blender 进程核对已保存的配置。成功时会打印工程目录。打开 Blender 的 3D 视图，按 **N**，选择 **Charloom** 标签即可使用面板。若通过界面安装，在 **偏好设置 → 获取扩展 → 从磁盘安装** 选择同一个 ZIP，启用 Charloom，然后在插件偏好中把 **Charloom 工程目录** 设为仓库根目录。

```sh
"$BLENDER" -b --python-exit-code 1 --python-expr '
import bpy, os
p = bpy.context.preferences.addons["bl_ext.user_default.charloom"].preferences
assert os.path.samefile(p.checkout, os.environ["CHARLOOM_ROOT"])
assert os.access(p.python, os.X_OK) and os.access(p.bun, os.X_OK)
assert hasattr(bpy.types, "CHARLOOM_PT_character")
print("Charloom ready:", p.checkout)
'
```

### 2. 安装 agent skills

使用 Codex 时，将每个完整的 skill 目录链接到用户 skills 目录。这样配套脚本和参考文档会一起安装，更新仓库后也会同步生效。若只在本项目使用，先设置 `CHARLOOM_SKILLS_DIR="$CHARLOOM_ROOT/.agents/skills"`。选择一种安装范围即可。若已有同名条目指向其他位置，命令会提示检查。[Codex skills 加载规则](https://developers.openai.com/codex/skills)

```sh
skills_dir="${CHARLOOM_SKILLS_DIR:-$HOME/.agents/skills}"
mkdir -p "$skills_dir"
for src in "$CHARLOOM_ROOT"/skills/charloom*; do
  dest="$skills_dir/${src##*/}"
  if [ ! -e "$dest" ] && [ ! -L "$dest" ]; then
    ln -s "$src" "$dest"
  fi
  if [ ! "$dest" -ef "$src" ]; then
    echo "Check existing skill: $dest" >&2
  fi
done
```

在仓库根目录打开 Codex，在技能选择器里输入 `$charloom`，应能看到 `charloom` 和 `charloom-mh`。新技能尚未出现时，重启 Codex。使用其他技能加载方式的 agent 可以直接读取这两个入口，再按文中的相对链接操作。所有 shell 命令均从仓库根目录运行。

| 任务 | 技能入口 |
| --- | --- |
| 从原始扫描制作角色 | [charloom-mh](skills/charloom-mh/SKILL.zh-CN.md) |
| 编辑已绑定角色或追加动作 | [charloom](skills/charloom/SKILL.zh-CN.md) |
| 检查交付模型和实际动作 | [charloom-gallery-review](skills/charloom-gallery-review/SKILL.zh-CN.md) |

## 从原始扫描构建角色

准备静态扫描 GLB（单张 UV 图集）、MakeHuman 身体模板和 FBX 或 GLB 动作。用下列命令创建标注文件，然后在 `guidance.json` 中标记人体关键点和衣物分区。按扫描标定 skill 制作特写，描出唇线和眼睑边界。[模板准备](docs/mpfb-prior.zh-CN.md) · [扫描标定](skills/charloom-scan-guide/SKILL.zh-CN.md) · [资产目录](assets/README.zh-CN.md)

```sh
"$CHARLOOM_CV_PYTHON" scripts/scan-guide.py scan.glb --out out/guide
cp out/guide/guidance.template.json out/guide/guidance.json
```

复核标注、完成身体拟合和分件检查后，把下列路径替换为实际输入并构建。构建命令依赖前序人工或 agent 对量体表的复核，导出后还要按同一定义复测。

```sh
"$CHARLOOM_CV_PYTHON" scripts/scan-guide.py scan.glb --check out/guide/guidance.json
"$BLENDER" -b --factory-startup --python-exit-code 1 --python scripts/blender_character.py -- \
  scan.glb out/new-character --name character \
  --prior prior.glb --guidance out/guide/guidance.json \
  --mouth-trace out/guide/mouth-trace.json --motion walking.fbx \
  --head-binding scan-surface --clip face_talk --export-glb
```

打开 `out/new-character/character.blend` 编辑场景，或检查同目录的 `character.glb`。标定迁移数据在 `transfer/`，裸身和穿衣模型分别在 `build/3-skin/body.glb` 和 `build/4-garments/character.glb`，动作检查记录在 `build/animated-checks.json`。已安装适用的 MakeHuman 基础内衣时，裸模导出会装配该衣物，并在材质中记录许可信息。

## 通过脚本操作 Blender 并复核交付

从仓库根目录启动 Blender，在它的 Python 环境中运行以下示例。请把两个路径换成含 `0-guide/resolved.json` 的标定目录和新的输出目录。

```python
import sys
from pathlib import Path
import bpy

checkout = Path.cwd()
sys.path.insert(0, str(checkout / "blender"))
from charloom import api

transfer = Path("out/calibrated-transfer")
output = Path("out/edited-character")
built = api.build_character(transfer, output, checkout=checkout, head_binding="scan-surface")
rig, body = api.import_character(built / "4-garments/character.glb")
api.preview(rig, "face_talk")
bpy.context.scene.frame_set(18)
api.export_character(rig, built / "edited.glb")
```

原扫描可用 `api.prepare_guide` 和 `api.transfer_scan` 处理，也可以将扫描及配套输入传给 `api.build_character`。身体修整用 `api.fit_surface`，衣物绑定用 `api.bind_clothing`。处理动作用 `api.retarget_animation`、`api.solve_pose` 或 `api.build_animation`。[在已有骨架上追加动作](skills/charloom/references/walk-replay.zh-CN.md)

要直接运行命令行动画工具，请准备配方引用的本地绑定模型。下例读取足部锁定配方中的资产路径。

```sh
"$CHARLOOM_CV_PYTHON" scripts/animate.py build recipes/myth-explicit-foot-lock.json --out out/foot-lock
"$CHARLOOM_CV_PYTHON" scripts/animate.py validate out/foot-lock/character.glb
"$CHARLOOM_CV_PYTHON" scripts/skin.py analyze out/foot-lock/character.glb
```

要在只读 Gallery 登记成品，先用 `python3 scripts/mh-ide-sync.py BUILD_PARENT --dry-run` 预览登记结果，再去掉 `--dry-run` 执行。运行 `(cd viewer && bun run dev)`，打开终端打印的会话链接，即可对比原扫描、裸模、穿衣版和实际动画。生产预览在 `viewer/` 下执行 `bun run build` 和 `bun run preview`。[导入已有项目](docs/local-workspace.zh-CN.md)
