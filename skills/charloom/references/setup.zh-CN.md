# 环境与输入准备

[English](setup.md) · 简体中文

从仓库根目录执行命令。需要 Python 3.12、uv、Bun 和 Blender，版本与平台要求以项目依赖文件和 Blender 扩展清单为准。

```sh
uv venv --python 3.12 out/tools/cv/.venv
uv pip install --python out/tools/cv/.venv/bin/python -r scripts/requirements-blender.txt
(cd viewer && bun install --frozen-lockfile)
export PY="$PWD/out/tools/cv/.venv/bin/python"
export CHARLOOM_CV_PYTHON="$PY"
```

命令行 Blender 接口默认使用上述 CV 目录。直接调用 Python API 时可传 `python=...` 选择其他已安装依赖的环境。Bun 面部动作脚本也需要 CV Python，保留导出的 `CHARLOOM_CV_PYTHON`。设置 `BLENDER` 为可执行文件路径，在 shell 命令中用 `"$BLENDER"` 调用。

## 原扫描与动作

新角色准备原始扫描、项目选定的角色骨架和动作素材。已有角色追加动作只需现有角色 GLB、新动作和映射配置。原头路线使用静态、单网格、单三角 primitive、单贴图图集。输入有蒙皮时，在明确的参考姿势应用变换和蒙皮后导出静态副本。保留原文件和坐标、UV、材质，检查合并后的纹理与面数。完成预处理后再生成绑定该副本哈希的 guide。

复现已有步行先核对[原动作与驱动骨架](walk-replay.zh-CN.md)，打包原构建实际使用的动作源和绑定参数。下列说明适用于为模板骨架重新定向。动作应包含待转移的骨架、参考姿态和所选片段。骨名映射与参考链对齐见 `scripts/transfer/motion.py` 及 `recipes/mixamo/`。校对实际骨名与目标链，明确传 `--motion`。扫描和动作文件由任务指定或从可用且许可适用的资源取得。

## MakeHuman 模板

从仓库的固定版本清单下载并生成模板，设置 `PRIOR_OUTPUT` 为新目录，`BLENDER` 为实际可执行文件。

```sh
"$PY" scripts/mpfb-reference.py "$PRIOR_OUTPUT" --blender "$BLENDER" --gender male --muscles 0.5
```

按角色选择 gender、muscles、weight、height 和 age。所选 preset 子目录的 `body.glb` 作为 `--prior`，同时保留 `editable.blend`、模板资源和生成报告。

## 图片服务依赖

衣甲遮挡裸身且缺少可复用的有效配准参考时，必须先准备图像服务并生成参考，才能完成这条身体制作路线的量体。生成紧身衣参考需要 ImageMagick 的 `magick`、`genmedia` CLI，以及 `FAL_KEY` 或已配置的 genmedia 凭据。当前脚本通过 fal 的 `openai/gpt-image-2.5/sunburst/edit`，显式使用 medium 和约 2K 的多视图尺寸。运行前核对端点 schema 与任务的图像服务授权。请求 ID、参数和缓存收据保存在当前 guide 的 `views/measurement-cache/`，中断后复用同一请求。已配准的有效测量图可直接复用。遮挡身体在生成图中的尺寸属于估计，需结合原扫描的可见皮肤和解剖位置复查。

局部纹理补绘按任务选定的图像编辑服务准备凭据。资料缺失时明确记录尚缺的输入，完成其余可执行步骤。
