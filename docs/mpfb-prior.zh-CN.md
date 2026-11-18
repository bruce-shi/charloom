# MakeHuman 身体模板（MPFB）

[English](mpfb-prior.md) · 简体中文

`charloom mh-transfer` 使用已绑定的 MakeHuman 身体作为拟合模板，包括 MPFB 默认骨架（163 根骨骼，含脸部和手指骨链）及其蒙皮权重。模板在 Blender 后台进程中用固定版本的输入生成。

上游已固定的版本、下载链接和SHA256和值存储在[`scripts/mpfb/sources.json`](../scripts/mpfb/sources.json)中（包括MPFB 2.0.18以及MakeHuman的CC0系统资产）。下载内容及附加数据存储在`out/tools/mpfb`中；您自己的Blender配置不会受到影响。

## 生成

```sh
python3 scripts/mpfb-reference.py out/examples/mpfb-male-NEW --muscles 0.5 1.0
python3 scripts/mpfb-reference.py out/examples/mpfb-female-NEW --gender female --muscles 0.5 0.85
```

`--weight`、`--height`、`--age` 和 `--muscles` 是 MakeHuman 自带的 0–1 宏参数；年龄 0.5 代表年轻成年人。传递 `--blender PATH` 或设置 `BLENDER`。输出目录必须为 new。

每个预设文件夹（例如 `female-muscle-050/`）包含：
- `body.glb` 是已蒙皮的裸身模型。**在 `mh-transfer` 中用它作为 `--prior`。**
- `editable.blend`：完整的MakeHuman参数、形状键和绑定。
- `clothed.blend` / `clothed.glb` 并查看渲染效果。

正在使用的优先级：`out/examples/mpfb-female-reference-v3/female-muscle-050/body.glb` 和 `out/examples/mpfb-reference-v7/male-muscle-050/body.glb`。

## 打开或查看参考文档

```sh
# Edit in Blender with the project-private MPFB add-on (MPFB side panel)
BLENDER_USER_CONFIG="$PWD/out/tools/mpfb/runtime/config" \
BLENDER_USER_EXTENSIONS="$PWD/out/tools/mpfb/runtime/extensions" \
/Applications/Blender.app/Contents/MacOS/Blender --factory-startup --disable-autoexec \
  --python scripts/mpfb/open-reference.py -- out/examples/mpfb-reference-v7/male-muscle-100/editable.blend

# Re-import the exports and compare surfaces, triangle counts and the 163 bones
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --disable-autoexec \
  --python-exit-code 1 --python scripts/mpfb/check-exports.py -- out/examples/mpfb-male-NEW
```

## 形状目标 (`body-shape-build.glb`)

身体形状拟合（`charloom-mh` 的 `shape.rs`）使用 MakeHuman 自带的形态目标作为 glTF morph，包括腹部、围度、头、手、脚、五官和胸部（`pectoral`、`dorsi`；女性身体另有 `breast-size` / `breast-firmness`）。整体体型使用 MPFB 的体重及肌肉宏参数 `build-weight` / `build-muscle`。衣下身体由这些形态控制。先按可见皮肤拟合体型，决定粗壮或纤细程度，再以胸部目标调整胸形。`body-shape-chest.glb` 是未加整体体型目标时的导出。

```sh
for d in mpfb-female-reference-v3/female-muscle-050 mpfb-reference-v7/male-muscle-050; do
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
    --disable-autoexec --python-exit-code 1 --python scripts/mpfb/export-shape-targets.py -- \
    out/examples/$d/editable.blend out/examples/$d/body-shape-build.glb
done
```

## 耳部目标 (`ear-targets.glb`)

`charloom mh-transfer` 使用 MakeHuman 左右耳各自的形态目标，将耳形拟合到标注轮廓（`ear_{l,r}_*` 标记）。将 `ear-targets.glb` 与模板（该目录下任一 `body*.glb`）放在一起。该文件采用相同的身体网格，每个 `ears/` 目标按官方名称存为一个 morph。

```sh
for d in mpfb-female-reference-v3/female-muscle-050 mpfb-reference-v7/male-muscle-050; do
  CHARLOOM_TARGET_SET=ears /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
    --disable-autoexec --python-exit-code 1 --python scripts/mpfb/export-shape-targets.py -- \
    out/examples/$d/editable.blend out/examples/$d/ear-targets.glb
done
```

缺少耳部形态文件时，仍可按轮廓用相似变换定位耳朵。若附带模型的顶点与身体模板无法逐点对应，程序会忽略该文件。
