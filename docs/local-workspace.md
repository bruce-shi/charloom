# 本地工作区

在 Blender 中制作和保存角色。Gallery 从登记目录读取原扫描、裸模、穿衣版和动画。启动时会打印本机会话链接，打开后可以刷新、直接访问角色页面和使用浏览器的前进与后退。

```sh
cd viewer
bun install --frozen-lockfile
bun run dev
```

旧版本把项目保存在 `~/.charloom/projects/<id>/project.json`，资源位于同一工作区的 `blobs/<sha256>`。自定义工作区使用原来的 `CHARLOOM_DATA_DIR`。用后台 Blender 将已有项目导入新文件。

```sh
blender -b --factory-startup --python-exit-code 1 --python scripts/blender_project.py -- \
  /path/to/projects/PROJECT_ID/project.json /path/to/character.blend
```

脚本验证资源摘要，按资源建立 Blender Collection，打包外部贴图，并将原始项目文档和修改记录保存为 Text 数据块。拟合参数、标注和历史网格补丁保留在项目文档中供脚本读取。导入后的场景显示已保存的资产几何。字符制作和登记流程见[项目说明](../README.md)。
