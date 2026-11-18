# Charloom 画廊

[English](README.md) · 简体中文

本地画廊会显示已注册的角色结果，将原始扫描图与未修饰和已修饰的导出文件进行比较，并播放它们的动画片段。打开由 Vite 打印出的会话 URL。直接角色链接支持刷新和浏览器导航。

```sh
bun install --frozen-lockfile
bun run dev
```

对于生产版本，请使用`bun run build`和`bun run preview`。只读目录服务器在两种模式下都能运行。在结账过程中，使用`scripts/mh-ide-sync.py`注册结果。[Blender扩展和Python脚本](../README.zh-CN.md)则用于编辑操作。
