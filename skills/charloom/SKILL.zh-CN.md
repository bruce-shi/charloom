---
name: charloom
description: "在 Blender Python 中组装和编辑扫描角色，重定向 FBX 或 GLB 动作，求解姿态、烘焙动画配方、检查蒙皮变形并浏览已发布的角色。标定量体请从 charloom-mh 开始。"
---

# Blender中的Charloom

[English](SKILL.md) · 简体中文

对于新的环境，请遵循[环境和输入设置](references/setup.zh-CN.md)。

在 Blender 中运行 Python，使用 `blender -b --factory-startup --python-exit-code 1 --python your_script.py`。从 checkout 的 `blender/` 目录中导入 `charloom.api`。相同的函数也支持 Charloom 面板。有关安装、先决条件、代理脚本和示例命令，请参见 [项目说明文件](../../README.zh-CN.md)。

- 对于已存在的角色，将每个新的动作添加到其现有的绑定GLB中，使用`api.retarget_animation`。保持目标绑定和皮肤不变。如果角色有头发、布料或道具骨骼，请遵循[添加片段并审查次要动作](references/walk-replay.zh-CN.md#在同一角色骨架上增加动作)来处理这些连接，并通过动作循环检查渲染结果。
- 对于新角色建模，使用 [扫描工作流程](../charloom-mh/SKILL.zh-CN.md)。 `api.prepare_guide` 会生成扫描图。 `api.transfer_scan` 适用于 MakeHuman 框架，并分离出需要审核的服装部分。 `api.build_character(..., head_binding="scan-surface")` 用于连接已观察到的头部，并固定其原始面部表情。为每个操作指定一个新的输出目录。
- `rig, body = api.import_character(path)` 导入一个已装配的GLB文件。`api.fit_surface(body, scan, group)` 在裸露的皮肤上添加形状关键点。`api.bind_clothing(body, rig, objects)` 将每个顶点最多可转移四个身体影响至对齐的贴合面料上。对于刚性设备或松散的面板，在重新定位或预览自动装配的导出文件之前，请遵循[每件物品装配工作流程](../charloom-garment-fit/SKILL.zh-CN.md) 的说明。
- 要复制现有的行走动画，需要先确定其原始绑定、绑定方式、原始片段以及后续的轨道编辑内容，然后再进行适配或重新目标化。请遵循[行走重建与轨迹比较](references/walk-replay.zh-CN.md)步骤。
- `api.retarget_animation(source, target, output, clip=..., config=...)`、`api.solve_pose(model, effectors, output, clip=..., time=...)`和`api.build_animation(recipe, output)`会返回报告以及一个导出文件。请检查关节覆盖情况和效应器残差。将T-姿态源重新定位到手臂较短的角色时，需要参考链对齐。
- `api.preview(rig, clip)`用于选择绑定和变形动作。使用`bpy.context.scene.frame_set(frame)`进行样本测试。使用`bpy.ops.wm.save_as_mainfile`保存可编辑的场景，并使用`api.export_character(rig, path)`进行导出。
- 在CV环境中，使用`scripts/animate.py inspect|sample|validate`进行GLB操作。使用`scripts/skin.py analyze|contacts`进行皮肤和接缝测量，或者使用`start|step|inspect`进行版本化修复。修复计划需要包含所需的半径和诊断信息。同时检查诊断数字和实际播放情况。
- 运行 `python3 scripts/mh-ide-sync.py BUILD_PARENT --dry-run` 以检查注册信息，并在注册时删除 `--dry-run`。使用 `cd viewer && bun run dev` 浏览图库资源。

步行在世界空间穿地或离地时，见[用鞋底核对行走接地](references/walking-ground.zh-CN.md)，在穿衣和裸身上复查同一根平移。
