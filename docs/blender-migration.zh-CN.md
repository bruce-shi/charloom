# Blender 制作工具

[English](blender-migration.md) · 简体中文

在 Blender 中编辑场景，再到本地 Gallery 检查导出结果。插件及其 Python 环境的安装步骤见 [项目说明文件](../README.zh-CN.md)。

| 任务 | 入口点 | 需要检查的内容 |
| --- | --- | --- |
| 校准并拟合扫描结果 | `scripts/scan-guide.py`, `scripts/scan-transfer.py` | 扫描与相机哈希、人体标记、已配准的身体轮廓及衣物面归属 |
| 绑定原扫描头 | `scripts/roster/build_blender_character.py` | 唇缝线、上下睑缘、牙弓，以及两版模型的颈部接缝 |
| 调整身体形状或衣物蒙皮权重 | `blender/charloom/api.py` | Blender 形态键、关节附近的权重、衣下被遮挡的身体面、衣物与皮肤间隙 |
| 重定向或叠加动作 | `scripts/animate.py` | 静止变换、关节映射、脚底接地、形态键轨道和道具约束 |
| 检查蒙皮变形 | `scripts/skin.py` | 采样帧中的边长变化、表面相交及接缝裂开情况 |
| 打开保存的工作空间 | `scripts/blender_project.py` | 导入的集合、已打包的纹理和原项目数据 |
| 审核交付 | [图库评审技能](../skills/charloom-gallery-review/SKILL.zh-CN.md) | 实际加载的 GLB、当前层、相机、动作及播放时间 |

新角色从[扫描工作流程](mh-transfer.zh-CN.md)开始；已有项目可按[工作区导入](local-workspace.zh-CN.md)导入。
