# 检查新角色模型的兼容性

[English](model-compatibility.md) · 简体中文

记录待测模型的原始网格、贴图、原有绑定、动作来源和再分发条件。若需预处理，请保留原件，并将 guide 绑定到实际用于迁移的文件。[扫描校准](../skills/charloom-scan-guide/SKILL.zh-CN.md)

用相同相机和灯光比较原模型、裸身版与穿衣版。如果原模型自带可用绑定，让它播放同一动作作为对照。风格化角色重点检查各肢体截面和头颈过渡；写实角色先将静止侧脸、眼形与唇线对照原扫描，再检查对白及眨眼帧。[身体拟合](mh-body-fitting.zh-CN.md) · [面部绑定](../skills/charloom-face-dialogue/SKILL.zh-CN.md)

对每个指标注明对应的网格、制作阶段、采样时刻、单位及所比较的表面。成对颈部顶点的间隙为零时，附近三角面仍可能相交。GLB 可以通过结构校验，却仍有牙齿错位、领口接缝或眼睑未闭严。到 [图库](../skills/charloom-gallery-review/SKILL.zh-CN.md) 播放完整动作，检查最终交付的模型并记录遗留问题。
