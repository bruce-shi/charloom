---
name: charloom-gallery-review
description: "核对 Charloom 最终 GLB、Gallery 条目、分层与实际动作。用于角色交付及修复后同帧复查。"
---

# 验证实际交付的角色

[English](SKILL.md) · 简体中文

1. 从构建报告确定原扫描、输出路径和哈希。检查最终 `build/3-skin/body.glb`、`build/4-garments/character.glb` 或本次实际选用的替代导出。将 `PY` 指向项目 CV 环境的 Python，用 `"$PY" scripts/animate.py validate FILE.glb` 验证文件。
2. 按改动选择复查范围。衣甲看完整步态、抬臂和转身；面部看眨眼、说话和转头；下颌、发束与衣领看正、侧、后及斜下视角。在原问题的相机、姿势、光照下重现修前画面，再拍修后和互补视角。
   对新增动作，在穿衣层播放完整周期，查看摆臂、转身和发束回摆时的接触。若[部位相交检查](../charloom/references/walk-replay.zh-CN.md#动作中的局部穿插)检出交叉，定位到报告中的时刻，在穿衣层从外侧和背侧看穿出位置，再切层寻找相交面。分别记录几何交叉和最终画面中的可见范围。使用像素数时注明镜头、分辨率和抽查帧。
3. 需要更新本地 Gallery 时，按项目的条目格式登记所选导出。从根目录运行 `python3 scripts/mh-ide-sync.py BUILD_PARENT --dry-run` 预览，再去掉 `--dry-run` 同步。使用已有本地 Gallery，或按[项目说明](../../README.zh-CN.md)启动。
4. 打开实际角色 URL，核对源窗与原扫描坐标，核对结果窗请求的 GLB 与已验文件哈希。裸模加载完整的裸模导出；穿衣和纯衣服使用装配 GLB，纯衣服按部件过滤。在暂停的同一帧切换三层，核对动作、时间及衣下身体的完整性，再验证播放中切层。播放所需动作，按任务核对骨架叠加。
5. 记录导出、视角、动作时间、测量和遗留问题。分别记下文件检查、Gallery 实际加载以及用户已给出的视觉反馈。
