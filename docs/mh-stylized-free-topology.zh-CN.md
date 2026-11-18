# 组装并复查风格化角色

[English](mh-stylized-free-topology.md) · 简体中文

从正面、侧面和四分之三视角检查原扫描头及其贴图。辨认眼球、眼睑、睫毛、口腔表面和面部毛发。在眼裂和唇缝周围保留原有头部表面和 UV。将牙齿和舌头绑定到已测量的口腔内部，核对拟合后的颈部与身体、领口的接合。[面部组合与对话](../skills/charloom-face-dialogue/SKILL.zh-CN.md)

按原扫描的几何结构选择身体路线。常规路线拟合[MakeHuman模板](mh-body-fitting.zh-CN.md)；原扫描含独立躯干及四个完整肢体壳体时，可用[扫描体积身体](../skills/charloom-scan-volume/SKILL.zh-CN.md)。组装前保存源哈希及已复核的各壳体种子面。

单个角色可用[Blender构建命令](../README.zh-CN.md)，输入已复核的原扫描资料或已有 transfer 目录。若要批量从标定迁移结果重建指定角色，准备自己的 roster JSON，在 `characters` 数组中列出带名称的条目。迁移目录下每个角色使用与条目名称相同的子目录。

```sh
"$CHARLOOM_CV_PYTHON" scripts/roster/rebuild.py "$NAME" \
  --roster "$ROSTER_JSON" --transfers "$TRANSFERS" --out "$NEW_OUTPUT"
```

分别在裸身和穿衣版核对 M/B/P 闭唇、F/V 时下唇与上齿接触、闭眼、视线和下颌至颈部的轮廓。播放整段步行，检查手部、运动中的颈缝及衣物与皮肤间隙。按[图库评审技能](../skills/charloom-gallery-review/SKILL.zh-CN.md)确认查看器实际加载的文件和帧。
