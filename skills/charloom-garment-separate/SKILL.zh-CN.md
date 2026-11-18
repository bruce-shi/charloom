---
name: charloom-garment-separate
description: "拆分 Charloom 扫描中的皮肤、头发、衣物和配件。用于服装层残留旧手指、悬浮指甲或头发与衣领粘连。"
---

# 判定扫描三角面的归属

[English](SKILL.md) · 简体中文

1. 用原扫描、裸模、纯衣服和穿衣版复现同一姿势、视角。隐藏整件后再次检查；同一物体含多种材质时，在编辑模式逐面隔离。依据连接、深度、解剖位置和贴图一起判断。
2. 将确认的原始三角面或组件 ID 绑定到本次 `source_hash`。有待判定组件时在 review JSON 的 `decisions` 中按组件 ID 标注 `skin`、`internal`、`brow` 或带衣物名称的 `piece`。在新输出目录用 `--segment-evidence`、`--segment-review` 重放。
3. **手边有碎片**时，分别查看左右袖口到指尖。若隐藏新手后，旧指甲和肤色碎面仍留在纯服装层，移除确认属于原手的面。检查修复后的空袖口、手套边和两手轮廓。
4. 角色有甲油时，采样原甲色并写到当前手指的甲床 UV 或已绑定的指甲上。MakeHuman 手可用与当前 UV 匹配的指甲遮罩。逐指核对遮罩落点，在同一帧切换裸模、纯衣服和穿衣版，再播放手腕运动，确认甲色始终贴合对应指尖。

头发与衣领共用面片时，沿可见材质边界切分，复查发束背面和空衣领。[衣领、皮肤和头发共面的处理](references/collar-and-hair.zh-CN.md)

## 拟合前独立分件

```sh
"$PY" skills/charloom-garment-separate/scripts/review_parts.py "$SCAN" --guide "$GUIDE" --out "$PARTS"
```

输出 `components.json`（每组件的原面范围与摘要）、`components.png`、每件正侧后视图和 `decisions.template.json`。复制模板为 `decisions.json` 后逐项填写 `action`：`skin`、`internal`、`brow`，或 `piece` 加 `name`（guide 中已存在的衣物名）。模板空值表示尚未判定，不默认塞入衣服。

混合组件不要整体归类：先在 guide 正侧后图用皮肤/衣物多边形或笔刷细分，再重做分件。若需要明确子面范围，在证据中用这些子面的半开范围保存新候选，其 id 是有序面号数组 `<u8` 字节的 SHA-256，并用相同 id 写 decision。范围不能重叠冲突。不能把整个含头发和衣领的连通片标为头发。

在新目录复放并检查：

```sh
"$PY" skills/charloom-garment-separate/scripts/review_parts.py "$SCAN" --guide "$GUIDE" --evidence "$PARTS/components.json" --review "$PARTS/decisions.json" --out "$PARTS_REPLAY"
```

输出 `owners.npy`、`review.json`、`skin-regions.json`。最后一个文件的 `triangle_ranges` 可交给量体脚本作为可见皮肤候选，但须排除没有真正可见的隐蔽面，眉毛单列。沿用同一证据与决定传给 `scan-transfer.py --segment-evidence ... --segment-review ...`。改变原网格后重做对应；更名衣物后核对 guide 与 decision。
