"""Show mh-transfer runs side by side with their source scans in the Web IDE.

    python3 scripts/mh-ide-review.py ROUND RUN_DIR... [--source NAME=SCAN.glb]

Each RUN_DIR is an `mh-transfer` output (as written by scripts/mh-review.sh;
its name, e.g. `orc-final-fit`, becomes the case id `ROUND-orc-final-fit`).
The run's files are copied to out/review/ide/ROUND/NAME/ (the IDE serves only
out/ and fixtures/), the source scan is written there in the run's working
frame (`report.working_frame.source_to_working`), and the case is added to
viewer/public/head-fit.json and viewer/dist/head-fit.json with suite "mh".
Open it at /characters/ROUND-NAME/editor: source left, result right, linked
cameras, and a version menu (dressed, naked body, fit diagnostic, scan labels).
"""
from pathlib import Path
import json
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("character", "穿衣角色（带走路动画）", "4-garments/character.glb"),
    ("body", "裸体身体 · 皮肤贴图", "3-skin/body.glb"),
    ("diagnostic", "拟合诊断 · 绿色为新折痕、白色为凸起", "2-body-rig/fit-diagnostic.glb"),
    ("segmentation", "扫描分区 · 皮肤/衣服/头发", "1-segment/segmentation.glb"),
]


def read_glb(path):
    b = path.read_bytes()
    assert b[:4] == b"glTF", path
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n]), b[20 + n:]


def write_glb(path, doc, rest):
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    chunk = struct.pack("<II", len(text), 0x4E4F534A) + text
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(chunk) + len(rest)) + chunk + rest)


def framed_source(scan, matrix, out):
    """The scan with one root node carrying `source_to_working`."""
    doc, rest = read_glb(scan)
    scene = doc["scenes"][doc.get("scene", 0)]
    doc["nodes"].append({"name": "source_to_working", "matrix": matrix, "children": scene["nodes"]})
    scene["nodes"] = [len(doc["nodes"]) - 1]
    write_glb(out, doc, rest)


def summary(r):
    b = r["stage2_body_fit"]
    f = b.get("face_landmark_residual_m") or {}
    eye = [v for k, v in f.items() if "eye" in k] or [0]
    head = r.get("stage2_head", {})
    route = head.get("route", "fit")
    if head.get("fallback_reason"):
        route += f"（keep_scan 未成功：{head['fallback_reason']}）"
    return (f"头部：{route} · 翻转三角形 {b['inverted_triangles']} · 最大边拉伸 "
            f"{b['distortion_vs_prior_rest']['edge_ratio']['max']:.2f} · 皮肤距离 p95 "
            f"{b['scan_skin_to_body_m']['p95'] * 1000:.1f} mm · 眼部残差 {max(eye) * 1000:.1f} mm · "
            f"面部凸起 {b.get('face_above_scan', {}).get('vertices_over_1mm')} · 新折痕 "
            f"{b.get('face_new_fold_vertices', {}).get('fitted')} · 走路穿模 "
            f"{r['stage5_checks'][0]['inside_fraction_over_frames']['max'] * 100:.2f}%")


def main(argv):
    sources = dict(a.split("=", 1) for a in argv if "=" in a)
    argv = [a for a in argv if "=" not in a and a != "--source"]
    rnd, runs = argv[0], [Path(a).resolve() for a in argv[1:]]
    entries = []
    for run in runs:
        name = run.name
        r = json.loads((run / "report.json").read_text())
        folder = ROOT / "out/review/ide" / rnd / name
        folder.mkdir(parents=True, exist_ok=True)
        scan = Path(sources.get(name) or r["source"])
        if not scan.is_absolute():
            scan = (ROOT / scan).resolve()
        framed_source(scan, r["working_frame"]["source_to_working"], folder / "source.glb")
        stages = []
        for sid, label, rel in STAGES:
            if (run / rel).exists():
                shutil.copyfile(run / rel, folder / f"{sid}.glb")
                stages.append({"id": sid, "label": label, "file": url(folder / f"{sid}.glb")})
        shutil.copyfile(run / "report.json", folder / "report.json")
        height = r["working_frame"]["height_m"]
        center = [0, height / 2, 0]
        title = name.replace("-final-fit", " · MakeHuman 头").replace("-final-ks", " · 扫描原头").replace("-ks1", " · 扫描原头")
        entries.append({
            "id": f"{rnd}-{name}", "suite": "mh", "title": f"{rnd} {title}",
            "resultSha256": "", "folder": url(folder), "report": url(folder / "report.json"),
            "sourceFile": url(folder / "source.glb"), "stages": stages,
            "hasFullBody": True, "hasClothingShell": False, "diagnosticOnly": False,
            "neckGeometryPassed": True, "bodyCenter": center, "center": center, "neckCenter": center,
            "appearanceNote": summary(r), "note": "MakeHuman 迁移结果：左侧为原扫描，右侧为结果。",
            "seconds": r.get("total_s", 0), "newCrossings": 0, "landmarkRmse": {},
        })
        print("registered", entries[-1]["id"])
    ids = {e["id"] for e in entries}
    for listing in [ROOT / "viewer/public/head-fit.json", ROOT / "viewer/dist/head-fit.json"]:
        if not listing.parent.exists():
            continue
        old = json.loads(listing.read_text()) if listing.exists() else []
        listing.write_text(json.dumps([e for e in old if e["id"] not in ids] + entries, ensure_ascii=False, indent=2) + "\n")


def url(path):
    return "/files/" + "/".join(Path(path).relative_to(ROOT).parts)


if __name__ == "__main__":
    main(sys.argv[1:])
