"""Register local transfer results in the character viewer."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys

STAGES = [
    ("character", "穿衣角色（带走路动画）", "4-garments/character.glb"),
    ("body", "裸体身体 · 皮肤贴图", "3-skin/body.glb"),
    ("diagnostic", "拟合诊断 · 绿色为新折痕、白色为凸起", "2-body-rig/fit-diagnostic.glb"),
    ("segmentation", "扫描分区 · 皮肤/衣服/头发", "1-segment/segmentation.glb"),
]
                                                                             
                                                                        
                                        
CLOSEUPS = [
                                                                        
                                                                        
                                                   
    "renders/evidence-6-heights.png", "renders/evidence-4-match.jpg", "renders/evidence-5-skin-error.png", "renders/evidence-1-sheet.jpg", "renders/evidence-2-front.jpg",
    "renders/evidence-3-side.jpg",
    "0-guide/qa-front.png", "0-guide/qa-side.png",
    "renders/eyes-scan-front.png", "renders/eyes-open-front.png", "renders/eyes-blink-front.png",
    "renders/eyes-scan-three-quarter.png", "renders/eyes-open-three-quarter.png",
    "renders/eyes-blink-three-quarter.png", "neck/montage.jpg",
]
RENDERS = [
    "4-garments-character-eevee-front-rest-crop.png",
    "4-garments-character-front-rest.png",
    "4-garments-character-eevee-side-rest-crop.png",
    "4-garments-character-side-rest.png",
    "4-garments-character-back-rest.png",
    "4-garments-character-front-0.25.png",
    "4-garments-character-side-0.60.png",
    "neck-character-quarter-rest.png",
    "neck-character-front-rest.png",
    "3-skin-body-eevee-front-rest-crop.png",
    "3-skin-body-front-rest.png",
    "1-segment-pieces-front-rest.png",
]
                                                                             
GENERIC = {"model", "raw-body", "source", "body", "scan"}


def repo_root():
    here = Path(__file__).resolve().parents[1]
    if (here / "out").is_dir():
        return here
                                                             
    common = subprocess.run(["git", "-C", str(here), "rev-parse", "--path-format=absolute",
                             "--git-common-dir"], capture_output=True, text=True).stdout.strip()
    return Path(common).parent if common else here


def glb_json(path):
    """The glTF JSON chunk of a GLB, without reading its binary chunk."""
    with open(path, "rb") as f:
        head = f.read(20)
        if head[:4] != b"glTF":
            return {}
        return json.loads(f.read(struct.unpack("<I", head[12:16])[0]))


def garment_parts(path):
    """Part kinds a character GLB marks on its meshes (`charloom_part`)."""
    try:
        return {(m.get("extras") or {}).get("charloom_part") for m in glb_json(path).get("meshes", [])} - {None}
    except (OSError, ValueError):
        return set()


def read_glb(path):
    b = path.read_bytes()
    assert b[:4] == b"glTF", path
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n]), b[20 + n:]


def write_glb(path, doc, rest):
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    chunk = struct.pack("<II", len(text), 0x4E4F534A) + text
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(chunk) + len(rest)) + chunk + rest)
    tmp.replace(path)


def framed_source(scan, matrix, out):
    doc, rest = read_glb(scan)
    scene = doc["scenes"][doc.get("scene", 0)]
    doc["nodes"].append({"name": "source_to_working", "matrix": matrix, "children": scene["nodes"]})
    scene["nodes"] = [len(doc["nodes"]) - 1]
    write_glb(out, doc, rest)


def get(d, *path, default=None):
    for k in path:
        if isinstance(d, list):
            d = d[k] if -len(d) <= k < len(d) else None
        elif isinstance(d, dict):
            d = d.get(k)
        else:
            return default
        if d is None:
            return default
    return d


def metrics(r):
    b = r.get("stage2_body_fit") or {}
    f = b.get("face_landmark_residual_m") or {}
    eye = [v for k, v in f.items() if "eye" in k]
    walk = get(r, "stage5_checks", 0, "inside_fraction_over_frames", "max")
    m = {
        "inverted": b.get("inverted_triangles"),
        "edgeMax": get(b, "distortion_vs_prior_rest", "edge_ratio", "max"),
        "skinP95Mm": None if get(b, "scan_skin_to_body_m", "p95") is None else b["scan_skin_to_body_m"]["p95"] * 1000,
        "eyeMm": max(eye) * 1000 if eye else None,
        "faceBumps": get(b, "face_above_scan", "vertices_over_1mm"),
        "newFolds": get(b, "face_new_fold_vertices", "fitted"),
        "walkPct": None if walk is None else walk * 100,
        "seconds": r.get("total_s"),
    }
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}


def head_route(r):
    head = r.get("stage2_head") or {}
    route = head.get("route", "fit")
    if head.get("fallback_reason"):
        route += f"（keep_scan 未成功：{head['fallback_reason']}）"
    return route


def summary(r, m):
    def fmt(v, f="{:.1f}"):
        return "—" if v is None else f.format(v)
    return (f"头部：{head_route(r)} · 翻转三角形 {fmt(m['inverted'], '{}')} · 最大边拉伸 {fmt(m['edgeMax'], '{:.2f}')} · "
            f"皮肤距离 p95 {fmt(m['skinP95Mm'])} mm · 眼部残差 {fmt(m['eyeMm'])} mm · 面部凸起 {fmt(m['faceBumps'], '{}')} · "
            f"新折痕 {fmt(m['newFolds'], '{}')} · 走路穿模 {fmt(m['walkPct'], '{:.2f}')}%")


def character_of(source):
    p = Path(source)
    for name in [p.stem, p.parent.name, p.parent.parent.name]:
        if name and name not in GENERIC:
            return re.sub(r"-unrigged$", "", name)
    return p.stem


def change_of(tag, group):
    return ""


def find_runs(base, depth=3):
    """Run directories under `base`, as (path, parts relative to base)."""
    out = []
    def walk(d, parts):
        if (d / "report.json").is_file():
            if (d / "4-garments/character.glb").is_file() or (d / "character.glb").is_file():
                out.append((d, parts))
            return
        if len(parts) >= depth:
            return
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir() and not c.name.startswith("."))
        except OSError:
            return
        for c in children:
            if c.name in {"sources", "renders", "0-guide", "1-segment", "2-body-rig", "3-skin", "4-garments", "5-animation"}:
                continue
            walk(c, parts + [c.name])
    walk(base, [])
    return out


def main(argv):
    dry = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]
    root = repo_root()
    if "--root" in argv:
        i = argv.index("--root")
        root = Path(argv[i + 1]).resolve()
        del argv[i:i + 2]
    keep = 1
    if "--keep" in argv:
        i = argv.index("--keep")
        keep = int(argv[i + 1])
        del argv[i:i + 2]
    forget = set()
    while "--forget" in argv:
        i = argv.index("--forget")
        forget.add(str(Path(argv[i + 1]).resolve()))
        del argv[i:i + 2]
    out = (root / "out").resolve()
    ide = out / "review/ide"
    remembered = ide / "sync-dirs.txt"
    extras = [l.strip() for l in remembered.read_text().splitlines() if l.strip()] if remembered.exists() else []
    extras = [e for e in dict.fromkeys(extras + [str(Path(a).resolve()) for a in argv]) if e not in forget]
    bases = [(ide, "ide"), (out / "review/agents", "agents")] + [(Path(e), "tmp") for e in extras]
    served = [out, (root / "fixtures").resolve()]

    def url(path):
        path = Path(path).resolve()
        if not path.is_relative_to(root.resolve()):
            return str(path)                                             
        return "/files/" + "/".join(path.relative_to(root.resolve()).parts)

    def images(look, alias, tag):
        standard = [url(look / "renders" / n) for n in RENDERS if (look / "renders" / n).exists()]
        closeups = [url(look / rel) for rel in CLOSEUPS if (look / rel).exists()]
        return standard, closeups

    seen, entries, skipped, copied = {}, [], [], 0
    for base, kind in bases:
        if not base.is_dir():
            if kind == "tmp":
                print("missing extra dir (kept in sync-dirs.txt; use --forget to drop):", base)
            continue
        for run, parts in find_runs(base):
            try:
                raw = (run / "report.json").read_bytes()
                r = json.loads(raw)
                assert isinstance(r, dict) and r.get("working_frame")
            except Exception as e:
                skipped.append(f"{run}: unreadable report ({e.__class__.__name__})")
                continue
            digest = hashlib.sha256(raw).hexdigest()
            name = run.name
            alias, _, tag = name.partition("-")
            tag = tag or name
            if digest in seen:
                                                                             
                                                                                            
                first = seen[digest]
                if not first["renders"] and run.resolve().is_relative_to(out):
                    standard, closeups = images(run, alias, first["tag"])
                    first["thumbnail"] = standard[0] if standard else None
                    first["renders"] = closeups + standard
                continue
            group = "/".join(parts[:-1])
            if kind == "ide" and group == "tmp":
                group = ""                                    
            rid = ("tmp-" if kind == "tmp" else "") + "-".join(parts)
            inside = run.resolve().is_relative_to(out)
            folder = run
            if not inside:
                                                                                     
                folder = ide / "tmp" / name
                if not dry:
                    folder.mkdir(parents=True, exist_ok=True)
                    for sid, _, rel in STAGES:
                        src = run / rel if (run / rel).exists() else run / f"{sid}.glb"
                        dst = folder / f"{sid}.glb"
                        if src.exists() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
                            shutil.copy2(src, dst)
                            copied += 1
                    shutil.copy2(run / "report.json", folder / "report.json")
                    for rel in CLOSEUPS + [f"renders/{n}" for n in RENDERS]:
                        if (run / rel).exists():
                            (folder / rel).parent.mkdir(exist_ok=True)
                            shutil.copy2(run / rel, folder / rel)
                                                                                                
                rid = "tmp-" + name
            look = run if dry and not inside else folder
            stages = []
            for sid, label, rel in STAGES:
                for cand in [look / rel, look / f"{sid}.glb"]:
                    if cand.exists():
                        stages.append({"id": sid, "label": label, "file": url(cand)})
                                                                                        
                        if sid == "character" and "garment" in garment_parts(cand):
                            stages.append({"id": "clothes", "label": "只看衣服", "file": url(cand), "parts": ["garment"]})
                        break
            source = Path(r.get("source", ""))
            if not source.is_absolute():
                source = (root / source).resolve()
            matrix = get(r, "working_frame", "source_to_working")
            entry_source, source_matrix = None, None
            if (folder / "source.glb").exists():
                entry_source = url(folder / "source.glb")
            elif source.exists() and any(source.resolve().is_relative_to(s) for s in served):
                entry_source, source_matrix = url(source), matrix
            elif source.exists():
                key = hashlib.sha256((r.get("source_hash", str(source)) + json.dumps(matrix)).encode()).hexdigest()[:10]
                framed = ide / "sources" / f"{character_of(source)}-{key}" / "source.glb"
                if not framed.exists() and not dry:
                    framed.parent.mkdir(parents=True, exist_ok=True)
                    framed_source(source, matrix, framed)
                entry_source = url(framed) if framed.exists() else None
            if not entry_source:
                skipped.append(f"{run}: source scan not found ({source})")
                continue
            standard, closeups = images(look, alias, tag)
            height = get(r, "working_frame", "height_m", default=1.75)
            center = [0, height / 2, 0]
            m = metrics(r)
            stamp = (run / "report.json").stat().st_mtime
            character = character_of(source)
            entries.append({
                "id": rid, "suite": "mh",
                "title": f"{character} · {tag}" + (f" ({group})" if group else ""),
                "character": character, "alias": alias, "tag": tag, "round": group,
                "change": change_of(tag, group.split("/")[-1] if group else ""),
                "headRoute": head_route(r), "metrics": m,
                "timestamp": datetime.fromtimestamp(stamp).isoformat(timespec="seconds"),
                "runDir": str(run), "location": kind,
                "resultSha256": digest, "folder": url(folder), "report": url(folder / "report.json"),
                "sourceFile": entry_source, **({"sourceMatrix": source_matrix} if source_matrix else {}),
                "stages": stages, "thumbnail": standard[0] if standard else None, "renders": closeups + standard,
                "hasFullBody": True, "hasClothingShell": False, "diagnosticOnly": False,
                "neckGeometryPassed": True, "bodyCenter": center, "center": center, "neckCenter": center,
                "appearanceNote": summary(r, m),
                "note": "MakeHuman 迁移结果：左侧为原扫描，右侧为结果。",
                "seconds": r.get("total_s", 0), "newCrossings": 0, "landmarkRmse": {},
            })
            seen[digest] = entries[-1]
                                                                             
                                                                              
    reviewed = {}
    for manifest in sorted(out.glob("review/*/gallery-entries.json")):
        for entry in json.loads(manifest.read_text()):
            paths = [entry["report"], entry["sourceFile"], *(stage["file"] for stage in entry["stages"])]
            if all(path.startswith("/files/") and (root / path[7:]).is_file() for path in paths):
                previous = reviewed.get(entry["id"])
                if previous is None or (datetime.fromisoformat(entry["timestamp"]).timestamp()
                                        > datetime.fromisoformat(previous["timestamp"]).timestamp()):
                    reviewed[entry["id"]] = entry
                                                                         
                                                               
    entries = [entry for entry in entries if entry["id"] not in reviewed]
    entries.extend(reviewed.values())
                                                                    
    ids = {}
    for e in entries:
        n = ids.get(e["id"], 0)
        ids[e["id"]] = n + 1
        if n:
            e["id"] += f"-{n + 1}"
    if keep > 0:
        newest = sorted(entries, key=lambda e: e["timestamp"], reverse=True)
        count, listed = {}, []
        for e in newest:
            count[e["character"]] = count.get(e["character"], 0) + 1
            if count[e["character"]] <= keep:
                listed.append(e)
        print(f"listing the {keep} newest runs per character ({len(entries) - len(listed)} older runs left out)")
        entries = listed
    entries.sort(key=lambda e: (e["character"], e["timestamp"]), reverse=False)
    for s in skipped:
        print("skip", s)
    chars = sorted({e["character"] for e in entries})
    print(f"{len(entries)} runs, {len(chars)} characters ({', '.join(chars)}); copied {copied} files; skipped {len(skipped)}")
    if dry:
        return
    ide.mkdir(parents=True, exist_ok=True)
    remembered.write_text("".join(e + "\n" for e in extras))
    target = ide / "results.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"), "runs": entries},
                              ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, target)
    print("wrote", target)


if __name__ == "__main__":
    main(sys.argv[1:])
