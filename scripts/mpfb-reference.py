#!/usr/bin/env python3
"""Generate a real MPFB reference in an isolated Blender process, using pinned inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
parser.add_argument("--blender", default=os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender"))
parser.add_argument("--gender", choices=["male", "female"], default="male")
parser.add_argument("--muscles", type=float, nargs="+", default=[.5, 1.])
parser.add_argument("--weight", type=float, default=.5)
parser.add_argument("--height", type=float, default=.5)
parser.add_argument("--age", type=float, default=.5)
parser.add_argument("--muscle-proxy", action="store_true")
args = parser.parse_args()
out = args.out.resolve()
if out.exists():
    parser.error(f"Use a new output directory: {out}")
if not Path(args.blender).is_file() and not shutil.which(args.blender):
    parser.error("Blender not found; pass --blender or set BLENDER")
manifest = json.loads((ROOT / "scripts/mpfb/sources.json").read_text())
tool_dir = ROOT / "out/tools/mpfb"
tool_dir.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_archive(name, info):
    archive = tool_dir / name
    if not archive.exists():
        partial = archive.with_suffix(".download")
        print(f"Downloading pinned MPFB input: {name}", flush=True)
        with urllib.request.urlopen(info["url"], timeout=120) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256(partial) != info["sha256"]:
            raise RuntimeError(f"Downloaded archive checksum mismatch: {partial}")
        partial.replace(archive)
    if sha256(archive) != info["sha256"]:
        raise RuntimeError(f"Archive checksum mismatch: {archive}")
    return archive


source_zip = ensure_archive("source.zip", manifest["mpfb"])
ensure_archive("makehuman_system_assets_cc0.zip", manifest["system_assets"])
revision = manifest["mpfb"]["revision"]
source = tool_dir / "source" / ("mpfb2-" + revision) / "src/mpfb"
if not source.is_dir():
    with zipfile.ZipFile(source_zip) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe archive entry")
        archive.extractall(tool_dir / "source")
revision_file = tool_dir / "revision.json"
if revision_file.exists():
    if json.loads(revision_file.read_text())["sha"] != revision:
        raise RuntimeError("Existing MPFB runtime uses a different revision")
else:
    revision_file.write_text(json.dumps({"sha": revision}, indent=2) + "\n")

env = os.environ.copy()
env["BLENDER_USER_CONFIG"] = str(tool_dir / "runtime/config")
env["BLENDER_USER_EXTENSIONS"] = str(tool_dir / "runtime/extensions")
command = [args.blender, "-b", "-t", "4", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
           "--python", str(ROOT / "scripts/mpfb/create-reference.py"), "--", str(out),
           "--gender", args.gender, "--weight", str(args.weight), "--height", str(args.height), "--age", str(args.age),
           "--muscles", *map(str, args.muscles)]
if args.muscle_proxy:
    command.append("--muscle-proxy")
out.parent.mkdir(parents=True, exist_ok=True)
log = out.with_name(out.name + ".log")
print(f"Generating MPFB reference; log: {log}", flush=True)
with log.open("w") as stream:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
if result.returncode:
    raise SystemExit(f"Blender failed ({result.returncode}); see {log}")
report = json.loads((out / "report.json").read_text())
print(json.dumps({"output": str(out), "presets": [r["preset"] for r in report["muscle_variants"]],
                  "mesh_to_meshy_fitting_complete": report["mesh_to_meshy_fitting_complete"]}, ensure_ascii=False))
