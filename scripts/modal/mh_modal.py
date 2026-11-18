"""Run Python scan transfers on Modal CPU containers and the Blender
review renders on Modal GPU containers.

    out/tools/modal/.venv/bin/modal run scripts/modal/mh_modal.py::main \
        --tag x8 --names orc,troll,goblin,brute [--config '{"texture_size": 2048}'] [--no-quick]

    out/tools/modal/.venv/bin/modal run scripts/modal/mh_modal.py::smoke_check   # GPU/EEVEE check

Per batch:
  1. the characters' source models and guidance folders are synced to the
     persistent volume `charloom-mh-assets` (`/sources/<name>/model.glb`,
     `/guides/<folder>/...`): only files new or changed since the last sync
     (size and modification time against `/manifest.json`) are uploaded;
  2. one CPU container per character runs the Python fit into
     `/<batch>/runs/<name>-<tag>/` on the volume `charloom-mh-jobs`;
  3. one GPU container per finished run renders it with Blender
     (`scripts/mh-render-review.py`, EEVEE and Workbench);
  4. every run folder is downloaded to `out/review/agents/<name>-<tag>/`;
  5. `/<batch>/` is deleted from the jobs volume: it holds only the batch in
     flight. The assets volume keeps the sources for the next batch.

The static assets (MakeHuman priors, underwear, the walking motion, the
garment frame) are baked into the images. The characters' sources, priors
and guidance folders follow `scripts/mh-batch.sh` (CHARACTERS below mirrors
its tables).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import modal

# The checkout on this machine; inside a container the module sits at
# /root/mh_modal.py and the image definitions below are not rebuilt there.
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/")
REPO = "/repo"
JOBS = "/jobs"
ASSETS = "/assets"
BLENDER = "/opt/blender/blender"

F = "out/examples/mpfb-female-reference-v3/female-muscle-050/body-shape-build.glb"
M = "out/examples/mpfb-reference-v7/male-muscle-050/body-shape-build.glb"
def characters():
    path = Path(os.environ.get("CHARLOOM_BATCH_CONFIG", ROOT / "assets/private/batch.json"))
    return json.loads(path.read_text())["characters"]


app = modal.App("charloom-mh")
jobs = modal.Volume.from_name("charloom-mh-jobs", create_if_missing=True)
assets = modal.Volume.from_name("charloom-mh-assets", create_if_missing=True)

# The renders: Blender (Linux build of the version used locally) on a GPU.
base_image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("curl", "xz-utils", "libx11-6", "libxi6", "libxrender1", "libxkbcommon0", "libsm6",
                 "libgl1", "libegl1", "libxxf86vm1", "libxfixes3", "libxext6", "libvulkan1", "libglib2.0-0t64")
    .run_commands(
        "curl -sSfL https://download.blender.org/release/Blender5.2/blender-5.2.1-linux-x64.tar.xz -o /tmp/b.tar.xz",
        "mkdir -p /opt/blender && tar -xJf /tmp/b.tar.xz -C /opt/blender --strip-components=1 && rm /tmp/b.tar.xz",
    )
    .env({"BLENDER": BLENDER})
)
render_image = (
    base_image.env({"NVIDIA_DRIVER_CAPABILITIES": "all"})
    .add_local_file(ROOT / "scripts/mh-render-review.py", f"{REPO}/scripts/mh-render-review.py", copy=True)
)


fit_image = (
    base_image.pip_install_from_requirements(str(ROOT / "scripts/requirements-blender.txt"))
    .add_local_dir(ROOT / "scripts", f"{REPO}/scripts", copy=True,
                   ignore=["**/__pycache__/**", "**/node_modules/**"])
    .add_local_file(ROOT / "fixtures/mixamo/walking.fbx", f"{REPO}/fixtures/mixamo/walking.fbx", copy=True)
    .add_local_dir(ROOT / "out/tools/mpfb/underwear-candidates", f"{REPO}/out/tools/mpfb/underwear-candidates", copy=True, ignore=["*.log", "*.py", "*.blend", "*.glb"])
    .add_local_file((next((ROOT / "out/tools/mpfb/source").glob("*/src/mpfb/data/3dobjs/base.obj")) if modal.is_local() else Path("/base.obj")),
                    f"{REPO}/out/tools/mpfb/underwear-candidates/base.obj", copy=True)
    .add_local_file(ROOT / F, f"{REPO}/{F}", copy=True)
    .add_local_file(ROOT / M, f"{REPO}/{M}", copy=True)
)


@app.function(image=fit_image, cpu=16, memory=32768, timeout=3600, volumes={JOBS: jobs, ASSETS: assets})
def fit(batch: str, name: str, tag: str, config: dict, prior: str, guide: str) -> dict:
    """One character's mh-transfer into /jobs/<batch>/runs/<name>-<tag>."""
    jobs.reload()
    source = Path(ASSETS) / "sources" / name / "model.glb"
    guidance = Path(ASSETS) / "guides" / guide / "guidance.json"
    run = Path(JOBS) / batch / "runs" / f"{name}-{tag}"
    run.parent.mkdir(parents=True, exist_ok=True)
    cfg = run.parent / f"{name}-{tag}.config.json"
    cfg.write_text(json.dumps(config))
    t = time.time()
    if set(config) - {"texture_size"}:
        raise ValueError(f"Unsupported transfer options: {sorted(set(config) - {'texture_size'})}")
    p = subprocess.run(
        [sys.executable, f"{REPO}/scripts/scan-transfer.py", str(source),
         "--prior", f"{REPO}/{prior}", "--guidance", str(guidance),
         "--texture-size", str(config.get("texture_size", 1024)), "--motion", f"{REPO}/fixtures/mixamo/walking.fbx", "--out", str(run)],
        cwd=REPO, capture_output=True, text=True)
    (run.parent / f"{name}-{tag}.log").write_text(p.stdout[-200000:] + "\n" + p.stderr[-200000:])
    jobs.commit()
    return {"name": name, "ok": p.returncode == 0, "seconds": round(time.time() - t), "tail": p.stderr[-600:]}


# Any of these GPUs (in order of preference): waiting for L4s alone left
# renders queued for over half an hour.
RENDER_GPUS = ["L4", "A10G", "L40S", "T4"]


@app.function(image=render_image, gpu=RENDER_GPUS, cpu=8, memory=16384, timeout=1800, volumes={JOBS: jobs, ASSETS: assets})
def render(batch: str, name: str, tag: str) -> dict:
    """Blender review renders of one run, written into its renders/."""
    jobs.reload()
    run = Path(JOBS) / batch / "runs" / f"{name}-{tag}"
    t = time.time()
    p = subprocess.run([BLENDER, "-b", "--factory-startup", "--python", f"{REPO}/scripts/mh-render-review.py",
                        "--", str(run)], capture_output=True, text=True)
    (run / "renders.log").write_text(p.stdout[-200000:] + "\n" + p.stderr[-200000:])
    jobs.commit()
    return {"name": name, "ok": p.returncode == 0, "seconds": round(time.time() - t), "tail": (p.stdout + p.stderr)[-600:]}


@app.function(image=render_image, gpu="L4", timeout=600)
def smoke() -> str:
    """EEVEE on the GPU container: render a cube, report the device."""
    script = (
        "import bpy;bpy.ops.wm.read_factory_settings(use_empty=False);s=bpy.context.scene;"
        "e=[x.identifier for x in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items];"
        "s.render.engine='BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in e else 'BLENDER_EEVEE';"
        "s.render.resolution_x=s.render.resolution_y=256;s.render.filepath='/tmp/smoke.png';"
        "bpy.ops.render.render(write_still=True);import gpu;print('GPU', gpu.platform.renderer_get(), gpu.platform.backend_type_get())"
    )
    p = subprocess.run([BLENDER, "-b", "--python-expr", script], capture_output=True, text=True)
    nv = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return f"exit {p.returncode}\n{nv.stdout}\n" + "\n".join(
        l for l in (p.stdout + p.stderr).splitlines() if "GPU" in l or "rror" in l or "EGL" in l or "Saved" in l)[-3000:]


def sync_assets(names: list[str], table: dict) -> tuple[int, int]:
    """Upload the characters' sources and guidance folders to the assets
    volume, only what is new or changed since the last sync; returns
    (files uploaded, bytes uploaded)."""
    try:
        manifest = json.loads(b"".join(assets.read_file("manifest.json")))
    except Exception:
        manifest = {}
    wanted = {}
    for n in names:
        src, _, guide = table[n]
        wanted[f"sources/{n}/model.glb"] = ROOT / src
        g = ROOT / "out/examples" / f"mh-guide-{guide}"
        for f in g.rglob("*"):
            if f.is_file() and "/check/" not in str(f):
                wanted[f"guides/{guide}/{f.relative_to(g)}"] = f
    changed = {k: f for k, f in wanted.items()
               if manifest.get(k) != [f.stat().st_size, f.stat().st_mtime_ns]}
    if changed:
        with assets.batch_upload(force=True) as up:
            for k, f in changed.items():
                up.put_file(str(f), f"/{k}")
        for k, f in changed.items():
            manifest[k] = [f.stat().st_size, f.stat().st_mtime_ns]
        with assets.batch_upload(force=True) as up:
            up.put_file(__import__("io").BytesIO(json.dumps(manifest).encode()), "/manifest.json")
    return len(changed), sum(f.stat().st_size for f in changed.values())


def download(batch: str, run: str, dest: Path):
    """The volume's /<batch>/runs/<run>[...] into dest (a run folder and its
    .config.json / .log beside it), with the `modal volume get` CLI: its
    parallel transfer took 36 s for four runs, where reading file by file
    through the client stalled for an hour on 2 MB."""
    exe = Path(sys.executable).parent / "modal"
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in ("", ".log", ".config.json"):
        p = subprocess.run([str(exe), "volume", "get", "--force", "charloom-mh-jobs",
                            f"/{batch}/runs/{run}{suffix}", str(dest)], capture_output=True, text=True)
        if p.returncode != 0 and not suffix:
            print("download failed", run, p.stderr[-400:])


@app.local_entrypoint()
def main(tag: str, names: str, config: str = "{}", quick: bool = True, render_runs: bool = True):
    names_l = [n.strip() for n in names.split(",") if n.strip()]
    table = characters()
    unknown = [n for n in names_l if n not in table]
    if unknown:
        raise SystemExit(f"unknown characters: {unknown}")
    cfg = {"texture_size": 1024 if quick else 2048}
    cfg.update(json.loads(config))
    batch = f"batch-{tag}-{uuid.uuid4().hex[:8]}"
    dest = ROOT / "out/review/agents"
    print("batch", batch)
    try:
        files, size = sync_assets(names_l, table)
        print(f"synced assets: {files} files, {size / 1e6:.1f} MB uploaded")
        fits = list(fit.map([batch] * len(names_l), names_l, [tag] * len(names_l), [cfg] * len(names_l),
                            [table[n][1] for n in names_l], [table[n][2] for n in names_l]))
        for r in fits:
            print("fit", r["name"], "ok" if r["ok"] else "FAILED", f'{r["seconds"]} s', "" if r["ok"] else r["tail"])
        ok = [r["name"] for r in fits if r["ok"]]
        if render_runs and ok:
            for r in render.map([batch] * len(ok), ok, [tag] * len(ok)):
                print("render", r["name"], "ok" if r["ok"] else "FAILED", f'{r["seconds"]} s', "" if r["ok"] else r["tail"])
        for n in names_l:
            run = f"{n}-{tag}"
            if (dest / run).exists():
                shutil.rmtree(dest / run)
            download(batch, run, dest)
            # Paths in the report point into the volume; the review tools
            # read `source` from it, so point it back at the local source.
            rep = dest / run / "report.json"
            if rep.exists():
                r = json.loads(rep.read_text())
                r["source"] = str(ROOT / table[n][0])
                rep.write_text(json.dumps(r))
            print("downloaded", dest / run)
    finally:
        # Clean the jobs volume: nothing of this batch stays on it (the
        # assets volume keeps the sources).
        try:
            jobs.remove_file(f"/{batch}", recursive=True)
        except Exception as e:
            print("nothing to remove:", e)
        left = [e.path for e in jobs.listdir("/")]
        print("removed", batch, "from the jobs volume; left on it:", left)


@app.local_entrypoint()
def smoke_check():
    """`modal run scripts/modal/mh_modal.py::smoke_check`: print the GPU check."""
    print(smoke.remote())
