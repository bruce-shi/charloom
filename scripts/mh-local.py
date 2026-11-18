"""Run local Python scan-transfer batches and render their staged exports.

    python3 scripts/mh-local.py --tag review [--names dimlight,orc] [--config JSON]
                                [--full] [--no-render] [--jobs N] [--render-jobs N]
                                [--fast] [--smoke] [--sync]

The batch reads the same character/prior/guide table as scripts/modal/mh_modal.py.
Texture size defaults to 1024, or 2048 with --full. Override texture_size with
--config. --fast caps it at 512 and defaults to three concurrent fits and the
CPU renderer. MH_RENDER selects blender or cpu. --smoke selects five characters.
Run directories and logs go under out/review/agents. --sync registers results.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CV_PYTHON = Path(os.environ.get("CHARLOOM_CV_PYTHON",ROOT/"out/tools/cv/.venv/bin/python"))
BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
DEST = ROOT / "out/review/agents"


# Threads per fit when several run at once (`--jobs`); 0 = every core.
THREADS = 0
def fit(name, tag, config, table):
    src, prior, guide = table[name]
    run = DEST / f"{name}-{tag}"
    if run.exists():
        raise FileExistsError(run)
    cfg = DEST / f"{name}-{tag}.config.json"
    config = dict(config)
    cfg.write_text(json.dumps(config))
    env = os.environ.copy()
    if THREADS:
        env["OPENBLAS_NUM_THREADS"] = str(THREADS)
        env["OMP_NUM_THREADS"] = str(THREADS)
    if env.get("MH_DUMP_STAGES"):
        env["CHARLOOM_DUMP_STAGES"] = str(Path(env["MH_DUMP_STAGES"]) / f"{name}-{tag}-stages")
    t = time.time()
    p = subprocess.run(
        [str(CV_PYTHON), str(ROOT/"scripts/scan-transfer.py"), str(ROOT / src),
         "--prior", str(ROOT / prior),
         "--guidance", str(ROOT / "out/examples" / f"mh-guide-{guide}" / "guidance.json"),
         "--texture-size", str(config.get("texture_size",1024)), "--motion", str(ROOT / "fixtures/mixamo/walking.fbx"), "--out", str(run)],
        cwd=ROOT, capture_output=True, text=True, env=env)
    (DEST / f"{name}-{tag}.log").write_text(p.stdout[-200000:] + "\n" + p.stderr[-200000:])
    return p.returncode == 0, round(time.time() - t), p.stderr[-600:]


def render(name, tag):
    run = DEST / f"{name}-{tag}"
    t = time.time()
    backend = os.environ.get("MH_RENDER", "blender")
    if backend == "cpu":
        command = [str(CV_PYTHON), str(ROOT/"scripts/render-character.py"), str(run)]
    elif backend == "blender":
        command = [BLENDER, "-b", "--factory-startup", "--python",
                   str(ROOT / "scripts/mh-render-review.py"), "--", str(run)]
    else:
        raise ValueError(f"unknown MH_RENDER={backend!r}; use blender or cpu")
    p = subprocess.run(command, capture_output=True, text=True)
    (run / "renders.log").write_text(p.stdout[-200000:] + "\n" + p.stderr[-200000:])
    (run / "renders").mkdir(exist_ok=True)
    (run / "renders/backend.json").write_text(json.dumps({
        "backend": backend, "command": command, "returncode": p.returncode,
        "seconds": round(time.time() - t, 2),
    }, indent=2) + "\n")
    return p.returncode == 0, round(time.time() - t), (p.stdout + p.stderr)[-600:]


def main(argv):
    def opt(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            return argv[i + 1]
        return default

    global THREADS
    tag = opt("--tag")
    if not tag:
        raise SystemExit(__doc__)
    fast = "--fast" in argv
    path = Path(os.environ.get("CHARLOOM_BATCH_CONFIG", ROOT / "assets/private/batch.json"))
    batch = json.loads(path.read_text())
    table = batch["characters"]
    default_names = ",".join(batch.get("smoke", list(table)[:5]) if "--smoke" in argv else table)
    names = [n.strip() for n in opt("--names", default_names).split(",") if n.strip()]
    unknown = [n for n in names if n not in table]
    if unknown:
        raise SystemExit(f"unknown characters: {unknown}")
    config = {"texture_size":2048 if "--full" in argv else 1024}
    config.update(json.loads(opt("--config", "{}")))
    if set(config)-{"texture_size"}:
        raise ValueError("The Python batch accepts texture_size as a config override")
    if fast:
        config["texture_size"] = min(config["texture_size"],512)
        os.environ.setdefault("MH_RENDER", "cpu")
    jobs = max(1, int(opt("--jobs", "3" if fast else "1")))
    render_jobs = max(1, int(opt("--render-jobs", "2" if fast else "1")))
    if jobs > 1:
        THREADS = max(2, (os.cpu_count() or 8) // jobs)
    if not CV_PYTHON.exists():
        raise SystemExit(f"Missing CV Python interpreter: {CV_PYTHON}")
    started = time.time()
    renders = []
    failures = []

    def one(n):
        ok, sec, tail = fit(n, tag, config, table)
        print("fit", n, "ok" if ok else "FAILED", f"{sec} s", "" if ok else tail, flush=True)
        return n, ok

    with ThreadPoolExecutor(max_workers=render_jobs) as renderer, ThreadPoolExecutor(max_workers=jobs) as fitter:
        for n, ok in fitter.map(one, names):
            if not ok:
                failures.append(f"fit {n}")
            if ok and "--no-render" not in argv:
                renders.append((n, renderer.submit(render, n, tag)))
        for n, f in renders:
            ok, sec, tail = f.result()
            print("render", n, "ok" if ok else "FAILED", f"{sec} s", "" if ok else tail, flush=True)
            if not ok:
                failures.append(f"render {n}")
    print(f"batch: {len(names)} characters in {round(time.time() - started)} s "
          f"({jobs} fits at once, {render_jobs} renders at once, {os.environ.get('MH_RENDER', 'blender')} renderer)", flush=True)
    if failures:
        raise SystemExit("batch failed: " + ", ".join(failures))
    if "--sync" in argv:
        subprocess.run([sys.executable, str(ROOT / "scripts/mh-ide-sync.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
