"""Aggregate mh-qa over the 16 characters: per metric, how many got better or worse than a base tag.

    python3 scripts/mh-qa-compare.py BASE_TAG TAG [TAG...]
"""
import os, json, subprocess, sys, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
table = Path(os.environ.get("CHARLOOM_BATCH_CONFIG", ROOT / "assets/private/batch.json"))
NAMES = list(json.loads(table.read_text())["characters"])
base, tags = sys.argv[1], sys.argv[2:]
KEYS = ["inverted triangles", "edge stretch max", "face new folds", "skin p95 torso", "skin p95 arm", "skin p95 leg",
        "asymmetry p95 hidden", "asymmetry p95 mixed", "walk poke-through max", "worst garment in walk",
        "worst garment at rest", "limb length L/R (rig,", "new creases torso", "chest body/dressed",
        "waist body/dressed", "hips body/dressed", "thigh body/dressed", "upper_arm body/dressed"]


def qa(tag):
    runs = [str(ROOT / "out/review/agents" / f"{n}-{tag}") for n in NAMES]
    missing = [Path(r).name for r in runs if not (Path(r) / "report.json").is_file()]
    if missing:
        raise SystemExit(f"incomplete {tag} batch: missing reports for {', '.join(missing)}")
    out = ROOT / f"out/review/ensemble-logs/qa-{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/mh-qa.py"), *runs, "--json", str(out)],
                   capture_output=True, text=True, check=True)
    data = json.loads(out.read_text())
    res = {}
    for run, rows in data.items():
        n = run.rsplit("-", 1)[0]
        d = {}
        for grade, cls, what, value, unit in rows:
            for k in KEYS:
                if what.startswith(k) and isinstance(value, (int, float)):
                    d[k] = max(d.get(k, value), value) if k.startswith("limb") else value
        d["_fails"] = sum(1 for r in rows if r[0] == "FAIL")
        d["_warns"] = sum(1 for r in rows if r[0] == "WARN")
        res[n] = d
    return res


B = qa(base)
for t in tags:
    T = qa(t)
    common = [n for n in NAMES if n in B and n in T]
    print(f"\n### {t} vs {base} ({len(common)} characters)")
    print(f"  {'metric':30} better worse  median change   sum base -> {t}")
    for k in KEYS + ["_fails", "_warns"]:
        pairs = [(B[n][k], T[n][k]) for n in common if k in B[n] and k in T[n]]
        if not pairs:
            continue
        # body/dressed: closer to 1 is not the point; report only.
        better = sum(1 for a, b in pairs if b < a * 0.97)
        worse = sum(1 for a, b in pairs if b > a * 1.03)
        rel = [(b - a) / a for a, b in pairs if a]
        med = statistics.median(rel) if rel else 0
        print(f"  {k:30} {better:6} {worse:5}  {med*100:+7.1f} %   {sum(a for a,_ in pairs):9.1f} -> {sum(b for _,b in pairs):9.1f}")
    print("  per character fails/warns:", ", ".join(f"{n} {B[n]['_fails']}/{B[n]['_warns']}->{T[n]['_fails']}/{T[n]['_warns']}" for n in common))
