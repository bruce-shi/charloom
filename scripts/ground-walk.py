"""Apply an explicit locomotion ground config to new layered outputs."""

import argparse, json
from pathlib import Path
from animation.ground_walk import ground_walk

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    report = ground_walk(out=a.out, **config)
    print(json.dumps(report, indent=2))
