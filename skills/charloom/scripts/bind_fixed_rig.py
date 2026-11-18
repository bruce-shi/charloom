"""Fit a specified rig to a static scan, preserving its animation curves."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from animation.fixed_rig import fit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scan", "donor", "config", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fit(args.scan, args.donor, json.loads(args.config.read_text()), args.output)
