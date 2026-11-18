"""Attach fitted geometry and morphs to a previously bound character rig."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from animation.assemble_fixed_rig import assemble

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("layer", "rig", "config", "output"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assemble(args.layer, args.rig, json.loads(args.config.read_text()), args.output)
