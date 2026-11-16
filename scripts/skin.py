"""Analyze a surface or manage a versioned skin-repair session."""

import argparse
import json
from pathlib import Path

from animation.repair import start, step
from animation.surface import Surface


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("analyze", "contacts", "start", "step", "inspect"):
        op = sub.add_parser(command)
        op.add_argument("path", type=Path)
        if command in ("analyze", "contacts", "start"):
            op.add_argument("--config", type=Path, required=command == "contacts")
        if command == "start":
            op.add_argument("--out", type=Path, required=True)
        if command == "step":
            op.add_argument("--plan", type=Path, required=True)
    args = p.parse_args()
    config = (
        json.loads(args.config.read_text()) if getattr(args, "config", None) else {}
    )
    if args.command == "start":
        result = start(args.path, args.out, config)
    elif args.command == "step":
        result = step(args.path, json.loads(args.plan.read_text()))
    elif args.command == "inspect":
        result = json.loads((args.path / "session.json").read_text())
    elif args.command == "contacts":
        result = Surface(args.path).contacts(config)
    else:
        result = Surface(args.path).analyze(config)
    print(json.dumps(result))
