"""Read or register a reusable garment attachment and material preset."""

import argparse
import json
from pathlib import Path
from transfer.cloth import preset_path, register_preset, validate_rig, fabric_motion


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, default=Path("garment-presets"))
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("register").add_argument("manifest", type=Path)
    sub.add_parser("get").add_argument("id")
    args = p.parse_args()
    if args.action == "register":
        path = register_preset(args.library, json.loads(args.manifest.read_text()))
    else:
        path = preset_path(args.library, args.id)
    data = json.loads(path.read_text())
    if data["id"] != path.stem:
        raise ValueError("Garment preset filename differs from its ID")
    validate_rig(data["rig"])
    fabric_motion(data.get("fabric", "woven"))
    print(json.dumps(data))
