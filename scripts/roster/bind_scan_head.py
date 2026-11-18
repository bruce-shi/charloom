"""Assemble the original scan head and bind expressions directly to its surface."""

import argparse
import json
from pathlib import Path

from volume_head import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    entries = json.loads(args.config.read_text()) if args.config else {}
    for name in args.names:
        config = dict(entries.get(name, {}), face_binding="scan_surface")
        run(name, config)


if __name__ == "__main__":
    main()
