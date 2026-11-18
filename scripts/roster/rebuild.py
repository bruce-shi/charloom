"""Build original-scan heads for named roster entries, optionally from their raw GLBs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from build_blender_character import build

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--roster", type=Path, help="JSON character inventory")
    parser.add_argument("--out", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--transfers", type=Path)
    group.add_argument("--from-scan", action="store_true")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--reviews", type=Path, help="Previous scan transfers containing NAME/1-segment/review.json")
    parser.add_argument(
        "--config",
        type=Path,
        help="Reviewed original-head and body-volume settings keyed by character name",
    )
    parser.add_argument("--bun", default="bun")
    args = parser.parse_args()
    if args.from_scan != bool(args.work):
        parser.error("--from-scan requires --work for its new transfer directories")
    root = args.root.resolve()
    entries = json.loads((args.roster or root / "scripts/roster/inputs/roster.json").read_text())[
        "characters"
    ]
    entries = {entry["name"]: entry for entry in entries}
    transfers = (args.transfers or root / "out/transfers").resolve()
    unknown = set(args.names) - entries.keys()
    if unknown:
        parser.error(f"Unknown roster entries {sorted(unknown)}")
    for name in args.names:
        out = args.out.resolve() / name
        if out.exists():
            raise FileExistsError(out)
        run = transfers / name
        if args.from_scan:
            row = entries[name]
            run = args.work.resolve() / name
            source, prior, guide = (
                root / row[k] for k in ("source", "prior", "guidance")
            )
            trace = next(
                (
                    f
                    for f in (
                        guide.parent / "mouth-trace.json",
                        root / "scripts/roster/inputs" / (name + "-trace.json"),
                        transfers / name / "mouth-trace.json",
                    )
                    if f.exists()
                ),
                None,
            )
            if trace is None:
                raise FileNotFoundError(
                    f"Missing reviewed lip pixels on the original scan for {name}"
                )
            previous = (args.reviews or transfers) / name / "1-segment/review.json"
            subprocess.run(
                [
                    sys.executable,
                    root / "scripts/scan-transfer.py",
                    source,
                    "--prior",
                    prior,
                    "--guidance",
                    guide,
                    "--out",
                    run,
                    "--motion",
                    root / "fixtures/mixamo/walking.fbx",
                    "--mouth-trace",
                    trace,
                    *(["--segment-evidence", previous.resolve()] if previous.is_file() else []),
                ],
                check=True,
                cwd=root,
            )
        build(run, out, args.config.resolve() if args.config else None, args.bun)
        print(
            json.dumps(
                dict(
                    name=name,
                    path=str(out),
                    validation=str(out / "animated-checks.json"),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
