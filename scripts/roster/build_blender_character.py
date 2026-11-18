"""Assemble and verify an observed scan head for the Blender extension."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def build(transfer, output, config, bun, head_binding="anatomical"):
    scripts = Path(__file__).resolve().parent
    root = scripts.parents[1]
    name = transfer.name
    if not (transfer / "0-guide/resolved.json").is_file():
        raise ValueError(
            "Select the calibrated transfer containing 0-guide/resolved.json"
        )
    if output.exists():
        raise FileExistsError(output)
    if config:
        entries = json.loads(config.read_text())
        if name not in entries or entries[name].get("raw_head"):
            raise ValueError(f"Expected an original-head configuration for {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        env = os.environ.copy()
        env.pop("CHARLOOM_VOLUME_DIR", None)
        env.update(
            CHARLOOM_TRANSFER_DIR=str(transfer.parent),
            CHARLOOM_HEAD_DIR=tmp,
            CHARLOOM_CV_PYTHON=sys.executable,
            PYTHONNOUSERSITE="1",
        )
        # Embedded Blender Python has its own library paths. The geometry worker
        # and the clip writer both run under the project's CV interpreter.
        for key in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(key, None)

        def run(args):
            subprocess.run(args, check=True, cwd=root, env=env)

        builder = "bind_scan_head.py" if head_binding == "scan-surface" else "volume_head.py"
        args = [sys.executable, scripts / builder, name]
        if config:
            args.extend(["--config", config])
        run(args)
        run([bun, scripts / "embed-clips.ts", name])
        run([bun, scripts / "verify-face.ts", name])
        (Path(tmp) / name).rename(output)


def build_scan(source, output, config, bun, options):
    """Publish the calibrated transfer and the verified character together."""
    root = Path(__file__).resolve().parents[2]
    if output.exists():
        raise FileExistsError(output)
    name = options.name or source.stem
    if name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("Character name must be a single directory name")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)
        transfer = stage / "transfer" / name
        command = [
            sys.executable,
            root / "scripts/scan-transfer.py",
            source,
            "--prior",
            options.prior,
            "--guidance",
            options.guidance,
            "--out",
            transfer,
            "--review-out",
            output.with_name(output.name + ".segment-review.json"),
            "--motion",
            options.motion or root / "fixtures/mixamo/walking.fbx",
        ]
        for key in ("mouth_trace", "segment_review", "segment_evidence"):
            if value := getattr(options, key):
                command.extend(["--" + key.replace("_", "-"), value])
        subprocess.run(command, cwd=root, check=True)
        built = stage / "assembled"
        build(transfer, built, config, bun, options.head_binding)
        (stage / "transfer").rename(built / "transfer")
        report_file = built / "report.json"
        report = json.loads(report_file.read_text())
        report["input_transfer"] = str(output / "transfer" / name)
        report_file.write_text(json.dumps(report, indent=2))
        built.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transfer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bun", default="bun")
    parser.add_argument("--head-binding", choices=("scan-surface", "anatomical"), default="anatomical")
    parser.add_argument("--name")
    for key in (
        "prior",
        "guidance",
        "motion",
        "mouth-trace",
        "segment-review",
        "segment-evidence",
    ):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    if args.prior or args.guidance:
        if not args.prior or not args.guidance:
            parser.error("Raw scan assembly needs both --prior and --guidance")
        build_scan(
            args.transfer.resolve(), args.output.resolve(), args.config, args.bun, args
        )
        sys.exit(0)
    build(
        args.transfer.resolve(),
        args.output.resolve(),
        args.config.resolve() if args.config else None,
        args.bun,
        args.head_binding,
    )
