"""Build or import a character, select an animation and save an editable Blender scene."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
from charloom import api


def main(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="A calibrated transfer directory or a character GLB"
    )
    parser.add_argument("output", type=Path, help="A new output directory")
    parser.add_argument(
        "--config", type=Path, help="Original-head configuration for the transfer"
    )
    parser.add_argument(
        "--head-binding",
        choices=("scan-surface", "anatomical"),
        help="Head assembly route when building a raw scan or calibrated transfer",
    )
    parser.add_argument(
        "--clip", help="Timeline action, e.g. walking, face_test or face_talk"
    )
    parser.add_argument(
        "--export-glb",
        action="store_true",
        help="Also export the Blender character as GLB",
    )
    parser.add_argument(
        "--prior", type=Path, help="MakeHuman template GLB for a raw scan"
    )
    parser.add_argument("--guidance", type=Path, help="Reviewed calibrated scan guide")
    parser.add_argument(
        "--mouth-trace", type=Path, help="Hash-bound lip seam on the scan"
    )
    parser.add_argument(
        "--segment-review", type=Path, help="Hash-bound component decisions"
    )
    parser.add_argument(
        "--segment-evidence",
        type=Path,
        help="Candidate triangle ranges accompanying decisions",
    )
    parser.add_argument("--motion", type=Path, help="Walk FBX or GLB for the scan")
    parser.add_argument("--name", help="Transfer directory name and configuration key")
    options = parser.parse_args(args)
    source, output = options.source.resolve(), options.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    raw = options.prior is not None
    if raw != bool(options.guidance) or raw and source.is_dir():
        parser.error("A raw scan requires a source GLB, --prior and --guidance")
    if not raw and any(
        (
            options.motion,
            options.mouth_trace,
            options.name,
            options.segment_review,
            options.segment_evidence,
        )
    ):
        parser.error("--motion, --mouth-trace and --name apply to raw scans")
    if options.config and not (raw or source.is_dir()):
        parser.error("--config requires a raw scan or transfer directory")
    if options.head_binding and not (raw or source.is_dir()):
        parser.error("--head-binding requires a raw scan or transfer directory")
    name = options.name or source.stem
    if not name or name in (".", "..") or Path(name).name != name:
        parser.error("--name must be a single directory name")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        stage = Path(scratch) / "result"
        stage.mkdir()
        if raw:
            source = api.transfer_scan(
                source,
                options.prior,
                options.guidance,
                stage / "transfer" / name,
                motion=options.motion or ROOT / "fixtures/mixamo/walking.fbx",
                mouth_trace=options.mouth_trace,
                segment_review=options.segment_review,
                segment_evidence=options.segment_evidence,
                review_out=output.with_name(output.name + ".segment-review.json"),
            )
        model = source
        if source.is_dir():
            built = api.build_character(
                source, stage / "build", config=options.config,
                head_binding=options.head_binding,
            )
            model = built / "4-garments/character.glb"
            if raw:
                report_file = built / "report.json"
                report = json.loads(report_file.read_text())
                report["input_transfer"] = str(output / "transfer" / name)
                report_file.write_text(json.dumps(report, indent=2))
        bpy.ops.wm.read_factory_settings(use_empty=True)
        rig, body = api.import_character(model)
        if options.clip:
            api.preview(rig, options.clip)
        bpy.ops.wm.save_as_mainfile(
            filepath=str(stage / "character.blend"), compress=True
        )
        result = {
            "blend": str(output / "character.blend"),
            "rig": rig.name,
            "body": body.name if body else None,
        }
        if options.export_glb:
            api.export_character(rig, stage / "character.glb")
            result["glb"] = str(output / "character.glb")
        stage.rename(output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
