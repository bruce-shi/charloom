"""Prepare or generate a full-figure side measurement plate from the registered bodysuit view."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from measurement_images import sheets

PROMPT = (
    "Create a single side elevation for a 3D character turnaround. Show the same "
    "fully clothed adult character in the green motion capture costume against the "
    "same grey background. Use a neutral standing pose with arms hanging down and "
    "each palm resting vertically against the outside of its thigh, fingers pointing "
    "towards the floor. From the side, keep the fingers inside the thigh outline "
    "and the front waistline visible. Match the reference camera, facing direction, "
    "head position, trunk, legs, foot placement, scale, margins and lighting. "
    "Show the full character without text or dimension lines."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guide", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--generate", action="store_true", help="Submit or resume the cached edit"
    )
    args = parser.parse_args()
    guide = args.guide.resolve()
    annotations = json.loads((guide / "guidance.json").read_text())
    views = json.loads((guide / "views.json").read_text())
    if views["source_hash"] != annotations["source_hash"]:
        raise ValueError("Guide and calibrated views belong to different scans")
    yaw = int(annotations["front_view"].removeprefix("body_y"))
    view = f"body_y{(yaw + 90) % 360:03d}"
    source = guide / "views" / f"bodysuit_{view}-reg.png"
    receipt = json.loads((guide / "views/bodysuit.json").read_text())
    if receipt.get("accepted") is not True:
        raise ValueError("Inspect and register the base measurement sheet first")
    with Image.open(source) as original:
        if original.size != (768, 1024):
            raise ValueError("Expected a calibrated 768 x 1024 profile")
    fields = dict(
        model="openai/gpt-image-2.5/sunburst/edit",
        prompt=PROMPT,
        quality="medium",
        size="1536x2048",
        n="1",
    )
    metadata = dict(
        source_hash=annotations["source_hash"],
        view=view,
        source=str(source),
        image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        parameters=fields,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    plan = args.out / "plan.json"
    if plan.exists() and metadata != json.loads(plan.read_text()):
        raise FileExistsError("Use a new directory for different inputs")
    plan.write_text(json.dumps(metadata, indent=2))
    if args.generate:
        images, usage = sheets(guide, source, fields)
        native = args.out / "profile-native.png"
        native.write_bytes(images[0])
        with Image.open(native) as edited:
            edited.convert("RGB").resize((768, 1024), Image.Resampling.LANCZOS).save(
                args.out / "profile-candidate.png"
            )
        (args.out / "generation.json").write_text(
            json.dumps(
                dict(
                    **metadata,
                    receipt=usage,
                    native_sha256=hashlib.sha256(images[0]).hexdigest(),
                    status="pending_registration_review",
                ),
                indent=2,
            )
        )
    print(args.out)


if __name__ == "__main__":
    main()
