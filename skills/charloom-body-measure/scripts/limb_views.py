"""Generate three calibrated full-figure reference panels with contrast at the arm/trunk boundary."""

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from measurement_images import sheets

PROMPT = (
    "Make a technical three-view orthographic character sheet for 3D animation. "
    "Use exactly the same adult character and the same three full-figure camera views "
    "as the input, in the same order: front, one side, opposite side. Wear one "
    "consistent fitted athletic motion capture outfit. Both sleeves are bright green "
    "from the shoulder seams to the wrists, with matching thin green five-finger gloves. "
    "The trunk is dark blue, both legs light blue, both feet covered by thin light blue socks. "
    "Keep a crisp visible green-to-blue boundary from the armpit to the wrist, "
    "including where an arm is in front of the trunk. Show the near arm of each side "
    "view separately as seen by that camera. Preserve the source shoulder, elbow, wrist, "
    "fingertips, hips, knees, ankles, feet and their original poses in each view. "
    "Match the build and proportions in the first two reference panels when changing "
    "the costume in the third panel. Preserve the exact head position, direction, "
    "height, ground line, panel scale and margins in every camera. Plain grey background "
    "without shadows. No text, no guide lines, no extra people or panels."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guide", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    guide = args.guide.resolve()
    marks = json.loads((guide / "guidance.json").read_text())
    views = json.loads((guide / "views.json").read_text())
    base = json.loads((guide / "views/bodysuit.json").read_text())
    if marks["source_hash"] != views["source_hash"] or not base.get("accepted"):
        raise ValueError(
            "Use calibrated views and a registered base sheet from the same scan"
        )
    yaw = int(marks["front_view"].removeprefix("body_y"))
    ids = [f"body_y{(yaw + angle) % 360:03d}" for angle in (0, 90, 270)]
    cameras = {c["id"]: c for c in views["cameras"]}
    inputs = [
        guide / "views" / (f"bodysuit_{view}-reg.png" if i < 2 else f"{view}.png")
        for i, view in enumerate(ids)
    ]
    sheet = Image.new("RGB", (2304, 1024))
    records = []
    for i, (view, source) in enumerate(zip(ids, inputs)):
        with Image.open(source) as im:
            if im.size != (768, 1024) or im.size != (
                cameras[view]["width"],
                cameras[view]["height"],
            ):
                raise ValueError(f"Unsupported camera canvas for {view}")
            sheet.paste(im.convert("RGB"), (i * 768, 0))
        records.append(
            dict(
                view=view,
                file=str(source),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        )
    fields = dict(
        model="openai/gpt-image-2.5/sunburst/edit",
        prompt=PROMPT,
        quality="medium",
        size="2560x1136",
        n="1",
    )
    plan = dict(source_hash=marks["source_hash"], inputs=records, parameters=fields)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "plan.json"
    if manifest.is_file() and json.loads(manifest.read_text()) != plan:
        raise ValueError("Use a separate output directory for different inputs")
    manifest.write_text(json.dumps(plan, indent=2))
    source = args.out / "sheet-input.png"
    sheet.resize((2560, 1136), Image.Resampling.LANCZOS).save(source)
    if args.generate:
        raw, usage = sheets(guide, source, fields)
        (args.out / "sheet-native.png").write_bytes(raw[0])
        with Image.open(BytesIO(raw[0])) as native:
            canvas = native.convert("RGB").resize(sheet.size, Image.Resampling.LANCZOS)
        generation = dict(
            **plan,
            receipt=usage,
            native_sha256=hashlib.sha256(raw[0]).hexdigest(),
            status="pending_per_view_review",
        )
        (args.out / "generation.json").write_text(json.dumps(generation, indent=2))
        for i, record in enumerate(records):
            folder = args.out / record["view"]
            folder.mkdir(exist_ok=True)
            panel = canvas.crop((i * 768, 0, (i + 1) * 768, 1024))
            panel.save(folder / "profile-candidate.png")
            receipt = dict(
                **generation,
                source=record["file"],
                view=record["view"],
                image_sha256=record["sha256"],
            )
            (folder / "generation.json").write_text(json.dumps(receipt, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
