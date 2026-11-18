"""Prepare a Gallery motion preview from a free-topology result and its original rig.

Usage in the CV Python environment:
    python scripts/mh-motion-preview.py RUN OUTPUT [--mouth-curves SOURCE_CURVES]
"""

import argparse
from roster.walk import restore_walk
from roster.sculpt_mouth import apply
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(run, output, mouth_curves):
    run = run.resolve()
    origin = Path(json.loads((run / "candidate-report.json").read_text())["input"])
    curves = None
    if mouth_curves:
        trace = json.loads(mouth_curves.read_text())
        mapping = json.loads(Path(trace["source"]).read_text())
        original = json.loads((origin / "report.json").read_text())
        if (
            digest(Path(original["source"])) != mapping["source_sha256"]
            or digest(Path(mapping["mouth_source_image"]))
            != mapping["mouth_source_image_sha256"]
        ):
            raise ValueError("Mouth trace belongs to a different scan or camera image")
        curves = trace["curves"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "source": str(run),
        "walk_source": str(origin),
        "mouth_curves": str(mouth_curves) if curves else None,
        "exports": [],
    }
    for rel in ("3-skin/body.glb", "4-garments/character.glb"):
        dest = output / rel
        count = restore_walk(run / rel, origin / rel, dest)
        changed = {}
        if curves:
            changed = apply(dest, curves)
        report["exports"].append(
            {
                "file": rel,
                "sha256": digest(dest),
                "walk_channels": count,
                "affected_mouth_vertices": changed,
            }
        )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mouth-curves", type=Path)
    main(**vars(parser.parse_args()))
