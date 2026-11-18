#!/usr/bin/env python3
"""Copy pinned, hash-checked myth sources into an isolated local fixture folder."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = (ROOT.parent / "games/apps/swiftcarrot/app/routes/games/mythic-rts-demo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--myth-root", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=ROOT / "fixtures/local-myth")
    args = parser.parse_args()
    inventory = args.myth_root / "source-art/style-v3/runtime-integration/inventory.json"
    data = json.loads(inventory.read_text())
    inputs = []
    for unit in data["units"]:
        if unit["id"] in {"crossbow-guard", "starbow-adept", "longfeather-guardian", "ford-scout"}:
            source = unit["pinnedSource"]
            inputs.append((unit["id"], Path(source["path"]), source["sha256"],
                           "myth pinned runtime source", unit.get("sourceApproval")))
    assert len(inputs) == 4, "Selected runtime source is missing from inventory"
    flame = args.myth_root / "source-art/style-v3/flame-guard-simplification-test-20261025"
    inputs.extend([
        ("flame-unrigged", flame / "tripo-image/downloads/model.glb", None, "downloaded candidate; runtime adoption not claimed", None),
        ("flame-walk", flame / "weapon-and-animation/rig/downloads/walking-glb.glb", None, "downloaded animation; runtime adoption not claimed", None),
    ])
    # Verify all sources and existing snapshots before copying any bytes.
    records = []
    for name, source, expected, provenance, review in inputs:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if expected and digest != expected:
            raise RuntimeError(f"Pinned source hash mismatch: {source}")
        target = args.out / f"{name}.glb"
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Existing snapshot differs; choose a new --out folder: {target}")
        records.append({"id": name, "source": str(source), "file": target.name, "sha256": digest,
                        "provenance": provenance, "source_review": review, "charloom_visual_review": "pending",
                        "license": "Local user project asset; not licensed here for redistribution"})
    args.out.mkdir(parents=True, exist_ok=True)
    for record in records:
        target = args.out / record["file"]
        if not target.exists():
            shutil.copy2(record["source"], target)
    (args.out / "manifest.json").write_text(json.dumps({"inventory": str(inventory), "assets": records}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"assets": len(records), "output": str(args.out)}))


if __name__ == "__main__":
    main()
