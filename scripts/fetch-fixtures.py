#!/usr/bin/env python3
"""Fetch a pinned, attributed set of small public interoperability fixtures."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8f492390ddc7a2ae221a1369ef560589cf8c70fb"
BASE = f"https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/{REVISION}/Models"
DEST = ROOT / "fixtures" / "reference"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in ("CesiumMan", "Fox"):
        for relative, local in (
            (f"{name}/glTF-Binary/{name}.glb", f"{name}.glb"),
            (f"{name}/README.md", f"{name}.LICENSE.md"),
        ):
            url = f"{BASE}/{relative}"
            path = DEST / local
            if not path.exists():
                with urllib.request.urlopen(url, timeout=40) as response:
                    data = response.read(32 * 1024 * 1024 + 1)
                if len(data) > 32 * 1024 * 1024:
                    raise RuntimeError("fixture exceeded size limit")
                path.write_bytes(data)
            data = path.read_bytes()
            manifest.append({"file": local, "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (DEST / "manifest.json").write_text(json.dumps({"repository_revision": REVISION, "assets": manifest}, indent=2) + "\n")
    print(json.dumps({"directory": str(DEST), "files": len(manifest)}))


if __name__ == "__main__":
    main()
