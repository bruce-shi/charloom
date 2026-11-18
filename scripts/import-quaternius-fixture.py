"""Import one CC0 human from the downloaded standard pack, recording URI repairs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("archive", type=Path)
parser.add_argument("--out", type=Path, default=ROOT / "fixtures/quaternius")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
prefix = "Universal Base Characters[Standard]/"
folder = prefix + "Base Characters/Godot - UE/"
asset = folder + "Superhero_Male_FullBody.gltf"
records = []
with zipfile.ZipFile(args.archive) as archive:
    document = json.loads(archive.read(asset))
    resources = [(Path(asset).name, asset, None)]
    for group in ("buffers", "images"):
        for resource in document.get(group, []):
            uri = resource.get("uri", "")
            if not uri or uri.startswith("data:"):
                continue
            if Path(uri).name != uri:
                raise ValueError("Unexpected nested resource URI")
            source = folder + uri
            repair = None
            if source not in archive.namelist():
                # The standard pack refers to two *_Normal_png.png files,
                # but provides their bytes as *_Normal.png in this same folder.
                expected = resource.get("name", "")
                if expected != uri.replace("_png.png", ".png") or folder + expected not in archive.namelist():
                    raise FileNotFoundError(uri)
                source = folder + expected
                repair = "Copied the provided image to the exported URI; pixels unchanged"
            resources.append((uri, source, repair))
    for local, source, repair in resources:
        info = archive.getinfo(source)
        if info.file_size > 64 * 1024 * 1024:
            raise ValueError("Resource exceeds fixture limit")
        data = archive.read(source)
        path = args.out / local
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(path)
        path.write_bytes(data)
        records.append({"file": local, "archive_path": source, "sha256": hashlib.sha256(data).hexdigest(), "repair": repair})
    (args.out / "LICENSE.txt").write_bytes(archive.read(prefix + "License_Standard.txt"))
manifest = {"source": "https://quaternius.com/packs/universalbasecharacters.html",
    "download_page": "https://quaternius.itch.io/universal-base-characters", "pack": "Universal Base Characters Standard",
    "license": "CC0 1.0", "date": "2026-10-28", "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(), "files": records}
(args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"output": str(args.out), "files": len(records), "repaired_uri_names": [r["file"] for r in records if r["repair"]]}))
