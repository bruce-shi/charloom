# Charloom

The public tools fit a rigged MakeHuman scaffold to an annotated humanoid scan, extract its wardrobe and transfer animation. The local Gallery browses registered character exports.

```sh
uv venv --python 3.12 out/tools/cv/.venv
uv pip install --python out/tools/cv/.venv/bin/python -r scripts/requirements-blender.txt
(cd viewer && bun install --frozen-lockfile)
```

Use a locally installed Blender for FBX motion input. Supply your own rigged body, scan and guidance. See the [annotation guide](docs/scan-annotation.md) for camera coordinates.

```sh
PY=out/tools/cv/.venv/bin/python
"$PY" scripts/scan-guide.py scan.glb --out out/guide
"$PY" scripts/scan-transfer.py scan.glb --prior body.glb --guidance out/guide/guidance.json --motion walk.fbx --out out/character
"$PY" scripts/animate.py validate out/character/4-garments/character.glb
python3 scripts/mh-ide-sync.py out/character
(cd viewer && bun run dev)
```

Use `scripts/animate.py` for joint mapping, pose sampling, IK and recipe baking. Use `scripts/skin.py` for surface diagnostics and versioned weight repair.
