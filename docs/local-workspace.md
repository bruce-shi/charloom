# Local workspace

English · [简体中文](local-workspace.zh-CN.md)

Create and save characters in Blender. The Gallery reads registered scans, bare bodies, dressed models and animations. Open the local session link printed at startup. Character pages support refresh, direct links and browser back and forward.

```sh
cd viewer
bun install --frozen-lockfile
bun run dev
```

Earlier versions stored projects at `~/.charloom/projects/<id>/project.json`, with assets in the same workspace at `blobs/<sha256>`. For a custom workspace, keep its original `CHARLOOM_DATA_DIR`. Run Blender in background mode to import the project into a new file.

```sh
blender -b --factory-startup --python-exit-code 1 --python scripts/blender_project.py -- \
  /path/to/projects/PROJECT_ID/project.json /path/to/character.blend
```

The importer verifies asset hashes, creates a Blender collection for each asset, packs textures and stores the original project and change records as Text data blocks. Scripts can read fitting parameters, annotations and earlier mesh patches from that project data. The imported scene displays the saved geometry. See [Project README](../README.md) for character builds and Gallery registration.
