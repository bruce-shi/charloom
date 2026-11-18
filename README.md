# Charloom

English · [简体中文](README.zh-CN.md)

Charloom turns textured scans into editable Blender characters. Fit a MakeHuman body, bind the original scanned head and clothing, then add motion to the character’s existing rig. The Blender extension provides the editing panel. The agent skills guide scanning, fitting, animation and review.

## Install from a public checkout

Use a macOS or Linux shell with Git, uv, Bun and Blender 4.4 or newer. Keep the checkout in a permanent location. Both the extension and the skills run scripts from it. Run the following blocks in the same shell.

```sh
git clone https://github.com/bruce-shi/charloom.git
cd charloom
export CHARLOOM_ROOT="$(pwd -P)"
export BLENDER="${BLENDER:-blender}"
if [ "$(uname -s)" = Darwin ] && [ "$BLENDER" = blender ] && ! command -v blender >/dev/null; then
  export BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
fi
"$BLENDER" --version
uv venv --python 3.12 out/tools/cv/.venv
export CHARLOOM_CV_PYTHON="$CHARLOOM_ROOT/out/tools/cv/.venv/bin/python"
uv pip install --python "$CHARLOOM_CV_PYTHON" -r scripts/requirements-blender.txt
(cd viewer && bun install --frozen-lockfile)
```

### 1. Install and enable the Blender extension

Close other Blender sessions before changing saved preferences. The commands build the ZIP, install it in Blender’s default user extension repository and save the checkout, Python and Bun paths. Set `BLENDER` to the executable you use for editing if you have multiple versions.

```sh
mkdir -p out/extensions
"$BLENDER" -b -c extension build --source-dir blender/charloom \
  --output-filepath "$CHARLOOM_ROOT/out/extensions/charloom.zip"
"$BLENDER" -b -c extension install-file --repo user_default --enable \
  "$CHARLOOM_ROOT/out/extensions/charloom.zip"
"$BLENDER" -b --python-exit-code 1 --python-expr '
import bpy, os, shutil
prefs = bpy.context.preferences.addons["bl_ext.user_default.charloom"].preferences
prefs.checkout = os.environ["CHARLOOM_ROOT"]
prefs.python = os.environ["CHARLOOM_CV_PYTHON"]
prefs.bun = shutil.which("bun")
bpy.ops.wm.save_userpref()
'
```

Verify the saved settings in a fresh Blender process. Success prints the checkout path. In the GUI, open the 3D Viewport, press **N** and select **Charloom**. For a GUI installation, use **Preferences → Get Extensions → Install from Disk**, select the same ZIP, enable Charloom and set **Charloom 工程目录** to the checkout in its add-on preferences.

```sh
"$BLENDER" -b --python-exit-code 1 --python-expr '
import bpy, os
p = bpy.context.preferences.addons["bl_ext.user_default.charloom"].preferences
assert os.path.samefile(p.checkout, os.environ["CHARLOOM_ROOT"])
assert os.access(p.python, os.X_OK) and os.access(p.bun, os.X_OK)
assert hasattr(bpy.types, "CHARLOOM_PT_character")
print("Charloom ready:", p.checkout)
'
```

### 2. Install the agent skills

For Codex, link each complete skill directory into your user skills folder. This keeps its scripts and references together and picks up updates from the checkout. To install only for this project, set `CHARLOOM_SKILLS_DIR="$CHARLOOM_ROOT/.agents/skills"` first. Choose one scope. Existing entries pointing elsewhere are reported for review. [Codex skill discovery](https://developers.openai.com/codex/skills)

```sh
skills_dir="${CHARLOOM_SKILLS_DIR:-$HOME/.agents/skills}"
mkdir -p "$skills_dir"
for src in "$CHARLOOM_ROOT"/skills/charloom*; do
  dest="$skills_dir/${src##*/}"
  if [ ! -e "$dest" ] && [ ! -L "$dest" ]; then
    ln -s "$src" "$dest"
  fi
  if [ ! "$dest" -ef "$src" ]; then
    echo "Check existing skill: $dest" >&2
  fi
done
```

Open Codex in the checkout and type `$charloom` in the skill picker. Look for `charloom` and `charloom-mh`. Restart Codex if the new skills have not appeared. Agents with another skill loader can open these files directly and follow their relative links. Run shell commands from the checkout root.

| Task | Start here |
| --- | --- |
| Build a character from a raw scan | [charloom-mh](skills/charloom-mh/SKILL.md) |
| Edit a rigged character or add a motion | [charloom](skills/charloom/SKILL.md) |
| Inspect the delivered model and motion | [charloom-gallery-review](skills/charloom-gallery-review/SKILL.md) |

## Build from a scan

Supply a static scan GLB with a single texture atlas, a MakeHuman body prior and a motion FBX or GLB. Prepare the guide below, then mark the landmarks and clothing regions in `guidance.json`. Follow the scan-guide skill to refine the views and trace the mouth and eyelids. [Template setup](docs/mpfb-prior.md) · [Scan annotation](skills/charloom-scan-guide/SKILL.md) · [Asset locations](assets/README.md)

```sh
"$CHARLOOM_CV_PYTHON" scripts/scan-guide.py scan.glb --out out/guide
cp out/guide/guidance.template.json out/guide/guidance.json
```

After reviewing the guide, fitting the body and checking the separated parts, run the build with your own input paths. Repeat the measurements on the exported character. The build command assumes the measurement table has already been reviewed.

```sh
"$CHARLOOM_CV_PYTHON" scripts/scan-guide.py scan.glb --check out/guide/guidance.json
"$BLENDER" -b --factory-startup --python-exit-code 1 --python scripts/blender_character.py -- \
  scan.glb out/new-character --name character \
  --prior prior.glb --guidance out/guide/guidance.json \
  --mouth-trace out/guide/mouth-trace.json --motion walking.fbx \
  --head-binding scan-surface --clip face_talk --export-glb
```

Open `out/new-character/character.blend` to edit the scene or `character.glb` in the same directory to review the export. The calibrated transfer is in `transfer/`. The bare and dressed models are in `build/3-skin/body.glb` and `build/4-garments/character.glb`, with animation checks in `build/animated-checks.json`. An installed MakeHuman base underwear asset can be fitted to the bare export with its license recorded in the material.

## Script Blender and review the result

Run this example inside Blender from the checkout root. Replace the two paths with a calibrated directory containing `0-guide/resolved.json` and a fresh output directory.

```python
import sys
from pathlib import Path
import bpy

checkout = Path.cwd()
sys.path.insert(0, str(checkout / "blender"))
from charloom import api

transfer = Path("out/calibrated-transfer")
output = Path("out/edited-character")
built = api.build_character(transfer, output, checkout=checkout, head_binding="scan-surface")
rig, body = api.import_character(built / "4-garments/character.glb")
api.preview(rig, "face_talk")
bpy.context.scene.frame_set(18)
api.export_character(rig, built / "edited.glb")
```

For raw input, use `api.prepare_guide` and `api.transfer_scan`, or pass the scan and its inputs to `api.build_character`. Use `api.fit_surface` for body edits and `api.bind_clothing` for garments. For motion, use `api.retarget_animation`, `api.solve_pose` or `api.build_animation`. [Adding motion to an existing rig](skills/charloom/references/walk-replay.md)

To run the animation tools directly, supply a recipe pointing to a local rigged model. The example below uses the asset path in the foot-lock recipe.

```sh
"$CHARLOOM_CV_PYTHON" scripts/animate.py build recipes/myth-explicit-foot-lock.json --out out/foot-lock
"$CHARLOOM_CV_PYTHON" scripts/animate.py validate out/foot-lock/character.glb
"$CHARLOOM_CV_PYTHON" scripts/skin.py analyze out/foot-lock/character.glb
```

To register a build in the read-only Gallery, preview the change with `python3 scripts/mh-ide-sync.py BUILD_PARENT --dry-run`, then repeat without `--dry-run`. Run `(cd viewer && bun run dev)` and open the printed session URL to compare the scan, bare body, dressed model and animation. For a production preview, run `bun run build` and `bun run preview` in `viewer/`. [Import an existing workspace](docs/local-workspace.md)
