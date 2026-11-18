"""Open a reference in a separate Blender instance with the project-local MPFB enabled."""
from pathlib import Path
import argparse
import importlib
import json
import sys

import addon_utils
import bpy

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("blend", type=Path)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
path = args.blend.resolve()
if not path.is_file():
    parser.error(f"Blender file not found: {path}")
revision = json.loads((ROOT / "scripts/mpfb/sources.json").read_text())["mpfb"]["revision"]
repository = ROOT / "out/tools/mpfb/runtime/repository"
source = ROOT / "out/tools/mpfb/source" / ("mpfb2-" + revision) / "src/mpfb"
if not source.is_dir() or not (repository / "mpfb").is_dir():
    parser.error("Generate a reference with scripts/mpfb-reference.py first")
bpy.context.preferences.extensions.repos.new(name="Charloom MPFB", module="charloom_mpfb", custom_directory=str(repository))
bpy.utils.refresh_script_paths()
package = "bl_ext.charloom_mpfb.mpfb"
addon_utils.enable(package, default_set=True, persistent=False)
assert importlib.import_module(package).MPFB_CONTEXTUAL_INFORMATION
bpy.ops.wm.open_mainfile(filepath=str(path), use_scripts=False)
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            space.region_3d.view_location = (0, 0, .85)
            space.region_3d.view_distance = 2.5
            space.region_3d.view_rotation = bpy.context.scene.camera.rotation_euler.to_quaternion()
            space.region_3d.view_perspective = "ORTHO"
            space.show_region_ui = True
