"""Export a reference body with MakeHuman shape targets as glTF morph targets.

    Blender -b --factory-startup --python scripts/mpfb/export-shape-targets.py -- \
        out/examples/mpfb-female-reference-v3/female-muscle-050/editable.blend \
        out/examples/mpfb-female-reference-v3/female-muscle-050/body-shape.glb

Same rig, weights and macro shape as body.glb; the selected MPFB targets
(belly, girths, head, hands, feet, face features) become morph targets named
after the parameter (`stomach-pregnant+`, `head-scale-horiz-`, ...). Left and
right targets are combined into one morph. charloom-mh fits their weights
before its fine fit (a bounded forward shape model, as the old workflow did).
Run with the project's MPFB add-on environment (see docs/mpfb-prior.md);
CHARLOOM_MPFB_TOOLS points at out/tools/mpfb of another checkout (worktrees).

CHARLOOM_TARGET_SET=ears exports only MakeHuman's per-side ear targets, one
morph per official target file (`ears/l-ear-scale-incr`, ...), into a sidecar
(`ear-targets.glb` next to the prior): charloom-mh shapes each ear with them
to the marked ear outline (fit::ears), not in the body-wide shape fit.

The chest: `pectoral` and `dorsi` (MakeHuman's chest and back muscle targets,
both sexes) and, on a female body, `breast-size` and `breast-firmness`: the
breast macros (MPFB's cup size and firmness, 0.5 default) moved to 1 (`+`)
or 0 (`-`) for this body's own age, muscle and weight, i.e. the `breast/`
macro targets MPFB itself would load, scaled by the body's female weight
(MPFB leaves it out, so a male body would get a female bust).

The build: `build-weight` and `build-muscle` are MPFB's weight and muscle
macros moved to 1 (`+`) or 0 (`-`): the difference of MPFB's whole macro
target stack to this body's, so one parameter makes the whole body heavier
or more muscular the way MakeHuman does (charloom-mh fits it first, from the
visible body, and carries it under the clothes).
"""
from pathlib import Path
import importlib
import json
import os
import sys

import addon_utils
import bpy

ROOT = Path(__file__).resolve().parents[2]
blend, out = [Path(p).resolve() for p in sys.argv[sys.argv.index("--") + 1:][:2]]
revision = json.loads((ROOT / "scripts/mpfb/sources.json").read_text())["mpfb"]["revision"]
tools = Path(os.environ.get("CHARLOOM_MPFB_TOOLS", ROOT / "out/tools/mpfb"))
repository = tools / "runtime/repository"
source = tools / "source" / ("mpfb2-" + revision) / "src/mpfb"
targets_dir = source / "data/targets"
bpy.context.preferences.extensions.repos.new(name="Charloom MPFB", module="charloom_mpfb", custom_directory=str(repository))
bpy.utils.refresh_script_paths()
package = "bl_ext.charloom_mpfb.mpfb"
addon_utils.enable(package, default_set=True, persistent=False)
assert importlib.import_module(package).MPFB_CONTEXTUAL_INFORMATION, "MPFB failed to register"
TargetService = importlib.import_module(package + ".services.targetservice").TargetService
ObjectService = importlib.import_module(package + ".services.objectservice").ObjectService
ExportService = importlib.import_module(package + ".services.exportservice").ExportService
bpy.ops.wm.open_mainfile(filepath=str(blend), use_scripts=False)

# (morph name, target files; "{s}" expands to l and r and is combined)
PAIRS = [
    ("stomach-pregnant", "stomach/stomach-pregnant-{d}"),
    ("torso-scale-depth", "torso/torso-scale-depth-{d}"),
    ("torso-scale-horiz", "torso/torso-scale-horiz-{d}"),
    ("waist-circ", "torso/measure-waist-circ-{d}"),
    ("hips-circ", "torso/measure-hips-circ-{d}"),
    ("bust-circ", "torso/measure-bust-circ-{d}"),
    ("pectoral", "torso/torso-muscle-pectoral-{d}"),
    ("dorsi", "torso/torso-muscle-dorsi-{d}"),
    ("buttocks-volume", "buttocks/buttocks-volume-{d}"),
    ("upperarm-circ", "arms/measure-upperarm-circ-{d}"),
    ("thigh-circ", "legs/measure-thigh-circ-{d}"),
    ("calf-circ", "legs/measure-calf-circ-{d}"),
    ("hand-scale", "hands/{s}-hand-scale-{d}"),
    ("foot-scale", "feet/{s}-foot-scale-{d}"),
    ("neck-circ", "neck/measure-neck-circ-{d}"),
    ("head-scale-horiz", "head/head-scale-horiz-{d}"),
    ("head-scale-vert", "head/head-scale-vert-{d}"),
    ("head-scale-depth", "head/head-scale-depth-{d}"),
    ("nose-volume", "nose/nose-volume-{d}"),
    ("mouth-scale-horiz", "mouth/mouth-scale-horiz-{d}"),
    ("mouth-scale-vert", "mouth/mouth-scale-vert-{d}"),
    ("eye-scale", "eyes/{s}-eye-scale-{d}"),
    ("cheek-volume", "cheek/{s}-cheek-volume-{d}"),
]
SINGLES = [
    ("fingers-length+", "hands/{s}-hand-fingers-length-incr"),
    ("chin-width+", "chin/chin-width-incr"),
]
plan = []
for name, pattern in PAIRS:
    for d, sign in (("incr", "+"), ("decr", "-")):
        plan.append((name + sign, pattern.replace("{d}", d)))
plan += SINGLES
# Breast macros: (morph, macro, value) -> MPFB's own breast/ target stack.
BREAST = [("breast-size+", "cupsize", 1.0), ("breast-size-", "cupsize", 0.0),
          ("breast-firmness+", "firmness", 1.0), ("breast-firmness-", "firmness", 0.0)]
# Build macros: the whole body's weight and muscle (MPFB's own macro target
# stack moved to 1 for `+`, 0 for `-`, as the difference to this body's).
BUILD = [("build-weight+", "weight", 1.0), ("build-weight-", "weight", 0.0),
         ("build-muscle+", "muscle", 1.0), ("build-muscle-", "muscle", 0.0)]
if os.environ.get("CHARLOOM_TARGET_SET") == "ears":
    # Ear targets, one morph per official target file (per side).
    plan = [(f, f) for f in sorted(str(p.relative_to(targets_dir)).removesuffix(".target.gz")
                                   for p in (targets_dir / "ears").glob("*.target.gz"))]

body = next(o for o in bpy.data.objects if ObjectService.object_is(o, "Basemesh") and not o.name.endswith("_export"))
export_root = ExportService.create_character_copy(body, name_suffix="_export")
export_body = ObjectService.find_object_of_type_amongst_nearest_relatives(export_root, "Basemesh")
ObjectService.activate_blender_object(export_body, deselect_all=True)
TargetService.bake_targets(export_body)
if export_body.data.shape_keys is None:
    export_body.shape_key_add(name="Basis", from_mix=False)
report = []
for name, pattern in plan:
    files = [pattern.replace("{s}", s) for s in ("l", "r")] if "{s}" in pattern else [pattern]
    paths = [targets_dir / (f + ".target.gz") for f in files]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        report.append({"morph": name, "skipped": missing})
        continue
    keys = export_body.data.shape_keys.key_blocks
    before = set(k.name for k in keys)
    for p in paths:
        TargetService.load_target(export_body, str(p), weight=1.0)
    added = [k for k in export_body.data.shape_keys.key_blocks if k.name not in before]
    for k in export_body.data.shape_keys.key_blocks:
        k.value = 1.0 if k in added else 0.0
    combined = export_body.shape_key_add(name=name, from_mix=True)
    combined.value = 0.0
    for k in added:
        export_body.shape_key_remove(k)
    report.append({"morph": name, "targets": files})
if os.environ.get("CHARLOOM_TARGET_SET") != "ears":
    macro = TargetService.get_macro_info_dict_from_basemesh(body)
    def breast_stack(info):
        return {n: w for n, w in TargetService.calculate_target_stack_from_macro_info_dict(info) if n.startswith("breast/")}
    base_stack = breast_stack(macro)
    for name, key, value in BREAST:
        info = json.loads(json.dumps(macro))
        info[key] = value
        stack = breast_stack(info)
        rows = [(n, w - base_stack.get(n, 0.0)) for n, w in stack.items()] + [(n, -w) for n, w in base_stack.items() if n not in stack]
        # MPFB leaves the gender weight out (its breast targets are female
        # only); on a male body that would put a female bust on it.
        female = max(0.0, 1.0 - macro["gender"])
        rows = [(n, w * female) for n, w in rows if abs(w * female) > 0.05]
        if not rows:
            report.append({"morph": name, "skipped": "no breast macro targets for this body (male)"})
            continue
        keys = export_body.data.shape_keys.key_blocks
        before = set(k.name for k in keys)
        weights = {}
        for n, w in rows:
            k = TargetService.load_target(export_body, str(targets_dir / (n + ".target.gz")), weight=w)
            weights[k.name] = w
        for k in export_body.data.shape_keys.key_blocks:
            k.value = weights.get(k.name, 0.0) if k.name not in before else 0.0
        combined = export_body.shape_key_add(name=name, from_mix=True)
        combined.value = 0.0
        for k in [k for k in export_body.data.shape_keys.key_blocks if k.name in weights]:
            export_body.shape_key_remove(k)
        report.append({"morph": name, "macro": {key: value}, "targets": rows})
    full_base = dict(TargetService.calculate_target_stack_from_macro_info_dict(macro))
    for name, key, value in BUILD:
        info = json.loads(json.dumps(macro))
        info[key] = value
        stack = dict(TargetService.calculate_target_stack_from_macro_info_dict(info))
        rows = [(n, stack.get(n, 0.0) - full_base.get(n, 0.0)) for n in sorted(set(stack) | set(full_base))]
        # Breast targets are female only (MPFB leaves the gender weight out).
        female = max(0.0, 1.0 - macro["gender"])
        rows = [(n, w * female if n.startswith("breast/") else w) for n, w in rows]
        rows = [(n, w) for n, w in rows if abs(w) > 1e-3 and (targets_dir / (n + ".target.gz")).exists()]
        keys = export_body.data.shape_keys.key_blocks
        before = set(k.name for k in keys)
        weights = {}
        for n, w in rows:
            k = TargetService.load_target(export_body, str(targets_dir / (n + ".target.gz")), weight=0.0)
            k.slider_min = -1.0
            weights[k.name] = w
        for k in export_body.data.shape_keys.key_blocks:
            k.value = weights.get(k.name, 0.0) if k.name not in before else 0.0
        combined = export_body.shape_key_add(name=name, from_mix=True)
        combined.value = 0.0
        for k in [k for k in export_body.data.shape_keys.key_blocks if k.name in weights]:
            export_body.shape_key_remove(k)
        report.append({"morph": name, "macro": {key: value}, "targets": rows})
for modifier in list(export_body.modifiers):
    if modifier.type == "MASK" and modifier.name.startswith("Delete."):
        export_body.modifiers.remove(modifier)
ExportService.bake_modifiers_remove_helpers(export_body, bake_masks=True, bake_subdiv=False, remove_helpers=True)
bpy.ops.object.select_all(action="DESELECT")
export_root.select_set(True)
for child in ObjectService.get_list_of_children(export_root):
    if not ObjectService.object_is(child, "Clothes"):
        child.hide_set(False)
        child.select_set(True)
bpy.context.view_layer.objects.active = export_root
bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", use_selection=True, export_animations=False,
                          export_morph=True, export_morph_normal=False)
keys = [k.name for k in export_body.data.shape_keys.key_blocks] if export_body.data.shape_keys else []
out.with_suffix(".targets.json").write_text(json.dumps({"source": str(blend), "morphs": report, "shape_keys": keys}, indent=1) + "\n")
print("EXPORTED", out, len(keys) - 1, "morphs")
