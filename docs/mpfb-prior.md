# MakeHuman (MPFB) prior bodies

`charloom mh-transfer` reshapes a rigged MakeHuman body: the MPFB default rig
(163 bones, with face and finger chains) and its weights. The prior bodies
are generated in a background Blender process with pinned inputs.

Pinned upstream versions, download URLs and SHA256 sums are in
[`scripts/mpfb/sources.json`](../scripts/mpfb/sources.json) (MPFB 2.0.18 and
the MakeHuman CC0 system assets). Downloads and the add-on data live in
`out/tools/mpfb`; your own Blender configuration is not touched.

## Generate

```sh
python3 scripts/mpfb-reference.py out/examples/mpfb-male-NEW --muscles 0.5 1.0
python3 scripts/mpfb-reference.py out/examples/mpfb-female-NEW --gender female --muscles 0.5 0.85
```

`--weight`, `--height`, `--age` and `--muscles` are MakeHuman's own 0–1
macro parameters; age 0.5 is a young adult. Pass `--blender PATH` or set
`BLENDER`. The output directory must be new.

Each preset folder (e.g. `female-muscle-050/`) holds:
- `body.glb`: the naked skinned body. **This is the `--prior` for
  `mh-transfer`.**
- `editable.blend`: the full MakeHuman parameters, shape keys and rig.
- `clothed.blend` / `clothed.glb` and review renders.

Priors in use: `out/examples/mpfb-female-reference-v3/female-muscle-050/body.glb`
and `out/examples/mpfb-reference-v7/male-muscle-050/body.glb`.

## Open or check a reference

```sh
# Edit in Blender with the project-private MPFB add-on (MPFB side panel)
BLENDER_USER_CONFIG="$PWD/out/tools/mpfb/runtime/config" \
BLENDER_USER_EXTENSIONS="$PWD/out/tools/mpfb/runtime/extensions" \
/Applications/Blender.app/Contents/MacOS/Blender --factory-startup --disable-autoexec \
  --python scripts/mpfb/open-reference.py -- out/examples/mpfb-reference-v7/male-muscle-100/editable.blend

# Re-import the exports and compare surfaces, triangle counts and the 163 bones
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --disable-autoexec \
  --python-exit-code 1 --python scripts/mpfb/check-exports.py -- out/examples/mpfb-male-NEW
```

## Shape targets (`body-shape-build.glb`)

The shape fit (`charloom-mh` `shape.rs`) uses MakeHuman's own targets as
glTF morphs: belly, girths, head, hands, feet, face features and the chest
(`pectoral`, `dorsi`, and on a female body the breast macros as
`breast-size` / `breast-firmness`), and the whole-body build (MPFB's
weight and muscle macros as `build-weight` / `build-muscle`). The body under
the clothes is only this shape: the build (fitted first, from the visible
body) makes it stocky or slim, the chest targets give it a chest
(`body-shape-chest.glb` is the export without the build):

```sh
for d in mpfb-female-reference-v3/female-muscle-050 mpfb-reference-v7/male-muscle-050; do
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
    --disable-autoexec --python-exit-code 1 --python scripts/mpfb/export-shape-targets.py -- \
    out/examples/$d/editable.blend out/examples/$d/body-shape-build.glb
done
```

## Ear targets (`ear-targets.glb`)

`charloom mh-transfer` shapes each ear to its marked outline (`ear_{l,r}_*`
landmarks) with MakeHuman's per-side ear targets. They are read from a
sidecar `ear-targets.glb` next to the prior (any `body*.glb` in that folder),
the same body with every `ears/` target as one morph under its official name:

```sh
for d in mpfb-female-reference-v3/female-muscle-050 mpfb-reference-v7/male-muscle-050; do
  CHARLOOM_TARGET_SET=ears /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
    --disable-autoexec --python-exit-code 1 --python scripts/mpfb/export-shape-targets.py -- \
    out/examples/$d/editable.blend out/examples/$d/ear-targets.glb
done
```

Without it the outline still places the ear (similarity only); a sidecar
whose body does not match the prior's vertex for vertex is ignored.
