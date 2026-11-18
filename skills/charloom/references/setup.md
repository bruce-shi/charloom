# Environment and Input Preparation

English · [简体中文](setup.zh-CN.md)

Execute the command from the root directory of the repository. Python 3.12, uv, Bun, and Blender are required. The version and platform requirements are subject to the project dependency files and the Blender extension list.

```sh
uv venv --python 3.12 out/tools/cv/.venv
uv pip install --python out/tools/cv/.venv/bin/python -r scripts/requirements-blender.txt
(cd viewer && bun install --frozen-lockfile)
export PY="$PWD/out/tools/cv/.venv/bin/python"
export CHARLOOM_CV_PYTHON="$PY"
```

The Blender command line interface uses the aforementioned CV directory by default. When directly calling the Python API, you can pass `python=...` to select other installed dependent environments. The Bun facial motion script also requires CV Python, and the exported `CHARLOOM_CV_PYTHON` should be preserved. Set `BLENDER` as the executable file path, and use `"$BLENDER"` in the shell command to call it.

## Original Scan and Actions

Prepare the original scan for the new character, select the character skeleton and action materials for the project. For existing characters with additional actions, only the existing character GLB, new actions, and mapping configurations are required. The original head route uses static, single-mesh, single-triangle primitives, and a single texture atlas. When the input has skinned data, apply transformations and skinning to the explicit reference pose, and then export a static copy. Retain the original files, coordinates, UVs, and materials. Check the merged textures and the number of faces. After the preprocessing is completed, generate a guide that binds the corresponding copy hash.

To reproduce a walk, first check [Original Action and Driving Skeleton](walk-replay.md) and retain the actual motion source and binding parameters from the original build. The following instructions apply to retargeting onto the template rig. The motion must include its source skeleton, reference pose and selected clip. For joint-name mapping and reference-chain alignment see `scripts/transfer/motion.py` and `recipes/mixamo/`. Verify actual bone names and target chains and explicitly pass `--motion`. Use the specified scan and motion or obtain inputs with suitable permissions.

## MakeHuman Template

Download and generate the template from the fixed version list in the repository. Set `PRIOR_OUTPUT` as the new directory, and `BLENDER` as the actual executable file.

```sh
"$PY" scripts/mpfb-reference.py "$PRIOR_OUTPUT" --blender "$BLENDER" --gender male --muscles 0.5
```

Select gender, muscles, weight, height, and age based on the character. The `body.glb` of the selected preset sub-directory will be used as `--prior`, while `editable.blend`, template resources, and generation reports will also be retained.

## Image service requirements

When clothing hides the body and a valid registered reference is unavailable, this measurement route requires a newly generated reference. To generate it, install ImageMagick’s `magick` and the `genmedia` CLI, and supply `FAL_KEY` or configured genmedia credentials. The script calls fal’s `openai/gpt-image-2.5/sunburst/edit` at medium quality and roughly 2K multi-view resolution. Check the endpoint schema and the task’s image-service authorization. The guide stores the request ID, parameters and receipt under `views/measurement-cache/`. Resume the same request after an interruption or reuse a previously registered reference. Treat measurements of hidden anatomy as estimates and compare them with visible scan skin and anatomical landmarks.

Local texture rendering requires the image editing service selected in the task, along with the necessary credentials. If the materials are missing, clearly note the missing inputs, and complete the remaining executable steps.
