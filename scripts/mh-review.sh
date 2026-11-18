#!/bin/zsh
# Fit a marked scan in the CV interpreter and render the staged wardrobe.
# scripts/mh-review.sh OUT_DIR SOURCE.glb PRIOR_BODY.glb GUIDANCE.json
set -eu
O=$1; SRC=$2; PRIOR=$3; G=$4
ROOT=${0:A:h:h}
cd $ROOT
PY=${CHARLOOM_CV_PYTHON:-$ROOT/out/tools/cv/.venv/bin/python}
B=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
MOTION=${MOTION:-fixtures/mixamo/walking.fbx}
mkdir -p ${O:h}
ARGS=(--prior "$PRIOR" --guidance "$G" --motion "$MOTION" --out "$O")
[[ -n "${MH_TEXTURE_SIZE:-}" ]] && ARGS+=(--texture-size "$MH_TEXTURE_SIZE")
[[ -n "${MH_MOUTH_TRACE:-}" ]] && ARGS+=(--mouth-trace "$MH_MOUTH_TRACE")
"$PY" scripts/scan-transfer.py "$SRC" "${ARGS[@]}" > "$O.log" 2>&1 || { tail -6 "$O.log"; exit 1; }
if [[ "${MH_RENDER:-blender}" == cpu ]]; then
  "$PY" scripts/render-character.py "$O" --only "${MH_RENDER_ONLY:-segment,body,dressed,walk}" > "$O/renders.log" 2>&1
  exit
fi
# At most CHARLOOM_BLENDER_SLOTS (default 3) Blender jobs across all concurrent reviews (and worktrees:
# set CHARLOOM_BLENDER_LOCKS to one shared folder): take a slot.
LOCKS=${CHARLOOM_BLENDER_LOCKS:-out}
mkdir -p $LOCKS
SLOT=""
while [ -z "$SLOT" ]; do
  for s in $(seq 1 ${CHARLOOM_BLENDER_SLOTS:-3}); do mkdir $LOCKS/.blender-slot-$s 2>/dev/null && { SLOT=$LOCKS/.blender-slot-$s; break; }; done
  [ -z "$SLOT" ] && sleep 2
done
trap 'rmdir $SLOT 2>/dev/null' EXIT INT TERM
# One Blender session for all passes: each GLB is imported once.
$B -b --factory-startup --python-exit-code 1 --python scripts/mh-render-review.py -- $O > $O/renders.log 2>&1 || { tail -5 $O/renders.log; exit 1; }
echo "renders: $O/renders"
