#!/bin/zsh
# Run the review characters (seven tuned ones by default; held-out:
# stocky-woman, starbow-adept, and the four-digit fox) through scripts/mh-review.sh and print one
# summary line each.
#
#   scripts/mh-batch.sh TAG [NAME...]
#
# Writes REVIEW_DIR/NAME-TAG/ (and .log, .summary); REVIEW_DIR defaults to
# out/review in the current checkout.
# Sources, priors (body-shape.glb, with MakeHuman's shape targets) and guidance
# are read from the main checkout, so this works from a worktree too. Three
# characters run at a time; Blender slots are shared across worktrees.
# GUIDE_SUFFIX (e.g. -v2) picks alternative
# guidance where it exists. MOTION, MH_HEAD, MH_CONFIG and the rest of mh-review.sh's environment pass through.
set -e
TAG=$1; shift
ROOT=${0:A:h:h}
MAIN=${$(git -C $ROOT rev-parse --path-format=absolute --git-common-dir):h}
export CHARLOOM_BLENDER_LOCKS=${CHARLOOM_BLENDER_LOCKS:-$MAIN/out}
export MOTION=${MOTION:-$MAIN/fixtures/mixamo/walking.fbx}
# body-shape-build.glb adds MakeHuman's chest targets (pectoral, dorsi, the
# female breast macros) and the build macros (weight, muscle) to the shape
# fit; SHAPE_PRIOR=body-shape-chest or body-shape keeps an older export.
SP=${SHAPE_PRIOR:-body-shape-build}
F=$MAIN/out/examples/mpfb-female-reference-v3/female-muscle-050/$SP.glb
M=$MAIN/out/examples/mpfb-reference-v7/male-muscle-050/$SP.glb
[ -f $F ] || F=${F:h}/body-shape.glb
[ -f $M ] || M=${M:h}/body-shape.glb
# Supply the character input, prior and guide arrays for this batch.
source "${CHARLOOM_BATCH_TABLE:-$MAIN/assets/private/batch.zsh}"
cd $ROOT
R=${REVIEW_DIR:-out/review}
mkdir -p $R
i=0
for n in $NAMES; do
  g=$MAIN/out/examples/mh-guide-${GUIDE[$n]:-$n}/guidance.json
  # GUIDE_SUFFIX=-v2 reads mh-guide-NAME-v2 where it exists (else the plain one).
  gs=$MAIN/out/examples/mh-guide-${GUIDE[$n]:-$n}$GUIDE_SUFFIX/guidance.json
  [ -n "$GUIDE_SUFFIX" ] && [ -f $gs ] && g=$gs
  (scripts/mh-review.sh $R/$n-$TAG ${SRC[$n]} ${PRIOR[$n]} $g > $R/$n-$TAG.summary 2>&1; echo "$n exit $?") &
  i=$((i + 1)); [ $((i % 3)) -eq 0 ] && wait
done
wait
for n in $NAMES; do
  python3 - $n $R/$n-$TAG/report.json <<'EOF'
import json, sys
n, path = sys.argv[1:]
try:
    r = json.load(open(path))
except OSError:
    print(n.ljust(10), 'no report'); sys.exit()
print(n, r["stage2_rig"], "registered silhouette distance (px)", r["stage2_body_fit"].get("median_silhouette_distance_px"))
EOF
done
