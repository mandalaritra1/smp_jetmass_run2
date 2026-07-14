#!/usr/bin/env bash
# Combine the condor outputs into one CH3 prediction:
#   1. per slice: inverse-variance AVERAGE the seeds   (chunk_merge.py -- same dsigma/dx)
#   2. across slices: cross-section SUM the slices      (merge_slices.py -- disjoint pT)
#   ./combine_herwig_ch3.sh [out/herwig_ch3] [out/herwig_ch3.yoda]
# chunk_merge.py is pure stdlib+numpy; merge_slices.py needs the `yoda` module
# (run under a python with yoda, e.g. inside the LCG view or the rivet image).
set -euo pipefail
IDIR=${1:-../out/herwig_ch3}; OUT=${2:-../out/herwig_ch3.yoda}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PYTHON:-python3}

slice_yodas=()
for s in 0 1 2 3; do
  seeds=( "$IDIR"/herwig_ch3_s${s}_seed*.yoda )
  [ -e "${seeds[0]}" ] || { echo "no seeds for slice $s, skipping"; continue; }
  sy="$IDIR/herwig_ch3_s${s}.yoda"
  echo "### slice $s: inverse-variance avg of ${#seeds[@]} seeds -> $sy ###"
  $PY ../gen/chunk_merge.py "$sy" "${seeds[@]}"
  slice_yodas+=("$sy")
done

[ ${#slice_yodas[@]} -gt 0 ] || { echo "no slices to stitch"; exit 1; }
echo "### stitch ${#slice_yodas[@]} disjoint pT slices (xsec sum) -> $OUT ###"
$PY ../gen/merge_slices.py "$OUT" "${slice_yodas[@]}"
echo "done: $OUT"
