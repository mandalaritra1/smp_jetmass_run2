#!/usr/bin/env bash
# After all four run_mg_sliced_syst.sh slices finish (on whatever nodes), stitch the
# per-slice yodas into one flat-statistics prediction per variation. The per-slice
# yodas are absolute dsigma/dx (pb) -- merge_slices.py sums the disjoint slices.
# Writes out/mglo_<name>.yoda (the name plot_model_uncertainty.py reads), so:
#   ./stitch_sliced_syst.sh
#   python3 plot_model_uncertainty.py        # (or the GluonJetMass venv python)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"

for name in pythia pythia_vincia pythia_cr1 pythia_cr2 pythia_fragsoft pythia_fraghard; do
  slices=$(ls out/slices/mglo_${name}_s*.yoda 2>/dev/null)
  [ -z "$slices" ] && { echo "skip $name (no slices)"; continue; }
  python3 gen/merge_slices.py "out/mglo_${name}.yoda" $slices
  echo "stitched $name <- $(echo $slices | wc -w) slices -> out/mglo_${name}.yoda"
done
echo "### stitched. now: python3 plot_model_uncertainty.py ###"
