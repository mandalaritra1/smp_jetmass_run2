#!/usr/bin/env bash
# Overlay the Pythia modelling-variation yodas (colour reconnection + hadronization)
# against the nominal Pythia prediction -- a dedicated figure for the ARC's
# "parton shower vs hadronisation" modelling-uncertainty question, kept separate
# from the multi-generator plot.sh comparison.
#
#   ./plot_syst.sh [output_dir] [linear]      (default: out/plots_syst, log y)
#
# Generate the inputs first, e.g. (flat high-pT stats via slices):
#   for g in pythia pythia_cr1 pythia_cr2 pythia_fragsoft pythia_fraghard; do
#     ./run_slices.sh "$g" 20000
#   done
# (or ./run_all.sh "$g" for a quick inclusive run). Then run this inside an image
# with rivet-mkhtml.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
OUTDIR=${1:-out/plots_syst}
CFG=gen/compare.plot
[ "${2:-}" = "linear" ] && CFG=gen/compare_linear.plot

mkdir -p out/_norm
args=()
# Prefer the flat-statistics sliced yoda if present, else the inclusive one; then
# area-normalise (shape comparison). Trailing ||true so a missing yoda is skipped.
add() {
  local f="$1"; local s="${1%.yoda}_sliced.yoda"
  [ -f "$s" ] && f="$s"
  [ -f "$f" ] || return 0
  local nf="out/_norm/$(basename "$f")"
  python3 gen/normalize_shapes.py "$f" "$nf"
  args+=("$nf:Title=$2") || true
}
add out/pythia.yoda          "Pythia8_nominal"
add out/pythia_cr1.yoda      "CR_QCD"
add out/pythia_cr2.yoda      "CR_gluon-move"
add out/pythia_fragsoft.yoda "Frag_soft"
add out/pythia_fraghard.yoda "Frag_hard"

if [ ${#args[@]} -eq 0 ]; then
  echo "no out/pythia*.yoda found - run run_slices.sh / run_all.sh first" >&2; exit 1
fi

rivet-mkhtml -c "$CFG" -o "$OUTDIR" "${args[@]}"
echo "plots in $OUTDIR/index.html"
