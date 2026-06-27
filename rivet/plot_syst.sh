#!/usr/bin/env bash
# Overlay the modelling variations (colour reconnection + Lund hadronization)
# against the nominal prediction -- a dedicated figure for the ARC's "parton
# shower vs hadronisation" question, kept separate from the multi-generator plot.sh.
#
# Auto-selects the family present in out/:
#   * MG-LO + Pythia   (mglo_pythia*.yoda, from ./run_mg_local.sh)   <- preferred
#   * standalone Pythia (pythia*.yoda,    from ./run_slices.sh / ./run_all.sh)
#
#   ./plot_syst.sh [output_dir] [linear]      (default: out/plots_syst, log y)
#
# Run inside an image with rivet-mkhtml (run_mg_local.sh calls it for you).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
OUTDIR=${1:-out/plots_syst}
CFG=gen/compare.plot
[ "${2:-}" = "linear" ] && CFG=gen/compare_linear.plot

mkdir -p out/_norm
args=()
# Prefer the flat-statistics sliced yoda if present, else the plain one; then
# area-normalise (shape comparison). Trailing ||true so a missing yoda is skipped.
add() {
  local f="$1"; local s="${1%.yoda}_sliced.yoda"
  [ -f "$s" ] && f="$s"
  [ -f "$f" ] || return 0
  local nf="out/_norm/$(basename "$f")"
  python3 gen/normalize_shapes.py "$f" "$nf"
  args+=("$nf:Title=$2") || true
}

# MG-LO+Pythia family if present, else standalone Pythia family.
if ls out/mglo_pythia*.yoda >/dev/null 2>&1; then PFX="mglo_"; LBL="MGLO+Pythia8"; else PFX=""; LBL="Pythia8_nominal"; fi
add "out/${PFX}pythia.yoda"          "$LBL"
add "out/${PFX}pythia_cr1.yoda"      "CR_QCD"
add "out/${PFX}pythia_cr2.yoda"      "CR_gluon-move"
add "out/${PFX}pythia_fragsoft.yoda" "Frag_soft"
add "out/${PFX}pythia_fraghard.yoda" "Frag_hard"

if [ ${#args[@]} -eq 0 ]; then
  echo "no out/(mglo_)pythia*.yoda found - run run_mg_local.sh / run_slices.sh first" >&2; exit 1
fi

rivet-mkhtml -c "$CFG" -o "$OUTDIR" "${args[@]}"
echo "plots in $OUTDIR/index.html"
