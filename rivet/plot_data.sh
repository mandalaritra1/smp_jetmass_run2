#!/usr/bin/env bash
# Overlay the generator predictions with the unfolded DATA (refdata/), rho only,
# with ratios taken w.r.t. data. The data /REF/ objects are already per-pT-bin
# unit-area density; the MC curves are area-normalised to match.
#   ./plot_data.sh [outdir] [linear]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
OUTDIR=${1:-out/plots_data}
CFG=gen/compare.plot
[ "${2:-}" = "linear" ] && CFG=gen/compare_linear.plot

[ -f refdata/CMS_ZJET_JETMASS.yoda ] || python3 gen/make_data_ref.py
mkdir -p out/_norm

args=()
add() {
  local f="$1"; local s="${1%.yoda}_sliced.yoda"
  [ -f "$s" ] && f="$s"
  [ -f "$f" ] || return 0
  local nf="out/_norm/$(basename "$f")"
  python3 gen/normalize_shapes.py "$f" "$nf"
  args+=("$nf:Title=$2") || true
}
add out/pythia.yoda      "Pythia8"
add out/vincia.yoda      "Vincia"
add out/herwig.yoda      "Herwig7"
add out/sherpa.yoda      "Sherpa"
add out/mglo_pythia.yoda "MGLO+Pythia8"
add out/mglo_herwig.yoda "MGLO+Herwig7"

# data reference first; rivet-mkhtml treats /REF/ as data and ratios MC -> data.
rivet-mkhtml -c "$CFG" -o "$OUTDIR" -m "rho_" \
  refdata/CMS_ZJET_JETMASS.yoda:"Title=CMS (unfolded)" "${args[@]}"
echo "plots in $OUTDIR/index.html"
