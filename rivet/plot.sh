#!/usr/bin/env bash
# Overlay the available per-generator yodas into shape-comparison plots.
#   ./plot.sh [output_dir]      (default: out/plots)
# Run inside any image that has rivet-mkhtml, e.g.:
#   docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work \
#       hepstore/rivet-pythia:latest ./plot.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
OUTDIR=${1:-out/plots}
# second arg "linear" -> linear y-axis (default: log, via compare.plot)
CFG=gen/compare.plot
[ "${2:-}" = "linear" ] && CFG=gen/compare_linear.plot

# NOTE: rivet-mkhtml's per-file "path:Title=..." option does not tolerate
# spaces/parens in the title -> use single-token labels.
mkdir -p out/_norm
args=()
# Prefer the flat-statistics sliced yoda if present, else the plain one; then
# area-normalise it (NormalizeToIntegral is a no-op on finalized estimates).
# Trailing ||true so a missing yoda doesn't trip set -e and abort the script.
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

if [ ${#args[@]} -eq 0 ]; then echo "no out/*.yoda found - run run_all.sh first" >&2; exit 1; fi

rivet-mkhtml -c "$CFG" -o "$OUTDIR" "${args[@]}"
echo "plots in $OUTDIR/index.html"
