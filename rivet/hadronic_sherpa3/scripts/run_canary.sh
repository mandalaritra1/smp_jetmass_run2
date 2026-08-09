#!/usr/bin/env bash
# Run one Sherpa+Rivet canary. This script cannot submit a production.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
OUTDIR=${1:-$HERE/canary/run}
EVENTS=${2:-200}
SLICE_INDEX=${3:-2}
PTLOW=${4:-480}
PTHIGH=${5:-570}
SHOWER=${6:-Dire}
SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:?SOURCE_BUNDLE_SHA256 must be set}

if [[ -e "$OUTDIR" ]] && [[ -n "$(ls -A "$OUTDIR" 2>/dev/null)" ]]; then
  echo "FATAL: refusing to reuse non-empty canary directory: $OUTDIR" >&2
  exit 19
fi
mkdir -p "$OUTDIR"
# Absolutize before anything exports it: several steps below run with the cwd
# already inside $OUTDIR, so a relative path (as passed by the Condor wrapper)
# would resolve to $OUTDIR/$OUTDIR and silently miss the Rivet plugin.
OUTDIR=$(cd "$OUTDIR" && pwd)
python3 "$HERE/scripts/render_card.py" \
  --template "$HERE/configs/Sherpa.yaml.tmpl" \
  --output "$OUTDIR/Sherpa.yaml" \
  --pt-low "$PTLOW" --pt-high "$PTHIGH" --events "$EVENTS" \
  --slice-index "$SLICE_INDEX" --shower "$SHOWER" \
  > "$OUTDIR/card.json"

command -v Sherpa >/dev/null || { echo "FATAL: Sherpa is not on PATH" >&2; exit 20; }
command -v rivet-build >/dev/null || { echo "FATAL: rivet-build is not on PATH" >&2; exit 21; }
command -v rivet >/dev/null || { echo "FATAL: rivet is not on PATH" >&2; exit 22; }
python3 -c 'import numpy' >/dev/null || {
  echo "FATAL: the selected Python cannot import numpy" >&2
  exit 23
}
[[ "$(rivet --version | awk '{print $2}')" == v4.* ]] || {
  echo "FATAL: this routine is canaried only with Rivet 4" >&2
  exit 24
}
lhapdf list > "$OUTDIR/lhapdf_sets.txt"
grep -x PDF4LHC21_40_pdfas "$OUTDIR/lhapdf_sets.txt" >/dev/null || {
  echo "FATAL: PDF4LHC21_40_pdfas is not available to LHAPDF" >&2
  exit 25
}

Sherpa --version > "$OUTDIR/sherpa_version.txt" 2>&1
{
  echo "hostname=$(hostname -f 2>/dev/null || hostname)"
  echo "uname=$(uname -srm)"
  echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "LCG_VIEW=${LCG_VIEW:-unset}"
  echo "SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_SHA256:-unset}"
  echo "Sherpa=$(command -v Sherpa)"
  echo "rivet=$(command -v rivet)"
  echo "rivet-build=$(command -v rivet-build)"
  command -v Sherpa | xargs sha256sum
  rivet --version
  rivet-build --version 2>/dev/null || true
  command -v yoda-config >/dev/null && yoda-config --version || true
  command -v lhapdf-config >/dev/null && lhapdf-config --version || true
  command -v lhapdf-config >/dev/null && lhapdf-config --datadir || true
  echo "SHERPA_SHARE_PATH=${SHERPA_SHARE_PATH:-unset}"
  echo "LHAPDF_DATA_PATH=${LHAPDF_DATA_PATH:-unset}"
  echo "RIVET_ANALYSIS_PATH=${RIVET_ANALYSIS_PATH:-unset}"
} > "$OUTDIR/environment.txt" 2>&1
(cd "$OUTDIR" && Sherpa SHOW_SHOWER_GENERATORS=1) \
  > "$OUTDIR/shower_generators.log" 2>&1
grep -Eq "^[[:space:]]*${SHOWER}[[:space:]]" "$OUTDIR/shower_generators.log" || {
  echo "FATAL: requested shower $SHOWER was not advertised by Sherpa" >&2
  exit 26
}

(cd "$OUTDIR" && rivet-build \
  RivetCMS_HADRONIC_PAIR_SPLIT.so "$HERE/CMS_HADRONIC_PAIR_SPLIT.cc") \
  > "$OUTDIR/rivet_build.log" 2>&1

export RIVET_ANALYSIS_PATH="$OUTDIR:${RIVET_ANALYSIS_PATH:-}"
export HAD_PAIR_SPLIT_NTUPLE="$OUTDIR/pairsplit"
export HAD_PAIR_SPLIT_EVENT_NTUPLE="$OUTDIR/events.txt"

START=$(date +%s)
(cd "$OUTDIR" && Sherpa Sherpa.yaml) > "$OUTDIR/sherpa.log" 2>&1
END=$(date +%s)
echo $((END - START)) > "$OUTDIR/wall_seconds.txt"

(cd "$OUTDIR" && rivet -a CMS_HADRONIC_PAIR_SPLIT events \
  -o pair_split.yoda) > "$OUTDIR/rivet.log" 2>&1

python3 "$HERE/scripts/validate_pair_split.py" "$OUTDIR/events.txt" \
  --output "$OUTDIR/selection_validation.json"
python3 "$HERE/scripts/make_manifest.py" "$OUTDIR" \
  --source "$HERE/CMS_HADRONIC_PAIR_SPLIT.cc" \
  --shower "$SHOWER" --slice-index "$SLICE_INDEX" \
  --pt-low "$PTLOW" --pt-high "$PTHIGH" --requested-events "$EVENTS" \
  --source-bundle-sha256 "$SOURCE_BUNDLE_SHA256"

echo "Canary complete: $OUTDIR"
