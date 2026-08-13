#!/usr/bin/env bash
# Submit a manifest-gated Pythia 8.313 internal-variation campaign.
#
#   ./submit_internal_variations.sh validation mgmlm_internal_trijet_20260802_v1
#   ./submit_internal_variations.sh five_million mgmlm_internal_trijet_20260802_v1
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

MODE=${1:-validation}
CAMPAIGN=${2:-mgmlm_internal_trijet_20260802_v1}
JOBLIST=joblist_internal.txt
: > "$JOBLIST"

add_jobs() {
  local bin=$1 jobs=$2 nevt=$3 bin_offset=$4 base=$5
  local index seed
  for ((index=0; index<jobs; ++index)); do
    seed=$((base + bin_offset * 100000 + index * 131))
    printf '%s,%s,%s,%s,%s\n' "$MODE" "$CAMPAIGN" "$bin" "$seed" "$nevt" >> "$JOBLIST"
  done
}

case "$MODE" in
  validation)
    BASE=17000000
    add_jobs QCD_HT300to500 1 2000 0 "$BASE"
    add_jobs QCD_HT500to700 1 2000 1 "$BASE"
    add_jobs QCD_HT700to1000 1 2000 2 "$BASE"
    add_jobs QCD_HT1000to1500 1 2000 3 "$BASE"
    add_jobs QCD_HT1500to2000 1 2000 4 "$BASE"
    add_jobs QCD_HT2000toInf 1 2000 5 "$BASE"
    ;;
  five_million)
    BASE=19000000
    # Variance-oriented first tranche from the pooled legacy CP5 sample.
    # It preserves the dijet low-pT HT300/500 slices while feeding the
    # trijet bottlenecks HT500/700, HT700/1000, and HT1000/1500.
    add_jobs QCD_HT300to500 120 5000 0 "$BASE"
    add_jobs QCD_HT500to700 220 5000 1 "$BASE"
    add_jobs QCD_HT700to1000 260 5000 2 "$BASE"
    add_jobs QCD_HT1000to1500 220 5000 3 "$BASE"
    add_jobs QCD_HT1500to2000 150 5000 4 "$BASE"
    add_jobs QCD_HT2000toInf 30 5000 5 "$BASE"
    ;;
  *) echo "usage: $0 {validation|five_million} [campaign]"; exit 2 ;;
esac

NJOBS=$(wc -l < "$JOBLIST")
NEVENTS=$(awk -F, '{s += $5} END {print s+0}' "$JOBLIST")
cp "$JOBLIST" "joblist_internal_${MODE}.txt"
echo "### mode=$MODE campaign=$CAMPAIGN jobs=$NJOBS requested_lhe=$NEVENTS configs=cp5,cr1,cr2,fraghard,fragsoft"
if [ "${DRY_RUN:-0}" = 1 ]; then
  exit 0
fi
[ -x py313/pythia_rivet ] || { echo "FATAL: missing py313/pythia_rivet"; exit 1; }
[ -s py313/RivetCMS_HADRONIC_JETMASS.so ] || { echo "FATAL: missing py313 Rivet analysis library"; exit 1; }
condor_submit submit_internal_variations.jdl
