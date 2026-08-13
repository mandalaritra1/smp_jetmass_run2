#!/usr/bin/env bash
# Submit MESS + Vincia validation or an additive trijet production wave.
#
#   ./submit_mess.sh validation mess_trijet_20260802_v1
#   ./submit_mess.sh million    mess_trijet_20260802_v1
#   ./submit_mess.sh five_million mess_trijet_20260802_v1
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

MODE=${1:-validation}
CAMPAIGN=${2:-mess_trijet_20260802_v1}
JOBLIST=joblist_mess.txt
: > "$JOBLIST"

add_jobs() {
  local bin=$1 jobs=$2 nevt=$3 bin_offset=$4 base=$5
  local index seed
  for ((index=0; index<jobs; ++index)); do
    # Deterministic and unique across bin, mode, and job index. Retries reuse
    # the same seed by design and safely overwrite the same chunk.
    seed=$((base + bin_offset * 100000 + index * 131))
    printf '%s,%s,%s,%s\n' "$CAMPAIGN" "$bin" "$seed" "$nevt" >> "$JOBLIST"
  done
}

case "$MODE" in
  validation)
    BASE=7300000
    add_jobs QCD_HT300to500 1 2000 0 "$BASE"
    add_jobs QCD_HT500to700 1 2000 1 "$BASE"
    add_jobs QCD_HT700to1000 1 2000 2 "$BASE"
    add_jobs QCD_HT1000to1500 1 2000 3 "$BASE"
    add_jobs QCD_HT1500to2000 1 2000 4 "$BASE"
    add_jobs QCD_HT2000toInf 1 2000 5 "$BASE"
    ;;
  million)
    BASE=8200000
    # Variance-oriented first tranche, informed by the six-bin MESS
    # validation. HT300-500 and HT500-700 contributed no trijets above
    # 200 GeV (the latter is also negligible in the larger CP5 sample).
    add_jobs QCD_HT700to1000 75 5000 2 "$BASE"
    add_jobs QCD_HT1000to1500 65 5000 3 "$BASE"
    add_jobs QCD_HT1500to2000 45 5000 4 "$BASE"
    add_jobs QCD_HT2000toInf 15 5000 5 "$BASE"
    ;;
  five_million)
    # Additive five-million-LHE wave.  This is the measured one-million-wave
    # allocation scaled uniformly by five, so its HT mixture and pooled
    # estimator remain unchanged.  The disjoint seed base prevents collisions
    # when both waves are staged into the same campaign directories.
    BASE=9200000
    add_jobs QCD_HT700to1000 375 5000 2 "$BASE"
    add_jobs QCD_HT1000to1500 325 5000 3 "$BASE"
    add_jobs QCD_HT1500to2000 225 5000 4 "$BASE"
    add_jobs QCD_HT2000toInf 75 5000 5 "$BASE"
    ;;
  *) echo "usage: $0 {validation|million|five_million} [campaign]"; exit 2 ;;
esac

[ -x py313/pythia_rivet_mess ] || { echo "FATAL: missing py313/pythia_rivet_mess"; exit 1; }
[ -s py313/RivetCMS_HADRONIC_JETMASS.so ] || { echo "FATAL: missing py313 Rivet analysis library"; exit 1; }
NJOBS=$(wc -l < "$JOBLIST")
NEVENTS=$(awk -F, '{s += $4} END {print s+0}' "$JOBLIST")
echo "### mode=$MODE campaign=$CAMPAIGN jobs=$NJOBS requested_lhe=$NEVENTS"
if [ "${DRY_RUN:-0}" = 1 ]; then
  exit 0
fi
condor_submit submit_mess.jdl
