#!/usr/bin/env bash
# Build a deterministic job list and submit the pair_split MESS+Vincia dijet
# campaign. Allocations are LHE-event targets, rounded to 5k-event jobs.
#
#   ./submit_mess_pairsplit.sh canary mess_pairsplit_dijet_20260813_v1
#   ./submit_mess_pairsplit.sh pilot  mess_pairsplit_dijet_20260813_v1
#   ./submit_mess_pairsplit.sh pilot2 mess_pairsplit_dijet_20260813_v1
#   ./submit_mess_pairsplit.sh full   mess_pairsplit_dijet_20260813_v1
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

MODE=${1:-canary}
CAMPAIGN=${2:-mess_pairsplit_dijet_20260813_v1}
JOBLIST=joblist_mess_pairsplit.txt
: > "$JOBLIST"

add_jobs() {
  local phase=$1 bin=$2 jobs=$3 nevt=$4 bin_offset=$5 base=$6
  local index seed
  for ((index=0; index<jobs; ++index)); do
    seed=$((base + bin_offset * 100000 + index * 131))
    printf '%s,%s,%s,%s,%s\n' \
      "$CAMPAIGN" "$phase" "$bin" "$seed" "$nevt" >> "$JOBLIST"
  done
}

case "$MODE" in
  canary)
    BASE=13100000
    add_jobs canary QCD_HT300to500 1 2000 0 "$BASE"
    add_jobs canary QCD_HT500to700 1 2000 1 "$BASE"
    add_jobs canary QCD_HT700to1000 1 2000 2 "$BASE"
    add_jobs canary QCD_HT1000to1500 1 2000 3 "$BASE"
    add_jobs canary QCD_HT1500to2000 1 2000 4 "$BASE"
    add_jobs canary QCD_HT2000toInf 1 2000 5 "$BASE"
    ;;
  pilot)
    BASE=14100000
    add_jobs pilot QCD_HT300to500 4 5000 0 "$BASE"
    add_jobs pilot QCD_HT500to700 4 5000 1 "$BASE"
    add_jobs pilot QCD_HT700to1000 4 5000 2 "$BASE"
    add_jobs pilot QCD_HT1000to1500 4 5000 3 "$BASE"
    add_jobs pilot QCD_HT1500to2000 4 5000 4 "$BASE"
    add_jobs pilot QCD_HT2000toInf 4 5000 5 "$BASE"
    ;;
  pilot2)
    # Distinct deterministic recovery pilot after the concurrent diagnostic
    # product-name collision was found in the first pilot.
    BASE=14200000
    add_jobs pilot2 QCD_HT300to500 4 5000 0 "$BASE"
    add_jobs pilot2 QCD_HT500to700 4 5000 1 "$BASE"
    add_jobs pilot2 QCD_HT700to1000 4 5000 2 "$BASE"
    add_jobs pilot2 QCD_HT1000to1500 4 5000 3 "$BASE"
    add_jobs pilot2 QCD_HT1500to2000 4 5000 4 "$BASE"
    add_jobs pilot2 QCD_HT2000toInf 4 5000 5 "$BASE"
    ;;
  full)
    BASE=15100000
    # Goal allocation: 8.00M requested LHE events across six official QCD-HT
    # gridpack slices. The pilot gate may change these numbers only with a
    # recorded acceptance/throughput justification.
    add_jobs full QCD_HT300to500 200 5000 0 "$BASE"
    add_jobs full QCD_HT500to700 200 5000 1 "$BASE"
    add_jobs full QCD_HT700to1000 450 5000 2 "$BASE"
    add_jobs full QCD_HT1000to1500 390 5000 3 "$BASE"
    add_jobs full QCD_HT1500to2000 270 5000 4 "$BASE"
    add_jobs full QCD_HT2000toInf 90 5000 5 "$BASE"
    ;;
  *) echo "usage: $0 {canary|pilot|pilot2|full} [campaign]"; exit 2 ;;
esac

NJOBS=$(wc -l < "$JOBLIST")
NEVENTS=$(awk -F, '{sum += $5} END {print sum+0}' "$JOBLIST")
mkdir -p condor_logs
echo "### mode=$MODE campaign=$CAMPAIGN jobs=$NJOBS requested_lhe=$NEVENTS"
if [ "${DRY_RUN:-0}" = 1 ]; then
  exit 0
fi
[ -x py313/pythia_rivet_mess ] || {
  echo "FATAL: missing py313/pythia_rivet_mess"; exit 1;
}
[ -s py313/RivetCMS_HADRONIC_PAIR_SPLIT.so ] || {
  echo "FATAL: missing py313 pair_split Rivet analysis library"; exit 1;
}
[ -s py313/CMS_HADRONIC_PAIR_SPLIT.cc ] || {
  echo "FATAL: missing frozen pair_split analysis source"; exit 1;
}
[ -s py313/pythia_rivet_mess.cc ] || {
  echo "FATAL: missing frozen MESS driver source"; exit 1;
}
condor_submit submit_mess_pairsplit.jdl
