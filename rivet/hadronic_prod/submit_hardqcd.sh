#!/usr/bin/env bash
# Submit the high-statistics HardQCD production for the model variations.
#
#   ./submit_hardqcd.sh test                 # 1 seed/slice, small N
#   ./submit_hardqcd.sh full                 # the tuned per-slice allocation below
#   ./submit_hardqcd.sh flat 40 10000        # uniform: 40 seeds/slice, N=10000
#   CONFIGS="cp5 vincia" ./submit_hardqcd.sh full
#
# ONE CONFIG PER JOB. Running all configs in one job produced multi-GB Condor
# MemoryUsage and mass holds; the real cause was the 8.312 cr1 segfault (see
# inner_hardqcd.sh header), but one-config-per-job also keeps jobs short and
# limits the blast radius of any single failure. The variation/nominal PAIRING
# survives: the same (slice, seed) is submitted for every config and drives
# Random:seed, so each config showers identical hard events.
#
# VINCIA IS INCLUDED HERE. The MLM-matching defect that blocks it in the
# gridpack path does not exist in this one -- HardQCD is a plain 2->2 process
# with no matching veto at all, so the antenna shower runs unencumbered.
#
# PER-SLICE ALLOCATION (tuned 2026-08-01 from diag_slice_weights.py).
# Uniform allocation is badly wrong for TRIJET: the measured third jet is soft
# relative to pThat, so each pT bin is fed by rare, very-high-weight entries
# from the lowest contributing slice. Measured for trijet pT 200-290: slice
# pthat170_300 supplied 51% of the yield but 95% of the VARIANCE from just 851
# entries, while pthat1400_0 supplied 16k entries and ~0% of the variance.
# Statistical power therefore comes almost entirely from the low slices, and
# generating more at high pThat is wasted effort. Columns: ptlo pthi seeds nevt
SLICES=(
  "170  300  100 50000"
  "300  470   80 40000"
  "470  600   40 20000"
  "600  800   20 10000"
  "800  1000  10 10000"
  "1000 1400  10 10000"
  "1400 0     10 10000"
)
CONFIGS=${CONFIGS:-"cp5 vincia cr1 cr2 fraghard fragsoft"}
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
MODE=${1:-full}
: > joblist_hardqcd.txt
if [ "$MODE" = flat ] || [ "$MODE" = test ]; then
  if [ "$MODE" = test ]; then SEEDS_PER=1; NEVT=4000; else SEEDS_PER=${2:-40}; NEVT=${3:-10000}; fi
  for s in "${SLICES[@]}"; do
    set -- $s
    for j in $(seq 0 $((SEEDS_PER-1))); do
      seed=$(( 60000 + j*137 ))
      for cf in $CONFIGS; do echo "$1, $2, $seed, $NEVT, $cf" >> joblist_hardqcd.txt; done
    done
  done
else
  for s in "${SLICES[@]}"; do
    set -- $s          # ptlo pthi seeds nevt
    for j in $(seq 0 $(($3-1))); do
      seed=$(( 20000 + j*137 ))
      for cf in $CONFIGS; do echo "$1, $2, $seed, $4, $cf" >> joblist_hardqcd.txt; done
    done
  done
fi
n=$(wc -l < joblist_hardqcd.txt)
echo "### $MODE: $n jobs (1 config each), configs: $CONFIGS"
awk -F', ' '{print $1"_"$2": "$4" ev"}' joblist_hardqcd.txt | sort | uniq -c | sort -k2 -n
[ -x py313/pythia_rivet ] && [ -f py313/RivetCMS_HADRONIC_JETMASS.so ] || { echo "FATAL: build the 8.313 binaries first (build_rivet_313.sh)"; exit 1; }
condor_submit -append "REQMEM = ${REQMEM:-2000}" submit_hardqcd.jdl
echo "### submitted; watch:  condor_watch_q"
