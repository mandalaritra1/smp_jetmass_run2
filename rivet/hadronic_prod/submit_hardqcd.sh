#!/usr/bin/env bash
# Submit the high-statistics HardQCD production for the HAD leg.
#
#   ./submit_hardqcd.sh test                 # 1 seed/slice x all configs
#   ./submit_hardqcd.sh full 40 10000        # 40 seeds/slice, N=10000 per job
#   CONFIGS="cp5 cr1" ./submit_hardqcd.sh full 10 10000
#
# ONE CONFIG PER JOB (changed 2026-08-01). Running all five configs inside a
# single job produced multi-GB Condor MemoryUsage and mass holds even though
# the process itself peaks at ~83 MB (measured over 73 samples of a full
# 10k-event run) -- the reported usage tracked ~2x whatever request_memory
# was set to, i.e. reclaimable page cache counted against the cgroup, not
# real allocation. Splitting per config keeps each job short and its I/O
# small, and a failure now costs one config instead of the whole set.
#
# The variation/nominal PAIRING IS PRESERVED: the same (slice, seed) is
# submitted for every config, and inner_hardqcd.sh sets Random:seed from it,
# so each config showers the SAME hard events and the LHE-level noise still
# cancels in the ratio.
#
# pThat slices follow the official QCD_Pt binning over the measured range;
# each is normalized by its own sigma/sumw at harvest time. The open top
# slice uses pthi=0 (no upper bound).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
# ptlo pthi
SLICES=("170 300" "300 470" "470 600" "600 800" "800 1000" "1000 1400" "1400 0")
CONFIGS=${CONFIGS:-"cp5 cr1 cr2 fraghard fragsoft"}
MODE=${1:-full}
if [ "$MODE" = test ]; then
  SEEDS_PER=1; NEVT=4000; BASE=60000
else
  SEEDS_PER=${2:-40}; NEVT=${3:-10000}; BASE=20000
fi
: > joblist_hardqcd.txt
for s in "${SLICES[@]}"; do
  set -- $s
  for j in $(seq 0 $((SEEDS_PER-1))); do
    seed=$(( BASE + j*137 ))          # deterministic: same seed for every config
    for cf in $CONFIGS; do
      echo "$1, $2, $seed, $NEVT, $cf" >> joblist_hardqcd.txt
    done
  done
done
n=$(wc -l < joblist_hardqcd.txt)
echo "### $MODE: $n jobs (1 config each; ${SEEDS_PER} seeds/slice x $(echo $CONFIGS | wc -w) configs x ${#SLICES[@]} slices, NEVT=$NEVT)"
echo "### configs: $CONFIGS"
[ -x pythia_rivet ] && [ -f RivetCMS_HADRONIC_JETMASS.so ] || { echo "FATAL: run build_rivet.sh first"; exit 1; }
condor_submit -append "REQMEM = ${REQMEM:-2000}" submit_hardqcd.jdl
echo "### submitted; watch:  condor_watch_q"
