#!/usr/bin/env bash
# Build joblist.txt and submit the dijet/trijet model-uncertainty production.
#   6 HT bins x 6 configs x JOBS_PER jobs.
#   Canary first:  ./submit_all.sh canary     (cp5+vincia, 1 job/bin, N=800 -> 12 jobs)
#   Full:          ./submit_all.sh full 8 5000  (6 configs x 6 bins x 8 = 288 jobs)
# HT bins below 300 are skipped: at jet pT > 185 they contribute nothing
# (a 185 GeV dijet already has parton HT ~ 370).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
BINS=(QCD_HT300to500 QCD_HT500to700 QCD_HT700to1000 QCD_HT1000to1500 QCD_HT1500to2000 QCD_HT2000toInf)
MODE=${1:-full}
if [ "$MODE" = canary ]; then
  CONFIGS=(cp5 vincia); JOBS_PER=1; NEVT=800; BASE=90000
else
  CONFIGS=(cp5 vincia cr1 cr2 fraghard fragsoft); JOBS_PER=${2:-8}; NEVT=${3:-5000}; BASE=10000
fi
: > joblist.txt
for cf in "${CONFIGS[@]}"; do
  for b in "${BINS[@]}"; do
    for j in $(seq 0 $((JOBS_PER-1))); do
      seed=$(( BASE + j*131 + RANDOM % 97 ))
      echo "$b, $seed, $NEVT, $cf" >> joblist.txt
    done
  done
done
n=$(wc -l < joblist.txt)
echo "### $MODE: $n jobs (configs=${CONFIGS[*]}, ${JOBS_PER}/bin, NEVT=$NEVT)"
[ -x pythia_rivet ] && [ -f RivetCMS_HADRONIC_JETMASS.so ] || { echo "FATAL: run build_rivet.sh first (need pythia_rivet + .so)"; exit 1; }
condor_submit submit_hadronic.jdl
echo "### submitted; watch:  condor_watch_q"
