#!/usr/bin/env bash
# Build joblist.txt and submit the dijet/trijet model-uncertainty production.
#
#   ./submit_all.sh canary            # 1 job/bin, N=800   (validates the chain)
#   ./submit_all.sh full 20 5000      # 20 jobs/bin, N=5000
#   CONFIGS="cp5 vincia" ./submit_all.sh full 8 5000       # override the set
#
# Each job generates ONE LHE and showers it with EVERY config in $CONFIGS, so
# the variation/nominal ratio sees identical hard events (noise cancels) and
# runcmsgrid runs once instead of once per config.
#
# vincia is EXCLUDED by default: the MLM veto does not fire correctly under
# the antenna shower (accepted xsec 4.2x the official value; same defect in
# the zjet vincia_prod sample). See ai-wiki
# wiki/bugs/mlm_matching_broken_under_vincia.md. Re-add it with CONFIGS=...
# once that is resolved.
#
# HT bins below 300 are skipped: at jet pT > 185 they contribute nothing
# (a 185 GeV dijet already has parton HT ~ 370).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
BINS=(QCD_HT300to500 QCD_HT500to700 QCD_HT700to1000 QCD_HT1000to1500 QCD_HT1500to2000 QCD_HT2000toInf)
CONFIGS=${CONFIGS:-"cp5 cr1 cr2 fraghard fragsoft"}
MODE=${1:-full}
if [ "$MODE" = canary ]; then
  JOBS_PER=1; NEVT=800; BASE=90000
else
  JOBS_PER=${2:-20}; NEVT=${3:-5000}; BASE=10000
fi
: > joblist.txt
for b in "${BINS[@]}"; do
  for j in $(seq 0 $((JOBS_PER-1))); do
    seed=$(( BASE + j*131 + RANDOM % 97 ))
    echo "$b, $seed, $NEVT" >> joblist.txt
  done
done
n=$(wc -l < joblist.txt)
echo "### $MODE: $n jobs x $(echo $CONFIGS | wc -w) configs each (${JOBS_PER}/bin, NEVT=$NEVT)"
echo "### configs: $CONFIGS"
[ -x pythia_rivet ] && [ -f RivetCMS_HADRONIC_JETMASS.so ] || { echo "FATAL: run build_rivet.sh first (need pythia_rivet + .so)"; exit 1; }
condor_submit -append "HADCONFIGS = $CONFIGS" submit_hadronic.jdl
echo "### submitted; watch:  condor_watch_q"
