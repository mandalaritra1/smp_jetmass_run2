#!/usr/bin/env bash
# Submit the high-statistics HardQCD production for the HAD leg.
#
#   ./submit_hardqcd.sh canary               # 1 job/slice, N=2000 (times + yields)
#   ./submit_hardqcd.sh full 40 20000        # 40 jobs/slice, N=20000 per config
#   CONFIGS="cp5 cr1" ./submit_hardqcd.sh full 10 20000
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
if [ "$MODE" = canary ]; then
  JOBS_PER=1; NEVT=2000; BASE=70000
else
  JOBS_PER=${2:-40}; NEVT=${3:-20000}; BASE=20000
fi
: > joblist_hardqcd.txt
for s in "${SLICES[@]}"; do
  set -- $s
  for j in $(seq 0 $((JOBS_PER-1))); do
    seed=$(( BASE + j*137 + RANDOM % 89 ))
    echo "$1, $2, $seed, $NEVT" >> joblist_hardqcd.txt
  done
done
n=$(wc -l < joblist_hardqcd.txt)
echo "### $MODE: $n jobs x $(echo $CONFIGS | wc -w) configs each (${JOBS_PER}/slice, NEVT=$NEVT)"
echo "### slices: ${SLICES[*]}"
echo "### configs: $CONFIGS"
[ -x pythia_rivet ] && [ -f RivetCMS_HADRONIC_JETMASS.so ] || { echo "FATAL: run build_rivet.sh first"; exit 1; }
condor_submit -append "HADCONFIGS = $CONFIGS" submit_hardqcd.jdl
echo "### submitted; watch:  condor_watch_q"
