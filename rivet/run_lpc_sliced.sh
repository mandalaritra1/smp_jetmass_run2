#!/usr/bin/env bash
# Sliced (flat-statistics) production on lxplus/LPC via a cvmfs LCG view.
# Runs pythia + vincia in pThat slices (run_slices.sh) in parallel, each slice
# stitched with gen/merge_slices.py -> out/<gen>_sliced.yoda.
#
#   ./run_lpc_sliced.sh [events_per_slice]
set -uo pipefail

VIEW=${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt}
NPS=${1:-8000}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
set +u; source "$VIEW/setup.sh"; set -u
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
mkdir -p logs out

# pre-build so the parallel slice jobs don't race
rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
    $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet

# vincia is ~8x slower per event -> fewer events/slice (still flat coverage)
nice -n 10 ./run_slices.sh pythia "$NPS"        > logs/pythia_sliced.log 2>&1 &
p=$!
nice -n 10 ./run_slices.sh vincia "$((NPS/2))"  > logs/vincia_sliced.log 2>&1 &
v=$!

rc=0
wait $p && echo "OK pythia_sliced"  || { echo "FAIL pythia_sliced"; rc=1; }
wait $v && echo "OK vincia_sliced"  || { echo "FAIL vincia_sliced"; rc=1; }
ls -la out/*_sliced.yoda 2>/dev/null
exit $rc
