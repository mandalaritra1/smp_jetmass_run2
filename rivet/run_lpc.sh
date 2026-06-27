#!/usr/bin/env bash
# Run the generator suite natively on lxplus/LPC via a cvmfs LCG view.
# One environment (Rivet 4 + Pythia8 + Herwig + Sherpa) for all generators,
# run in parallel (one core each) with per-generator logs in logs/.
#
#   ./run_lpc.sh [nevents] [gen ...]      # default: pythia vincia herwig sherpa
set -euo pipefail

VIEW=${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt}
N=${1:-30000}
shift || true
GENS=("$@"); [ ${#GENS[@]} -eq 0 ] && GENS=(pythia vincia herwig sherpa)

HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
# LCG setup scripts are not nounset-clean -> relax -u around the source.
set +u
# shellcheck disable=SC1091
source "$VIEW/setup.sh"
set -u
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
mkdir -p logs out
echo "rivet $(rivet --version) | view $VIEW | N=$N | gens: ${GENS[*]}"

# Pre-build plugin + Pythia driver ONCE so parallel jobs don't race.
rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
    $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet

# Launch generators in parallel (nice, one core each).
pids=()
for g in "${GENS[@]}"; do
  nice -n 10 ./run_all.sh "$g" "$N" > "logs/$g.log" 2>&1 &
  pids+=($!)
  echo "launched $g (pid $!)"
done

rc=0
for i in "${!GENS[@]}"; do
  if wait "${pids[$i]}"; then echo "OK   ${GENS[$i]}"; else echo "FAIL ${GENS[$i]}"; rc=1; fi
done

echo "yodas:"; ls -la out/*.yoda 2>/dev/null
exit $rc
