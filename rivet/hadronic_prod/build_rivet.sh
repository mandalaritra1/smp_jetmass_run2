#!/usr/bin/env bash
# One-time build (run once on lxplus, in ~/hadronic_model_prod) of the Rivet
# analysis plugin + the pythia_rivet driver against the LCG_106 view
# (Pythia 8.312 + Rivet 4.0.0 + YODA 2.0.0, the co-versioned trio).
# The .so/binary are cached on AFS and reused by every condor job (which
# sources the SAME LCG view so the ABI matches).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh; set -u
echo "pythia8-config: $(command -v pythia8-config)  ($(pythia8-config --version 2>/dev/null))"
echo "rivet-config  : $(command -v rivet-config)    ($(rivet-config --version 2>/dev/null))"
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
rivet-build RivetCMS_HADRONIC_JETMASS.so CMS_HADRONIC_JETMASS.cc
# -std=c++20 LAST so it overrides the -std=c++11 that pythia8-config injects
g++ pythia_rivet.cc -o pythia_rivet \
    $(pythia8-config --cxxflags --libs) $(rivet-config --cppflags --ldflags --libs) -std=c++20
echo "BUILT: $(ls -la RivetCMS_HADRONIC_JETMASS.so pythia_rivet)"
