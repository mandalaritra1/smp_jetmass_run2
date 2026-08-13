#!/usr/bin/env bash
# Build the HardQCD and MESS binaries against LCG_107
# (Pythia 8.313 + Rivet 4.0.2).
# Kept SEPARATE from the LCG_106 build used by the gridpack path
# (inner_hadronic.sh) -- the two are ABI-incompatible, so they must not share
# a binary. 8.313 is required: 8.312 segfaults in cr1 (mode=1 + remnantMode=1)
# about once per ~2000 HardQCD events.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
ANALYSIS_SRC=CMS_HADRONIC_JETMASS.cc
[ -f "$ANALYSIS_SRC" ] || ANALYSIS_SRC=../CMS_HADRONIC_JETMASS.cc
[ -f "$ANALYSIS_SRC" ] || { echo "FATAL: cannot find CMS_HADRONIC_JETMASS.cc"; exit 1; }
mkdir -p py313
cp "$ANALYSIS_SRC" py313/CMS_HADRONIC_JETMASS.cc
cp pythia_rivet.cc pythia_rivet_mess.cc py313/
cd py313
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh; set -u
echo "pythia8-config: $(pythia8-config --version)   rivet: $(rivet-config --version)"
export RIVET_ANALYSIS_PATH="$PWD:${RIVET_ANALYSIS_PATH:-}"
rivet-build RivetCMS_HADRONIC_JETMASS.so CMS_HADRONIC_JETMASS.cc
g++ pythia_rivet.cc -o pythia_rivet \
    $(pythia8-config --cxxflags --libs) $(rivet-config --cppflags --ldflags --libs) -std=c++20
g++ pythia_rivet_mess.cc -o pythia_rivet_mess \
    $(pythia8-config --cxxflags --libs) $(rivet-config --cppflags --ldflags --libs) -std=c++20
echo "BUILT: $(ls -la RivetCMS_HADRONIC_JETMASS.so pythia_rivet pythia_rivet_mess)"
