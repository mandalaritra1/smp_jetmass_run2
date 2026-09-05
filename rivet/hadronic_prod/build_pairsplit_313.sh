#!/usr/bin/env bash
# Build the dedicated pair_split Rivet plugin and MESS driver against LCG_107.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

ANALYSIS_SOURCE=CMS_HADRONIC_PAIR_SPLIT.cc
[ -f "$ANALYSIS_SOURCE" ] || ANALYSIS_SOURCE=../CMS_HADRONIC_PAIR_SPLIT.cc
[ -f "$ANALYSIS_SOURCE" ] || {
  echo "FATAL: cannot find CMS_HADRONIC_PAIR_SPLIT.cc"; exit 1;
}
mkdir -p py313
cp "$ANALYSIS_SOURCE" py313/CMS_HADRONIC_PAIR_SPLIT.cc
cp pythia_rivet_mess.cc py313/pythia_rivet_mess.cc
cd py313

set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
set -u
[ "$(pythia8-config --version)" = "8.313" ]
[ "$(rivet-config --version)" = "4.0.2" ]
export RIVET_ANALYSIS_PATH="$PWD:${RIVET_ANALYSIS_PATH:-}"
rivet-build RivetCMS_HADRONIC_PAIR_SPLIT.so CMS_HADRONIC_PAIR_SPLIT.cc
g++ pythia_rivet_mess.cc -o pythia_rivet_mess \
  $(pythia8-config --cxxflags --libs) \
  $(rivet-config --cppflags --ldflags --libs) -std=c++20
set +e
DISCOVERY_OUTPUT=$(rivet --show-analysis CMS_HADRONIC_PAIR_SPLIT 2>&1)
set -e
DISCOVERED_ANALYSIS=$(printf '%s\n' "$DISCOVERY_OUTPUT" | sed -n '1p')
[ "$DISCOVERED_ANALYSIS" = "CMS_HADRONIC_PAIR_SPLIT" ] || {
  echo "FATAL: local plugin discovery returned: $DISCOVERED_ANALYSIS"; exit 1;
}
sha256sum CMS_HADRONIC_PAIR_SPLIT.cc pythia_rivet_mess.cc \
  RivetCMS_HADRONIC_PAIR_SPLIT.so pythia_rivet_mess
