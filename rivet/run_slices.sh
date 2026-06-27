#!/usr/bin/env bash
# Generate a Pythia-family generator in pThat SLICES and stitch the per-slice
# YODAs (each normalised to its own cross section) into one prediction with flat
# statistics across the steeply-falling jet-pT spectrum.
#
#   ./run_slices.sh <pythia|vincia> [events_per_slice]
#
# Each slice's YODA is already absolute (pb) via the routine's finalize(); the
# slices are disjoint in pThat, so the correct stitch is a cross-section sum.
set -euo pipefail

GEN=${1:?usage: run_slices.sh <pythia|vincia> [events_per_slice]}
NPS=${2:-20000}

HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
mkdir -p out/slices

if [ ! -f RivetCMS_ZJET_JETMASS.so ] || [ CMS_ZJET_JETMASS.cc -nt RivetCMS_ZJET_JETMASS.so ]; then
  rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
fi
if [ ! -x pythia_rivet ] || [ gen/pythia_rivet.cc -nt pythia_rivet ]; then
  g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
      $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet
fi

# pThat slice edges [GeV]; last slice is open-ended (very large upper bound).
EDGES=(120 200 300 450 100000)

slices=()
for ((i=0; i<${#EDGES[@]}-1; i++)); do
  lo=${EDGES[i]}; hi=${EDGES[i+1]}
  cmnd="out/slices/_${GEN}_s${i}.cmnd"
  cp "gen/${GEN}.cmnd" "$cmnd"
  # later settings override earlier -> these win over the base pTHatMin
  echo "PhaseSpace:pTHatMin = ${lo}." >> "$cmnd"
  echo "PhaseSpace:pTHatMax = ${hi}." >> "$cmnd"
  echo "Random:seed = $((12345 + i))" >> "$cmnd"   # independent slices
  y="out/slices/${GEN}_s${i}.yoda"
  echo "### slice $i: pThat [$lo, $hi]  ($NPS events) ###"
  ./pythia_rivet "$cmnd" "$y" "$NPS"
  slices+=("$y")
done

# Stitch disjoint slices: bin-by-bin sum of the per-slice absolute (pb) dsigma/dx
# (rivet-merge drops the RAW histos for cross-section-weighted merges, so we sum
# the finalized estimates directly -- validated to reproduce an inclusive run).
python3 gen/merge_slices.py "out/${GEN}_sliced.yoda" "${slices[@]}"
