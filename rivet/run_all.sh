#!/usr/bin/env bash
# Generate events for ONE generator and run the CMS_ZJET_JETMASS Rivet routine.
#
#   ./run_all.sh <pythia|vincia|herwig|sherpa> [nevents] [seed]
#
# Assumes rivet / pythia8-config / Herwig / Sherpa are on PATH (native cvmfs/LPC
# install or a sourced LCG view -- see run_lpc.sh). Output: out/<generator>.yoda
set -euo pipefail

GEN=${1:?usage: run_all.sh <pythia|vincia|herwig|sherpa> [nevents] [seed]}
N=${2:-20000}
SEED=${3:-12345}

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
mkdir -p out

# Build the Rivet plugin (only when stale, so parallel runs don't race on it).
if [ ! -f RivetCMS_ZJET_JETMASS.so ] || [ CMS_ZJET_JETMASS.cc -nt RivetCMS_ZJET_JETMASS.so ]; then
  rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
fi

case "$GEN" in
  pythia*|vincia*)
    # tiny Pythia->Rivet driver (no prebuilt main144 in LCG views).
    # Matches the nominal pythia/vincia cards and the modelling-variation cards
    # (pythia_cr1, pythia_cr2, pythia_fragsoft, pythia_fraghard, ...).
    if [ ! -x pythia_rivet ] || [ gen/pythia_rivet.cc -nt pythia_rivet ]; then
      g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
          $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet
    fi
    ./pythia_rivet "gen/$GEN.cmnd" "out/$GEN.yoda" "$N"
    ;;
  herwig)
    # In an LCG view the repository path is baked to a build-time location, so
    # pass the view's HerwigDefaults.rpo explicitly (--repo) and add share/Herwig
    # to the read search path (-I) so 'read snippets/...' resolves.
    HWSHARE="$(cd "$(dirname "$(command -v Herwig)")/../share/Herwig" && pwd)"
    Herwig read --repo="$HWSHARE/HerwigDefaults.rpo" -I "$HWSHARE" gen/herwig.in
    [ -f herwig.run ] || { echo "Herwig read failed" >&2; exit 1; }
    Herwig run herwig.run -N "$N" -s "$SEED"
    mv "herwig-S${SEED}.yoda" out/herwig.yoda
    rm -f "herwig-S${SEED}.out" "herwig-S${SEED}.tex" "herwig-S${SEED}.log"
    ;;
  sherpa)
    # local fallback PDF (downloaded into ../.lhapdf); on cvmfs/LPC PDF4LHC21
    # is already on the default LHAPDF path so this dir simply won't exist.
    [ -d "$HERE/.lhapdf/PDF4LHC21_40_pdfas" ] && \
      export LHAPDF_DATA_PATH="$HERE/.lhapdf:$(lhapdf-config --datadir 2>/dev/null):${LHAPDF_DATA_PATH:-}"
    rm -rf Process Results Results.zip Status__* References.tex  # stale ME cache
    # config is a POSITIONAL arg in Sherpa 3 (-f is deprecated/ignored)
    Sherpa gen/sherpa.yaml -e "$N"
    # EVENT_OUTPUT: HepMC3_GenEvent[out/sherpa] -> file literally named out/sherpa
    rivet -a CMS_ZJET_JETMASS out/sherpa -o out/sherpa.yoda
    rm -f out/sherpa
    ;;
  *)
    echo "unknown generator: $GEN" >&2; exit 1 ;;
esac

echo "wrote out/$GEN.yoda"
