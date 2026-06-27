#!/usr/bin/env bash
# Native-LPC sliced modelling-variation production for ONE jet-pT slice, so the
# four slices can run on four different LPC nodes in parallel (they share /uscms
# and write distinct out/slices/*_s<idx>.yoda, so no collision). Stitch afterward
# with stitch_sliced_syst.sh, then plot_model_uncertainty.py.
#
# Generates this slice's MG5 LHE and showers ALL variations through it:
#   pythia(nominal) vincia(shower) cr1/cr2(colour reconnection) fragsoft/fraghard(hadronization)
#
# Source an LCG view first (mg5_aMC, pythia8, rivet on PATH -- see run_lpc.sh), then:
#   ./run_mg_sliced_syst.sh <slice_idx 0..3> [events_per_slice]
set -uo pipefail

IDX=${1:?usage: run_mg_sliced_syst.sh <slice_idx 0..3> [events]}
N=${2:-50000}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
mkdir -p out/slices

# jet-pT slice edges (match gen/madgraph/mg5_zjet_sliced.dat and the standalone slices)
LO=(120 200 300 450); HI=(200 300 450 100000)
lo=${LO[$IDX]}; hi=${HI[$IDX]}

# build plugin + Pythia->Rivet driver (idempotent; rebuild if source changed)
[ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
if [ ! -x pythia_rivet ] || [ gen/pythia_rivet.cc -nt pythia_rivet ]; then
  g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
      $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet
fi

# 1. MG5 LO Z+jet -> LHE for THIS slice --------------------------------------
OUT=zjet_mglo_s$IDX
rm -rf "$OUT"
card=out/slices/_mg5_s$IDX.dat
{
  echo "set nb_core 1"
  echo "generate p p > l+ l- j"
  echo "output $OUT"
  echo "launch $OUT"
  echo "set nevents $N"
  echo "set iseed $((1000 + IDX))"
  echo "set ebeam1 6500"
  echo "set ebeam2 6500"
  echo "set ptj $lo"
  [ "$hi" -lt 100000 ] && echo "set ptjmax $hi"
  echo "set mmll 60"
  echo "set mmllmax 120"
  echo "set systematics_program none"
} > "$card"
echo "### slice $IDX: MG5 LHE ptj in [$lo, $hi] ###"
mg5_aMC "$card"
LHE=$(ls "$OUT/Events/run_01/unweighted_events.lhe"* 2>/dev/null | head -1)
case "${LHE:-}" in *.gz) gunzip -f "$LHE"; LHE=${LHE%.gz};; esac
[ -z "${LHE:-}" ] && { echo "no LHE for slice $IDX" >&2; exit 1; }

# 2. shower ALL variations through this slice's LHE --------------------------
var_settings() { case "$1" in
  pythia) : ;;
  pythia_vincia)   printf 'PartonShowers:model = 2\n' ;;
  pythia_cr1)      printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 1\nBeamRemnants:remnantMode = 1\n' ;;
  pythia_cr2)      printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 2\n' ;;
  pythia_fragsoft) printf 'StringZ:aLund = 0.78\nStringZ:bLund = 1.18\nStringPT:sigma = 0.365\n' ;;
  pythia_fraghard) printf 'StringZ:aLund = 0.58\nStringZ:bLund = 0.78\nStringPT:sigma = 0.305\n' ;;
esac; }

for name in pythia pythia_vincia pythia_cr1 pythia_cr2 pythia_fragsoft pythia_fraghard; do
  cmnd=out/slices/_${name}_s$IDX.cmnd
  { printf 'Beams:frameType = 4\nBeams:LHEF = %s\nPartonLevel:MPI = on\nHadronLevel:all = on\n' "$LHE"
    var_settings "$name"; } > "$cmnd"
  echo "### slice $IDX shower: $name ###"
  ./pythia_rivet "$cmnd" "out/slices/mglo_${name}_s$IDX.yoda" "$N"
done
echo "### slice $IDX DONE -> out/slices/mglo_*_s$IDX.yoda ###"
ls -la out/slices/mglo_*_s$IDX.yoda
