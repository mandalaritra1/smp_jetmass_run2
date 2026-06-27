#!/usr/bin/env bash
# Local (two Docker images) sliced modelling-variation production for ONE jet-pT
# slice. Mirrors run_mg_sliced_syst.sh but generates the LHE in scailfin and showers
# in hepstore (the Mac has no native MG5). Run a couple of slices here while the
# rest run native on LPC; stitch all of them together with stitch_sliced_syst.sh
# (per-slice filenames match, and the yodas just need to land in the same out/slices/).
#
#   ./run_mg_sliced_local.sh <slice_idx 0..3> [events]   (default 20000)
set -euo pipefail
IDX=${1:?usage: run_mg_sliced_local.sh <slice_idx 0..3> [events]}
N=${2:-20000}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
MG5_IMG=scailfin/madgraph5-amc-nlo:mg5_amc3.5.1
RIV_IMG=hepstore/rivet-pythia:latest
DRUN=(docker run --rm --platform linux/amd64 -v "$HERE":/work -w /work)
MGRUN=(docker run --rm --platform linux/amd64 -v "$HERE":/work -w /work --entrypoint bash "$MG5_IMG")
mkdir -p out/slices

LO=(120 200 300 450); HI=(200 300 450 100000)
lo=${LO[$IDX]}; hi=${HI[$IDX]}
OUT=zjet_mglo_s$IDX

# 1. MG5 LO Z+jet -> LHE for this slice (scailfin) ---------------------------
rm -rf "$OUT"
card=out/slices/_mg5_local_s$IDX.dat
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
echo "### slice $IDX: MG5 LHE ptj [$lo,$hi] (scailfin) ###"
"${MGRUN[@]}" -c "mg5_aMC $card"
LHE="$OUT/Events/run_01/unweighted_events.lhe"
[ -f "$LHE.gz" ] && gunzip -f "$LHE.gz"
[ -f "$LHE" ] || { echo "no LHE for slice $IDX" >&2; exit 1; }

# 2. build the Rivet driver once + shower all variations (hepstore) ----------
"${DRUN[@]}" "$RIV_IMG" bash -lc '
  export RIVET_ANALYSIS_PATH=/work
  [ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
  if [ ! -x pythia_rivet ] || [ gen/pythia_rivet.cc -nt pythia_rivet ]; then
    g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
        $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet
  fi
'
var_settings() { case "$1" in
  pythia) : ;;
  pythia_vincia)   printf 'PartonShowers:model = 2\n' ;;
  pythia_cr1)      printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 1\nBeamRemnants:remnantMode = 1\n' ;;
  pythia_cr2)      printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 2\n' ;;
  pythia_fragsoft) printf 'StringZ:aLund = 0.78\nStringZ:bLund = 1.18\nStringPT:sigma = 0.365\n' ;;
  pythia_fraghard) printf 'StringZ:aLund = 0.58\nStringZ:bLund = 0.78\nStringPT:sigma = 0.305\n' ;;
esac; }
for name in pythia pythia_vincia pythia_cr1 pythia_cr2 pythia_fragsoft pythia_fraghard; do
  cmnd=out/slices/_${name}_local_s$IDX.cmnd
  { printf 'Beams:frameType = 4\nBeams:LHEF = %s\nPartonLevel:MPI = on\nHadronLevel:all = on\n' "$LHE"
    var_settings "$name"; } > "$cmnd"
  echo "### slice $IDX shower: $name ###"
  "${DRUN[@]}" "$RIV_IMG" bash -lc "export RIVET_ANALYSIS_PATH=/work; ./pythia_rivet '$cmnd' 'out/slices/mglo_${name}_s$IDX.yoda' $N"
done
echo "### slice $IDX DONE -> out/slices/mglo_*_s$IDX.yoda ###"
ls -la out/slices/mglo_*_s$IDX.yoda