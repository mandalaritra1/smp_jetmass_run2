#!/usr/bin/env bash
# Sliced MadGraph-LO production: generate the LHE in jet-pT slices, shower each
# slice through BOTH Pythia8 and Herwig7, and stitch ->
#   out/mglo_pythia_sliced.yoda   out/mglo_herwig_sliced.yoda
#
#   ./gen/madgraph/run_mg_sliced.sh [events_per_slice]
# Run inside a sourced LCG view (mg5_aMC, pythia8, Herwig, rivet on PATH).
set -uo pipefail

N=${1:-8000}
HERE=$(cd "$(dirname "$0")/../.." && pwd); cd "$HERE"   # rivet/
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
mkdir -p out/slices

[ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
[ -x pythia_rivet ] || g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
    $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet
HWSHARE="$(cd "$(dirname "$(command -v Herwig)")/../share/Herwig" && pwd)"

# 1. sliced LHEs --------------------------------------------------------------
echo "### MadGraph sliced LHE generation ###"
rm -rf zjet_mglo_sliced
mg5_aMC gen/madgraph/mg5_zjet_sliced.dat

pslices=(); hslices=()
for i in 0 1 2 3; do
  run=$(printf "run_%02d" $((i + 1)))
  LHE=$(ls "zjet_mglo_sliced/Events/$run/unweighted_events.lhe"* 2>/dev/null | head -1)
  [ -z "$LHE" ] && { echo "missing LHE for $run" >&2; continue; }
  case "$LHE" in *.gz) gunzip -f "$LHE"; LHE=${LHE%.gz};; esac
  echo "### slice $i ($run): $LHE ###"

  # Pythia shower
  cmnd="out/slices/_mglo_p_s$i.cmnd"
  printf 'Beams:frameType = 4\nBeams:LHEF = %s\nPartonLevel:MPI = on\nHadronLevel:all = on\n' "$LHE" > "$cmnd"
  ./pythia_rivet "$cmnd" "out/slices/mglo_pythia_s$i.yoda" "$N"
  pslices+=("out/slices/mglo_pythia_s$i.yoda")

  # Herwig shower (same slice LHE)
  shin="out/slices/_mglo_h_s$i.in"
  sed "s#__LHE__#$LHE#" gen/madgraph/shower_herwig.in > "$shin"
  if Herwig read --repo="$HWSHARE/HerwigDefaults.rpo" -I "$HWSHARE" "$shin" \
     && Herwig run mglo_herwig.run -N "$N" -s $((2000 + i)); then
    mv mglo_herwig-S*.yoda "out/slices/mglo_herwig_s$i.yoda"
    hslices+=("out/slices/mglo_herwig_s$i.yoda")
  else
    echo "WARN: herwig shower failed for slice $i" >&2
  fi
  rm -f mglo_herwig-S*.{out,tex,log}
done

# 2. stitch -------------------------------------------------------------------
[ ${#pslices[@]} -gt 0 ] && python3 gen/merge_slices.py out/mglo_pythia_sliced.yoda "${pslices[@]}"
[ ${#hslices[@]} -gt 0 ] && python3 gen/merge_slices.py out/mglo_herwig_sliced.yoda "${hslices[@]}"
echo "### sliced MadGraph done ###"
ls -la out/mglo_*_sliced.yoda 2>/dev/null
