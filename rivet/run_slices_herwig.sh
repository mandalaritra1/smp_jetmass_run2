#!/usr/bin/env bash
# Standalone Herwig7 in jet-kT SLICES (MEZJet with JetKtCut MinKT/MaxKT windows),
# stitched into a flat-statistics prediction -> out/herwig_sliced.yoda.
#
#   ./run_slices_herwig.sh [events_per_slice]
# Run inside a sourced LCG view (Herwig + rivet on PATH).
set -uo pipefail

NPS=${1:-8000}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
mkdir -p out/slices

[ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
HWSHARE="$(cd "$(dirname "$(command -v Herwig)")/../share/Herwig" && pwd)"

# jet-kT slice edges [GeV]; last slice ~open-ended
EDGES=(120 200 300 450 100000)

# base input minus its saverun line (we add a per-slice saverun + cut overrides)
sed '/saverun/d' gen/herwig.in > out/slices/_hbase.in

slices=()
for ((i=0; i<${#EDGES[@]}-1; i++)); do
  lo=${EDGES[i]}; hi=${EDGES[i+1]}
  sin="out/slices/_herwig_s${i}.in"
  cp out/slices/_hbase.in "$sin"
  # override the base MinKT (150) and add a MaxKT -> disjoint window
  echo "set /Herwig/Cuts/JetKtCut:MinKT ${lo}.*GeV" >> "$sin"
  echo "set /Herwig/Cuts/JetKtCut:MaxKT ${hi}.*GeV" >> "$sin"
  echo "saverun herwig_s${i} /Herwig/Generators/EventGenerator" >> "$sin"
  echo "### herwig slice $i: kT [$lo, $hi]  ($NPS events) ###"
  Herwig read --repo="$HWSHARE/HerwigDefaults.rpo" -I "$HWSHARE" "$sin"
  if Herwig run "herwig_s${i}.run" -N "$NPS" -s $((3000 + i)); then
    mv "herwig_s${i}-S"*.yoda "out/slices/herwig_s${i}.yoda"
    slices+=("out/slices/herwig_s${i}.yoda")
  fi
  rm -f "herwig_s${i}-S"*.{out,tex,log}
done

python3 gen/merge_slices.py out/herwig_sliced.yoda "${slices[@]}"
echo "wrote out/herwig_sliced.yoda"
