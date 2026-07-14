#!/usr/bin/env bash
# CP5 shower on LPC (native, inside a sourced LCG view). No MG5 gen.
# Builds .cmnd = Beams/LHEF + gen/cp5.cmnd + variation knob; runs native pythia_rivet.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/setup.sh
#   ./run_cp5_lpc.sh <lhe_template> <out_dir> <maxjobs> <idx:var> [<idx:var> ...]
set -uo pipefail
LHE_TMPL=${1:?lhe template with %IDX%}; ODIR=${2:?out dir}; MAXJOBS=${3:?maxjobs}
shift 3; JOBS=("$@")
N=${N:-50000}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
mkdir -p "$ODIR" logs
PROG=logs/cp5_progress.log

[ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
if [ ! -x pythia_rivet ] || [ gen/pythia_rivet.cc -nt pythia_rivet ]; then
  g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
      $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet
fi

var_settings() { case "$1" in
  pythia) : ;;
  pythia_vincia)   printf 'PartonShowers:model = 2\n' ;;
  pythia_cr1)      printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 1\nBeamRemnants:remnantMode = 1\n' ;;
  pythia_cr2)      printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 2\n' ;;
  pythia_fragsoft) printf 'StringZ:aLund = 0.78\nStringZ:bLund = 1.18\nStringPT:sigma = 0.365\n' ;;
  pythia_fraghard) printf 'StringZ:aLund = 0.58\nStringZ:bLund = 0.78\nStringPT:sigma = 0.305\n' ;;
esac; }

shower_one() {  # idx name
  local idx=$1 name=$2
  local LHE=${LHE_TMPL//%IDX%/$idx}
  [ -f "$LHE" ] || { echo "[$(date +%H:%M:%S)] cp5 $name s$idx NO-LHE ($LHE)" >> "$PROG"; return 1; }
  local cmnd=$ODIR/_${name}_s$idx.cmnd
  { printf 'Beams:frameType = 4\nBeams:LHEF = %s\nPartonLevel:MPI = on\nHadronLevel:all = on\n' "$LHE"
    cat gen/cp5.cmnd; var_settings "$name"; } > "$cmnd"
  ./pythia_rivet "$cmnd" "$ODIR/mglo_cp5_${name}_s$idx.yoda" "$N" > "logs/cp5_${name}_s$idx.log" 2>&1
  if [ -f "$ODIR/mglo_cp5_${name}_s$idx.yoda" ]; then
    echo "[$(date +%H:%M:%S)] cp5 $name s$idx done" >> "$PROG"
  else
    echo "[$(date +%H:%M:%S)] cp5 $name s$idx FAILED" >> "$PROG"
  fi
}

echo "### CP5 LPC: ${#JOBS[@]} jobs, $MAXJOBS at a time -> $ODIR ###"
running=0
for job in "${JOBS[@]}"; do
  idx=${job%%:*}; name=${job#*:}
  ( shower_one "$idx" "$name" ) &
  running=$((running+1))
  [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }
done
wait
echo "### CP5 LPC batch done -> $ODIR ###"; ls -la "$ODIR"/mglo_cp5_*.yoda 2>/dev/null | tail
