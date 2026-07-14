#!/usr/bin/env bash
# CP5 shower of cached LHE through Pythia8 (rivet-pythia Docker), no MG5 gen.
# Builds each .cmnd = Beams/LHEF block + gen/cp5.cmnd (CP5 tune) + variation knob.
# Works native (akasha) or amd64-emulated (Mac) -- always passes --platform linux/amd64.
#
#   ./run_cp5_docker.sh <lhe_template> <out_dir> <maxjobs> <idx:var> [<idx:var> ...]
# lhe_template uses %IDX% for the slice, e.g.
#   "zjet_50k_s%IDX%/Events/run_01/unweighted_events.lhe"   (akasha, cached 50k LHE)
#   "lhe_cp5/s%IDX%.lhe"                                     (Mac, scp'd LHE)
# Pass jobs with vincia LAST. Outputs <out_dir>/mglo_cp5_<var>_s<idx>.yoda.
set -uo pipefail
LHE_TMPL=${1:?lhe template with %IDX%}; ODIR=${2:?out dir}; MAXJOBS=${3:?maxjobs}
shift 3; JOBS=("$@")
N=${N:-50000}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
IMG=hepstore/rivet-pythia:latest
DRUN=(docker run --rm --platform linux/amd64 -v "$HERE":/work -w /work)
mkdir -p "$ODIR" logs
PROG=logs/cp5_progress.log

# build plugin + driver once (idempotent)
"${DRUN[@]}" "$IMG" bash -lc '
  export RIVET_ANALYSIS_PATH=/work
  [ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
  [ -x pythia_rivet ] || g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
      $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet
' > logs/cp5_build.log 2>&1

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
  "${DRUN[@]}" "$IMG" bash -lc "export RIVET_ANALYSIS_PATH=/work; ./pythia_rivet '$cmnd' '$ODIR/mglo_cp5_${name}_s$idx.yoda' $N" \
      > "logs/cp5_${name}_s$idx.log" 2>&1
  if [ -f "$ODIR/mglo_cp5_${name}_s$idx.yoda" ]; then
    echo "[$(date +%H:%M:%S)] cp5 $name s$idx done" >> "$PROG"
  else
    echo "[$(date +%H:%M:%S)] cp5 $name s$idx FAILED" >> "$PROG"
  fi
}

echo "### CP5 docker: ${#JOBS[@]} jobs, $MAXJOBS at a time -> $ODIR ###"
# pool throttle compatible with macOS bash 3.2 (no `wait -n`): cap running bg jobs
for job in "${JOBS[@]}"; do
  idx=${job%%:*}; name=${job#*:}
  while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do sleep 2; done
  ( shower_one "$idx" "$name" ) &
done
wait
echo "### CP5 docker batch done -> $ODIR ###"; ls -la "$ODIR"/mglo_cp5_*.yoda 2>/dev/null | tail
