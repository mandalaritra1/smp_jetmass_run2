#!/usr/bin/env bash
# MG-LO + Herwig7: shower the SAME cached MG-LO LHE (zjet_50k_s0..3, from the 50k
# Pythia run) through Herwig's angular-ordered shower + CLUSTER hadronization into the
# CMS_ZJET_JETMASS Rivet routine. Same hard events as the Pythia nominal -> the clean
# string<->cluster (shower+hadronization) / alternate-generator probe that Vincia and the
# Lund variations do NOT provide (they stay inside Pythia's string framework).
# Native-amd64 hepstore/rivet-herwig. Pool of MAXJOBS (default 3 -- Herwig is RAM-heavy
# and akasha has 6.7 GB; 3 keeps headroom). Outputs out/herwig/mglo_herwig_s<idx>.yoda.
#   ./run_akasha_herwig.sh [N=50000] [MAXJOBS=3]
set -uo pipefail
N=${1:-50000}; MAXJOBS=${2:-3}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
IMG=hepstore/rivet-herwig:latest
ODIR=out/herwig
mkdir -p "$ODIR" logs
: > logs/herwig_progress.log

# build the Rivet plugin once inside the herwig image (shared /work volume)
echo "### build RivetCMS_ZJET_JETMASS.so in rivet-herwig ###"
docker run --rm -v "$HERE":/work -w /work "$IMG" bash -lc '
  export RIVET_ANALYSIS_PATH=/work
  rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
' > logs/herwig_build.log 2>&1 || { echo "build failed (see logs/herwig_build.log)"; exit 1; }

run_slice() {  # idx
  local idx=$1
  local LHE="zjet_50k_s$idx/Events/run_01/unweighted_events.lhe"
  [ -f "$LHE" ] || { echo "no LHE s$idx" >> logs/herwig_s$idx.log; return 1; }
  # per-slice .in: substitute the LHE path + a unique saverun name (avoid collisions)
  sed -e "s#__LHE__#$LHE#" -e "s#saverun mglo_herwig #saverun mglo_herwig_s$idx #" \
      gen/madgraph/shower_herwig.in > "$ODIR/_shw_s$idx.in"
  docker run --rm -v "$HERE":/work -w /work "$IMG" bash -lc "
    export RIVET_ANALYSIS_PATH=/work; cd /work
    Herwig read $ODIR/_shw_s$idx.in
    Herwig run mglo_herwig_s$idx.run -N $N -s $((4000+idx))
    mv mglo_herwig_s${idx}-S*.yoda $ODIR/mglo_herwig_s$idx.yoda
  " > "logs/herwig_s$idx.log" 2>&1
  if [ -f "$ODIR/mglo_herwig_s$idx.yoda" ]; then
    echo "[$(date +%H:%M:%S)] herwig s$idx done" >> logs/herwig_progress.log
  else
    echo "[$(date +%H:%M:%S)] herwig s$idx FAILED (see logs/herwig_s$idx.log)" >> logs/herwig_progress.log
  fi
}

echo "### showering 4 slices through Herwig, $MAXJOBS at a time ###"
running=0
for idx in 0 1 2 3; do
  run_slice "$idx" &
  running=$((running+1))
  [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }
done
wait
echo "### Herwig DONE -> $ODIR/mglo_herwig_s*.yoda ###"
ls -la $ODIR/mglo_herwig_s*.yoda 2>/dev/null
