#!/usr/bin/env bash
# Re-shower the EXISTING 300k MG5 LO Z+jet LHE the 6 modelling ways, robustly.
#
# Why this exists (vs run_mg_local.sh): the high-stats run is ~1.5h PER shower
# under amd64 emulation, so a single serial chain (~10h) that dies loses hours.
# This driver is:
#   * resumable     -- per-variation sentinel out/.done_<name>, written ONLY after
#                      a clean pythia_rivet exit; completed showers are skipped.
#   * isolated      -- each shower is its own docker run; one crash != lose all.
#   * bounded-||    -- JOBS showers at once (default 2; Docker VM has 7.75 GiB).
#   * LHE-reusing   -- never regenerates; asserts the 300k LHE is present.
# Meant to be launched fully detached:  setsid nohup bash reshower.sh ... &
#
#   ./reshower.sh [nevents] [jobs]      (defaults: N=300000, JOBS=2)
set -uo pipefail

N=${1:-300000}; JOBS=${2:-2}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
RIV_IMG=hepstore/rivet-pythia:latest
DRUN=(docker run --rm --platform linux/amd64 -v "$HERE":/work -w /work)
LHE=zjet_mglo/Events/run_01/unweighted_events.lhe
mkdir -p out

# --- preconditions ----------------------------------------------------------
[ -f "$LHE" ] || { echo "FATAL: no LHE at $LHE (generate with run_mg_local.sh --regen)"; exit 1; }
nev=$(grep -c "<event>" "$LHE" || echo 0)
echo "### LHE: $LHE has $nev events; showering $N each, JOBS=$JOBS ###"
[ "$nev" -ge "$N" ] || { echo "FATAL: LHE has $nev < requested $N events"; exit 1; }

# per-variation Pythia deltas (ME fixed by the LHE) -- mirrors run_mg_local.sh
var_settings() {
  case "$1" in
    pythia) : ;;
    pythia_vincia) printf 'PartonShowers:model = 2\n' ;;
    pythia_cr1) printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 1\nBeamRemnants:remnantMode = 1\n' ;;
    pythia_cr2) printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 2\n' ;;
    pythia_fragsoft) printf 'StringZ:aLund = 0.78\nStringZ:bLund = 1.18\nStringPT:sigma = 0.365\n' ;;
    pythia_fraghard) printf 'StringZ:aLund = 0.58\nStringZ:bLund = 0.78\nStringPT:sigma = 0.305\n' ;;
  esac
}

# build plugin + driver once if missing (they should already exist)
if [ ! -f RivetCMS_ZJET_JETMASS.so ] || [ ! -x pythia_rivet ]; then
  echo "### building plugin + pythia_rivet (missing) ###"
  "${DRUN[@]}" "$RIV_IMG" bash -lc '
    export RIVET_ANALYSIS_PATH=/work:${RIVET_ANALYSIS_PATH:-}
    [ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
    [ -x pythia_rivet ] || g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
        $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet'
fi

shower_one() {  # $1 = variation name
  local name="$1" cmnd out
  cmnd="out/_mglo_${name}.cmnd"; out="out/mglo_${name}.yoda"
  { echo "Beams:frameType = 4"
    echo "Beams:LHEF = $LHE"
    echo "PartonLevel:MPI = on"
    echo "HadronLevel:all = on"
    var_settings "$name"; } > "$cmnd"
  echo "$(date '+%H:%M:%S') START  $name -> $out"
  if "${DRUN[@]}" "$RIV_IMG" bash -lc "export RIVET_ANALYSIS_PATH=/work; ./pythia_rivet '$cmnd' '$out' $N" \
       > "out/_log_${name}.txt" 2>&1; then
    touch "out/.done_${name}"
    echo "$(date '+%H:%M:%S') DONE   $name ($(du -h "$out" | cut -f1))"
  else
    echo "$(date '+%H:%M:%S') FAIL   $name (see out/_log_${name}.txt)"
  fi
}

# pending = variations without a completion sentinel
ALL="pythia pythia_vincia pythia_cr1 pythia_cr2 pythia_fragsoft pythia_fraghard"
pending=""
for name in $ALL; do
  if [ -f "out/.done_${name}" ]; then echo "skip (done): $name"; else pending="$pending $name"; fi
done
echo "### pending:${pending:- none} ###"

# bounded-parallel batches (showers are ~equal length, so batching is ~optimal)
running=0
for name in $pending; do
  shower_one "$name" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait; running=0; fi
done
wait

echo "### all showers handled. yodas: ###"; ls -la out/mglo_pythia*.yoda 2>/dev/null
echo "### done sentinels: ###"; ls out/.done_* 2>/dev/null
