#!/usr/bin/env bash
# Fully-local "mglo+pythia" baseline + CR/hadronization modelling variations,
# using TWO amd64 Docker images (no LPC needed):
#   1. MadGraph5 LO Z(->ll)+jet -> LHE        scailfin/madgraph5-amc-nlo (tested Python)
#   2. Pythia8 shower x{nominal,CR,hadr} -> Rivet yodas   hepstore/rivet-pythia
# The hepstore image's Python (3.14) is too new for MG5, hence the split.
#
# The SINGLE LHE (fixed LO matrix element) is showered N ways that differ ONLY in
# colour reconnection / Lund fragmentation, so each jet-mass shape difference
# cleanly isolates that effect -- exactly the ARC's "shower vs hadronisation"
# question, on the analysis-relevant MG baseline (not Pythia's internal ME).
#
#   ./run_mg_local.sh [nevents] [--regen]   (default 30000; --regen forces a new LHE)
# NOTE: written for the macOS stock /bin/bash 3.2 (no associative arrays).
set -euo pipefail

N=30000; REGEN=0; MLM=0
for a in "$@"; do case "$a" in --regen) REGEN=1 ;; --mlm) MLM=1 ;; *[0-9]*) N=$a ;; esac; done
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
MG5_IMG=scailfin/madgraph5-amc-nlo:mg5_amc3.5.1
RIV_IMG=hepstore/rivet-pythia:latest
DRUN=(docker run --rm --platform linux/amd64 -v "$HERE":/work -w /work)
# scailfin's ENTRYPOINT is mg5_aMC, so override it to bash to run our own command.
MGRUN=(docker run --rm --platform linux/amd64 -v "$HERE":/work -w /work --entrypoint bash "$MG5_IMG")
mkdir -p out

# single-mult (default) vs MLM-merged (--mlm): pick the MG5 card + the matching block
# prepended to every shower cmnd (pythia_rivet activates the JetMatchingMadgraph hook
# when JetMatching:merge=on; setMad reads xqcut/nJetMax from the merged LHE header).
if [ "$MLM" -eq 1 ]; then
  MG5CARD=gen/madgraph/mg5_zjet_mlm.dat; OUTDIR_MG=zjet_mglo_mlm
  MATCH=$'JetMatching:merge = on\nJetMatching:scheme = 1\nJetMatching:setMad = on\nJetMatching:nJetMax = 2'
else
  MG5CARD=gen/madgraph/mg5_zjet.dat; OUTDIR_MG=zjet_mglo; MATCH=""
fi
LHE=$OUTDIR_MG/Events/run_01/unweighted_events.lhe

# extra Pythia settings per variation (deltas only; the ME is fixed by the LHE).
# Mirrors the standalone gen/pythia_{cr1,cr2,fragsoft,fraghard}.cmnd cards.
var_settings() {
  case "$1" in
    pythia) : ;;   # nominal -- no extra settings
    pythia_vincia) printf 'PartonShowers:model = 2\n' ;;   # shower source (Vincia)
    pythia_cr1) printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 1\nBeamRemnants:remnantMode = 1\n' ;;
    pythia_cr2) printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 2\n' ;;
    pythia_fragsoft) printf 'StringZ:aLund = 0.78\nStringZ:bLund = 1.18\nStringPT:sigma = 0.365\n' ;;
    pythia_fraghard) printf 'StringZ:aLund = 0.58\nStringZ:bLund = 0.78\nStringPT:sigma = 0.305\n' ;;
  esac
}

# 1. MadGraph LO Z+jet -> LHE (reuse if present unless --regen) --------------
if [ "$REGEN" -eq 0 ] && [ -f "$LHE" ]; then
  echo "### [1/2] reusing existing LHE: $LHE  (--regen to rebuild) ###"
else
  echo "### [1/2] MadGraph LHE (scailfin) ###"
  rm -rf "$OUTDIR_MG"                   # MG5 'output' refuses to overwrite an existing dir
  # MG5's recursive make (GNU make 4.4+) hits a jobserver 'Bad file descriptor'
  # bug when compiling in parallel under amd64 emulation -> force serial builds.
  mg5card=out/_mg5_local.dat
  # serial build (nb_core=1) + size the LHE to N events (the LHE caps shower stats)
  { echo "set nb_core 1"; sed "s/^set nevents.*/set nevents     $N/" "$MG5CARD"; } > "$mg5card"
  "${MGRUN[@]}" -c "mg5_aMC $mg5card"
  [ -f "$LHE.gz" ] && gunzip -f "$LHE.gz"
  [ -f "$LHE" ] || { echo "no LHE produced at $LHE" >&2; exit 1; }
fi
echo "LHE: $LHE"

# 2. Shower the SAME LHE N ways (hepstore) ----------------------------------
# build the Rivet plugin + pythia_rivet driver once (so per-variation runs don't race)
"${DRUN[@]}" "$RIV_IMG" bash -lc '
  export RIVET_ANALYSIS_PATH=/work:${RIVET_ANALYSIS_PATH:-}
  [ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
  if [ ! -x pythia_rivet ] || [ gen/pythia_rivet.cc -nt pythia_rivet ]; then
    g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
        $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet
  fi
'
for name in pythia pythia_vincia pythia_cr1 pythia_cr2 pythia_fragsoft pythia_fraghard; do
  cmnd="out/_mglo_${name}.cmnd"
  { echo "Beams:frameType = 4"
    echo "Beams:LHEF = $LHE"
    echo "PartonLevel:MPI = on"
    echo "HadronLevel:all = on"
    [ -n "$MATCH" ] && printf '%b\n' "$MATCH"
    var_settings "$name"; } > "$cmnd"
  out="out/mglo_${name}.yoda"
  echo "### shower: $name -> $out ###"
  "${DRUN[@]}" "$RIV_IMG" bash -lc "export RIVET_ANALYSIS_PATH=/work; ./pythia_rivet '$cmnd' '$out' $N"
done

echo "### done. yodas: ###"; ls -la out/mglo_*.yoda
# overlay (nominal vs CR vs hadronization) into out/plots_syst/index.html
"${DRUN[@]}" "$RIV_IMG" bash -lc "./plot_syst.sh"
