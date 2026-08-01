#!/usr/bin/env bash
# High-statistics worker for the dijet/trijet HAD-leg model variations.
#
#   inner_hardqcd.sh <PTHATLO> <PTHATHI> <SEED> <NEVT> [CONFIG ...]
#     PTHATHI = 0 means no upper bound (the open top slice)
#
# WHY THIS EXISTS (see LAUNCH.md "Two production paths"): the gridpack+MLM
# path yields only ~0.13 measured dijet jets per generated event, because MLM
# matching accepts 7.5%. Generating the hard process directly with
# HardQCD:all -- no gridpack, no runcmsgrid, no matching veto -- yields ~1.5
# dijet jets per generated event and skips the expensive MadGraph step
# entirely, so it buys the ~50-100x statistics the HAD leg needs.
#
# COST: the ME is 2->2, not the nominal 0-4-parton madgraphMLM. That is
# acceptable for a variation/nominal RATIO applied as a reweight (and is what
# the zjet exploratory wave did), but it is NOT a same-ME sample -- the
# gridpack production stays as the same-ME reference. The caveat is larger
# for TRIJET, whose measured third jet is entirely shower-generated here
# rather than ME-accurate.
#
# pThat SLICING: cross sections span ~6 decades over the measured pT range, so
# a single inclusive run starves the high-pT bins. Each slice is generated
# separately and normalized by its own sigma/sumw (written into the ntuple
# footer by the Rivet routine), exactly as the HT-binned path is stitched.
#
# All configs in a job share ONE explicit random seed, so the hard events are
# the same and the variation/nominal ratio is statistically correlated.
set -uo pipefail
PTLO=$1; PTHI=$2; SEED=$3; NEVT=$4; shift 4
CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(cp5 cr1 cr2 fraghard fragsoft)
SUBMITDIR=$(pwd)
AFS=/afs/cern.ch/user/a/amandal/hadronic_model_prod
find_helper() { [ -f "$SUBMITDIR/$1" ] && echo "$SUBMITDIR/$1" || echo "$AFS/$1"; }
PYRIVET=$(find_helper py313/pythia_rivet); RIVETSO=$(find_helper py313/RivetCMS_HADRONIC_JETMASS.so)
CP5=$(find_helper cp5.cmnd); RIVETDIR=$(dirname "$RIVETSO")
EOSBASE=/eos/user/a/amandal/hadronic_model_prod_hardqcd
SLICE="pthat${PTLO}_${PTHI}"
export TMPDIR=${TMPDIR:-/tmp/$USER.$$}; mkdir -p "$TMPDIR"
WORK=$(mktemp -d "$TMPDIR/hq.XXXXXX"); cd "$WORK"
echo "### $(date) job SLICE=$SLICE SEED=$SEED NEVT=$NEVT CONFIGS=${CONFIGS[*]} host=$(hostname)"

#### LCG_107 (Pythia 8.313 + Rivet 4.0.2), NOT LCG_106 (8.312).
#### 8.312 SEGFAULTS in cr1 (ColourReconnection:mode=1 + BeamRemnants:
#### remnantMode=1) roughly once per ~2000 HardQCD events, which killed every
#### cr1 job of cluster 11947394 across four different slices and nodes. The
#### identical job (seed 60000, 4000 events) completes cleanly under 8.313, so
#### the proper QCD-CR definition is kept rather than dropping remnantMode.
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh; set -u
export RIVET_ANALYSIS_PATH="$RIVETDIR:${RIVET_ANALYSIS_PATH:-}"

RC=0
for CFG in "${CONFIGS[@]}"; do
  {
    #### Pythia's default eCM is 14000 -- MUST be set explicitly for Run 2.
    echo "Beams:eCM = 13000."
    echo "HardQCD:all = on"
    echo "PhaseSpace:pTHatMin = ${PTLO}."
    [ "$PTHI" != "0" ] && echo "PhaseSpace:pTHatMax = ${PTHI}."
    #### same seed for every config in this job -> same hard events
    echo "Random:setSeed = on"
    echo "Random:seed = $SEED"
    echo "PartonLevel:MPI = on"
    echo "HadronLevel:all = on"
    cat "$CP5"
    case "$CFG" in
      cp5) ;;
      vincia)   echo "PartonShowers:model = 2" ;;
      cr1)      echo "ColourReconnection:mode = 1"; echo "BeamRemnants:remnantMode = 1" ;;
      cr2)      echo "ColourReconnection:mode = 2" ;;
      fraghard) echo "StringZ:aLund = 0.58"; echo "StringZ:bLund = 0.78"; echo "StringPT:sigma = 0.305" ;;
      fragsoft) echo "StringZ:aLund = 0.78"; echo "StringZ:bLund = 1.18"; echo "StringPT:sigma = 0.365" ;;
      *) echo "FATAL: unknown CONFIG=$CFG"; exit 40 ;;
    esac
  } > run_${CFG}.cmnd

  OUT=out_${SLICE}_${CFG}_${SEED}.yoda
  NTP=ntuple_${SLICE}_${CFG}_${SEED}
  export HAD_NTUPLE="$WORK/$NTP"
  if ! "$PYRIVET" run_${CFG}.cmnd "$OUT" "$NEVT" CMS_HADRONIC_JETMASS > rivet_${CFG}.log 2>&1; then
    echo "ERROR: pythia_rivet failed for $CFG"; tail -20 rivet_${CFG}.log; RC=44; continue
  fi
  if [ ! -s "$OUT" ] || [ ! -s "${NTP}_dijet.txt" ]; then
    echo "ERROR: empty output for $CFG"; tail -20 rivet_${CFG}.log; RC=45; continue
  fi
  echo "### $CFG: sumw=$(grep '^# sumw' ${NTP}_dijet.txt | awk '{print $3}') xsec_pb=$(grep '^# xsec_pb' ${NTP}_dijet.txt | awk '{print $3}') dijet=$(grep -c '^[0-9]' ${NTP}_dijet.txt) trijet=$(grep -c '^[0-9]' ${NTP}_trijet.txt)"

  EOS="$EOSBASE/${CFG}/${SLICE}"
  eos root://eosuser.cern.ch mkdir -p "$EOS" 2>/dev/null || mkdir -p "$EOS" 2>/dev/null || true
  for F in "$OUT" "${NTP}_dijet.txt" "${NTP}_trijet.txt"; do
    if ! cp "$F" "$EOS/" 2>/dev/null; then
      xrdcp -f "$F" "root://eosuser.cern.ch/$EOS/$F" || { echo "ERROR: stageout $F failed"; RC=46; }
    fi
  done
done

echo "### DONE $(date) rc=$RC -> $EOSBASE/<config>/$SLICE"
cd /; rm -rf "$WORK"
exit $RC
