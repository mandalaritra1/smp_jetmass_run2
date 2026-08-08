#!/usr/bin/env bash
# Self-contained condor worker for the dijet/trijet model-uncertainty
# standalone production (QCD_HT madgraphMLM gridpacks, shower/hadronization
# configs on the CP5 base).  Per job:
#   1. extract the official UL QCD_HT gridpack to $TMPDIR and run runcmsgrid.sh
#      inside the cmssw-el7 (slc7) container  ->  cmsgrid_final.lhe
#   2. source the LCG_106 view (el9, Pythia 8.312 + Rivet 4.0.0) and shower
#      THAT SAME LHE once per config with MLM jet matching
#      (CombineMatchingInput), running the CMS_HADRONIC_JETMASS Rivet
#      analysis -> yoda + 2 ntuples per config
#   3. copy the outputs to EOS.
#
#   inner_hadronic.sh <BIN> <SEED> <NEVT> [CONFIG ...]
#     BIN     e.g. QCD_HT700to1000
#     CONFIG  cp5 | vincia | cr1 | cr2 | fraghard | fragsoft
#             (default: all six)
#
# ONE LHE, MANY SHOWERS -- deliberate.  The variation/nominal ratio is the
# measured quantity, so every config must see the SAME hard events: then the
# LHE statistical noise cancels in the ratio instead of adding to it (this is
# how the zjet exploratory wave was run).  It also runs the expensive
# runcmsgrid step once instead of once per config.
#
# MLM matching parameters COPIED from the official UL fragment
# (JME-RunIISummer20UL18wmLHEGEN-00015): qCut=14, setMad=off, scheme=1,
# jetAlgorithm=2, etaJetMax=5, coneRadius=1, slowJetPower=1, nQmatch=5,
# nJetMax=4.  These differ from the DY production (qCut=19, setMad=on) --
# do NOT copy blindly between the two.
set -uo pipefail
BIN=$1; SEED=$2; NEVT=$3; shift 3
CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(cp5 vincia cr1 cr2 fraghard fragsoft)
SUBMITDIR=$(pwd)                          # condor drops transfer_input_files here
AFS=/afs/cern.ch/user/a/amandal/hadronic_model_prod
find_helper() { [ -f "$SUBMITDIR/$1" ] && echo "$SUBMITDIR/$1" || echo "$AFS/$1"; }
PYRIVET=$(find_helper pythia_rivet); RIVETSO=$(find_helper RivetCMS_HADRONIC_JETMASS.so)
CP5=$(find_helper cp5.cmnd); RIVETDIR=$(dirname "$RIVETSO")
EOSBASE=/eos/user/a/amandal/hadronic_model_prod
GP=/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.1/QCD_HT_Binned/${BIN}/v1/${BIN}_MG261_NNPDF31_slc7_amd64_gcc700_CMSSW_10_6_0_tarball.tar.xz
export TMPDIR=${TMPDIR:-/tmp/$USER.$$}; mkdir -p "$TMPDIR"
WORK=$(mktemp -d "$TMPDIR/hm.XXXXXX"); cd "$WORK"
echo "### $(date) job BIN=$BIN SEED=$SEED NEVT=$NEVT CONFIGS=${CONFIGS[*]} host=$(hostname) work=$WORK"

# --- 1. gridpack -> LHE inside cmssw-el7 (ONCE) -----------------------------
mkdir gp && tar xaf "$GP" -C gp || { echo "FATAL: gridpack untar failed"; exit 41; }
cat > _genlhe.sh <<EOF
set -e
cd "$WORK/gp"
export SCRAM_ARCH=slc7_amd64_gcc700
# runcmsgrid.sh <nevents> <seed> [ncpu]
./runcmsgrid.sh $NEVT $SEED 1
EOF
/cvmfs/cms.cern.ch/common/cmssw-el7 -- bash "$WORK/_genlhe.sh" > genlhe.log 2>&1 || { echo "FATAL: runcmsgrid failed"; tail -40 genlhe.log; exit 42; }
LHE="$WORK/gp/cmsgrid_final.lhe"
[ -f "$LHE" ] || { echo "FATAL: no LHE produced"; tail -40 genlhe.log; exit 43; }
NLHE=$(grep -c "<event>" "$LHE")
echo "### LHE has $NLHE events (requested $NEVT)"
[ "$NLHE" -gt 0 ] || { echo "FATAL: empty LHE"; exit 43; }

# --- 2. shower the SAME LHE once per config in LCG_106 (el9) -----------------
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh; set -u
export RIVET_ANALYSIS_PATH="$RIVETDIR:${RIVET_ANALYSIS_PATH:-}"

RC=0
for CFG in "${CONFIGS[@]}"; do
  {
    echo "Beams:frameType = 4"
    echo "Beams:LHEF = $LHE"
    echo "JetMatching:merge = on"
    echo "JetMatching:scheme = 1"
    echo "JetMatching:setMad = off"
    echo "JetMatching:jetAlgorithm = 2"
    echo "JetMatching:etaJetMax = 5."
    echo "JetMatching:coneRadius = 1."
    echo "JetMatching:slowJetPower = 1"
    echo "JetMatching:qCut = 14."
    echo "JetMatching:nQmatch = 5"
    echo "JetMatching:nJetMax = 4"
    echo "PartonLevel:MPI = on"
    echo "HadronLevel:all = on"
    #### CP5 base for EVERY config; the variation knob comes AFTER so it
    #### overrides CP5 where they touch the same parameter.  Knob values are
    #### the same brackets as the zjet modelling variations (exploratory
    #### brackets / mode flips, NOT retuned CP5-CR or eigentunes).
    cat "$CP5"
    case "$CFG" in
      cp5) ;;
      vincia)
        echo "PartonShowers:model = 2" ;;          # Vincia antenna shower
      cr1)
        echo "ColourReconnection:mode = 1"          # QCD-inspired CR
        echo "BeamRemnants:remnantMode = 1" ;;
      cr2)
        echo "ColourReconnection:mode = 2" ;;       # gluon-move CR
      fraghard)
        echo "StringZ:aLund = 0.58"
        echo "StringZ:bLund = 0.78"
        echo "StringPT:sigma = 0.305" ;;
      fragsoft)
        echo "StringZ:aLund = 0.78"
        echo "StringZ:bLund = 1.18"
        echo "StringPT:sigma = 0.365" ;;
      *) echo "FATAL: unknown CONFIG=$CFG"; exit 40 ;;
    esac
  } > run_${CFG}.cmnd

  OUT=out_${BIN}_${CFG}_${SEED}.yoda
  NTP=ntuple_${BIN}_${CFG}_${SEED}
  export HAD_NTUPLE="$WORK/$NTP"
  # shower ALL LHE events (nEvents >= NLHE; Pythia stops at end of file)
  if ! "$PYRIVET" run_${CFG}.cmnd "$OUT" "$NLHE" CMS_HADRONIC_JETMASS > rivet_${CFG}.log 2>&1; then
    echo "ERROR: pythia_rivet failed for $CFG"; tail -20 rivet_${CFG}.log; RC=44; continue
  fi
  if [ ! -s "$OUT" ] || [ ! -s "${NTP}_dijet.txt" ]; then
    echo "ERROR: empty output for $CFG"; tail -20 rivet_${CFG}.log; RC=45; continue
  fi
  SUMW=$(grep "^# sumw" ${NTP}_dijet.txt | awk '{print $3}')
  echo "### $CFG: sumw=$SUMW / NLHE=$NLHE (MLM acceptance) dijet=$(grep -c '^[0-9]' ${NTP}_dijet.txt) trijet=$(grep -c '^[0-9]' ${NTP}_trijet.txt)"

  # --- 3. stage out this config immediately (partial results survive) -------
  EOS="$EOSBASE/${CFG}/${BIN}"
  eos root://eosuser.cern.ch mkdir -p "$EOS" 2>/dev/null || mkdir -p "$EOS" 2>/dev/null || true
  for F in "$OUT" "${NTP}_dijet.txt" "${NTP}_trijet.txt"; do
    if ! cp "$F" "$EOS/" 2>/dev/null; then
      xrdcp -f "$F" "root://eosuser.cern.ch/$EOS/$F" || { echo "ERROR: stageout $F failed"; RC=46; }
    fi
  done
done

echo "### DONE $(date) rc=$RC -> $EOSBASE/<config>/$BIN"
cd /; rm -rf "$WORK"
exit $RC
