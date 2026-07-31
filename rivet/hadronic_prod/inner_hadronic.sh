#!/usr/bin/env bash
# Self-contained condor worker for the dijet/trijet model-uncertainty
# standalone production (QCD_HT madgraphMLM gridpacks, 6 shower/hadronization
# configs on the CP5 base).  Per job:
#   1. extract the official UL QCD_HT gridpack to $TMPDIR and run runcmsgrid.sh
#      inside the cmssw-el7 (slc7) container  ->  cmsgrid_final.lhe
#   2. source the LCG_106 view (el9, Pythia 8.312 + Rivet 4.0.0) and shower the
#      LHE with MLM jet matching (CombineMatchingInput) + the chosen config,
#      running the CMS_HADRONIC_JETMASS Rivet analysis -> yoda + 2 ntuples
#   3. copy the outputs to EOS.
#
#   inner_hadronic.sh <BIN> <SEED> <NEVT> <CONFIG>
#     BIN    e.g. QCD_HT700to1000
#     CONFIG cp5 | vincia | cr1 | cr2 | fraghard | fragsoft
#
# MLM matching parameters COPIED from the official UL fragment
# (JME-RunIISummer20UL18wmLHEGEN-00015): qCut=14, setMad=off, scheme=1,
# jetAlgorithm=2, etaJetMax=5, coneRadius=1, slowJetPower=1, nQmatch=5,
# nJetMax=4.  These differ from the DY production (qCut=19, setMad=on) --
# do NOT copy blindly between the two.
set -uo pipefail
BIN=$1; SEED=$2; NEVT=$3; CONFIG=$4
SUBMITDIR=$(pwd)                          # condor drops transfer_input_files here
AFS=/afs/cern.ch/user/a/amandal/hadronic_model_prod
find_helper() { [ -f "$SUBMITDIR/$1" ] && echo "$SUBMITDIR/$1" || echo "$AFS/$1"; }
PYRIVET=$(find_helper pythia_rivet); RIVETSO=$(find_helper RivetCMS_HADRONIC_JETMASS.so)
CP5=$(find_helper cp5.cmnd); RIVETDIR=$(dirname "$RIVETSO")
EOS=/eos/user/a/amandal/hadronic_model_prod/${CONFIG}/${BIN}
GP=/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.1/QCD_HT_Binned/${BIN}/v1/${BIN}_MG261_NNPDF31_slc7_amd64_gcc700_CMSSW_10_6_0_tarball.tar.xz
export TMPDIR=${TMPDIR:-/tmp/$USER.$$}; mkdir -p "$TMPDIR"
WORK=$(mktemp -d "$TMPDIR/hm.XXXXXX"); cd "$WORK"
echo "### $(date) job BIN=$BIN SEED=$SEED NEVT=$NEVT CONFIG=$CONFIG host=$(hostname) work=$WORK"

# --- 1. gridpack -> LHE inside cmssw-el7 ------------------------------------
mkdir gp && tar xaf "$GP" -C gp || { echo "FATAL: gridpack untar failed"; exit 41; }
cat > _genlhe.sh <<EOF
set -e
cd "$WORK/gp"
export SCRAM_ARCH=slc7_amd64_gcc700
# runcmsgrid.sh <nevents> <seed> [ncpu]
./runcmsgrid.sh $NEVT $SEED 1
EOF
/cvmfs/cms.cern.ch/common/cmssw-el7 -- bash "$WORK/_genlhe.sh" > genlhe.log 2>&1 || { echo "FATAL: runcmsgrid failed"; tail -40 genlhe.log; exit 42; }
LHE=$(ls "$WORK"/gp/cmsgrid_final.lhe 2>/dev/null || true)
[ -f "$LHE" ] || { echo "FATAL: no LHE produced"; tail -40 genlhe.log; exit 43; }
NEV=$(grep -c "<event>" "$LHE"); echo "### LHE has $NEV events"

# --- 2. shower + Rivet in LCG_106 (el9) -------------------------------------
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
  #### the same brackets as the zjet modelling variations (Monash-symmetric
  #### Lund brackets, CR mode flips -- exploratory, not retuned CP5-CR tunes).
  cat "$CP5"
  case "$CONFIG" in
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
    *) echo "FATAL: unknown CONFIG=$CONFIG"; exit 40 ;;
  esac
} > run.cmnd
echo "### cmnd:"; cat run.cmnd

set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh; set -u
export RIVET_ANALYSIS_PATH="$RIVETDIR:${RIVET_ANALYSIS_PATH:-}"
OUT=out_${BIN}_${CONFIG}_${SEED}.yoda
NTP=ntuple_${BIN}_${CONFIG}_${SEED}
# per-jet gen ntuples: ${NTP}_dijet.txt / ${NTP}_trijet.txt
# (jet_pt m_u m_g rho_u rho_g weight + xsec_pb/sumw footer)
export HAD_NTUPLE="$WORK/$NTP"
"$PYRIVET" run.cmnd "$OUT" "$NEVT" CMS_HADRONIC_JETMASS > rivet.log 2>&1 || { echo "FATAL: pythia_rivet failed"; tail -40 rivet.log; exit 44; }
[ -s "$OUT" ] || { echo "FATAL: empty yoda"; tail -40 rivet.log; exit 45; }
[ -s "${NTP}_dijet.txt" ] || { echo "FATAL: empty dijet ntuple"; tail -40 rivet.log; exit 45; }
echo "### yoda: $(ls -la $OUT)"
echo "### ntuples: $(wc -l ${NTP}_dijet.txt ${NTP}_trijet.txt)"

# --- 3. stage out to EOS (yoda + both ntuples) ------------------------------
eos root://eosuser.cern.ch mkdir -p "$EOS" 2>/dev/null || mkdir -p "$EOS" 2>/dev/null || true
for F in "$OUT" "${NTP}_dijet.txt" "${NTP}_trijet.txt"; do
  if ! cp "$F" "$EOS/" 2>/dev/null; then
    xrdcp -f "$F" "root://eosuser.cern.ch/$EOS/$F" || { echo "FATAL: stageout $F failed"; exit 46; }
  fi
done
echo "### DONE $(date) -> $EOS"
cd /; rm -rf "$WORK"
