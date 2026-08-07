#!/usr/bin/env bash
# Diagnostic: shower ONE QCD_HT LHE with each config and compare MLM
# acceptance + the groomed-rho shape.  Same LHE for every config, so any
# difference is the shower/hadronization, not LHE statistics.
#   diag_matching.sh <BIN> <SEED> <NEVT> [configs...]
set -uo pipefail
BIN=${1:-QCD_HT700to1000}; SEED=${2:-4242}; NEVT=${3:-2000}; shift 3 || true
CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(cp5 vincia)
AFS=/afs/cern.ch/user/a/amandal/hadronic_model_prod
GP=/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.1/QCD_HT_Binned/${BIN}/v1/${BIN}_MG261_NNPDF31_slc7_amd64_gcc700_CMSSW_10_6_0_tarball.tar.xz
export TMPDIR=${TMPDIR:-/tmp/$USER.$$}; mkdir -p "$TMPDIR"
WORK=$(mktemp -d "$TMPDIR/diag.XXXXXX"); cd "$WORK"
echo "### diag $BIN seed=$SEED nevt=$NEVT configs=${CONFIGS[*]} work=$WORK"

mkdir gp && tar xaf "$GP" -C gp || { echo "FATAL untar"; exit 41; }
cat > _genlhe.sh <<EOF
set -e
cd "$WORK/gp"
export SCRAM_ARCH=slc7_amd64_gcc700
./runcmsgrid.sh $NEVT $SEED 1
EOF
/cvmfs/cms.cern.ch/common/cmssw-el7 -- bash "$WORK/_genlhe.sh" > genlhe.log 2>&1 || { echo "FATAL runcmsgrid"; tail -20 genlhe.log; exit 42; }
LHE="$WORK/gp/cmsgrid_final.lhe"
NLHE=$(grep -c "<event>" "$LHE"); echo "### LHE events: $NLHE"

set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh; set -u
export RIVET_ANALYSIS_PATH="$AFS:${RIVET_ANALYSIS_PATH:-}"

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
    cat "$AFS/cp5.cmnd"
    case "$CFG" in
      cp5) ;;
      vincia)   echo "PartonShowers:model = 2" ;;
      cr1)      echo "ColourReconnection:mode = 1"; echo "BeamRemnants:remnantMode = 1" ;;
      cr2)      echo "ColourReconnection:mode = 2" ;;
      fraghard) echo "StringZ:aLund = 0.58"; echo "StringZ:bLund = 0.78"; echo "StringPT:sigma = 0.305" ;;
      fragsoft) echo "StringZ:aLund = 0.78"; echo "StringZ:bLund = 1.18"; echo "StringPT:sigma = 0.365" ;;
    esac
  } > run_${CFG}.cmnd
  export HAD_NTUPLE="$WORK/nt_${CFG}"
  "$AFS/pythia_rivet" run_${CFG}.cmnd out_${CFG}.yoda "$NEVT" CMS_HADRONIC_JETMASS > log_${CFG}.txt 2>&1
  SUMW=$(grep "^# sumw" nt_${CFG}_dijet.txt | awk '{print $3}')
  XS=$(grep "^# xsec_pb" nt_${CFG}_dijet.txt | awk '{print $3}')
  DJ=$(grep -c "^[0-9]" nt_${CFG}_dijet.txt)
  TJ=$(grep -c "^[0-9]" nt_${CFG}_trijet.txt)
  # Pythia's own matching veto bookkeeping, if printed
  VETO=$(grep -iE "vetoed|user hooks" log_${CFG}.txt | head -2 | tr '\n' ' ')
  echo "RESULT $CFG sumw=$SUMW acc=$(python3 -c "print(f'{$SUMW/$NLHE:.3f}')") xsec_pb=$XS dijet_rows=$DJ trijet_rows=$TJ | $VETO"
done
echo "### WORKDIR $WORK (logs + ntuples kept)"
