#!/usr/bin/env bash
# Condor worker: official UL QCD_HT MG gridpack -> MESS + Vincia -> Rivet.
#
# inner_mess.sh <CAMPAIGN> <BIN> <SEED> <NEVT>
set -euo pipefail

CAMPAIGN=$1
BIN=$2
SEED=$3
NEVT=$4
SUBMITDIR=$(pwd)
AFS=/afs/cern.ch/user/a/amandal/hadronic_model_prod
EOSBASE=/eos/user/a/amandal/hadronic_model_prod_mess

find_helper() {
  [ -f "$SUBMITDIR/$1" ] && printf '%s\n' "$SUBMITDIR/$1" || printf '%s\n' "$AFS/$1"
}

PYRIVET=$(find_helper pythia_rivet_mess)
RIVETSO=$(find_helper RivetCMS_HADRONIC_JETMASS.so)
CP5=$(find_helper cp5.cmnd)
RIVETDIR=$(dirname "$RIVETSO")
GRIDPACK=/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.1/QCD_HT_Binned/${BIN}/v1/${BIN}_MG261_NNPDF31_slc7_amd64_gcc700_CMSSW_10_6_0_tarball.tar.xz

export TMPDIR=${TMPDIR:-/tmp/$USER.$$}
mkdir -p "$TMPDIR"
WORK=$(mktemp -d "$TMPDIR/mess.XXXXXX")
cleanup() { cd /; rm -rf "$WORK"; }
trap cleanup EXIT
cd "$WORK"
echo "### $(date -u +%FT%TZ) campaign=$CAMPAIGN bin=$BIN seed=$SEED nevt=$NEVT host=$(hostname)"

# Generate one LHE chunk using the official UL gridpack.
mkdir gridpack
tar xaf "$GRIDPACK" -C gridpack
printf '%s\n' \
  'set -e' \
  "cd $WORK/gridpack" \
  'export SCRAM_ARCH=slc7_amd64_gcc700' \
  "./runcmsgrid.sh $NEVT $SEED 1" > genlhe.sh
/cvmfs/cms.cern.ch/common/cmssw-el7 -- bash "$WORK/genlhe.sh" > genlhe.log 2>&1
LHE=$WORK/gridpack/cmsgrid_final.lhe
[ -s "$LHE" ] || { echo "FATAL: missing LHE"; tail -40 genlhe.log; exit 43; }
NLHE=$(grep -c '<event>' "$LHE")
[ "$NLHE" -gt 0 ] || { echo "FATAL: empty LHE"; exit 43; }

# Shower/merge in the Pythia 8.313 view used to build the dedicated driver.
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
set -u
export RIVET_ANALYSIS_PATH="$RIVETDIR:${RIVET_ANALYSIS_PATH:-}"
PYTHIA_VERSION=$(pythia8-config --version)
RIVET_VERSION=$(rivet-config --version)
[ "$PYTHIA_VERSION" = "8.313" ] || { echo "FATAL: expected Pythia 8.313, got $PYTHIA_VERSION"; exit 47; }

{
  echo "Beams:frameType = 4"
  echo "Beams:LHEF = $LHE"
  echo "Random:setSeed = on"
  echo "Random:seed = $SEED"
  echo "Check:epTolErr = 2e-2"
  cat "$CP5"
  echo "PartonShowers:model = 2"
  echo "Vincia:sectorShower = on"
  echo "Vincia:kineMapFFsplit = 1"
  echo "Vincia:pTmaxMatch = 1"
  echo "Merging:TMS = 14."
  echo "Merging:Process = {p p > j j}"
  echo "Merging:nJetMax = 2"
  echo "Merging:doKTMerging = on"
  echo "Main:timesAllowErrors = 20"
} > run_mess.cmnd

STEM=${BIN}_mess_${SEED}
YODA=out_${STEM}.yoda
NTP=ntuple_${STEM}
METRICS=metrics_${STEM}.txt
export HAD_NTUPLE="$WORK/$NTP"
"$PYRIVET" run_mess.cmnd "$YODA" "$NLHE" "$METRICS" CMS_HADRONIC_JETMASS > rivet_mess.log 2>&1
[ -s "$YODA" ] && [ -s "${NTP}_dijet.txt" ] && [ -s "${NTP}_trijet.txt" ] && [ -s "$METRICS" ] || {
  echo "FATAL: incomplete MESS output"; tail -60 rivet_mess.log; exit 45;
}

rows() { awk '!/^#/ && NF {n++} END {print n+0}' "$1"; }
nonzero_rows() { awk '!/^#/ && NF && $6 != 0 {n++} END {print n+0}' "$1"; }
sum_col() { awk -v p="$2" '!/^#/ && NF && $6 != 0 {s += p == 1 ? $6 : $6*$6} END {printf "%.17g", s+0}' "$1"; }

DIJET_ROWS=$(rows "${NTP}_dijet.txt")
DIJET_NONZERO=$(nonzero_rows "${NTP}_dijet.txt")
TRIJET_ROWS=$(rows "${NTP}_trijet.txt")
TRIJET_NONZERO=$(nonzero_rows "${NTP}_trijet.txt")
TRIJET_SUMW=$(sum_col "${NTP}_trijet.txt" 1)
TRIJET_SUMW2=$(sum_col "${NTP}_trijet.txt" 2)
metric() { awk -F= -v key="$1" '$1 == key {print $2}' "$METRICS"; }
INPUT_XSEC_PB=$(metric input_xsec_pb)
SAMPLE_XSEC_PB=$(metric sample_xsec_pb)
SAMPLE_XSEC_ERR_PB=$(metric sample_xsec_err_pb)
CMND_SHA256=$(sha256sum run_mess.cmnd | awk '{print $1}')
PHYSICS_CONFIG_SHA256=$(grep -v -E '^(Beams:LHEF|Random:seed)' run_mess.cmnd | sha256sum | awk '{print $1}')
DRIVER_SHA256=$(sha256sum "$PYRIVET" | awk '{print $1}')
ANALYSIS_SHA256=$(sha256sum "$RIVETSO" | awk '{print $1}')

MANIFEST=manifest_${STEM}.json
cat > "$MANIFEST" <<EOF
{
  "schema": 1,
  "complete": true,
  "campaign": "$CAMPAIGN",
  "bin": "$BIN",
  "seed": $SEED,
  "requested": $NEVT,
  "lhe_events": $NLHE,
  "pythia_version": "$PYTHIA_VERSION",
  "rivet_version": "$RIVET_VERSION",
  "cmnd_sha256": "$CMND_SHA256",
  "physics_config_sha256": "$PHYSICS_CONFIG_SHA256",
  "driver_sha256": "$DRIVER_SHA256",
  "analysis_sha256": "$ANALYSIS_SHA256",
  "dijet_rows": $DIJET_ROWS,
  "dijet_nonzero_rows": $DIJET_NONZERO,
  "trijet_rows": $TRIJET_ROWS,
  "trijet_nonzero_rows": $TRIJET_NONZERO,
  "trijet_sumw": $TRIJET_SUMW,
  "trijet_sumw2": $TRIJET_SUMW2,
  "input_xsec_pb": $INPUT_XSEC_PB,
  "sample_xsec_pb": $SAMPLE_XSEC_PB,
  "sample_xsec_err_pb": $SAMPLE_XSEC_ERR_PB,
  "yoda": "$YODA",
  "dijet_ntuple": "${NTP}_dijet.txt",
  "trijet_ntuple": "${NTP}_trijet.txt",
  "metrics": "$METRICS"
}
EOF

# Stage the completion manifest last. Harvesters must ignore chunks without it.
EOS=$EOSBASE/$CAMPAIGN/$BIN
eos root://eosuser.cern.ch mkdir -p "$EOS"
for FILE in "$YODA" "${NTP}_dijet.txt" "${NTP}_trijet.txt" "$METRICS" rivet_mess.log genlhe.log run_mess.cmnd; do
  xrdcp -f "$FILE" "root://eosuser.cern.ch/$EOS/$FILE"
done
xrdcp -f "$MANIFEST" "root://eosuser.cern.ch/$EOS/$MANIFEST"
echo "### COMPLETE $(date -u +%FT%TZ) $EOS/$MANIFEST trijet_nonzero=$TRIJET_NONZERO"
