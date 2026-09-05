#!/usr/bin/env bash
# Condor worker: official UL QCD_HT MG gridpack -> MESS + Vincia ->
# CMS_HADRONIC_PAIR_SPLIT. The completion manifest is staged to EOS last.
#
# inner_mess_pairsplit.sh <CAMPAIGN> <PHASE> <BIN> <SEED> <NEVT>
set -euo pipefail

CAMPAIGN=$1
PHASE=$2
BIN=$3
SEED=$4
NEVT=$5
STEM=${BIN}_mess_${SEED}
COMMAND=run_${STEM}.cmnd
RIVET_LOG=rivet_${STEM}.log
LHE_LOG=genlhe_${STEM}.log
SUBMITDIR=$(pwd)
AFS=/afs/cern.ch/user/a/amandal/hadronic_model_prod
EOSBASE=/eos/user/a/amandal/hadronic_model_prod_mess

find_helper() {
  [ -f "$SUBMITDIR/$1" ] && printf '%s\n' "$SUBMITDIR/$1" || printf '%s\n' "$AFS/$1"
}

PYRIVET=$(find_helper pythia_rivet_mess)
RIVETSO=$(find_helper RivetCMS_HADRONIC_PAIR_SPLIT.so)
ANALYSIS_SOURCE=$(find_helper CMS_HADRONIC_PAIR_SPLIT.cc)
DRIVER_SOURCE=$(find_helper pythia_rivet_mess.cc)
CP5=$(find_helper cp5.cmnd)
RIVETDIR=$(dirname "$RIVETSO")
GRIDPACK=/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.1/QCD_HT_Binned/${BIN}/v1/${BIN}_MG261_NNPDF31_slc7_amd64_gcc700_CMSSW_10_6_0_tarball.tar.xz

export TMPDIR=${TMPDIR:-/tmp/$USER.$$}
mkdir -p "$TMPDIR"
WORK=$(mktemp -d "$TMPDIR/mess-pairsplit.XXXXXX")
cleanup() { cd /; rm -rf "$WORK"; }
trap cleanup EXIT
cd "$WORK"
echo "### $(date -u +%FT%TZ) campaign=$CAMPAIGN phase=$PHASE bin=$BIN seed=$SEED nevt=$NEVT host=$(hostname)"

mkdir gridpack
tar xaf "$GRIDPACK" -C gridpack
printf '%s\n' \
  'set -e' \
  "cd $WORK/gridpack" \
  'export SCRAM_ARCH=slc7_amd64_gcc700' \
  "./runcmsgrid.sh $NEVT $SEED 1" > genlhe.sh
/cvmfs/cms.cern.ch/common/cmssw-el7 -- bash "$WORK/genlhe.sh" > "$LHE_LOG" 2>&1
LHE=$WORK/gridpack/cmsgrid_final.lhe
[ -s "$LHE" ] || { echo "FATAL: missing LHE"; tail -40 "$LHE_LOG"; exit 43; }
NLHE=$(grep -c '<event>' "$LHE")
[ "$NLHE" -gt 0 ] || { echo "FATAL: empty LHE"; exit 43; }

set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
set -u
export RIVET_ANALYSIS_PATH="$RIVETDIR:${RIVET_ANALYSIS_PATH:-}"
PYTHIA_VERSION=$(pythia8-config --version)
RIVET_VERSION=$(rivet-config --version)
[ "$PYTHIA_VERSION" = "8.313" ] || {
  echo "FATAL: expected Pythia 8.313, got $PYTHIA_VERSION"; exit 47;
}
[ "$RIVET_VERSION" = "4.0.2" ] || {
  echo "FATAL: expected Rivet 4.0.2, got $RIVET_VERSION"; exit 47;
}

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
} > "$COMMAND"

YODA=out_${STEM}.yoda
NTP=ntuple_${STEM}
METRICS=metrics_${STEM}.txt
SELECTION_METRICS=selection_metrics_${STEM}.txt
export HAD_NTUPLE="$WORK/$NTP"
export HAD_SELECTION_METRICS="$WORK/$SELECTION_METRICS"
"$PYRIVET" "$COMMAND" "$YODA" "$NLHE" "$METRICS" \
  CMS_HADRONIC_PAIR_SPLIT > "$RIVET_LOG" 2>&1
[ -s "$YODA" ] && [ -s "${NTP}_dijet.txt" ] \
  && [ -s "${NTP}_trijet.txt" ] && [ -s "$METRICS" ] \
  && [ -s "$SELECTION_METRICS" ] || {
  echo "FATAL: incomplete MESS pair_split output"; tail -60 "$RIVET_LOG"; exit 45;
}

rows() { awk '!/^#/ && NF {n++} END {print n+0}' "$1"; }
nonzero_rows() { awk '!/^#/ && NF && $6 != 0 {n++} END {print n+0}' "$1"; }
sum_col() {
  awk -v power="$2" '!/^#/ && NF && $6 != 0 {s += power == 1 ? $6 : $6*$6} END {printf "%.17g", s+0}' "$1"
}
metric() { awk -F= -v key="$1" '$1 == key {print $2}' "$2"; }

DIJET_ROWS=$(rows "${NTP}_dijet.txt")
DIJET_NONZERO=$(nonzero_rows "${NTP}_dijet.txt")
DIJET_SUMW=$(sum_col "${NTP}_dijet.txt" 1)
DIJET_SUMW2=$(sum_col "${NTP}_dijet.txt" 2)
TRIJET_ROWS=$(rows "${NTP}_trijet.txt")
TRIJET_NONZERO=$(nonzero_rows "${NTP}_trijet.txt")
TRIJET_SUMW=$(sum_col "${NTP}_trijet.txt" 1)
TRIJET_SUMW2=$(sum_col "${NTP}_trijet.txt" 2)

REQUESTED=$(metric requested "$METRICS")
NEXT_OK=$(metric next_ok "$METRICS")
NEXT_FAILED=$(metric next_failed "$METRICS")
ZERO_WEIGHT_SKIPPED=$(metric zero_weight_skipped "$METRICS")
BROKEN_SKIPPED=$(metric broken_skipped "$METRICS")
ANALYZED=$(metric analyzed "$METRICS")
ANALYZED_SUMW=$(metric analyzed_sumw "$METRICS")
ANALYZED_SUMW2=$(metric analyzed_sumw2 "$METRICS")
INPUT_XSEC_PB=$(metric input_xsec_pb "$METRICS")
SAMPLE_XSEC_PB=$(metric sample_xsec_pb "$METRICS")
SAMPLE_XSEC_ERR_PB=$(metric sample_xsec_err_pb "$METRICS")
DIJET_SELECTED_EVENTS=$(metric dijet_selected_events "$SELECTION_METRICS")
TRIJET_SELECTED_EVENTS=$(metric trijet_selected_events "$SELECTION_METRICS")
OVERLAP_EVENTS=$(metric overlap_events "$SELECTION_METRICS")
NEITHER_SELECTED_EVENTS=$(metric neither_selected_events "$SELECTION_METRICS")

[ "$DIJET_ROWS" -eq "$DIJET_NONZERO" ] || { echo "FATAL: zero dijet rows"; exit 49; }
[ "$TRIJET_ROWS" -eq "$TRIJET_NONZERO" ] || { echo "FATAL: zero trijet rows"; exit 49; }
[ "$DIJET_ROWS" -eq $((2 * DIJET_SELECTED_EVENTS)) ] || {
  echo "FATAL: dijet row/event mismatch rows=$DIJET_ROWS events=$DIJET_SELECTED_EVENTS"; exit 49;
}
[ "$TRIJET_ROWS" -eq "$TRIJET_SELECTED_EVENTS" ] || {
  echo "FATAL: trijet row/event mismatch rows=$TRIJET_ROWS events=$TRIJET_SELECTED_EVENTS"; exit 49;
}
[ "$OVERLAP_EVENTS" -eq 0 ] || { echo "FATAL: pair_split overlap=$OVERLAP_EVENTS"; exit 49; }

CMND_SHA256=$(sha256sum "$COMMAND" | awk '{print $1}')
PHYSICS_CONFIG_SHA256=$(grep -v -E '^(Beams:LHEF|Random:seed)' "$COMMAND" | sha256sum | awk '{print $1}')
DRIVER_SHA256=$(sha256sum "$PYRIVET" | awk '{print $1}')
DRIVER_SOURCE_SHA256=$(sha256sum "$DRIVER_SOURCE" | awk '{print $1}')
ANALYSIS_SHA256=$(sha256sum "$RIVETSO" | awk '{print $1}')
ANALYSIS_SOURCE_SHA256=$(sha256sum "$ANALYSIS_SOURCE" | awk '{print $1}')
GRIDPACK_SHA256=$(sha256sum "$GRIDPACK" | awk '{print $1}')
YODA_SHA256=$(sha256sum "$YODA" | awk '{print $1}')
DIJET_NTUPLE_SHA256=$(sha256sum "${NTP}_dijet.txt" | awk '{print $1}')
TRIJET_NTUPLE_SHA256=$(sha256sum "${NTP}_trijet.txt" | awk '{print $1}')
METRICS_SHA256=$(sha256sum "$METRICS" | awk '{print $1}')
SELECTION_METRICS_SHA256=$(sha256sum "$SELECTION_METRICS" | awk '{print $1}')
COMMAND_SHA256=$(sha256sum "$COMMAND" | awk '{print $1}')
RIVET_LOG_SHA256=$(sha256sum "$RIVET_LOG" | awk '{print $1}')
LHE_LOG_SHA256=$(sha256sum "$LHE_LOG" | awk '{print $1}')

MANIFEST=manifest_${STEM}.json
cat > "$MANIFEST" <<EOF
{
  "schema": 2,
  "complete": true,
  "campaign": "$CAMPAIGN",
  "phase": "$PHASE",
  "selection": "CMS_HADRONIC_PAIR_SPLIT",
  "bin": "$BIN",
  "seed": $SEED,
  "requested": $NEVT,
  "driver_requested": $REQUESTED,
  "lhe_events": $NLHE,
  "next_ok": $NEXT_OK,
  "next_failed": $NEXT_FAILED,
  "zero_weight_skipped": $ZERO_WEIGHT_SKIPPED,
  "broken_skipped": $BROKEN_SKIPPED,
  "analyzed": $ANALYZED,
  "analyzed_sumw": $ANALYZED_SUMW,
  "analyzed_sumw2": $ANALYZED_SUMW2,
  "dijet_selected_events": $DIJET_SELECTED_EVENTS,
  "trijet_selected_events": $TRIJET_SELECTED_EVENTS,
  "overlap_events": $OVERLAP_EVENTS,
  "neither_selected_events": $NEITHER_SELECTED_EVENTS,
  "dijet_rows": $DIJET_ROWS,
  "dijet_nonzero_rows": $DIJET_NONZERO,
  "dijet_sumw": $DIJET_SUMW,
  "dijet_sumw2": $DIJET_SUMW2,
  "trijet_rows": $TRIJET_ROWS,
  "trijet_nonzero_rows": $TRIJET_NONZERO,
  "trijet_sumw": $TRIJET_SUMW,
  "trijet_sumw2": $TRIJET_SUMW2,
  "input_xsec_pb": $INPUT_XSEC_PB,
  "sample_xsec_pb": $SAMPLE_XSEC_PB,
  "sample_xsec_err_pb": $SAMPLE_XSEC_ERR_PB,
  "pythia_version": "$PYTHIA_VERSION",
  "rivet_version": "$RIVET_VERSION",
  "cmnd_sha256": "$CMND_SHA256",
  "physics_config_sha256": "$PHYSICS_CONFIG_SHA256",
  "gridpack_sha256": "$GRIDPACK_SHA256",
  "driver_sha256": "$DRIVER_SHA256",
  "driver_source_sha256": "$DRIVER_SOURCE_SHA256",
  "analysis_sha256": "$ANALYSIS_SHA256",
  "analysis_source_sha256": "$ANALYSIS_SOURCE_SHA256",
  "yoda": "$YODA",
  "dijet_ntuple": "${NTP}_dijet.txt",
  "trijet_ntuple": "${NTP}_trijet.txt",
  "metrics": "$METRICS",
  "selection_metrics": "$SELECTION_METRICS",
  "command": "$COMMAND",
  "rivet_log": "$RIVET_LOG",
  "lhe_log": "$LHE_LOG",
  "files_sha256": {
    "$YODA": "$YODA_SHA256",
    "${NTP}_dijet.txt": "$DIJET_NTUPLE_SHA256",
    "${NTP}_trijet.txt": "$TRIJET_NTUPLE_SHA256",
    "$METRICS": "$METRICS_SHA256",
    "$SELECTION_METRICS": "$SELECTION_METRICS_SHA256",
    "$COMMAND": "$COMMAND_SHA256",
    "$RIVET_LOG": "$RIVET_LOG_SHA256",
    "$LHE_LOG": "$LHE_LOG_SHA256"
  }
}
EOF

EOS=$EOSBASE/$CAMPAIGN/$BIN
eos root://eosuser.cern.ch mkdir -p "$EOS"
for FILE in "$YODA" "${NTP}_dijet.txt" "${NTP}_trijet.txt" \
  "$METRICS" "$SELECTION_METRICS" "$RIVET_LOG" "$LHE_LOG" "$COMMAND"; do
  xrdcp -f "$FILE" "root://eosuser.cern.ch/$EOS/$FILE"
done
xrdcp -f "$MANIFEST" "root://eosuser.cern.ch/$EOS/$MANIFEST"
echo "### COMPLETE $(date -u +%FT%TZ) $EOS/$MANIFEST dijet_events=$DIJET_SELECTED_EVENTS analyzed=$ANALYZED"
