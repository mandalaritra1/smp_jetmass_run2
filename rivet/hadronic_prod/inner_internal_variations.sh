#!/usr/bin/env bash
# Condor worker for same-ME Pythia internal variations:
# official UL QCD-HT MadGraph LHE -> Pythia 8.313 MLM -> Rivet.
#
# One LHE chunk is showered with every requested config so the nominal and
# variations share the hard events. A completion manifest is staged only after
# every config and output passes the local checks.
set -euo pipefail

MODE=$1
CAMPAIGN=$2
BIN=$3
SEED=$4
NEVT=$5
shift 5
CONFIGS=("$@")
[ ${#CONFIGS[@]} -gt 0 ] || { echo "FATAL: no configs supplied"; exit 40; }

SUBMITDIR=$(pwd)
AFS=/afs/cern.ch/user/a/amandal/hadronic_model_prod
EOSBASE=/eos/user/a/amandal/hadronic_model_prod_internal
GRIDPACK=/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.1/QCD_HT_Binned/${BIN}/v1/${BIN}_MG261_NNPDF31_slc7_amd64_gcc700_CMSSW_10_6_0_tarball.tar.xz

find_helper() {
  if [ -f "$SUBMITDIR/$1" ]; then
    printf '%s\n' "$SUBMITDIR/$1"
  elif [ -f "$SUBMITDIR/py313/$1" ]; then
    printf '%s\n' "$SUBMITDIR/py313/$1"
  elif [ -f "$AFS/py313/$1" ]; then
    printf '%s\n' "$AFS/py313/$1"
  else
    printf '%s\n' "$AFS/$1"
  fi
}

PYRIVET=$(find_helper pythia_rivet)
RIVETSO=$(find_helper RivetCMS_HADRONIC_JETMASS.so)
CP5=$(find_helper cp5.cmnd)
RIVETDIR=$(dirname "$RIVETSO")

export TMPDIR=${TMPDIR:-/tmp/$USER.$$}
mkdir -p "$TMPDIR"
WORK=$(mktemp -d "$TMPDIR/internal.XXXXXX")
STAGE=setup
SUCCESS=0

finish() {
  local rc=$?
  trap - EXIT
  if [ "$SUCCESS" -ne 1 ]; then
    local faildir=$EOSBASE/$CAMPAIGN/failures/$BIN
    local failure=$WORK/failure_${BIN}_${SEED}.txt
    {
      echo "campaign=$CAMPAIGN"
      echo "mode=$MODE"
      echo "bin=$BIN"
      echo "seed=$SEED"
      echo "requested=$NEVT"
      echo "stage=$STAGE"
      echo "exit_code=$rc"
      echo "host=$(hostname)"
      echo "time=$(date -u +%FT%TZ)"
    } > "$failure"
    eos root://eosuser.cern.ch mkdir -p "$faildir" >/dev/null 2>&1 || true
    xrdcp -f "$failure" "root://eosuser.cern.ch/$faildir/$(basename "$failure")" >/dev/null 2>&1 || true
    for logfile in "$WORK"/genlhe.log "$WORK"/rivet_*.log; do
      [ -f "$logfile" ] || continue
      xrdcp -f "$logfile" "root://eosuser.cern.ch/$faildir/${SEED}_$(basename "$logfile")" >/dev/null 2>&1 || true
    done
  fi
  cd /
  rm -rf "$WORK"
  exit "$rc"
}
trap finish EXIT

cd "$WORK"
echo "### $(date -u +%FT%TZ) mode=$MODE campaign=$CAMPAIGN bin=$BIN seed=$SEED nevt=$NEVT configs=${CONFIGS[*]} host=$(hostname)"

STAGE=gridpack
mkdir gridpack
tar xaf "$GRIDPACK" -C gridpack
printf '%s\n' \
  'set -e' \
  "cd $WORK/gridpack" \
  'export SCRAM_ARCH=slc7_amd64_gcc700' \
  "./runcmsgrid.sh $NEVT $SEED 1" > genlhe.sh
/cvmfs/cms.cern.ch/common/cmssw-el7 -- bash "$WORK/genlhe.sh" > genlhe.log 2>&1
LHE=$WORK/gridpack/cmsgrid_final.lhe
[ -s "$LHE" ] || { echo "FATAL: missing LHE"; exit 43; }
NLHE=$(grep -c '<event>' "$LHE")
[ "$NLHE" -gt 0 ] || { echo "FATAL: empty LHE"; exit 43; }

STAGE=environment
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
set -u
export RIVET_ANALYSIS_PATH="$RIVETDIR:${RIVET_ANALYSIS_PATH:-}"
PYTHIA_VERSION=$(pythia8-config --version)
RIVET_VERSION=$(rivet-config --version)
[ "$PYTHIA_VERSION" = 8.313 ] || { echo "FATAL: expected Pythia 8.313, got $PYTHIA_VERSION"; exit 47; }
[ "$RIVET_VERSION" = 4.0.2 ] || { echo "FATAL: expected Rivet 4.0.2, got $RIVET_VERSION"; exit 48; }

DRIVER_SHA256=$(sha256sum "$PYRIVET" | awk '{print $1}')
ANALYSIS_SHA256=$(sha256sum "$RIVETSO" | awk '{print $1}')
CONFIG_JSON=$WORK/configs.jsonl
: > "$CONFIG_JSON"
OUTPUTS=()

for CFG in "${CONFIGS[@]}"; do
  STAGE=shower_$CFG
  case "$CFG" in
    cp5|cr1|cr2|fraghard|fragsoft) ;;
    *) echo "FATAL: unknown config $CFG"; exit 40 ;;
  esac

  {
    echo "Beams:frameType = 4"
    echo "Beams:LHEF = $LHE"
    echo "Random:setSeed = on"
    echo "Random:seed = $SEED"
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
    echo "Main:timesAllowErrors = 20"
    cat "$CP5"
    case "$CFG" in
      cp5) ;;
      cr1)
        echo "ColourReconnection:mode = 1"
        echo "BeamRemnants:remnantMode = 1" ;;
      cr2)
        echo "ColourReconnection:mode = 2" ;;
      fraghard)
        echo "StringZ:aLund = 0.58"
        echo "StringZ:bLund = 0.78"
        echo "StringPT:sigma = 0.305" ;;
      fragsoft)
        echo "StringZ:aLund = 0.78"
        echo "StringZ:bLund = 1.18"
        echo "StringPT:sigma = 0.365" ;;
    esac
  } > "run_${CFG}.cmnd"

  STEM=${BIN}_${CFG}_${SEED}
  YODA=out_${STEM}.yoda
  NTP=ntuple_${STEM}
  export HAD_NTUPLE="$WORK/$NTP"
  "$PYRIVET" "run_${CFG}.cmnd" "$YODA" "$NLHE" CMS_HADRONIC_JETMASS > "rivet_${CFG}.log" 2>&1
  [ -s "$YODA" ] && [ -s "${NTP}_dijet.txt" ] && [ -s "${NTP}_trijet.txt" ] || {
    echo "FATAL: incomplete $CFG output"; exit 45;
  }

  rows() { awk '!/^#/ && NF {n++} END {print n+0}' "$1"; }
  zero_rows() { awk '!/^#/ && NF && $6 == 0 {n++} END {print n+0}' "$1"; }
  header() { awk -v key="$2" '$1 == "#" && $2 == key {print $3}' "$1"; }
  DIJET_ROWS=$(rows "${NTP}_dijet.txt")
  TRIJET_ROWS=$(rows "${NTP}_trijet.txt")
  DIJET_ZERO=$(zero_rows "${NTP}_dijet.txt")
  TRIJET_ZERO=$(zero_rows "${NTP}_trijet.txt")
  XSEC_PB=$(header "${NTP}_dijet.txt" xsec_pb)
  SUMW=$(header "${NTP}_dijet.txt" sumw)
  awk -v x="$XSEC_PB" -v s="$SUMW" 'BEGIN {exit !(x > 0 && s > 0)}' || {
    echo "FATAL: invalid xsec/sumw for $CFG: $XSEC_PB $SUMW"; exit 49;
  }
  [ "$DIJET_ZERO" -eq 0 ] && [ "$TRIJET_ZERO" -eq 0 ] || {
    echo "FATAL: zero-weight ntuple rows for $CFG"; exit 49;
  }
  PHYSICS_CONFIG_SHA256=$(grep -v -E '^(Beams:LHEF|Random:seed)' "run_${CFG}.cmnd" | sha256sum | awk '{print $1}')
  printf '{"config":"%s","xsec_pb":%.17g,"sumw":%.17g,"dijet_rows":%d,"trijet_rows":%d,"dijet_zero_rows":%d,"trijet_zero_rows":%d,"physics_config_sha256":"%s","yoda":"%s","dijet_ntuple":"%s","trijet_ntuple":"%s","cmnd":"%s","log":"%s"}\n' \
    "$CFG" "$XSEC_PB" "$SUMW" "$DIJET_ROWS" "$TRIJET_ROWS" "$DIJET_ZERO" "$TRIJET_ZERO" \
    "$PHYSICS_CONFIG_SHA256" "$YODA" "${NTP}_dijet.txt" "${NTP}_trijet.txt" "run_${CFG}.cmnd" "rivet_${CFG}.log" >> "$CONFIG_JSON"
  OUTPUTS+=("$YODA" "${NTP}_dijet.txt" "${NTP}_trijet.txt" "run_${CFG}.cmnd" "rivet_${CFG}.log")
done

STAGE=manifest
MANIFEST=manifest_${BIN}_${SEED}.json
{
  printf '{\n'
  printf '  "schema": 1,\n'
  printf '  "complete": true,\n'
  printf '  "campaign": "%s",\n' "$CAMPAIGN"
  printf '  "mode": "%s",\n' "$MODE"
  printf '  "bin": "%s",\n' "$BIN"
  printf '  "seed": %d,\n' "$SEED"
  printf '  "requested": %d,\n' "$NEVT"
  printf '  "lhe_events": %d,\n' "$NLHE"
  printf '  "pythia_version": "%s",\n' "$PYTHIA_VERSION"
  printf '  "rivet_version": "%s",\n' "$RIVET_VERSION"
  printf '  "driver_sha256": "%s",\n' "$DRIVER_SHA256"
  printf '  "analysis_sha256": "%s",\n' "$ANALYSIS_SHA256"
  printf '  "configs": [\n'
  awk 'NR > 1 {printf ",\n"} {printf "    %s", $0} END {printf "\n"}' "$CONFIG_JSON"
  printf '  ]\n'
  printf '}\n'
} > "$MANIFEST"

STAGE=stageout
EOS=$EOSBASE/$CAMPAIGN/$BIN
eos root://eosuser.cern.ch mkdir -p "$EOS"
for FILE in genlhe.log "${OUTPUTS[@]}"; do
  xrdcp -f "$FILE" "root://eosuser.cern.ch/$EOS/$FILE"
done
# Completion manifest is deliberately last. Harvesters ignore unmanifested files.
xrdcp -f "$MANIFEST" "root://eosuser.cern.ch/$EOS/$MANIFEST"
SUCCESS=1
echo "### COMPLETE $(date -u +%FT%TZ) $EOS/$MANIFEST"
