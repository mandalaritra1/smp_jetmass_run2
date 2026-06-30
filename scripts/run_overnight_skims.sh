#!/usr/bin/env bash
# Overnight Z+jet skims on casa: data (8 PDs, mm+ee) + NLO MC (16 ptZ bin x era).
# Resumable: a dataset whose merged.parquet already exists is skipped, so you can
# relaunch this any time (after a pod cull / disconnect / failure) and it finishes
# only what's missing.
#
#   cd ~/smp_jetmass_run2 && git pull
#   nohup bash scripts/run_overnight_skims.sh > ~/skims.log 2>&1 &
#   tail -f ~/skims.log
set -u
cd "$(dirname "$0")/.." || exit 1            # repo root, wherever it is checked out

NLO_CFG=configs/zjet_nlo_ptz_skim_casa.json
DATA_CFG=configs/zjet_data_skim_casa.json
NLO_OUT=/home/cms-jovyan/nlo_ptz_skims
DATA_OUT=/home/cms-jovyan/zjet_data_skims

run () {  # $1=config  $2=outdir  $3=dataset
  if [ -f "$2/$3/merged.parquet" ]; then echo "[skip] $3 (already done)"; return; fi
  echo "[run ] $3  $(date)"
  if python scripts/run_zjet_skim.py --config "$1" --only "$3"; then
    echo "[done] $3  $(date)"
  else
    echo "[FAIL] $3  $(date)"                # no merged.parquet -> retried next launch
  fi
}

# ---- DATA first (small/fast -- quick confirmation): 8 PDs, mm+ee ----
for ds in SingleMuon_UL2016 SingleElectron_UL2016 \
          SingleMuon_UL2016APV SingleElectron_UL2016APV \
          SingleMuon_UL2017 SingleElectron_UL2017 \
          SingleMuon_UL2018 EGamma_UL2018; do
  run "$DATA_CFG" "$DATA_OUT" "$ds"
done

# ---- MC: NLO 16 (ptZ bin x era), both channels ----
for era in UL16NanoAODv9 UL16NanoAODAPVv9 UL17NanoAODv9 UL18NanoAODv9; do
  for b in 100To250 250To400 400To650 650ToInf; do
    run "$NLO_CFG" "$NLO_OUT" "nlo_ptz_${b}_${era}"
  done
done

echo "ALL DONE  $(date)"
