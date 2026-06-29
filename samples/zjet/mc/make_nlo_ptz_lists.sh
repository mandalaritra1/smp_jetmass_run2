#!/usr/bin/env bash
#
# Generate the PtZ-binned NLO (amcatnloFXFX, MatchEWPDG20) DY NANOAODv9 file lists
# for all four UL eras x 6 LHE-ptZ bins (24 lists: ptz_<bin>_<era>.txt).
#
# Each bin becomes its own framework dataset; the processor postprocess normalizes
# every bin to its own XSDB cross section, so summing the 6 bins gives the stitched
# ptZ spectrum (see _postprocess_scale_for_dataset / the ptz_ histogram branch in
# zjet_processor.py, and configs/zjet_nlo_ptz_all.json, dataset="nlo_ptz").
#
# The framework reads these raw /store LFN lists and prepends the redirector itself
# (root://xcache/ on coffea.casa).
#
# Requires dasgoclient + a valid grid proxy (run on coffea.casa or the LPC):
#     voms-proxy-init --rfc --voms cms -valid 192:00
#     bash samples/zjet/mc/make_nlo_ptz_lists.sh           # missing lists only
#     bash samples/zjet/mc/make_nlo_ptz_lists.sh --force    # regenerate all 24
#
set -euo pipefail

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUFFIX="MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8"

BINS=(0To50 50To100 100To250 250To400 400To650 650ToInf)
ERAS=(UL16NanoAODv9 UL16NanoAODAPVv9 UL17NanoAODv9 UL18NanoAODv9)
declare -A PROC=(
  [UL16NanoAODv9]="RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-*"
  [UL16NanoAODAPVv9]="RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-*"
  [UL17NanoAODv9]="RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-*"
  [UL18NanoAODv9]="RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-*"
)

command -v dasgoclient >/dev/null || { echo "ERROR: dasgoclient not found (run on casa/LPC)"; exit 1; }

for era in "${ERAS[@]}"; do
  for b in "${BINS[@]}"; do
    out="$OUTDIR/ptz_${b}_${era}.txt"
    if [[ -s "$out" && $FORCE -eq 0 ]]; then
      echo ">>> [ptz_${b}_${era}] exists, skipping ($out). Use --force to regenerate."
      continue
    fi
    query="/DYJetsToLL_LHEFilterPtZ-${b}_${SUFFIX}/${PROC[$era]}/NANOAODSIM"
    ds=$(dasgoclient --query "dataset=$query" 2>/dev/null | sort | head -1)
    if [[ -z "$ds" ]]; then
      echo "!!! [ptz_${b}_${era}] no dataset found: $query"
      continue
    fi
    dasgoclient --query "file dataset=$ds" | sort > "$out"
    echo ">>> [ptz_${b}_${era}] $(wc -l < "$out") files -> $out  [$ds]"
  done
done
