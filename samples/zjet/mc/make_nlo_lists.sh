#!/usr/bin/env bash
#
# Generate the NLO (amcatnloFXFX) DY M-50 NANOAOD file lists for all four UL eras.
#
# The framework reads these .txt lists (one raw /store LFN per line) and prepends the
# redirector itself (root://xcache/ on coffea.casa). The 2016 postVFP list
# (inclusive_UL16NanoAODv9.txt) already exists and is skipped unless --force is given;
# this script produces the three missing eras (2016APV, 2017, 2018) so that
# `configs/zjet_nlo_all.json` (era="all") can run.
#
# Requires dasgoclient + a valid grid proxy (run on coffea.casa or the LPC):
#     voms-proxy-init --rfc --voms cms -valid 192:00
#     bash samples/zjet/mc/make_nlo_lists.sh          # missing eras only
#     bash samples/zjet/mc/make_nlo_lists.sh --force   # regenerate all four
#
set -euo pipefail

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMARY="DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8"

# era tag -> processing-version glob (wildcard tolerates -v1/-v2 differences)
ERAS=(UL16NanoAODv9 UL16NanoAODAPVv9 UL17NanoAODv9 UL18NanoAODv9)
declare -A PROC=(
  [UL16NanoAODv9]="RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-*"
  [UL16NanoAODAPVv9]="RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-*"
  [UL17NanoAODv9]="RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-*"
  [UL18NanoAODv9]="RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-*"
)

command -v dasgoclient >/dev/null || { echo "ERROR: dasgoclient not found (run on casa/LPC)"; exit 1; }

for tag in "${ERAS[@]}"; do
  out="$OUTDIR/inclusive_${tag}.txt"
  if [[ -s "$out" && $FORCE -eq 0 ]]; then
    echo ">>> [$tag] exists, skipping ($out). Use --force to regenerate."
    continue
  fi
  query="/${PRIMARY}/${PROC[$tag]}/NANOAODSIM"
  echo ">>> [$tag] resolving: $query"
  mapfile -t cands < <(dasgoclient --query "dataset=$query" | sort)
  if [[ ${#cands[@]} -eq 0 ]]; then
    echo "!!! [$tag] no dataset found. Inspect available versions with:"
    echo "    dasgoclient --query \"dataset=/${PRIMARY}/RunIISummer20${tag}-*/NANOAODSIM\""
    continue
  fi
  [[ ${#cands[@]} -gt 1 ]] && printf '    candidate: %s\n' "${cands[@]}"
  ds="${cands[0]}"   # prefer the lowest (-v1, no ext) — matches the existing 2016 list
  echo ">>> [$tag] dataset: $ds"
  dasgoclient --query "file dataset=$ds" | sort > "$out"
  echo ">>> [$tag] wrote $(wc -l < "$out") files -> $out"
done
