#!/usr/bin/env bash
# Flat-pT QCD (standalone Herwig7 CH3 / Pythia8 CP5, no MLM) dijet minimal_rho,
# 2018, on LPC condor (2026-09-17).  The merging-vs-shower test for the groomed
# mass pT drift: Flat-Herwig7/Flat-Pythia8 against MLM-Herwig7/MLM-Pythia8.
#
#   ./launch_flat_lpc.sh smoke            # first file of each sample, local futures
#   ./launch_flat_lpc.sh herwig_flat      # the 700-file Herwig7 flat sample
#   ./launch_flat_lpc.sh pythia_flat      # the 20-file Pythia8 flat sample
#   ./launch_flat_lpc.sh both             # pythia_flat then herwig_flat, chained
#
# Logs in logs/, outputs in outputs/minimal_rho_dijet_<dataset>_2018.pkl.
cd "$(dirname "$0")" || exit 1
mkdir -p logs outputs
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"
case "$1" in
  smoke)
    for ds in herwig_flat pythia_flat; do
      echo "=== $ds smoke start $(date -u +%FT%TZ)"
      ./lpc_container.sh "python -u scripts/run_analysis_cli.py --config configs/dijet_${ds}_2018_minimal_rho_lpc_smoke.json" > logs/smoke_dijet_${ds}_2018.log 2>&1
      echo "=== $ds smoke exit $? $(date -u +%FT%TZ)"
    done;;
  herwig_flat|pythia_flat)
    ds="$1"
    setsid nohup bash -c 'ds="$0"; echo "=== dijet ${ds} 2018 start $(date -u +%FT%TZ)";
        ./lpc_container.sh "python -u scripts/run_analysis_cli.py --config configs/dijet_${ds}_2018_minimal_rho_lpc.json" > logs/campaign_dijet_${ds}_2018.log 2>&1;
        echo "=== dijet ${ds} 2018 exit $? $(date -u +%FT%TZ)"' "$ds" > "logs/campaign_dijet_${ds}_chain.log" 2>&1 < /dev/null &
    echo "$ds started, pid $!";;
  both)
    setsid nohup bash -c 'for ds in pythia_flat herwig_flat; do
        echo "=== dijet ${ds} 2018 start $(date -u +%FT%TZ)";
        ./lpc_container.sh "python -u scripts/run_analysis_cli.py --config configs/dijet_${ds}_2018_minimal_rho_lpc.json" > logs/campaign_dijet_${ds}_2018.log 2>&1;
        echo "=== dijet ${ds} 2018 exit $? $(date -u +%FT%TZ)";
      done' > logs/campaign_dijet_flat_chain.log 2>&1 < /dev/null &
    echo "flat chain started, pid $!";;
  *) echo "usage: $0 smoke|herwig_flat|pythia_flat|both"; exit 2;;
esac
