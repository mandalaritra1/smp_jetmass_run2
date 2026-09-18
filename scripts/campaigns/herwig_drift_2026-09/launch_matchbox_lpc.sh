#!/usr/bin/env bash
# NLO Matchbox Herwig7 DY (no LHE, no MLM) zjet minimal_rho, 2018, on LPC condor
# (2026-09-18).  Does the GEN-proposed MG+Herwig7 replacement drift in pT?
#
#   ./launch_matchbox_lpc.sh smoke   # first file, local futures in the container
#   ./launch_matchbox_lpc.sh run     # the 620-file sample on condor (n_workers from config)
#
# Logs in logs/, output in outputs/*herwig_matchbox*2018.pkl.
cd "$(dirname "$0")" || exit 1
mkdir -p logs outputs
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"
case "$1" in
  smoke)
    echo "=== herwig_matchbox smoke start $(date -u +%FT%TZ)"
    ./lpc_container.sh "python -u scripts/run_analysis_cli.py --config configs/zjet_herwig_matchbox_2018_minimal_rho_lpc_smoke.json" > logs/smoke_zjet_herwig_matchbox_2018.log 2>&1
    echo "=== herwig_matchbox smoke exit $? $(date -u +%FT%TZ)";;
  run)
    setsid nohup bash -c 'echo "=== zjet herwig_matchbox 2018 start $(date -u +%FT%TZ)";
        ./lpc_container.sh "python -u scripts/run_analysis_cli.py --config configs/zjet_herwig_matchbox_2018_minimal_rho_lpc.json" > logs/campaign_zjet_herwig_matchbox_2018.log 2>&1;
        echo "=== zjet herwig_matchbox 2018 exit $? $(date -u +%FT%TZ)"' > logs/campaign_zjet_herwig_matchbox_chain.log 2>&1 < /dev/null &
    echo "herwig_matchbox started, pid $!";;
  *) echo "usage: $0 smoke|run"; exit 2;;
esac
