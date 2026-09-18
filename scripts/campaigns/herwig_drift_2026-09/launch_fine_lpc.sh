#!/usr/bin/env bash
# Fine-rho (minimal_rho_fine, 4x rho bins) 2018 Z+jet: Pythia8 MLM, Herwig7 MLM and
# the NLO Matchbox Herwig7 sample on LPC condor, produced together (2026-09-18).
#
#   ./launch_fine_lpc.sh smoke   # first file of each sample, local futures in the container
#   ./launch_fine_lpc.sh chain   # herwig -> herwig_matchbox -> pythia, sequential, background
#   ./launch_fine_lpc.sh all     # smoke, then chain if every smoke passed
#
# Logs in logs/, outputs in outputs/minimal_rho_fine_<dataset>_2018.pkl.
cd "$(dirname "$0")" || exit 1
mkdir -p logs outputs
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"
smoke() {
  rc=0
  for ds in herwig herwig_matchbox pythia; do
    echo "=== $ds fine smoke start $(date -u +%FT%TZ)"
    ./lpc_container.sh "python -u scripts/run_analysis_cli.py --config configs/zjet_${ds}_2018_minimal_rho_fine_lpc_smoke.json" > logs/smoke_zjet_${ds}_fine_2018.log 2>&1
    r=$?; echo "=== $ds fine smoke exit $r $(date -u +%FT%TZ)"; [ $r -ne 0 ] && rc=1
  done
  return $rc
}
chain() {
  setsid nohup bash -c 'for ds in herwig herwig_matchbox pythia; do
      echo "=== zjet ${ds} fine 2018 start $(date -u +%FT%TZ)";
      ./lpc_container.sh "python -u scripts/run_analysis_cli.py --config configs/zjet_${ds}_2018_minimal_rho_fine_lpc.json" > logs/campaign_zjet_${ds}_fine_2018.log 2>&1;
      echo "=== zjet ${ds} fine 2018 exit $? $(date -u +%FT%TZ)";
    done' > logs/campaign_zjet_fine_chain.log 2>&1 < /dev/null &
  echo "fine chain started, pid $!"
}
case "$1" in
  smoke) smoke;;
  chain) chain;;
  all) smoke && chain || { echo "smoke failed; chain not started"; exit 1; };;
  *) echo "usage: $0 smoke|chain|all"; exit 2;;
esac
