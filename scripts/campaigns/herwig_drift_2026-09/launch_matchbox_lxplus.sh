#!/usr/bin/env bash
# NLO Matchbox Herwig7 DY (no LHE, no MLM) zjet minimal_rho, 2018, on CERN lxplus
# HTCondor through dask-lxplus (2026-09-18; hedge against the LPC driver stall).
#
#   ./launch_matchbox_lxplus.sh smoke   # first file, local futures
#   ./launch_matchbox_lxplus.sh run     # the 620-file sample, fixed pool from the config's n_workers
#
# Logs in logs/, output in outputs/minimal_rho_herwig_matchbox_r2_2018.pkl.
cd "$(dirname "$0")" || exit 1
mkdir -p logs outputs
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"
export DASK_LXPLUS_JOB_FLAVOUR="${DASK_LXPLUS_JOB_FLAVOUR:-workday}"
export DASK_LXPLUS_LOG_DIR="${DASK_LXPLUS_LOG_DIR:-$HOME/dask_lxplus_logs}"
env_run() {  # env_run <config-stem>
  bash -c "source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1; \
           export PATH=\$HOME/.local/bin:\$PATH; cd $PWD; python -u scripts/run_analysis_cli.py --config configs/$1.json"
}
export -f env_run
case "$1" in
  smoke)
    echo "=== herwig_matchbox smoke start $(date -u +%FT%TZ)"
    env_run zjet_herwig_matchbox_2018_minimal_rho_lxplus_smoke > logs/smoke_zjet_herwig_matchbox_2018.log 2>&1
    echo "=== herwig_matchbox smoke exit $? $(date -u +%FT%TZ)";;
  run)
    setsid nohup bash -c 'echo "=== zjet herwig_matchbox 2018 start $(date -u +%FT%TZ)";
        env_run zjet_herwig_matchbox_2018_minimal_rho_lxplus > logs/campaign_zjet_herwig_matchbox_2018.log 2>&1;
        echo "=== zjet herwig_matchbox 2018 exit $? $(date -u +%FT%TZ)"' > logs/campaign_zjet_herwig_matchbox_chain.log 2>&1 < /dev/null &
    echo "herwig_matchbox started, pid $!";;
  *) echo "usage: $0 smoke|run"; exit 2;;
esac
