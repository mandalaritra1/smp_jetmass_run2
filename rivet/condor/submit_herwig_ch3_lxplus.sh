#!/usr/bin/env bash
# Build joblist.txt (slice x seed grid) and submit herwig_ch3_lxplus.jdl on lxplus.
#   ./submit_herwig_ch3_lxplus.sh [SEEDS_PER_SLICE=20] [NEVENTS_PER_JOB=50000] [SLICES="0 1 2 3"]
# Default 20 seeds x 50k x 4 slices = ~4M events (80 jobs). Needs a valid krb5 ticket
# (kinit -kt ~/private/…keytab amandal@CERN.CH) at submit time.
set -euo pipefail
SEEDS=${1:-20}; N=${2:-50000}; SLICES=${3:-"0 1 2 3"}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
mkdir -p logs ../out/herwig_ch3

: > joblist.txt
for s in $SLICES; do
  for ((k=1; k<=SEEDS; k++)); do
    echo "$s, $((1000*s + k)), $N" >> joblist.txt   # seed = 1000*slice + k (disjoint streams)
  done
done
echo "joblist.txt: $(wc -l < joblist.txt) jobs (slices [$SLICES] x $SEEDS seeds x $N evt)"

command -v condor_submit >/dev/null || { echo "condor_submit not found -- run on lxplus"; exit 1; }
condor_submit herwig_ch3_lxplus.jdl
echo "submitted. watch: condor_q ; combine: ./combine_herwig_ch3.sh (yodas) + the ntuple loader"
