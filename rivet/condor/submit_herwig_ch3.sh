#!/usr/bin/env bash
# Build joblist.txt (slice x seed grid) and submit herwig_ch3.jdl on LPC condor.
#   ./submit_herwig_ch3.sh [SEEDS_PER_SLICE=10] [NEVENTS_PER_JOB=50000] [SLICES="0 1 2 3"]
# Total events/slice = SEEDS_PER_SLICE * NEVENTS_PER_JOB (e.g. 10*50k = 500k/slice).
set -euo pipefail
SEEDS=${1:-10}; N=${2:-50000}; SLICES=${3:-"0 1 2 3"}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
mkdir -p logs ../out/herwig_ch3

: > joblist.txt
for s in $SLICES; do
  for ((k=1; k<=SEEDS; k++)); do
    # unique seed per (slice,seed): 1000*slice + k  -> disjoint event streams
    echo "$s, $((1000*s + k)), $N" >> joblist.txt
  done
done
njobs=$(wc -l < joblist.txt)
echo "joblist.txt: $njobs jobs (slices [$SLICES] x $SEEDS seeds x $N evt)"

command -v condor_submit >/dev/null || { echo "condor_submit not found -- run on an LPC login node"; exit 1; }
condor_submit herwig_ch3.jdl
echo "submitted. watch:  condor_q ; tail -f logs/condor.log"
echo "when done, combine: ./combine_herwig_ch3.sh"
