#!/usr/bin/env bash
# Local convenience wrapper: run run_all.sh for one generator inside the matching
# amd64-emulated hepstore Docker image (for Macs / machines without a native
# generator stack). On cvmfs/LPC just call ./run_all.sh directly instead.
#
#   ./run_local.sh <pythia|vincia|herwig|sherpa> [nevents]
set -euo pipefail

GEN=${1:?usage: run_local.sh <pythia|vincia|herwig|sherpa> [nevents]}
N=${2:-20000}
HERE=$(cd "$(dirname "$0")" && pwd)

case "$GEN" in
  pythia|vincia) IMG=hepstore/rivet-pythia:latest ;;
  herwig)        IMG=hepstore/rivet-herwig:latest ;;
  sherpa)        IMG=hepstore/rivet-sherpa:latest ;;
  *) echo "unknown generator: $GEN" >&2; exit 1 ;;
esac

docker run --rm --platform linux/amd64 -v "$HERE":/work -w /work "$IMG" \
  bash -lc "./run_all.sh $GEN $N"
