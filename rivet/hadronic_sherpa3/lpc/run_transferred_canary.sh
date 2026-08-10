#!/usr/bin/env bash
# Run one native LPC canary unit in Condor scratch and return one archive.
#
# Positional arguments are the physics configuration; the optional knobs
# (QCUT, CHUNK_INDEX, MODE) arrive through the Condor `environment` setting so
# the positional interface stays backward compatible with the v5 canary.
set -eo pipefail

PAIR_SPLIT_ARCHIVE=${1:?missing source archive}
PAIR_SPLIT_ARCHIVE_SHA256=${2:?missing source archive sha256}
PAIR_SPLIT_EVENTS=${3:?missing event count}
PAIR_SPLIT_SLICE_INDEX=${4:?missing slice index}
PAIR_SPLIT_PT_LOW=${5:?missing slice lower edge}
PAIR_SPLIT_PT_HIGH=${6:?missing slice upper edge}
PAIR_SPLIT_SHOWER=${7:?missing shower}
PAIR_SPLIT_RESULT_ARCHIVE=${8:?missing result archive name}
PAIR_SPLIT_GRID_ARCHIVE=${9:-}
PAIR_SPLIT_RUN_DIRECTORY=${PAIR_SPLIT_RESULT_ARCHIVE%.tar.gz}
PAIR_SPLIT_LCG_VIEW=/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt

printf '%s  %s\n' "$PAIR_SPLIT_ARCHIVE_SHA256" "$PAIR_SPLIT_ARCHIVE" \
  | sha256sum --check --strict
mkdir pair_split_source
tar -xzf "$PAIR_SPLIT_ARCHIVE" -C pair_split_source
test -x pair_split_source/hadronic_sherpa3/scripts/run_canary.sh

source "$PAIR_SPLIT_LCG_VIEW/setup.sh"
export LCG_VIEW="$PAIR_SPLIT_LCG_VIEW"
export SOURCE_BUNDLE_SHA256="$PAIR_SPLIT_ARCHIVE_SHA256"
export QCUT=${QCUT:-20}
export CHUNK_INDEX=${CHUNK_INDEX:-0}
export MODE=${MODE:-full}
export EVTMODE=${EVTMODE:-PartiallyUnweighted}

# Argument 9 is the grid archive name.  Its meaning follows MODE: an
# integrate job PRODUCES that file, a chunk job CONSUMES it.  Naming it
# explicitly on both sides is what lets a chunk find the grid its
# configuration built -- deriving the name from the run directory silently
# produced a name no chunk would ever ask for.
if [[ -n "$PAIR_SPLIT_GRID_ARCHIVE" && "$PAIR_SPLIT_GRID_ARCHIVE" != "NONE" ]]; then
  if [[ "$MODE" == "integrate" ]]; then
    export GRID_OUT="$PWD/$PAIR_SPLIT_GRID_ARCHIVE"
  else
    export GRID_IN="$PWD/$PAIR_SPLIT_GRID_ARCHIVE"
  fi
elif [[ "$MODE" == "integrate" ]]; then
  echo "FATAL: MODE=integrate needs a grid output name as argument 9" >&2
  exit 29
fi

set +e
pair_split_source/hadronic_sherpa3/scripts/run_canary.sh \
  "$PAIR_SPLIT_RUN_DIRECTORY" "$PAIR_SPLIT_EVENTS" \
  "$PAIR_SPLIT_SLICE_INDEX" "$PAIR_SPLIT_PT_LOW" \
  "$PAIR_SPLIT_PT_HIGH" "$PAIR_SPLIT_SHOWER"
PAIR_SPLIT_EXIT_CODE=$?
set -e

printf '%s\n' "$PAIR_SPLIT_EXIT_CODE" \
  > "$PAIR_SPLIT_RUN_DIRECTORY/canary_exit_code.txt"
# The HepMC event stream dominates the archive (~60 MB per 1k events) and is
# only needed to replay Rivet.  Drop it on success to keep stage-out small,
# but ALWAYS keep it on failure: cluster 85063614 lost its Rivet stage to a
# path bug, and the surviving event stream is what saved 6 h of generation.
if [[ "$PAIR_SPLIT_EXIT_CODE" -eq 0 && "${KEEP_EVENTS:-0}" != "1" ]]; then
  rm -f "$PAIR_SPLIT_RUN_DIRECTORY/events"
fi
tar -czf "$PAIR_SPLIT_RESULT_ARCHIVE" "$PAIR_SPLIT_RUN_DIRECTORY"
exit "$PAIR_SPLIT_EXIT_CODE"
