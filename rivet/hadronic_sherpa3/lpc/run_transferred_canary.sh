#!/usr/bin/env bash
# Run one native LPC canary in Condor scratch and return one result archive.
set -eo pipefail

PAIR_SPLIT_ARCHIVE=${1:?missing source archive}
PAIR_SPLIT_ARCHIVE_SHA256=${2:?missing source archive sha256}
PAIR_SPLIT_EVENTS=${3:?missing event count}
PAIR_SPLIT_SLICE_INDEX=${4:?missing slice index}
PAIR_SPLIT_PT_LOW=${5:?missing slice lower edge}
PAIR_SPLIT_PT_HIGH=${6:?missing slice upper edge}
PAIR_SPLIT_SHOWER=${7:?missing shower}
PAIR_SPLIT_RESULT_ARCHIVE=${8:?missing result archive name}
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

set +e
pair_split_source/hadronic_sherpa3/scripts/run_canary.sh \
  "$PAIR_SPLIT_RUN_DIRECTORY" "$PAIR_SPLIT_EVENTS" \
  "$PAIR_SPLIT_SLICE_INDEX" "$PAIR_SPLIT_PT_LOW" "$PAIR_SPLIT_PT_HIGH" \
  "$PAIR_SPLIT_SHOWER"
PAIR_SPLIT_EXIT_CODE=$?
set -e

printf '%s\n' "$PAIR_SPLIT_EXIT_CODE" \
  > "$PAIR_SPLIT_RUN_DIRECTORY/canary_exit_code.txt"
tar -czf "$PAIR_SPLIT_RESULT_ARCHIVE" "$PAIR_SPLIT_RUN_DIRECTORY"
exit "$PAIR_SPLIT_EXIT_CODE"
