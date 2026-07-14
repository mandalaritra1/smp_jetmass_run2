#!/bin/bash
# ===========================================================================
# Brief constraint #4 / acceptance checklist: for each bin read xqcut and the max
# parton multiplicity from the gridpack's run_card.dat, so we can confirm the
# fragment's FxFx/MLM merging (njetsmax 4, Qcut 19) matches — or flag a deviation.
#
# Run on lxplus (gridpacks live on cvmfs). Reads inputs/gridpacks.txt.
# ===========================================================================
set -euo pipefail
MAP="$(dirname "$0")/../inputs/gridpacks.txt"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

printf "%-16s %-8s %-10s %s\n" BIN xqcut maxjets "(from run_card.dat)"
while read -r BIN PATH_; do
  [[ "$BIN" =~ ^#|^$ ]] && continue
  [[ "$PATH_" == __NEEDS_INPUT* ]] && { printf "%-16s %s\n" "$BIN" "<no path yet>"; continue; }
  # run_card.dat lives inside the gridpack tarball; pull just that file.
  RC="$(tar -tJf "$PATH_" 2>/dev/null | grep -m1 'run_card.dat' || true)"
  if [ -z "$RC" ]; then printf "%-16s %s\n" "$BIN" "<run_card.dat not found in tarball>"; continue; fi
  tar -xJf "$PATH_" -C "$TMP" "$RC"
  F="$TMP/$RC"
  XQCUT=$(grep -iE '=\s*xqcut' "$F" | grep -oE '[0-9.]+' | head -1)
  # max parton multiplicity: highest 'pX:' / final-state count proxy from the proc; fall back to maxjetflavor/ickkw hints
  MAXJ=$(grep -iE '=\s*(maxjetflavor|ickkw)' "$F" | head -2 | tr '\n' ' ')
  printf "%-16s %-8s %-10s %s\n" "$BIN" "${XQCUT:-?}" "see-proc_card" "$MAXJ"
done < "$MAP"

echo
echo "NOTE: njetsmax = (number of extra partons in the process), read from proc_card.dat"
echo "      (the 'add process p p > ... @N' lines), not run_card. Qcut should be >= xqcut."
echo "      Standard DYJetsToLL_M-50_HT-* => xqcut 19, 4 partons, njetsmax 4."
