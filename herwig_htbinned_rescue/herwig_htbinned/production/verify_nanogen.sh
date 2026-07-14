#!/bin/bash
# ===========================================================================
# Decide the release question. Run on lxplus/cmslpc. Checks, for a given release,
# whether NanoGEN exists (step keyword + customize) and whether the Herwig7 CH3
# settings the fragment imports are present.
#
#   ./verify_nanogen.sh CMSSW_10_2_6
#   ./verify_nanogen.sh CMSSW_10_6_28
#   ./verify_nanogen.sh CMSSW_13_0_13
# ===========================================================================
set -uo pipefail
REL="${1:?usage: verify_nanogen.sh <CMSSW_RELEASE>}"

source /cvmfs/cms.cern.ch/cmsset_default.sh
# pick a sane default arch per release line; override with SCRAM_ARCH=... if needed
case "$REL" in
  CMSSW_10_2_*) export SCRAM_ARCH="${SCRAM_ARCH:-slc7_amd64_gcc700}";;
  CMSSW_10_6_*) export SCRAM_ARCH="${SCRAM_ARCH:-slc7_amd64_gcc700}";;
  CMSSW_12_*)   export SCRAM_ARCH="${SCRAM_ARCH:-el8_amd64_gcc10}";;
  CMSSW_13_*)   export SCRAM_ARCH="${SCRAM_ARCH:-el8_amd64_gcc11}";;
esac
echo "REL=$REL SCRAM_ARCH=$SCRAM_ARCH"

WORK="$(mktemp -d)"; cd "$WORK"
scram project CMSSW "$REL" >/dev/null 2>&1 || { echo "!! cannot set up $REL for $SCRAM_ARCH"; exit 1; }
cd "$REL/src"; eval "$(scram runtime -sh)"

echo "--- nanogen_cff present?"
NG="$CMSSW_RELEASE_BASE/python/PhysicsTools/NanoAOD/nanogen_cff.py"
[ -f "$NG" ] && echo "  YES: $NG" || echo "  NO  (no PhysicsTools/NanoAOD/nanogen_cff.py)"
[ -f "$NG" ] && grep -q customizeNanoGEN "$NG" && echo "  customizeNanoGEN: YES" || echo "  customizeNanoGEN: (not found)"

echo "--- NANOGEN step keyword known to cmsDriver?"
python3 -c "from Configuration.Applications.ConfigBuilder import defaultOptions" 2>/dev/null && \
  python3 - <<'PY' 2>/dev/null || echo "  (could not introspect; try: cmsDriver.py --help | grep -i nanogen)"
import Configuration.StandardSequences.Reconstruction as _
try:
    from Configuration.Applications.ConfigBuilder import ConfigBuilder
    print("  step table import OK — grep cmsDriver.py --help for NANOGEN")
except Exception as e:
    print("  introspection failed:", e)
PY

echo "--- Herwig7 CH3 settings the fragment imports present?"
for m in Herwig7LHECommonSettings Herwig7StableParticlesForDetector \
         Herwig7CH3TuneSettings Herwig7PSWeightsSettings Herwig7CommonMergingSettings; do
  F="$CMSSW_RELEASE_BASE/python/Configuration/Generator/Herwig7Settings/${m}_cfi.py"
  [ -f "$F" ] && echo "  YES $m" || echo "  NO  $m"
done

echo "--- el compatibility: $SCRAM_ARCH on an el9 worker may need a container"
echo "    (slc7_* -> run inside  cmssw-el7 ;  el8_* -> cmssw-el8 ; el9_* native)"
rm -rf "$WORK"
