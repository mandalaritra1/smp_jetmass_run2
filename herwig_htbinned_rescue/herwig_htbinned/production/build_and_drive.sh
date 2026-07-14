#!/bin/bash
# Runs INSIDE cmssw-el7. Builds CMSSW_10_6_28 (if absent), stages the GenProduction
# package, scram b, then cmsDriver for the bins passed as args -> prod_<BIN>.py.
set -e
export SCRAM_ARCH=slc7_amd64_gcc700
source /cvmfs/cms.cern.ch/cmsset_default.sh
BASE=~/herwig_htbinned
REL=$BASE/CMSSW_10_6_28
if [ ! -d "$REL" ]; then
  echo "### scram project CMSSW_10_6_28"
  ( cd "$BASE" && scram project CMSSW_10_6_28 )
fi
# stage GenProduction package (fragments + customise) into src
mkdir -p "$REL/src/Configuration/GenProduction/python"
cp "$BASE"/Configuration/GenProduction/python/*.py "$REL/src/Configuration/GenProduction/python/"
cd "$REL/src"
eval "$(scram runtime -sh)"
echo "### scram b (python-only, fast)"
scram b >/dev/null 2>&1 || scram b
cd "$BASE/cfgs" 2>/dev/null || { mkdir -p "$BASE/cfgs"; cd "$BASE/cfgs"; }
for BIN in "$@"; do
  echo "==================== cmsDriver HT-$BIN ===================="
  cmsDriver.py Configuration/GenProduction/python/Herwig7_DY_HT-${BIN}_CH3_cff.py \
    --python_filename prod_HT-${BIN}.py --fileout file:out.root \
    --mc --eventcontent NANOAODGEN --datatier NANOAODSIM --step LHE,GEN,NANOGEN \
    --conditions auto:mc --beamspot Realistic25ns13TeVEarly2018Collision --era Run2_2018 \
    --customise Configuration/GenProduction/nanogen_addsubjets.addSubGenJetAK8 \
    --nThreads 1 -n 1000 --no_exec 2>&1 | tail -6
  echo "--- self-containment check (should be NO GenProduction import at runtime):"
  grep -c "Configuration.GenProduction" prod_HT-${BIN}.py || true
  echo "--- customise applied marker:"
  grep -c "ak8GenJetsNoNuSoftDrop\|genSubJetAK8Table" prod_HT-${BIN}.py || true
  ls -la prod_HT-${BIN}.py
done
