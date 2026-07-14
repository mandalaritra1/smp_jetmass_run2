#!/bin/bash
set -e
BIN=$1; SEED=$2; NEVT=$3; OUTTAG=$4
SB=$PWD
export SCRAM_ARCH=slc7_amd64_gcc700
source /cvmfs/cms.cern.ch/cmsset_default.sh
[ -d CMSSW_10_6_28 ] || scram project CMSSW_10_6_28
mkdir -p CMSSW_10_6_28/src/Configuration/GenProduction/python
cp "$SB/nanogen_addsubjets.py" CMSSW_10_6_28/src/Configuration/GenProduction/python/
cd CMSSW_10_6_28/src
eval "$(scram runtime -sh)"
scram b >/dev/null 2>&1 || scram b
cd "$SB"
cat "prod_${BIN}.py" job_tail.py > job_cfg.py
export SEED NEVT OUTTAG
echo "[inner] cmsRun job_cfg.py  NEVT=$NEVT SEED=$SEED"
cmsRun job_cfg.py
