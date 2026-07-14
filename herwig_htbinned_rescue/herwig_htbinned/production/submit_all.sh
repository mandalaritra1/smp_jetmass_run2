#!/bin/bash
# Full Herwig7-CH3 production. Uniform ~300k OUTPUT events/bin:
#   nEvents=8000/job * 120 jobs/bin * ~0.32 merging acceptance ~= 307k written/bin.
# Distinct SEEDBASE block per bin so per-job seeds never collide.
cd ~/herwig_htbinned/production
NEVT=${NEVT:-8000}; NJOBS=${NJOBS:-120}
i=0
for BIN in HT-200to400 HT-400to600 HT-600to800 HT-800to1200 HT-1200to2500 HT-2500toInf; do
  SEEDBASE=$((3000000 + i*100000))
  mkdir -p logs/$BIN
  echo "### submit $BIN  NEVT=$NEVT NJOBS=$NJOBS SEEDBASE=$SEEDBASE"
  condor_submit BIN=$BIN NEVT=$NEVT SEEDBASE=$SEEDBASE NJOBS=$NJOBS submit.jdl 2>&1 | tail -1
  i=$((i+1))
done
