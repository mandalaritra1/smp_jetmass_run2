#!/bin/bash
set -e
BIN=$1; PROC=$2; NEVT=$3; SEEDBASE=$4
SEED=$(( SEEDBASE + PROC ))
OUTTAG="Herwig7_DY_${BIN}_CH3_${SEED}"
echo "[run_job] host=$(hostname) BIN=$BIN PROC=$PROC NEVT=$NEVT SEED=$SEED"
echo "[run_job] KRB5CCNAME=$KRB5CCNAME  (condor-delivered cred)"
# GEN+NANOGEN inside the el7 container (release is slc7_amd64_gcc700)
/cvmfs/cms.cern.ch/common/cmssw-el7 -- bash inner.sh "$BIN" "$SEED" "$NEVT" "$OUTTAG"
ls -lh "${OUTTAG}.root"
# stage to CERN EOS (back on el9; krb cred from MY.SendCredential lives here)
export EOS_MGM_URL=root://eosuser.cern.ch
DEST="root://eosuser.cern.ch//eos/user/a/amandal/herwig_ch3_dy/${BIN}/${OUTTAG}.root"
xrdcp -f -N "${OUTTAG}.root" "$DEST"
echo "[run_job] staged -> $DEST"
