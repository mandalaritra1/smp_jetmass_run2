#!/usr/bin/env bash
# lxplus condor payload: shower ONE Herwig-CH3 Z+jet jet-pT slice with ONE seed,
# writing a per-event gen ntuple (jet_pt m_u m_g rho_u rho_g weight + xsec/sumw
# footer) alongside the yoda. Runs NATIVELY on an el9 worker (lxplus is el9), so
# no apptainer wrapper -- just source the LCG_107 view from cvmfs.
#   herwig_ch3_job_lxplus.sh <slice_idx> <seed> <nevents>
set -euo pipefail
SLICE=${1:?slice}; SEED=${2:?seed}; N=${3:?nevents}

VIEW=${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt}
set +u; source "$VIEW/setup.sh"; set -u
export HERWIGPATH="$VIEW/share/Herwig"                       # repo + snippets
export LHAPDF_DATA_PATH="${LHAPDF_DATA_PATH:-}:/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current"
export RIVET_ANALYSIS_PATH="$PWD:${RIVET_ANALYSIS_PATH:-}"
export CH3_NTUPLE="herwig_ch3_s${SLICE}_seed${SEED}.ntuple"  # routine dumps per-event rows

echo "host=$(hostname) slice=$SLICE seed=$SEED N=$N view=$VIEW"
echo "rivet $(rivet --version 2>&1) | Herwig $(Herwig --version 2>&1 | head -1)"

LO=(120 200 300 450); HI=(200 300 450 100000)
rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
ln -sf "$HERWIGPATH/HerwigDefaults.rpo" .                    # Herwig has no --repo flag

RUN="hw_ch3_s${SLICE}_${SEED}"
sed -e "s/__MINKT__/${LO[$SLICE]}/" -e "s/__MAXKT__/${HI[$SLICE]}/" -e "s/__RUN__/$RUN/" \
    herwig_ch3_slice.in.tmpl > job.in
Herwig read -i "$HERWIGPATH" job.in
Herwig run "${RUN}.run" -N "$N" -s "$SEED"

yfile=$(ls "${RUN}"-S*.yoda 2>/dev/null | head -1 || true)
[ -z "$yfile" ] && yfile=$(ls "${RUN}.yoda" 2>/dev/null | head -1 || true)
[ -n "$yfile" ] || { echo "NO YODA produced"; exit 2; }
mv "$yfile" "herwig_ch3_s${SLICE}_seed${SEED}.yoda"
[ -s "$CH3_NTUPLE" ] || { echo "NO NTUPLE produced"; exit 3; }
echo "done: yoda + ntuple ($(grep -vc '^#' "$CH3_NTUPLE") event rows)"
