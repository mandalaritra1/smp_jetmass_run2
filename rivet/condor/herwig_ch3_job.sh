#!/usr/bin/env bash
# Condor payload: shower ONE Herwig-CH3 Z+jet jet-pT slice with ONE seed.
# Runs on an LPC el9 worker; Herwig/Rivet/LHAPDF come from cvmfs (LCG view +
# central lhapdfsets), so the input sandbox only ships the routine + template.
#   herwig_ch3_job.sh <slice_idx> <seed> <nevents>
# Produces: herwig_ch3_s<slice>_seed<seed>.yoda  (transferred back by condor).
set -euo pipefail

SLICE=${1:?slice idx}; SEED=${2:?seed}; N=${3:?nevents}

VIEW=${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt}
set +u; source "$VIEW/setup.sh"; set -u
# the LCG setup.sh does NOT set HERWIGPATH -> Herwig can't find HerwigDefaults.rpo
# or snippets/*.in. Point it at the view's Herwig share dir (repo + read search path).
export HERWIGPATH="$VIEW/share/Herwig"
# the view already sets LHAPDF_DATA_PATH (grid repo + lhapdf.conf dir); APPEND the
# central grid repo rather than replace it, or lhapdf.conf can't be found.
export LHAPDF_DATA_PATH="${LHAPDF_DATA_PATH:-}:/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current"
export RIVET_ANALYSIS_PATH="$PWD:${RIVET_ANALYSIS_PATH:-}"

echo "host=$(hostname) slice=$SLICE seed=$SEED N=$N view=$VIEW"
echo "rivet $(rivet --version 2>&1) | Herwig $(Herwig --version 2>&1 | head -1)"

# disjoint jet-pT slice edges [GeV] (last slice open-ended)
LO=(120 200 300 450); HI=(200 300 450 100000)
minkt=${LO[$SLICE]}; maxkt=${HI[$SLICE]}

rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc

# Herwig has no --repo flag in this build; it looks for HerwigDefaults.rpo in CWD
# before the (broken, compiled-in jenkins) default -> symlink the view's repo here.
ln -sf "$HERWIGPATH/HerwigDefaults.rpo" .

RUN="hw_ch3_s${SLICE}_${SEED}"          # Herwig runname (kept distinct from final)
sed -e "s/__MINKT__/$minkt/" -e "s/__MAXKT__/$maxkt/" -e "s/__RUN__/$RUN/" \
    herwig_ch3_slice.in.tmpl > job.in

Herwig read -i "$HERWIGPATH" job.in
Herwig run "${RUN}.run" -N "$N" -s "$SEED"

# Herwig+Rivet writes <run>-S<seed>.yoda (fallback: <run>.yoda)
yfile=$(ls "${RUN}"-S*.yoda 2>/dev/null | head -1 || true)
[ -z "$yfile" ] && yfile=$(ls "${RUN}.yoda" 2>/dev/null | head -1 || true)
[ -n "$yfile" ] || { echo "NO YODA produced"; exit 2; }
mv "$yfile" "herwig_ch3_s${SLICE}_seed${SEED}.yoda"
echo "wrote herwig_ch3_s${SLICE}_seed${SEED}.yoda"
