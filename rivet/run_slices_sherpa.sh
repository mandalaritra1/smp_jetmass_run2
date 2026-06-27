#!/usr/bin/env bash
# Sherpa in jet-pT SLICES ([PT, 93, lo, hi] window selectors), stitched into a
# flat-statistics prediction -> out/sherpa_sliced.yoda. Each slice runs in its
# own subdir (isolated Comix integration), writes HepMC, and is analysed by
# rivet. Intended to run inside hepstore/rivet-sherpa (Sherpa+Rivet).
#
#   ./run_slices_sherpa.sh [events_per_slice]
set -uo pipefail

NPS=${1:-5000}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
mkdir -p out/slices

[ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
[ -d "$HERE/.lhapdf/PDF4LHC21_40_pdfas" ] && \
  export LHAPDF_DATA_PATH="$HERE/.lhapdf:$(lhapdf-config --datadir 2>/dev/null):${LHAPDF_DATA_PATH:-}"

EDGES=(120 200 300 450 100000)

slices=()
for ((i=0; i<${#EDGES[@]}-1; i++)); do
  lo=${EDGES[i]}; hi=${EDGES[i+1]}
  d="out/slices/sherpa_run_s${i}"; rm -rf "$d"; mkdir -p "$d"
  cat > "$d/Sherpa.yaml" <<EOF
BEAMS: 2212
BEAM_ENERGIES: 6500
PDF_LIBRARY: LHAPDFSherpa
PDF_SET: PDF4LHC21_40_pdfas
ME_GENERATORS: [Comix]
RANDOM_SEED: $((12345 + i))
PROCESSES:
- 93 93 -> 11 -11 93:
    Order: {QCD: 1, EW: 2}
SELECTORS:
- [Mass, 11, -11, 60, 120]
- [PT, 93, ${lo}, ${hi}]
EVENT_OUTPUT: HepMC3_GenEvent[sherpa]
EOF
  echo "### sherpa slice $i: jet pT [$lo, $hi]  ($NPS events) ###"
  ( cd "$d" && Sherpa Sherpa.yaml -e "$NPS" )
  rivet -a CMS_ZJET_JETMASS "$d/sherpa" -o "out/slices/sherpa_s${i}.yoda"
  rm -f "$d/sherpa"
  slices+=("out/slices/sherpa_s${i}.yoda")
done

python3 gen/merge_slices.py out/sherpa_sliced.yoda "${slices[@]}"
echo "wrote out/sherpa_sliced.yoda"
