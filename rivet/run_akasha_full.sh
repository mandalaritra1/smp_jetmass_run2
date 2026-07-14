#!/usr/bin/env bash
# Full 4-slice modelling-variation run on a native-amd64 box at N events/variation,
# capped at MAXJOBS concurrent showers. Distinct LHE dirs (zjet_<TAG>_s*) and output
# dir (out/slices_<TAG>/) so it does NOT clash with the 20k run in out/slices/.
# Independent seeds (7000+idx) so this set can be inverse-variance AVERAGED with the
# LPC 50k on the overlapping high-pT slices (s2,s3) -> ~100k-equiv. Vincia LAST.
#
#   ./run_akasha_full.sh [N=50000] [MAXJOBS=4] [TAG=50k] [SLICES="0 1 2 3"]
set -uo pipefail
N=${1:-50000}; MAXJOBS=${2:-4}; TAG=${3:-50k}; SLICES=${4:-"0 1 2 3"}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
MG5_IMG=scailfin/madgraph5-amc-nlo:mg5_amc3.5.1
RIV_IMG=hepstore/rivet-pythia:latest
DRUN=(docker run --rm -v "$HERE":/work -w /work)
MGRUN=(docker run --rm -v "$HERE":/work -w /work --entrypoint bash "$MG5_IMG")
ODIR=out/slices_$TAG
mkdir -p "$ODIR" logs

LO=(120 200 300 450); HI=(200 300 450 100000)

# 1. generate (cache) LHE per slice -----------------------------------------
for IDX in $SLICES; do
  lo=${LO[$IDX]}; hi=${HI[$IDX]}
  OUT=zjet_${TAG}_s$IDX
  LHE="$OUT/Events/run_01/unweighted_events.lhe"
  if [ -f "$LHE" ]; then echo "### LHE $TAG s$IDX cached, skip gen ###"; continue; fi
  rm -rf "$OUT"
  card=$ODIR/_mg5_s$IDX.dat
  { echo "set nb_core 1"; echo "generate p p > l+ l- j"; echo "output $OUT";
    echo "launch $OUT"; echo "set nevents $N"; echo "set iseed $((7000+IDX))";
    echo "set ebeam1 6500"; echo "set ebeam2 6500"; echo "set ptj $lo";
    [ "$hi" -lt 100000 ] && echo "set ptjmax $hi";
    echo "set mmll 60"; echo "set mmllmax 120"; echo "set systematics_program none"; } > "$card"
  echo "### gen LHE $TAG slice $IDX (ptj $lo-$hi, N=$N) ###"
  "${MGRUN[@]}" -c "mg5_aMC $card" > logs/gen_${TAG}_s$IDX.log 2>&1
  [ -f "$LHE.gz" ] && gunzip -f "$LHE.gz"
  [ -f "$LHE" ] || { echo "no LHE $TAG s$IDX (see logs/gen_${TAG}_s$IDX.log)"; exit 1; }
  echo "### LHE $TAG s$IDX done ###"
done

# build plugin + driver once
echo "### build plugin + pythia_rivet ###"
"${DRUN[@]}" "$RIV_IMG" bash -lc '
  export RIVET_ANALYSIS_PATH=/work
  [ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
  [ -x pythia_rivet ] || g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
      $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet
' > logs/build_${TAG}.log 2>&1

var_settings() { case "$1" in
  pythia) : ;;
  pythia_vincia)   printf 'PartonShowers:model = 2\n' ;;
  pythia_cr1)      printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 1\nBeamRemnants:remnantMode = 1\n' ;;
  pythia_cr2)      printf 'ColourReconnection:reconnect = on\nColourReconnection:mode = 2\n' ;;
  pythia_fragsoft) printf 'StringZ:aLund = 0.78\nStringZ:bLund = 1.18\nStringPT:sigma = 0.365\n' ;;
  pythia_fraghard) printf 'StringZ:aLund = 0.58\nStringZ:bLund = 0.78\nStringPT:sigma = 0.305\n' ;;
esac; }

shower_one() {  # IDX name
  local IDX=$1 name=$2
  local LHE="zjet_${TAG}_s$IDX/Events/run_01/unweighted_events.lhe"
  local cmnd=$ODIR/_${name}_s$IDX.cmnd
  { printf 'Beams:frameType = 4\nBeams:LHEF = %s\nPartonLevel:MPI = on\nHadronLevel:all = on\n' "$LHE"
    var_settings "$name"; } > "$cmnd"
  "${DRUN[@]}" "$RIV_IMG" bash -lc "export RIVET_ANALYSIS_PATH=/work; ./pythia_rivet '$cmnd' '$ODIR/mglo_${name}_s$IDX.yoda' $N" \
      > "logs/shw_${TAG}_${name}_s$IDX.log" 2>&1
}

# 2. job list: FAST variations first (all slices), vincia LAST ---------------
JOBS=()
for name in pythia pythia_cr1 pythia_cr2 pythia_fragsoft pythia_fraghard; do
  for IDX in $SLICES; do JOBS+=("$IDX:$name"); done
done
for IDX in $SLICES; do JOBS+=("$IDX:pythia_vincia"); done

echo "### showering ${#JOBS[@]} jobs, $MAXJOBS at a time (vincia last) -> $ODIR ###"
running=0
for job in "${JOBS[@]}"; do
  IDX=${job%%:*}; name=${job#*:}
  ( t=$(date +%s); shower_one "$IDX" "$name"
    echo "[$(date +%H:%M:%S)] done $name s$IDX ($(( $(date +%s)-t ))s)" >> logs/akasha_${TAG}_progress.log ) &
  running=$((running+1))
  [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }
done
wait
echo "### $TAG DONE -> $ODIR/mglo_*.yoda ###"
ls -la $ODIR/mglo_*.yoda | tail