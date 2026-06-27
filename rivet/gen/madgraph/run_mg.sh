#!/usr/bin/env bash
# Native driver for the two MadGraph-LO setups (lxplus/LPC; mg5_aMC + Pythia8 +
# Herwig + Rivet all on PATH, e.g. an LCG view).
#
#   ./gen/madgraph/run_mg.sh [nevents]
#
# 1. MadGraph LO Z+jet -> LHE
# 2. shower with Pythia8 -> out/mglo_pythia.yoda      (via pythia_rivet)
# 3. shower with Herwig7 -> out/mglo_herwig.yoda      (best-effort)
set -uo pipefail

N=${1:-30000}
HERE=$(cd "$(dirname "$0")/../.." && pwd); cd "$HERE"   # rivet/
export RIVET_ANALYSIS_PATH="$HERE:${RIVET_ANALYSIS_PATH:-}"
mkdir -p out

# build plugin + Pythia driver if needed
[ -f RivetCMS_ZJET_JETMASS.so ] || rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
[ -x pythia_rivet ] || g++ gen/pythia_rivet.cc $(pythia8-config --cxxflags --libs) \
    $(rivet-config --cppflags --ldflags --libs) -std=c++17 -o pythia_rivet

# 1. Matrix element -> LHE -------------------------------------------------
echo "### MadGraph LHE generation ###"
mg5_aMC gen/madgraph/mg5_zjet.dat
LHE=$(ls zjet_mglo/Events/run_01/unweighted_events.lhe.gz 2>/dev/null | head -1)
[ -n "$LHE" ] && gunzip -f "$LHE" && LHE=${LHE%.gz}
[ -z "${LHE:-}" ] && LHE=$(ls zjet_mglo/Events/run_01/unweighted_events.lhe 2>/dev/null | head -1)
[ -z "${LHE:-}" ] && { echo "no LHE produced" >&2; exit 1; }
echo "LHE: $LHE"

# 2. Pythia8 shower --------------------------------------------------------
echo "### Pythia8 shower ###"
cat > gen/madgraph/_lhe_pythia.cmnd <<EOF
Beams:frameType = 4
Beams:LHEF = $LHE
PartonLevel:MPI = on
HadronLevel:all = on
EOF
./pythia_rivet gen/madgraph/_lhe_pythia.cmnd out/mglo_pythia.yoda "$N" \
  && echo "wrote out/mglo_pythia.yoda"

# 3. Herwig7 shower (best-effort) ------------------------------------------
echo "### Herwig7 shower ###"
HWSHARE="$(cd "$(dirname "$(command -v Herwig)")/../share/Herwig" && pwd)"
sed "s#__LHE__#$LHE#" gen/madgraph/shower_herwig.in > gen/madgraph/_sh.in
if Herwig read --repo="$HWSHARE/HerwigDefaults.rpo" -I "$HWSHARE" gen/madgraph/_sh.in \
   && Herwig run mglo_herwig.run -N "$N" -s 12345; then
  mv "mglo_herwig-S12345.yoda" out/mglo_herwig.yoda
  rm -f mglo_herwig-S12345.*
  echo "wrote out/mglo_herwig.yoda"
else
  echo "WARN: Herwig LHE shower failed; mglo_pythia still produced" >&2
fi
rm -f gen/madgraph/_lhe_pythia.cmnd gen/madgraph/_sh.in
echo "### MadGraph done ###"
