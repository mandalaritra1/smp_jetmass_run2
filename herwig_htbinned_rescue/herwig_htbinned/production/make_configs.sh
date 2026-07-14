#!/bin/bash
# ===========================================================================
# Generate one *_cfg.py per HT bin by ADAPTING the McM "Sequences" command
# (brief §2). Run inside CMSSW_RELEASE/src after the fragments are under
# Configuration/GenProduction/python/ and `scram b`.
#
# McM Sequences (the existing inclusive wmLHEGS request) was:
#   cmsDriver.py <frag> --fileout file:....root --mc \
#     --eventcontent RAWSIM,LHE --datatier GEN-SIM,LHE \
#     --conditions 102X_upgrade2018_realistic_v11 \
#     --beamspot Realistic25ns13TeVEarly2018Collision \
#     --step LHE,GEN,SIM --geometry DB:Extended --era Run2_2018
#
# Documented edits applied below (brief deliverable #2):
#   - fragment            -> per-bin Herwig7_DY_<bin>_CH3_cff.py
#   - --eventcontent      RAWSIM,LHE   -> NANOAODGEN
#   - --datatier          GEN-SIM,LHE  -> NANOAODSIM
#   - --step              LHE,GEN,SIM  -> LHE,GEN,NANOGEN   (drops SIM; GEN-only)
#   - --conditions/--era/--beamspot/--geometry kept VERBATIM
#   - add --no_exec --mc -n <NEVENTS_PER_JOB>
#
# !!! BLOCKER TO VERIFY FIRST (run production/verify_nanogen.sh):
#     CMSSW_10_2_6 most likely has NO NanoGEN. If so, either bump the release or
#     split into GEN (10_2_6) + NANOGEN (newer release). See README "NanoGEN".
#     If verify says the NANOGEN step is absent but customizeNanoGEN exists, switch
#     STEP to "LHE,GEN" and uncomment the --customise line.
# ===========================================================================
set -euo pipefail

NEVENTS_PER_JOB="${NEVENTS_PER_JOB:-5000}"

# Verbatim from McM Sequences (change only if the release is bumped — a 102X GT
# will not load in a 10_6+/12_X release, in which case use that release's GT):
CONDITIONS="${CONDITIONS:-102X_upgrade2018_realistic_v11}"
ERA="${ERA:-Run2_2018}"
BEAMSPOT="${BEAMSPOT:-Realistic25ns13TeVEarly2018Collision}"
GEOMETRY="${GEOMETRY:-DB:Extended}"
STEP="${STEP:-LHE,GEN,NANOGEN}"   # NANOGEN step keyword (modern releases)

BINS=(HT-400to600 HT-600to800 HT-800to1200 HT-1200to2500 HT-2500toInf)

for BIN in "${BINS[@]}"; do
  FRAG="Configuration/GenProduction/python/Herwig7_DY_${BIN}_CH3_cff.py"
  echo ">>> ${BIN}: ${FRAG} -> ${BIN}_cfg.py"

  cmsDriver.py "$FRAG" \
    --python_filename "${BIN}_cfg.py" \
    --eventcontent NANOAODGEN --datatier NANOAODSIM \
    --step "$STEP" \
    --conditions "$CONDITIONS" --era "$ERA" \
    --beamspot "$BEAMSPOT" --geometry "$GEOMETRY" \
    --no_exec --mc -n "$NEVENTS_PER_JOB"
  #   # --- alternative for releases without a NANOGEN step keyword: ---
  #   --step LHE,GEN \
  #   --customise PhysicsTools/NanoAOD/nanogen_cff.customizeNanoGEN \
done
