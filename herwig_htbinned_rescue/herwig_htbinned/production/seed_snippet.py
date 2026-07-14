# --- per-job seeding + unique output name -----------------------------------
# Appended to the generated *_cfg.py by run_job.sh. Operates on the already-built
# `process`. Reads JOBID / SEED_BASE / OUTTAG from the environment so the same
# cfg is reused by every job. Release-agnostic: it discovers the output module
# instead of hard-coding NANOAODGENoutput.
import os as _os

_jobid = int(_os.environ.get("JOBID", "0"))
_base  = int(_os.environ.get("SEED_BASE", "1000000"))
_seed  = _base + _jobid

# Both engines get the same per-job seed (distinct streams internally).
process.RandomNumberGeneratorService.externalLHEProducer.initialSeed = _seed
process.RandomNumberGeneratorService.generator.initialSeed           = _seed

# Unique output filename per job, e.g. Herwig7_DY_HT-400to600_CH3_42.root
_tag = _os.environ.get("OUTTAG", "nanogen")
for _name, _mod in process.outputModules_().items():
    _mod.fileName = "%s_%d.root" % (_tag, _jobid)
