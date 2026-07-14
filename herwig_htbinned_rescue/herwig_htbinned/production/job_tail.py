# appended to prod_HT-<BIN>.py by inner.sh; reads SEED/NEVT/OUTTAG from env.
# CRITICAL: externalLHEProducer.nEvents (how many LHE events the gridpack makes)
# must be >= maxEvents, else 'No lhe event found' crash. Keep them equal.
import os as _o
_n = int(_o.environ['NEVT'])
_s = int(_o.environ['SEED'])
process.maxEvents.input = _n
if hasattr(process, 'externalLHEProducer'):
    process.externalLHEProducer.nEvents = _n
_rng = process.RandomNumberGeneratorService
for _m in ('externalLHEProducer', 'generator'):
    if hasattr(_rng, _m):
        getattr(_rng, _m).initialSeed = _s
_tag = _o.environ['OUTTAG']
for _om in process.outputModules_().values():
    _om.fileName = '%s.root' % _tag
