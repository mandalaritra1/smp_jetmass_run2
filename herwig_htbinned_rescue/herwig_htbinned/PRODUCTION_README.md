# Herwig7-CH3 HT-binned DY (NanoGEN) — VALIDATED recipe, ready to scale

**Goal:** an alternate-parton-shower systematic for the zjet jet-mass measurement.
Nominal = official `DYJetsToLL_M-50_HT-*_TuneCP5_madgraphMLM-pythia8`. We imitate it with
the **same HT gridpacks + Herwig7 CH3** instead of Pythia8 CP5 → a clean shower/hadronization
variation on the *same* madgraphMLM (0–4j, MLM-merged) ME. Output NanoGEN → analyze with the
existing coffea processor (`smp_jetmass_run2/zjet_processor.py`) so selection + BOTH mass
definitions match the nominal automatically.

Status **2026-07-01: fully de-risked in a 100/200-event test.** Full 6-bin condor production
NOT yet built. Everything below is validated end-to-end (LHE→GEN→NANOGEN, branches, size).

## Environment / access
- lxplus (LPC is expiring). `ssh lxw` gives a **broken RC4 ticket** (no AFS token). Use **`lxwk`**
  (in `~/.zshrc`): keytab `~/.private/cern_amandal.keytab` (from `cern-get-keytab --user`) →
  native AES256 ticket. Automation pattern: base64-pipe the keytab per ssh, `kinit -kt <tmp> amandal@CERN.CH && aklog`.
  Full writeup: research-notes `topics/cern_lxplus_noninteractive_ssh.md` (troubleshooting §3).
- **Release: `CMSSW_10_6_28`, `SCRAM_ARCH=slc7_amd64_gcc700`, inside `cmssw-el7` container** (lxplus is el9).
  Has CH3 + `Herwig7MGMergingSettings` + NanoGEN. NANOGEN is a real `--step` (not in `--help`).

## Gridpacks (live on cvmfs, same as the Pythia nominal)
`/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.5/DYJets_HT/DYJets_HT-<BIN>_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz`
Use 6 bins: **200to400, 400to600, 600to800, 800to1200, 1200to2500, 2500toInf**
(DROP 70to100 AND 100to200 — HT<200 can't give a gen jet pT>200, contribute ~nothing to the measured bins).

## Files (this dir)
- `inputs/Herwig7_DY_HT_CH3_fragment.py` — the fragment. Generator block copied VERBATIM from
  CMSSW `GeneratorInterface/Herwig7Interface/test/DYToLL01234Jets_5FS_TuneCH3_13TeV_madgraphMLM_herwig7_cfg.py`
  (self-contained MG-merging block: creates LesHouchesHandler+FxFxLHReader+FxFxHandler; njetsmax=4,
  MergeMode TreeMG5, ETClus 20 GeV) + `externalLHEProducer` prepended. **Do NOT add Herwig7LHECommonSettings
  or PSWeights** — they double-create the handler → null-deref segfault in `caldel_mg`. Per-bin: swap the one gridpack path.
- `production/nanogen_addsubjets.py` — the customise `addSubGenJetAK8`: adds gen soft-drop subjets
  (genParticlesForJetsNoNu→ak8GenJetsNoNu→Constituents→ak8GenJetsNoNuSoftDrop:SubJets, defaults are
  β=0/zcut=0.1/R=0.8 ✓) and points the `SubGenJetAK8` table at them; then drops ONLY `genParticleTable`
  (GenPart, ~80% of size). Do NOT drop genJetTable etc. — their flavour EXTENSION tables error without the main.

## The command (per bin, validated)
```
cmsDriver.py <fragment> --python_filename prod.py --fileout file:out.root \
  --mc --eventcontent NANOAODGEN --datatier NANOAODSIM --step LHE,GEN,NANOGEN \
  --conditions auto:mc --beamspot Realistic25ns13TeVEarly2018Collision --era Run2_2018 \
  --customise Configuration/GenProduction/nanogen_addsubjets.addSubGenJetAK8 --nThreads 1 -n <N>
```
(fragment lives at `Configuration/GenProduction/python/`, customise at same package.)

## Verified NanoGEN branches (coffea-ready)
`SubGenJetAK8_{pt,eta,phi,mass}` (groomed), `GenJetAK8_{pt,eta,phi,mass,partonFlavour,hadronFlavour}`
(ungroomed), `GenDressedLepton_{pt,eta,phi,mass,pdgId,hasTauAnc}` (Z sel). Drop-in for the processor.

## Measured numbers (validated on lxplus condor, 2026-07-01)
- **~2–3 kB/event** at scale (tiny canary files ~3.2 kB/ev inflated by low-N ROOT overhead; GenPart drop ~80% cut).
- **Merging acceptance ~32%** (160 written / 500 nEvents; MLM veto, same as the Pythia nominal) → generate ~3× the final target.
- **~1.5–2 s/showered event**, RAM ~930 MB. `njetsmax` MUST stay 4 (matches gridpack + nominal; lowering biases the tail).
- Canary: 3/3 jobs `return 0`, files on EOS, `GenJetAK8_pt` max ~403 GeV for HT-200to400 (correct HT-cut kinematics).

## ⚠️ Per-job cfg bug (FIXED — see `production/job_tail.py`)
The per-job tail MUST override **`process.externalLHEProducer.nEvents = NEVT`** as well as
`process.maxEvents.input`. Otherwise the gridpack produces only the cmsDriver `-n` default number of LHE
events and cmsRun dies mid-run with `No lhe event found in ExternalLHEProducer::produce()` (exit 53 /
segfault). Keep `nEvents == maxEvents` (1 LHE read per event; ~32% survive the merging veto and get written).

## ⚠️ Normalization — do NOT trust the Herwig `GenXsecAnalyzer` σ
It reported **0.23 pb** for HT-200to400 (~150× low) — a known Herwig7 + external-LHE + merging readout
quirk (the LHE-header `# original cross-section 363 pb` is the raw *pre-matching* value). These are the
**same gridpacks as the Pythia nominal**, so stitch with the EXISTING analysis machinery:
`cms_utils.getXSweight(dataset, IOV, herwig=True)` already divides inclusive `z_xs=6077.22` ×
`xs_scale_dic[IOV][dataset]` by `numentries_herwig[dataset]`. Just populate `numentries_herwig` with the
per-bin Herwig event counts → stitching is correct by construction (same z_xs, same xs_scale_dic).

## Sizing (uniform stats for a reweight target — NOT xsec-proportional)
Generate **flat uniform OUTPUT events/bin** (÷~0.32 merging → nEvents/bin). Wave-1 LAUNCHED 2026-07-01:
6 bins × 120 jobs × `nEvents=8000` (workday, ~4 h/job) ≈ **~300k written/bin** (~2–3 GB/bin, ~15 GB total).
Clusters 11770825–30, SEEDBASE 3.0M–3.5M (100k block/bin). Top up a noisy bin later with a higher SEEDBASE.
`production/submit_all.sh` drives all 6 bins.

## REMAINING WORK
1. ✅ 6 per-bin fragments — `production/gen_fragments.py` (swaps only the gridpack bin token in the finalized fragment).
2. ✅ lxplus condor: `run_job.sh` (el9 launcher → `cmssw-el7` `inner.sh` → `xrdcp` to EOS), `submit.jdl`
   (`MY.SendCredential`, `+JobFlavour="workday"`), `submit_all.sh`. Output → EOS `/eos/user/a/amandal/herwig_ch3_dy/HT-<BIN>/`.
3. ✅ 1-bin canary confirmed size + 32% acceptance at scale (real σ deferred to `numentries_herwig`, above).
4. ▶ Monitor 720 jobs → count written events/bin → set `numentries_herwig` → run `zjet_processor.py`
   (`herwig=True`) → top up thin bins → build the mass/rho reweight vs the Pythia nominal → log to research-notes.
