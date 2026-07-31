# Gen-level closure: Rivet routine vs the coffea gen path, same FullSim events

Sal's ask. `CMS_2026_PAS_SMP_25_010` claims to reproduce the particle-level
selection of `zjet_processor.py`. This is the test of that claim: run both on the
**same events** of a FullSim sample and require them to agree.

Rivet cannot read NanoAOD (no particle-level constituents), so the two sides read
two different tiers of the *same* events:

| side | input | what it produces |
| --- | --- | --- |
| Rivet | MiniAOD (packed gen particles → HepMC) | `zjets_{un,}groomed` 2D + `_slice{0,1,2}` |
| coffea | the child NanoAOD of that MiniAOD | `ptjet_rhojet_{u,g}_gen` |

Because NanoAOD is produced from MiniAOD one-to-one, the event streams are
identical by construction — no seed matching, no re-generation.

## Sample

Nominal Z+jet MC is the HT-binned madgraphMLM DY stack. Use one file of

```
/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM
```

and its MiniAOD parent (`dasgoclient -query="parent file=<lfn>"`). HT-400to600
gives a decent rate of jets above 200 GeV without needing many files.

## Step 1 — build the plugin against CMSSW's Rivet

The routine is written against **Rivet 4** (`LeptonFinder`, `JetMuons`/
`JetInvisibles` enums, `normalize(h, norm, includeoverflows)`). If the CMSSW
release ships Rivet 3.x, back-port those three things — everything else is
version-neutral:

| Rivet 4 | Rivet 3 |
| --- | --- |
| `LeptonFinder(dR, cut, LeptonOrigin::NODECAY)` | `DressedLeptons(photons, bareleptons, dR, cut)` |
| `FastJets(fs, JetAlg::ANTIKT, 0.8, JetMuons::ALL, JetInvisibles::NONE)` | `FastJets(VisibleFinalState(...), FastJets::ANTIKT, 0.8)` |
| `normalize(h, 1.0, false)` | same signature, available since 3.0 |

```bash
cmsenv
rivet-build RivetCMS_2026_PAS_SMP_25_010.so CMS_2026_PAS_SMP_25_010.cc
export RIVET_ANALYSIS_PATH=$PWD:$RIVET_ANALYSIS_PATH
```

## Step 2 — Rivet side

```bash
cmsRun rivet_miniaod_cfg.py inputFiles=<miniaod-lfn> outputFile=closure_rivet.yoda maxEvents=-1
```

## Step 3 — coffea side

Run `zjet_processor.py` with `do_gen=True` over the **child NanoAOD file only**,
and keep `ptjet_rhojet_u_gen` / `ptjet_rhojet_g_gen`.

## Step 4 — compare

The two use different rho binnings, but both nest onto the published 10-bin
`log10(rho^2)` axis `[-10, -4.5, -4, ..., -0.5, 0]`:

- ungroomed gen axis `[-10, -6, -5, -4.5, -4, ...]` → merge the first three bins.
- groomed gen axis `[-10, -6, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3, ...,
  -1, 0]` → merge into the same edges, except that it has no `-0.5` edge, so the
  last two published bins must be merged on **both** sides.

Agreement expected:

- **ungroomed**: exact, bin by bin, up to the finite-precision of the two mass
  definitions. Any disagreement is a real selection mismatch.
- **groomed**: a few-% offset is expected and is not a bug — the routine applies
  `fastjet::contrib::SoftDrop` to the C/A-reclustered constituents, whereas the
  coffea gen path takes the `SubGenJetAK8` subjet-pair sum from NanoAOD.
- **yields**: the selected-event count is the sharpest test; compare it before
  looking at shapes.

## What this does not test

Anything reco-level: triggers, JEC/JER, lepton IDs, the response matrix. This is
a fiducial-definition closure only.

---

# Result (2026-07-28)

534146 events: one NanoAODv9 file of `DYJetsToLL_M-50_HT-400to600` UL18 and its
12 MiniAODv2 parents (each parent's only v9 child is that file, so the event sets
are identical by construction). CMSSW_15_0_9, Rivet 4.0.2.

**Ungroomed closes.** Normalized `log10(rho^2)` shape, rivet/coffea per bin:

| pT slice | populated bins | max deviation |
| --- | --- | --- |
| 200–290 | 6 | 0.04% (excl. one 8e-4-fraction bin at 5%) |
| 290–400 | 6 | 0.33% |
| >400 | 6 | 0.09% |

pT-slice fractions agree to 1e-4: `[0.42894, 0.39861, 0.17246]` (Rivet) vs
`[0.42891, 0.39863, 0.17246]` (coffea). The overall scale differs by a constant
1.5286 — the two sides normalize differently on purpose (coffea applies
xs·lumi/sumw in `postprocess`, Rivet fills raw generator weights) — and that
constant being identical across all three slices is itself evidence that the two
sides selected the same events.

**Groomed agrees to 0.3–2%**, with the expected sign: Rivet slightly high at low
rho and ~1.5–2.5% low in the top bin. This is the grooming-definition difference,
not a selection mismatch — the routine soft-drops the C/A-reclustered
constituents, the coffea gen path sums the `SubGenJetAK8` pair.

## Finding: the processor's "gen" histograms are matched-only

`allsel_gen` includes `is_matched_gen` (gen jet within dR<0.4 of the reco jet),
so every fill into `ptjet_rhojet_{u,g}_gen` and `ptjet_mjet_{u,g}_gen` is the
*matched* truth, not the fiducial gen spectrum. There is no unmatched gen fill in
`zjet_processor.py`. The closure above therefore uses a copy of the processor
with that one term dropped; comparing against the shipped version instead gives:

| pT slice | matched / all-gen |
| --- | --- |
| 200–290 | 0.872 |
| 290–400 | 0.945 |
| >400 | 0.961 |

The pT dependence distorts the slice composition by ~5%
(`[0.408, 0.411, 0.181]` matched vs `[0.429, 0.399, 0.172]` fiducial), which is
why a matched histogram cannot be compared to a particle-level prediction. Worth
checking where the unfold repo gets the `all_gen` efficiency denominator that the
comment at `zjet_processor.py:2192` refers to — this processor does not produce
one.
