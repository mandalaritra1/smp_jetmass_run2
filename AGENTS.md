# Repository Agent Instructions — smp_jetmass_run2

Unified Run 2 jet-mass cross-section framework for **three channels** (Z+jet,
dijet, trijet), built on the coffea-2025 `zjet_corrections` framework. dijet/trijet
were ported from `~/Projects/GluonJetMass` and refactored to the Z+jet
`QJetMassProcessor` conventions. See `README.md` for install + run instructions.

## Analysis context (compiled knowledge base)

For analysis context, read `~/Projects/ai-wiki/AGENTS.md`,
`~/Projects/ai-wiki/wiki/meta/index.md`, and the related repo card
`~/Projects/ai-wiki/wiki/repos/zjet_corrections.md`. When a code change creates
durable analysis knowledge, changes workflow behavior, fixes a reusable bug, or
changes project status, you may update the relevant `~/Projects/ai-wiki` pages
(follow its AGENTS.md; update `wiki/meta/index.md` + append `wiki/meta/log.md`).
Treat raw sources, PDFs, and other linked repositories as read-only unless asked.

## Paper & analysis note (local, read-only)

This framework is the analysis behind **CMS SMP-25-010** (Z+jet jet-mass cross
section) and its supporting note **AN-24-162**. Both LaTeX sources live locally:

- **Paper — SMP-25-010**: `~/Projects/SMP-25-010/` (`SMP-25-010.tex` / `main.tex`,
  `SMP-25-010.bib`, built `main.pdf`; a parallel copy is `~/Projects/smp_25_010_paper/`).
- **Analysis note — AN-24-162**: `~/Projects/AN-24-162/` (`AN-24-162.tex`,
  `AN-24-162.bib`, `AN-24-162_draft.pdf`).
- ARC / review context: `review/` in this repo, and
  `~/Projects/research-notes/papers/SMP-25-010_*`.

Consult these for definitions, binning, and result numbers; treat them as
read-only unless the user asks to edit them.

## Architecture (quick map)

- `smp_jetmass_run2/` — flat importable package (no `src/`).
  - `corrections.py` — shared zjet corrections + ported hadronic functions
    (`applyPrescales`, `getXSweight`, `MET_filters`, `applyjmsSF`/`applyjmrSF`,
    `GetQ2muF`/`GetQ2muR`, `getJetFlavors`, `applyBTag`, `HEMVeto`, …). Bundled
    correction data in `corrections/` (loaded via `importlib.resources`).
  - `hist_utils.py` — channel-aware `util_binning(channel=...)`, `register_hist`,
    `fill_hist`.
  - `notebook_utils.py` — `get_processor_class(mode, channel)` dispatch, `run_once`,
    fileset builders, executor/client setup, worker shipping, postprocess helpers.
  - `zjet_processor.py` / `dijet_processor.py` / `trijet_processor.py` — the three
    channel processors (all `processor.ProcessorABC`, `mode`-gated registration).
- `scripts/run_analysis_cli.py` — headless runner; reads a JSON `--config`.
- `configs/` — one example config per channel.
- `samples/zjet/{data,mc}` (+ `mc/backgrounds/`) — zjet `.txt` lists;
  `samples/hadronic/` — shared dijet+trijet JSON filesets.

## Hard invariants — do NOT break these without explicit instruction

1. **Selections are sacred.** The dijet/trijet `PackedSelection` cut sequences must
   stay byte-for-byte identical to GluonJetMass. Verify with the cutflow regression
   below — every step must match. The ONLY intentional selection change is HEM.
2. **HEM**: 2018 MC gets a flat lumi-fraction *weight* (`HEMVeto(..., isMC=True)`,
   added to the Weights object as `"HEM"`); 2018 data gets the hard veto. No
   `HEMCleaning` jet-systematic.
3. **Canonical systematic names** on the `systematic` axis: `pu`, `l1prefiring`,
   `pdf`, `isr`, `fsr`, `JES_<source>Up/Down`, `JER/JMS/JMRUp/Down`. (Kept extras:
   `Luminosity`, and the split `Q2muF`/`Q2muR` — not collapsed to `q2` yet.)
4. **JMS/JMR** use GluonJetMass values (`applyjmsSF`/`applyjmrSF`, flat sf=1.0,
   ±1% / ±2%) — NOT zjet's `jmssf`/`jmrsf`.
5. **rho** is `2*log10(m/(pt*R))` with `R=0.8` (`self._rho`), matching zjet — not the
   bare `2*log10(m/pt)`.
6. **XS/lumi normalization** (`xs*lumi*1000/sumw`) happens in `postprocess`, per
   dataset; `process` fills with the raw generator weight. The `dataset` axis carries
   the full dataset name so postprocess can infer IOV + look up xs/sumw.
7. **Binning** lives only in `util_binning(channel=...)`. Default `channel="zjet"`
   must stay byte-identical (zjet_processor depends on it). Hadronic hists have **no
   `channel` axis**; the `jk` axis appears only in `mass_jk`/`rho_jk` modes.
8. **Modes** gate registration; `fill_hist` no-ops on unregistered hists, so a fill
   call left in `process` is safe even when its hist isn't registered for that mode.
   `minimal`=mass, `minimal_rho`=rho, `validation`/`full`=diagnostics, `*_jk`=+jk,
   `mass_cov`=joint **gen+reco** groomed×ungroomed **mass and rho** per pT (groomed↔ungroomed covariance).
9. **2017 JEC uses V6 for all** (`_resolve_jec_tags`).

## Verify after changes (local files)

```bash
# cutflow + integrals must match GluonJetMass; mass unchanged, rho intentionally rebinned
PYTHONPATH=. <venv>/bin/python -c "..."   # run DijetProcessor on test_files MC/data
```
Local NanoAODv9 test files: `~/Projects/GluonJetMass/test_files/` (UL18 QCD pythia8
MC + Run2018A JetHT data). Do not run the unfolder (not enough files yet).
Use `~/Projects/GluonJetMass/.venv` (coffea 2026.5.0) for local runs.
