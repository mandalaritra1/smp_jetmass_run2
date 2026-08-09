# Sherpa 3 pair-split campaign — authoritative plan (Route A)

**Status:** canary phase. Full production is **not** authorized.
**Locked by Aritra 2026-08-09.** This file is the source of truth for the
campaign; update it in place rather than forking a second plan document.

Companion: [`CANARY_STATUS.md`](CANARY_STATUS.md) tracks live gate results.

---

## 0. Route A summary (approved)

- Sherpa 3.0.4, pinned by **CVMFS LCG view** (Aritra's preference, 2026-08-09 —
  this supersedes the original "immutable container digest" requirement).
  Pin = `/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt`, recorded
  together with the `Sherpa` binary sha256 so the build is identifiable even
  though the view is mutable in principle.
- Comix MEPS@LO with 2→2, 3, 4 QCD matrix elements.
- `Qcut = 20` GeV baseline.
- **Explicit Dire** shower baseline (never rely on the image default).
- **Explicit Ahadic** hadronization.
- A smaller CSS sample **only** as an internal shower diagnostic.
- Exclusive leading-ME-jet pT slices.
- A **new** pair-split Rivet routine.
- Particle-level Sherpa/Pythia comparison **plus** a Pythia-response
  forward-folded Sherpa proxy.
- **No full production before post-canary approval.**

Sherpa changes the matrix element, merging, shower, and hadronization
simultaneously. It is an independent model / stress test. Even if it brackets
the data it does **not** automatically become a nuisance parameter — that is a
separate physics and analysis-policy decision.

---

## 1. Readiness findings

- The live checkout (`dev`, at `8322520`, 36 commits ahead of `origin/dev`,
  with modified + untracked hadronic-production work) remains the
  **authoritative evidence checkout**. New implementation is prepared in an
  isolated directory and merged only after review.
- **`rivet/CMS_HADRONIC_JETMASS.cc` cannot be reused.** It implements the old
  legacy selection: `|y| < 2.5`; dijet `Δφ > 2`; asymmetry `< 0.3`;
  trijet-priority veto; legacy trijet definition. The old routine stays
  untouched for provenance but is **not** used by Sherpa.
- Sherpa is not on the Mac PATH. Historical checkout artifacts show Sherpa
  3.0.4, git revision `5d42e10d30910b435a6401f32ff3ba01b5cf4f3f`, Comix,
  PDF4LHC21, Ahadic — and a runtime setting of `SHOWER_GENERATOR=CSS`. That
  conflicts with the 3.0.4 manual (Dire is the documented default). Hence:
  **production must set the shower explicitly.**

---

## 2. Pair-split particle-level definition

Anti-kT, **R = 0.8**, visible stable particles.

### Dijet
- ≥ 2 jets
- leading two jets `pT > 200` GeV
- `|y_{1,2}| < 2.4`
- `pT_avg = (pT1 + pT2)/2`
- `Δφ_12 > π − 200 GeV / pT_avg`
- **fill both leading jets**
- **no asymmetry cut**
- **no trijet veto**

### Trijet
- ≥ 3 jets
- `pT_{1,2} > 200` GeV, `pT_3 > 185` GeV
- `|y_{1,2,3}| < 2.4`
- minimum pairwise `Δφ > 1.0`
- `Δφ_12 < π − 200 GeV / pT_avg`
- **fill only jet 3**

### Observable

    rho_g = 2 * log10( m_g / (pT * 0.8) )

`m_g` = Cambridge/Aachen-reclustered Soft Drop mass, `β = 0`, `z_cut = 0.1`.

### Binning

Groomed-rho truth edges (existing hadronic edges):

    [-10, -5, -4, -3.4, -2.85, -2.25, -1.8, -1.5,
     -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0]

Dijet pT edges (full):

    [200, 290, 400, 480, 570, 680, 760, 820, 1500]

Trijet pT edges (validated bins):

    [185, 200, 290, 400, 1500]

Primary study region **`rho_g >= -3`**, with explicit reporting for
`[-1.8, -1.5]` and `[-1.5, -1.3]`.

> **No explicit groomed-mass cut.** In particular do **not** introduce
> `m_g > 10` GeV. At `pT >= 400` GeV the `rho_g >= -3` region already implies
> a groomed mass around 10 GeV or greater.

---

## 3. Sherpa card

`configs/Sherpa.yaml.tmpl`:

```yaml
BEAMS: [2212, 2212]
BEAM_ENERGIES: 6500
EVENTS: $(NEVENTS)
RANDOM_SEED: $(SEED)
ME_GENERATORS: [Comix]
PDF_LIBRARY: LHAPDFSherpa
PDF_SET: PDF4LHC21_40_pdfas
ALPHAS:
  USE_PDF: 1
SHOWER_GENERATOR: $(SHOWER)
JET_CRITERION: $(SHOWER)
FRAGMENTATION: Ahadic
PROCESSES:
- 93 93 -> 93 93 93{2}:
    Order: {QCD: 2, EW: 0}
    CKKW: $(QCUT)
    Integration_Error: 0.02
SELECTORS:
- NJetFinder:
    N: 2
    PTMin: 20.0
    ETMin: 0.0
    R: 0.4
    Exp: -1
# Exclusive slice in the leading ME-level anti-kT R=0.4 jet.
# This exact expression syntax must pass the version canary before use.
- FastjetSelector:
    Expression: PPerp(p[2]) >= $(PTLOW) && PPerp(p[2]) < $(PTHIGH)
    Algorithm: antikt
    N: 2
    PTMin: 20.0
    ETMin: 0.0
    DR: 0.4
EVENT_OUTPUT: HepMC3_GenEvent[events]
```

Baselines: `SHOWER = Dire`, `QCUT = 20`.

`93 93 -> 93 93 93{2}` = two-parton core plus up to two additional ME partons,
so the trijet observable does not depend entirely on shower-generated third
jets. A separate core-process jet selector is required because CKKW does not
regularize the core 2→2 process.

**Canary-gated unknown:** the exact `FastjetSelector` expression spelling must
be verified against the pinned 3.0.4 executable before generating integration
grids.

---

## 4. Slicing and event allocation

Exclusive leading-ME-jet slices (GeV): 300–400, 400–480, 480–570, 570–680,
680–760, 760–820, 820–1000, 1000–1250, 1250–1500, 1500–2000, 2000–6500.

Add a **250–300 GeV migration canary**. Promote it to the full campaign only if
it contributes more than 0.5% of the Sherpa yield in the first reported
400–480 GeV particle-level bin after showering.

| Slice (GeV) | Events |
|---|---|
| 300–400 | 200k |
| 400–480 | 300k |
| 480–570 | 400k |
| 570–680 | 300k |
| 680–760 | 250k |
| 760–820 | 250k |
| 820–1000 | 250k |
| 1000–1250 | 200k |
| 1250–1500 | 150k |
| 1500–2000 | 100k |
| 2000–6500 | 50k |

**Total: 2.45 M requested Dire events.** Split into 10,000-event chunks.

Deterministic seeds:

    seed = 310000000 + 10000 * slice_index + chunk_index

**Adaptive stop.** Stop a slice only when `N_eff = (Σw)² / Σw²` is at least:

- 5,000 in each of the two key `rho_g` bins for 480–570 GeV;
- 2,500 per reported `rho_g >= -3` bin in every other dijet `pT >= 400` bin;
- 2,500 per reported `rho_g >= -3` bin in trijet 400–1500 GeV.

(≈ 1.4% and 2% standalone statistical precision before forming the
Sherpa/Pythia transfer.)

---

## 5. Merging-scale tests

Before production: 20k-event samples at `Qcut = 15, 20, 30` GeV for the
400–480, 480–570 and 820–1000 slices.

Acceptance:
- no discontinuity at slice boundaries beyond combined MC statistics;
- no localized > 5% distortion in normalized `rho_g >= -3` shapes under `Qcut`
  changes;
- stable ME multiplicity fractions and finite cross sections;
- no unexpected spike near a jet or grooming threshold.

A larger dependence is not automatically a failure, but must be reported as a
**Sherpa merging uncertainty** before interpreting Sherpa as a clean model
comparison.

---

## 6. Dire versus CSS

Dire is the explicit baseline (documented 3.0.4 default; most defensible
reproducible reading of "Sherpa 3"). But the historical `hepstore/rivet-sherpa`
result actually used CSS implicitly, so the version canary must run:

```
Sherpa --version
Sherpa 'SHOW_SHOWER_GENERATORS: 1' 'EVENTS: 0'
```

and inspect `Settings_Report/Settings_Report.md` after explicit Dire and CSS
test runs.

Run CSS only if: the pinned binary lists it; the same MEPS card integrates with
`JET_CRITERION: CSS`; slice and merging closure pass; and Dire shows an
interesting shift or the residual remains unbracketed. Budget 20–25% of Dire
statistics, concentrated in 400–480, 480–570, 570–680, 820–1000. It is an
**internal shower sensitivity study, not a second nominal model.**

Ahadic is also set explicitly even though it is the documented default.

---

## 7. File layout

```
rivet/hadronic_sherpa3/
├── README.md
├── CAMPAIGN_PLAN.md          <- this file
├── CANARY_STATUS.md
├── CMS_HADRONIC_PAIR_SPLIT.cc
├── configs/
│   ├── Sherpa.yaml.tmpl
│   ├── slices.json
│   └── campaign_2018_v1.json
├── scripts/
│   ├── check_environment.sh
│   ├── make_cards.py
│   ├── build_rivet.sh
│   ├── run_canary.sh
│   ├── inner_sherpa.sh
│   ├── submit_campaign.py
│   ├── validate_campaign.py
│   ├── harvest_slices.py
│   ├── build_particle_transfer.py
│   ├── forward_fold_sherpa.py
│   └── plot_sherpa_comparison.py
├── condor/
│   └── sherpa3.jdl
└── schemas/
    └── completion_manifest.schema.json
```

**Do not modify or reuse the active files under `rivet/hadronic_prod/` in
place.**

Remote output: `/eos/user/a/amandal/hadronic_sherpa3/pairsplit_2018_v1/` with
`dire/<slice>/<seed>/`, `css/<slice>/<seed>/`,
`integration/<shower>/<slice>/`, `harvest/`, `plots/`, `reports/`.

---

## 8. Canary gates

Full submission stays disabled until **all** pass.

**Environment gate** — exact Sherpa version and git revision recorded; **CVMFS
LCG view path plus `Sherpa` binary sha256** recorded (never produce from an
unpinned `:latest` image; CVMFS view + binary hash is the accepted pin here);
Rivet, YODA, FastJet, LHAPDF versions recorded; Dire and CSS availability
queried from the executable.

**Configuration gate** — final settings report says Comix, Dire, Ahadic,
requested PDF and `Qcut`; process multiplicities are 2→2, 3, 4; no silent
fallback from Dire to CSS; card and integration-cache SHA-256 recorded.

**Pair-split gate** — dumped jet kinematics reproduce the Python pair-split
cuts; dijet/trijet overlap exactly zero; boundary events follow the strict `>`
dijet and `<` trijet convention; `R = 0.8`, Soft Drop `(0, 0.1)` and `rho_g`
verified numerically; no asymmetry cut, no trijet veto, no `m_g > 10` cut.

**Slice gate** — each accepted event belongs to exactly one ME-level slice;
adjacent boundaries show no overlap or gap; the 250–300 migration test passes;
high-side migration from `pT > 1500` is included.

**Normalization gate** — cross section and error finite for every slice;
`sumw`, `sumw2`, generated and analyzed event counts agree with logs;
independent chunks using the same integration cache report consistent slice
cross sections; a duplicated chunk changes only statistics, not the normalized
prediction.

**Physics gate** — `Qcut = 15/20/30` checks pass; Dire/Ahadic produces
populated groomed-rho spectra in every target bin; transfer-factor
uncertainties meet the `N_eff` targets; standalone Pythia/Rivet and nominal
pair-split Pythia truth agree closely enough that the transfer denominator is
defensible.

> That last closure matters: Rivet grooming uses C/A reclustering while nominal
> NanoAOD truth uses `SubGenJetAK8` sums, and a few-percent difference was
> already noted in the existing routine. **If the normalized difference exceeds
> 2% in either key bin**, either use it as an explicit transfer-method
> uncertainty or stop and resolve the definition mismatch.

---

## 9. Normalization and harvesting

Each slice is a separate physical cross section. Multiple chunks within a slice
are partitions of that one cross section.

For slice `s`:

    H_{s,k} = sigma_s * ( sum_{c in s} sum_{events in k} w )
                      / ( sum_{c in s} sum_{all events} w )

    H_k = sum_s H_{s,k}

**Do not** give every chunk the full slice cross section and add the chunks.
**Do not** blindly `yodamerge` already-finalized chunk histograms.

Each completion manifest is staged **last** and contains:

```json
{
  "campaign": "pairsplit_2018_sherpa3_v1",
  "generator": "Sherpa",
  "generator_version": null,
  "generator_revision": null,
  "container_digest": null,
  "shower": "Dire",
  "hadronization": "Ahadic",
  "me_generator": "Comix",
  "me_multiplicities": ["2to2", "2to3", "2to4"],
  "qcut_GeV": 20,
  "slice_low_GeV": 480,
  "slice_high_GeV": 570,
  "seed": 310020000,
  "requested_events": 10000,
  "generated_events": null,
  "analyzed_events": null,
  "sumw": null,
  "sumw2": null,
  "cross_section_pb": null,
  "cross_section_error_pb": null,
  "config_sha256": null,
  "rivet_source_sha256": null,
  "integration_cache_sha256": null,
  "outputs": {
    "yoda": {"path": null, "sha256": null, "bytes": null},
    "dijet_ntuple": {"path": null, "sha256": null, "bytes": null},
    "trijet_ntuple": {"path": null, "sha256": null, "bytes": null}
  },
  "status": "complete"
}
```

---

## 10. Particle-level comparison and forward folding

First compare Sherpa and Pythia **at particle level** with the same pair-split
fiducial, jet algorithm and radius, grooming, pT and `rho_g` bins,
`rho_g >= -3` reporting range, and per-pT-bin shape normalization.
**Do not reweight generator pT to data.**

Gen-bin transfer:

    T_j = G_j^Sherpa / G_j^Pythia

Forward-fold through the nominal Pythia response:

    N_i^{Sherpa reco proxy} = F_i^nom + sum_j M_ij^nom * T_j

Event-level implementation follows `/tmp/forward_fold_hadronic_models.py`:
matched nominal Pythia events take the transfer from their gen-`(pT, rho_g)`
bin; unmatched reco fakes retain weight 1; misses enter through the response
normalization; histogram reco `(pT, rho_g)`; normalize every reco-pT row
exactly as the data; propagate transfer statistical uncertainties, preferably
with slice/chunk bootstrap replicas.

Label the output **"Sherpa particle-level transfer, forward-folded through
nominal Pythia response"**. It must **not** be called detector-level Sherpa.

### Coverage metrics

Per `pT >= 400` bin and per `rho_g >= -3` bin, save: data/Pythia; Sherpa
proxy/Pythia; statistical uncertainties; whether Sherpa lies between Pythia and
data; residual before and after Sherpa (`|D/P − 1|`, `|D/S − 1|`); signed
residual reduction; total-variation distance; maximum uncovered residual; and a
covariance-aware chi-square if the normalized-shape covariance is available.

The machine-readable report highlights **480–570 GeV** and the two bins from
**−1.8 to −1.3**. Existing evidence shows both are currently uncovered in the
nominal combined model envelope, at roughly **5.6%** and **7.2%** below Pythia.

---

## 11. Compute estimate

Baseline 2.45 M events plus ~180k merging-scale canary events. The actual
Sherpa MEPS event rate is **unknown and must be measured on a Condor worker**.

    core-hours = N_events * t_event / 3600 + integration cost

| Measured rate | Generation cost (2.45 M) |
|---|---|
| 0.5 s/event | 340 core-hours |
| 1 s/event | 681 core-hours |
| 2 s/event | 1,361 core-hours |
| 5 s/event | 3,403 core-hours |

Add 20–30% for retries, Rivet, stage-out and validation. Integration-grid
construction is separate — several to tens of core-hours per slice — and must
be measured, not guessed. Grids are built once per slice/shower configuration,
hashed, and reused by all chunk jobs.

---

## 12. Explicit approval boundary

The authorized step is **only** to prepare the isolated Sherpa directory and
run small version/syntax/physics canaries.

**Do not run `submit_campaign.py` for the 2.45 M-event wave** until Aritra has
reviewed: pinned Sherpa/container identity; the final Dire card and settings
report; pair-split selection-parity output; `Qcut` canary plots; slice closure;
and the measured runtime with the proposed final event allocation.

If approved later, *completion* means: canary, deterministic submission,
durable monitoring, manifest/hash validation, correct slice normalization,
harvesting, CMS-style rendered and **visually inspected** plots,
machine-readable coverage results, and a physics interpretation.
**Submission alone is not completion.**
