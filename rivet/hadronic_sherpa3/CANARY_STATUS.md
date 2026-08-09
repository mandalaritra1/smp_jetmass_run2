# Canary status — Sherpa 3 pair-split

Spec: [`CAMPAIGN_PLAN.md`](CAMPAIGN_PLAN.md). Gate definitions live there
(§8); this file records what has actually been measured.

**Bottom line as of 2026-08-09: the Dire 480–570 canary is green.** Full
production remains **unauthorized** pending the items in §"Still open" below.

---

## Run of record

`dire_pt480_570_200evt_v2` — LPC HTCondor cluster **85063614.0**, submitted
from `lpcschedd6.fnal.gov`, executed on `cmswn2199.fnal.gov` under the rhel9
apptainer image.

- Sherpa **3.0.4 (Erebus)**, LCG_110 `x86_64-el9-gcc15-opt`
- Rivet **4.1.2**, YODA 2.1.2, LHAPDF 6.5.5
- Sherpa binary sha256 `a87ac9ed…9542f`
- source bundle sha256 `5ae6f1e3…e45a4`
- 200 events, seed 310020000, slice index 2, 480–570 GeV, Dire
- cross section **314.93 pb ± 22.27 pb (7.07%)**
- result archive sha256 `27a909fb…44c32` (12,096,549 bytes)

Local salvaged artifacts: [`canary/dire_pt480_570_200evt_v2/`](canary/dire_pt480_570_200evt_v2/).

### The job originally exited 1 — two bugs, both fixed

1. **`RIVET_ANALYSIS_PATH` was relative.** `run_transferred_canary.sh` passes
   `OUTDIR` as a bare directory name, and several steps run with the cwd
   already inside `$OUTDIR`, so the plugin path resolved to
   `$OUTDIR/$OUTDIR` and Rivet reported *"'CMS_HADRONIC_PAIR_SPLIT' is not a
   known Rivet analysis"*. `set -euo pipefail` then aborted the script before
   validation and the manifest ran. Fixed in `scripts/run_canary.sh` by
   absolutizing `OUTDIR` immediately after `mkdir`. This never reproduced
   locally because the default `OUTDIR` is an absolute path.
2. **`scripts/validate_pair_split.py` read the wrong columns** — hardcoded
   indices 16/17 pointed at `pass_trijet` and `weight` once the routine began
   writing `pair_boundary`. It reported 185/190 mismatches against a routine
   that was in fact correct. Fixed by parsing the routine's own header line
   and asserting the expected layout.

The 5h58m of event generation was **not** lost: the 29 MB HepMC3 `events`
file survived in the archive, and the Rivet step was replayed over it in
**1 second**. No regeneration was needed.

---

## Gate results

| Gate | Status | Evidence |
|---|---|---|
| Environment | **pass** | versions + binary hash recorded above; Dire and CSS both advertised by `SHOW_SHOWER_GENERATORS` |
| Configuration | **pass** | see settings report below |
| Pair-split | **pass** | 0 mismatches, 0 overlap |
| Slice | **partial** | in-slice behaviour confirmed; boundary/migration tests not yet run |
| Normalization | **partial** | single chunk finite and consistent; multi-chunk closure not yet run |
| Physics | **open** | `Qcut` scan and Pythia closure not yet run |

### Configuration gate — the default really is CSS

`Settings_Report.md` reports **`SHOWER_GENERATOR | CSS | Dire | Dire`**
(default | user | used). This **confirms the plan's core suspicion**: the
pinned 3.0.4 build defaults to **CSS**, contradicting the 3.0.4 manual's claim
that Dire is the default. Setting the shower explicitly is not belt-and-braces
— it is load-bearing. Never rely on the image default.

Runtime confirmation of no silent fallback, from `sherpa.log`:
`From "Jet_Evolution:Dire": 2801 (4433) -> 63.1 %`.

Also confirmed used: `ME_GENERATORS = Comix` (default is
`Comix, Amegic, Internal`), `PDF_SET = PDF4LHC21_40_pdfas`,
`FRAGMENTATION = Ahadic`, `CKKW = 20`, `JET_CRITERION = Dire`.

### Pair-split gate

```json
{"events_with_two_jets": 200, "dijet_selected": 175, "trijet_selected": 10,
 "overlap": 0, "dijet_mismatches": 0, "trijet_mismatches": 0, "status": "pass"}
```

Independent NumPy re-implementation of the cuts agrees with the C++ routine on
every event, and dijet/trijet are disjoint.

### Slice gate — the FastjetSelector expression parses and bites

The version-sensitive expression `PPerp(p[2]) >= 480 && PPerp(p[2]) < 570`
**works** on this build. Particle-level AK8 leading-jet pT over the 200 events:

- min/max 373.3 / 736.0 GeV, mean 519.2, median 513.8
- 78.0% inside [480, 570]
- 0.5% below 400 GeV, 1.0% above 700 GeV

The spread is the expected ME(R=0.4) → particle-level(R=0.8) migration, not a
selector failure. Boundary overlap/gap and the 250–300 migration canary still
need dedicated runs.

---

## Measured numbers that change the plan

### Compute — event rate is ~2× the plan's worst case

`CAMPAIGN_PLAN.md` §11 budgets 0.5–5 s/event. **Measured: 9.75 s/event.**

The 5h56m wall splits into two very different costs:

| Phase | Cost | Scales with |
|---|---|---|
| Integration / grid construction | ~5.4 core-hours | once per slice+shower, **cacheable** |
| Event generation | 1950 s / 200 evt = **9.75 s/event** (8862 evt/day) | per event |

At face value the full 2.45 M-event wave is **~6,600 core-hours** of
generation plus ~60 core-hours of integration, so **~8,000–8,600 core-hours**
with the plan's 20–30% overhead. That is above the plan's 5 s/event row and
should be re-budgeted before approval.

### Statistics — the 480–570 allocation looks ~4× oversized

Weights are not unit: **7.5% are negative**, and the sample-level
`N_eff / N = 0.362` (126.7 of 350 dijet jet fills).

Per-bin `N_eff` from these 200 events, for the two key bins:

| rho bin | fills | N_eff | events needed for N_eff = 5000 |
|---|---|---|---|
| [-1.8, -1.5] | 29 | 11.3 | ~88k |
| [-1.5, -1.3] | 19 | 10.7 | ~93k |

Extrapolating linearly, the 480–570 slice reaches its 5,000-`N_eff` target at
**~95k events, not the 400k allocated** — roughly 4× headroom. 64.6% of dijet
fills land in the `rho_g >= -3` study region.

**Caveat:** these per-bin `N_eff` values come from 19–29 fills and are very
noisy, and small-sample `N_eff` is biased. Treat them as order-of-magnitude.
The right move is the plan's own adaptive rule — derive each slice's
allocation from a short per-slice canary rather than the a-priori table. Given
the measured 9.75 s/event, that re-derivation is worth real money.

---

## Still open before any production submission

- `Qcut = 15 / 20 / 30` scan (400–480, 480–570, 820–1000)
- slice boundary closure; 250–300 GeV migration canary; high-side
  migration from `pT > 1500`
- multi-chunk normalization closure (duplicate chunk must change only
  statistics)
- standalone Pythia/Rivet vs nominal pair-split Pythia truth closure, with the
  2% C/A-reclustering vs `SubGenJetAK8` threshold from the plan
- pinned **container digest** — this canary ran from an LCG CVMFS view, not a
  pinned image; the plan requires an immutable digest for production
- measured runtime re-budget and a revised per-slice allocation
