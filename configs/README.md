# configs/ — what's here and how to tune a run

Every JSON here feeds `scripts/run_analysis_cli.py --config <file>` (or the
notebook via its **load cfg** dropdown), except the `*_skim_*` ones, which feed
`scripts/run_zjet_skim.py`. `last_run.json` is the notebook's scratch file
(rewritten on every "lock" — never edit by hand).

## Naming scheme

`<channel>_<dataset>_<era>_<mode/variant>[_<site>].json`

- **site suffix** (`_lxplus`, `_casa2gb`, `_purdue`, none = coffea-casa
  defaults): same physics, different `executor_mode` / redirector /
  `worker_memory`. To run an existing config elsewhere, copy it and swap those
  three keys — see the tuning table below.
- `_r2` = ARC round-2 rho binning (`rho_refine: 2`).

## Families

| family | files | purpose |
|---|---|---|
| rho production | `zjet_{pythia,data}_all_minimal_rho_r2*` | the all-years groomed/ungroomed rho inputs (per site / syst profile) |
| reweight variants | `zjet_pythia_*_reweight_{cr1,cr2,fraghard,fragsoft,vincia}_*` | modelling-uncertainty reweight runs (one per Pythia variation) |
| NLO | `zjet_nlo_*` | amcatnloFXFX inclusive / ptZ-binned DY (rho and mass) |
| fine binning | `zjet_{pythia,data,herwig}_2018_finebins*` | 2018 fine-binned studies |
| hadronic | `dijet_*`, `trijet_*` | dijet/trijet channel runs (`*_test` = HT-filtered smoke test) |
| omnifold skims | `zjet_*_skim_casa` | per-chunk Parquet skims via `run_zjet_skim.py` |

## Tuning: `chunksize` / `worker_memory` / `worker_merge`

Since the streaming executor became the default for dask runs, workers never
merge histograms — worker memory only has to fit *processing one chunk*, and
the client merges finished chunks as they arrive. The three knobs interact
like this:

| run type | `chunksize` | `worker_memory` | `worker_merge` |
|---|---|---|---|
| **no-syst** MC, rho/mass (all years ≈ 3k chunks) | 50000 | `2 GiB` (validated on casa) | 4 |
| **no-syst** data (≈ 20k chunks) | 50000 | `2 GiB` | 8 |
| **all-syst** MC | 50000 (→ 25000 if workers die) | `3 GiB` | 4 |
| all-syst fallback: `"treereduce": true` | 50000 | `4–6 GiB` | (ignored) |
| local smoke test (`futures`/`iterative`, `test: true`) | 100000–200000 | (n/a) | 1 |

Rules of thumb:

- **`worker_merge`** = how many consecutive chunks one worker task processes
  and merges locally before shipping a single output. It divides client-bound
  traffic by that factor (the cure for "finished results pile up on workers,
  workers pause, run crawls" — seen with all-syst on casa). Keep
  `n_chunks / worker_merge ≳ 5–10 × n_workers`, and remember a failed chunk
  retries its whole group.
- **`chunksize`** is the *worker-memory* lever: halve it if you see
  KilledWorker with a named chunk. Raising it also shrinks client traffic,
  but (unlike `worker_merge`) raises worker memory needs — prefer
  `worker_merge` for that.
- **`worker_memory`** accepts `"4 GiB"` or a bare number (bare = GiB). More
  than 3 GiB is rarely needed with streaming; the old 6 GiB default only
  matters for `treereduce` runs.
- **Client memory** holds the full accumulated output regardless of executor.
  For all-years all-syst on a small submit machine, use
  `"group_mode": "per_group"` → one output file per year, one year in memory
  at a time.

## JMS/JMR unity reskim (2026-07-22) — comparison vs arc_r2

The arc_r2 pythia production ran the **old per-year JMS/JMR tables** (JMS
0.982/0.999 and JMR 1.09/1.108 centrals in 2017/2018; 2016 JMR ±20%); the UL
recommendation is **SF = 1 with 1% (JMS) / 2% (JMR)** for the groomed mass,
deviations doubled for ungroomed (see
`ai-wiki/wiki/bugs/zjet_stale_jmsjmr_tables_arc_r2.md`). The fix lives in
`smp_jetmass_run2/corrections.py` (`jmssf`/`jmrsf`) — **commit and push it
before running; the configs are identical to the arc_r2 ones and carry no
JMS/JMR knob of their own.**

Two configs:

- `zjet_pythia_all_minimal_rho_r2_casa2gb_jmsjmr_unity.json` — comparison run,
  `minimal_syst` (nominal + JER/JMS/JMR up/down). Enough to see the
  nominal-response shift and the new JMS/JMR bands. Run this first.
- `zjet_pythia_all_minimal_rho_r2_casa2gb_allsyst_jmsjmr_unity.json` — full 87
  systematics, to replace the arc_r2 pkls once the comparison is understood.

On coffea-casa (same session setup as the arc_r2 production):

```bash
git pull    # must include the corrections.py unity port
python scripts/run_analysis_cli.py --config configs/zjet_pythia_all_minimal_rho_r2_casa2gb_jmsjmr_unity.json
```

**Staging (filenames collide with arc_r2 — do not overwrite):** copy the
per-era `pythia_*.pkl` outputs into `unfold/inputs/zjet/rho/jmsjmr_unity/`,
pre-merge the four era pkls into `pythia_all.pkl` (same gotcha as arc_r2:
`_make_inputs_numpy` reads it before any merge step would write it), and copy
`data_all.pkl` + `herwig_all.pkl` over from `inputs/zjet/rho/arc_r2/` — data
carries no JMS/JMR, and herwig is overlay/bias-test only for this comparison.

Unfold-side comparison (tag registered in `unfolder_core.py`):

```bash
python scripts/run_unfolding.py --channel zjet --observable rho --tag jmsjmr_unity
```

then compare `outputs/zjet/rho/jmsjmr_unity/` against `outputs/zjet/rho/arc_r2/`
(response matrices, nominal unfolded, JMS/JMR uncertainty curves) in the
comparison app or via the `_previews/` PNG pairing. Expected differences:
2017/2018 response nominals shift (the 0.982 scale and 1.09/1.108 smearing go
away), the 2016 JMR band collapses ±20%→±2%, JMS bands go to a flat 1%.
