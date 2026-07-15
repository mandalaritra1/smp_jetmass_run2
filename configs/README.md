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
