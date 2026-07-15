# smp_jetmass_run2

Unified Run 2 jet-mass cross-section framework for **three channels** — Z+jet,
dijet, and trijet — built on the coffea-2025 `zjet_corrections` framework. The
dijet and trijet processors were ported from
[`GluonJetMass`](https://github.com/laurenhay/GluonJetMass) and refactored to the
Z+jet `QJetMassProcessor` conventions (mode-gated `register_hist`, channel-aware
`util_binning`, `fill_hist`, `Log`, XS normalization in `postprocess`). Their event
selections are byte-for-byte identical to the originals (the one intentional change
is HEM handling — see below).

## Layout (flat package)

```
smp_jetmass_run2/            <- repo root
  pyproject.toml             (distribution name: smp-jetmass-run2)
  smp_jetmass_run2/          <- the importable package
    corrections.py           shared corrections (zjet) + ported hadronic functions
    hist_utils.py            channel-aware util_binning, register_hist, fill_hist
    notebook_utils.py        runner + get_processor_class(mode, channel) dispatch
    zjet_processor.py        QJetMassProcessor (Z+jet)
    dijet_processor.py       DijetProcessor    (ported)
    trijet_processor.py      TrijetProcessor   (ported)
    corrections/             bundled JEC/JER/PU/golden-JSON + prescales
  configs/                   example run configs (one per channel)
  samples/
    zjet/{data,mc}/          Z+jet .txt file lists (+ mc/backgrounds/)
    hadronic/                dijet+trijet JSON filesets (shared: QCD MC, JetHT data)
  scripts/run_analysis_cli.py
```

## Install (Only for local environment, Ignore if running on coffea-casa of LPC)

```bash
pip install -e .            # core deps
pip install -e ".[dask]"    # + dask/distributed + fsspec-xrootd for cluster running
```

For notebook-driven running, also install the notebook extras:

```bash
pip install -e ".[dask,notebook]"
```

## Run Environments

This repository can run locally, on the LPC, on CERN **lxplus** (HTCondor), or on
[coffea.casa](https://coffea.casa). For large Run 2 jobs, prefer `dask-casa`,
`dask-lpc`, or `dask-lxplus` rather than the local/futures executors. The actual
analysis behavior is still controlled by the JSON config: `executor_mode`,
`redirector`, `prependstr`, and `casa` decide where the job runs and how input
files are resolved.

### LPC Setup

Log in with a forwarded port if you plan to use JupyterLab or the Dask dashboard:

```bash
ssh -Y -L 8XXX:127.0.0.1:8XXX LPCUSERNAME@cmslpc-el9.fnal.gov
```

It is usually best to work in your LPC `nobackup` area:

```bash
cd nobackup
```

Create a CMS proxy before reading remote NanoAOD files:

```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```

Clone this repository:

```bash
git clone https://github.com/mandalaritra1/smp_jetmass_run2.git
cd smp_jetmass_run2
```

Follow the `lpcjobqueue` setup instructions from
[CoffeaTeam/lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue). If you use
an LPC coffea Singularity helper, start a coffea-2025 image, for example:

```bash
./shell coffeateam/coffea-dask-almalinux9:2025.12.0-py3.12
```



For LPC runs, use configs with:

```json
"executor_mode": "dask-lpc",
"casa": false,
"redirector": "lpc",
"prependstr": "root://cmsxrootd.fnal.gov/"
```

Example:

```bash
python scripts/run_analysis_cli.py --config configs/dijet_pythia_2018.json
```

Optional JupyterLab from inside the environment/container:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8XXX
```

Then open the forwarded Jupyter link in your local browser.

### lxplus Setup (CERN HTCondor)

The lxplus analogue of `lpcjobqueue` is
[`dask-lxplus`](https://github.com/cernops/dask-lxplus), which wraps
`dask-jobqueue`'s `HTCondorCluster` in a `CernCluster` that knows about CERN's
batch quirks (JobFlavour walltime tiers, scheduler host/port so workers can dial
back, LCG-view environment shipping). This repo drives it through
`executor_mode: "dask-lxplus"`.

Log in with a forwarded port for the Dask dashboard / JupyterLab:

```bash
ssh -Y -L 8XXX:127.0.0.1:8XXX LXPLUSUSERNAME@lxplus.cern.ch
```

Get a CMS proxy before reading remote NanoAOD:

```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```

Source an LCG view, then put this package on the path. Because the cluster is
created with `lcg=True` + `container_runtime="none"`, the **submitting** environment
(this sourced view) is what runs on the batch workers — so coffea/dask must live in
it. A recent view already bundles everything you need: `LCG_110`
(`x86_64-el9-gcc13-opt`) ships coffea 2026.5.0, dask 2025.2.0, dask-awkward, and
**dask-lxplus 0.3.3**, so no cluster deps have to be installed — only this repo:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
git clone https://github.com/mandalaritra1/smp_jetmass_run2.git
cd smp_jetmass_run2
python -m pip install --user --no-deps -e .   # just this package; view provides the rest
```

(If you pick an older view without `dask-lxplus`, add `pip install --user dask-lxplus`.)

For lxplus runs, use configs with:

```json
"executor_mode": "dask-lxplus",
"casa": false,
"redirector": "lxplus",
"prependstr": "root://cms-xrd-global.cern.ch/"
```

Example (a ready-made config ships in `configs/`):

```bash
python scripts/run_analysis_cli.py --config configs/dijet_pythia_2018_lxplus.json
```

Two optional environment variables tune the batch submission (sensible defaults
otherwise):

| env var | default | meaning |
|---|---|---|
| `DASK_LXPLUS_JOB_FLAVOUR` | `longlunch` | HTCondor walltime tier: `espresso`(20m) · `microcentury`(1h) · `longlunch`(2h) · `workday`(8h) · `tomorrow`(1d) · `testmatch`(3d) · `nextweek`(1w) |
| `DASK_LXPLUS_LOG_DIR` | `~/dask_lxplus_logs` | where condor writes worker job logs (point at EOS for long-lived logs) |

The cluster auto-scales `adapt(minimum=1, maximum=100)`. If most of your input
files live on European sites, swap the redirector for the closer
`root://xrootd-cms.infn.it/`.

### Purdue Analysis Facility (stub — not yet usable)

**`dask-purdue`** targets the [Purdue AF](https://purdue-af.readthedocs.io/)
(`cms.geddes.rcac.purdue.edu/hub`; Purdue, CERN, or FNAL login) via Dask
Gateway inside their JupyterLab. Status 2026-07-15: login and the Gateway API
work, but the facility's global pixi env did not actually provide coffea when
tried, so **the worker software environment still has to be figured out** —
most likely a project pixi env in this repo pinning `coffea==2026.5.0`, passed
via `PURDUE_AF_PIXI_PROJECT` (or `PURDUE_AF_CONDA_ENV`).
`PURDUE_AF_MAX_WORKERS` caps adaptive scaling (default 200; facility limit 400
cores/user). Example config: `configs/zjet_pythia_all_minimal_rho_r2_purdue.json`.
The package ships via `client.upload_file` like the other dask modes; use the
`"global"` AAA redirector.

(An INFN CMS AF backend existed briefly and was removed — the facility needs a
registration we don't have. It's in git history if that changes.)

### coffea.casa Setup

Go to [coffea.casa](https://coffea.casa), log in, choose a recent coffea-2025
image, and press **Start**.

Clone this repository in the terminal:

```bash
git clone https://github.com/mandalaritra1/smp_jetmass_run2.git
cd smp_jetmass_run2
python -m pip install -e ".[dask,notebook]"
```

If a default Dask cluster is already running from the coffea.casa side panel, shut
it down before using this repo's `dask-casa` path. The runner creates its own
`CoffeaCasaCluster` unless `useDefault` is set in the config.

For coffea.casa runs, use configs with:

```json
"executor_mode": "dask-casa",
"casa": true,
"redirector": "casa",
"prependstr": "root://xcache/"
```

Example:

```bash
python scripts/run_analysis_cli.py --config configs/zjet_pythia_2018.json
```

For interactive configuration, open `notebooks/run_analysis.ipynb`, select
`executor = dask-casa`, lock the config, and run from the notebook. The notebook
and CLI both call the same `notebook_utils.run_from_config` path, so a
notebook-generated `configs/last_run.json` can be replayed headlessly.

## Running

### Notebook (interactive tuning) — `notebooks/run_analysis.ipynb`

`ipywidgets` UI to pick channel / dataset / era / mode / systematics / executor,
**lock** them to `configs/last_run.json`, then run + plot inline. The channel
dropdown updates the allowed dataset/mode options automatically, and the
**load cfg** dropdown pre-fills every widget from any saved `configs/*.json`.
It runs through the exact same `notebook_utils.run_from_config` path as the
CLI, so a notebook-tuned config can be replayed headlessly:
`--config configs/last_run.json`.

**[`configs/README.md`](configs/README.md) is the map of the configs folder**
— naming scheme, config families, and the recommended
`chunksize` / `worker_memory` / `worker_merge` settings per run type.

### CLI — all three channels, one entry point

```bash
python scripts/run_analysis_cli.py --config configs/dijet_pythia_2018.json
```

The config JSON selects everything; see `configs/` for one example per channel.
Key fields (defaults filled in by `validate_analysis_config`):

| field | meaning | values |
|---|---|---|
| `channel` | which processor | `zjet` · `dijet` · `trijet` |
| `dataset` | sample group | `pythia` · `mg_pythia8` · `herwig` · `data` (hadronic); zjet also `powheg`/`st`/`backgrounds`/`pythia_local` |
| `era` | data-taking period | `2016APV` · `2016` · `2017` · `2018` · `all` |
| `mode` | histogram set | `minimal` (mass+response) · `minimal_rho` (rho+response) · `validation`/`full` (diagnostics) · `mass_jk`/`rho_jk` (jackknife) |
| `systematic_profile` | which systematics | `all_syst` · `minimal_syst` · `no_syst` |
| `executor_mode` | where it runs | `iterative` · `futures` · `dask-local` · `dask-lpc` · `dask-lxplus` · `dask-casa` · `dask-purdue` (stub, see AF section) |
| `casa` / `redirector` / `prependstr` | file access | `dask-casa` uses `casa`+`root://xcache/`; LPC uses `lpc`+`root://cmsxrootd.fnal.gov/`; lxplus uses `lxplus`+`root://cms-xrd-global.cern.ch/` |
| `test` | 1 file / 1 chunk smoke run | `true`/`false` |
| `dataset_filter` | hadronic only: keep DAS names containing this substring | e.g. `"HT1000to1500"` |

> **Hadronic test tip:** the HT-binned MG+pythia / herwig samples start at low HT
> (HT200to300, …) where nothing passes the dijet/trijet `pt>200` / 3-jet selection,
> so `test: true` on the first bin is vacuous. Set `dataset_filter` to a high-HT bin
> (e.g. `HT1000to1500`) for a meaningful test — see `configs/dijet_mg_pythia8_2018_test.json`.

Channel → processor dispatch is `notebook_utils.get_processor_class(mode, channel)`;
hadronic filesets are built by `build_hadronic_fileset` from `samples/hadronic/*.json`
(ported from GluonJetMass `run.py`), zjet filesets from `samples/zjet/**` `.txt` lists.
On `dask-casa`/`dask-lpc`/`dask-lxplus` the whole package (incl. `corrections/`) is
shipped to the workers (`upload_package_if_casa` via `client.upload_file`, plus
`transfer_input_files` on LPC), so `importlib.resources` resolves the bundled
correction files remotely. Outputs are pickled to `outputs/`.

### Programmatic (notebook / quick local test)

```python
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from smp_jetmass_run2.notebook_utils import get_processor_class

cls = get_processor_class(mode="minimal", channel="dijet")   # or "trijet" / "zjet"
run = processor.Runner(executor=processor.IterativeExecutor(),
                       schema=NanoAODSchema, chunksize=50_000, maxchunks=1)
fileset = {"QCD_Pt_170to300_..._UL18": ["/path/to/file.root"]}
out = run(fileset, cls(do_gen=True, mode="minimal",
                       jet_systematics=["nominal"]), treename="Events")
out = cls(do_gen=True, mode="minimal").postprocess(out)   # applies xs*lumi/sumw
```

`mode="minimal"` → mass + `response_matrix_{u,g}`; `mode="minimal_rho"` → rho +
`response_matrix_rho_{u,g}`; `*_jk` modes add the `jk` axis.

## Channel orthogonality (DATA)

Each channel's data processor logs the `(run, luminosityBlock, event)` of every
**finally selected** data event into an `event_id` accumulator
(`processor.column_accumulator`). To check whether the three channels select
disjoint events, run each on the same data and compare:

```python
from smp_jetmass_run2.notebook_utils import event_id_set, channel_overlap
channel_overlap({"dijet": out_dijet, "trijet": out_trijet, "zjet": out_zjet})
# -> {'n[dijet]': ..., 'n[trijet]': ..., 'overlap[dijet&trijet]': ..., ...}
```

A non-zero `overlap[...]` means the selections share events (e.g. dijet and trijet
overlap because a ≥3-jet event whose leading two jets are back-to-back passes both
topologies). Only present for data runs (`do_gen=False`).

## Corrections reconciliation (key decisions)

The hadronic processors reuse the Z+jet correction functions
(`GetJetCorrections`, `get_pu_weights`, `GetL1PreFiringWeight`, `GetPDFweights`, …)
and emit **canonical** systematic-axis names (`pu`, `l1prefiring`, `pdf`, `isr`,
`fsr`, `JES_<source>Up/Down`). Deliberately kept from GluonJetMass:

- **JMS/JMR** use the GluonJetMass values (`applyjmsSF`/`applyjmrSF`: flat sf=1.0,
  ±1% JMS / ±2% JMR) — *not* zjet's `jmssf`/`jmrsf`.
- **q2** is kept split as `Q2muF` + `Q2muR` (not collapsed to a single `q2`).

**Binning** is a single channel-aware source of truth (`util_binning(channel=…)`):
hadronic keeps its pT edges and uses zjet's coarse mass scheme extended to a
high-mass tail; **rho** adopts zjet's definition `2*log10(m/(pt*R))` (R=0.8) and
edges (plus extra low-rho bins).

**HEM** is the only selection change: 2018 MC gets a flat lumi-fraction HEM
*weight* (`HEMVeto(..., isMC=True)`) instead of a hard veto; 2018 data still gets the
hard veto.

**Cross-section normalization** (xs·lumi·1000/sumw) is applied per dataset in
`postprocess` (zjet-style), so histograms are filled with the raw generator weight
during `process`.
