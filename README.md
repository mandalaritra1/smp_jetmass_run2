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

## Install

```bash
pip install -e .            # core deps
pip install -e ".[dask]"    # + dask/distributed + fsspec-xrootd for cluster running
```

## Running

### CLI (recommended) — all three channels, one entry point

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
| `executor_mode` | where it runs | `iterative` · `futures` · `dask-local` · `dask-lpc` · `dask-casa` |
| `casa` / `redirector` / `prependstr` | file access | `dask-casa` uses `casa`+`root://xcache/`; LPC/local use a redirector like `root://cmsxrootd.fnal.gov/` |
| `test` | 1 file / 1 chunk smoke run | `true`/`false` |

Channel → processor dispatch is `notebook_utils.get_processor_class(mode, channel)`;
hadronic filesets are built by `build_hadronic_fileset` from `samples/hadronic/*.json`
(ported from GluonJetMass `run.py`), zjet filesets from `samples/zjet/**` `.txt` lists.
On `dask-casa`/`dask-lpc` the whole package (incl. `corrections/`) is shipped to the
workers (`upload_package_if_casa` / `transfer_input_files`), so `importlib.resources`
resolves the bundled correction files remotely. Outputs are pickled to `outputs/`.

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
