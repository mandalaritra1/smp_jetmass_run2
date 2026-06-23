# Z+jet OmniFold skim workflow

Minimal AK8 Z(→μμ)+jet skim → flat per-event Parquet → OmniFold unfolding
(in the `omnifold` repo). Feature space for the first test: leading AK8 jet
`pt, eta, mass, phi` (skim also stores `msoftdrop` for reco and reconstructed
gen soft-drop mass).

## Pieces
- `smp_jetmass_run2/zjet_omnifold_skimmer.py` — `ZJetOmniFoldSkimmer`
  (adapted from `ZJetMinimalProcessor`; AK8 `FatJet`↔`GenJetAK8`, per-event
  Parquet columns). **Memory-flat**: `output_mode="parquet"` writes one file
  per chunk and returns only counts — no big `column_accumulator` merge.
- `scripts/run_zjet_skim.py` — runner; `executor_mode` ∈
  `iterative` / `futures` / `dask-local` / `dask-casa` / `dask-lpc`.
- `scripts/merge_skims.py` — stream `{dataset}/part-*.parquet` → one Parquet
  per dataset (row-group by row-group; no `hadd`).
- `configs/zjet_omnifold_skim_{local,casa,lpc}.json`.

## 1. Local debug (this Mac)
```bash
# uses the omnifold venv (coffea 2026.5 + pyarrow); 80 MB DYJetsToLL HT-400to600 file
/Users/aritra/Projects/omnifold/.venv/bin/python scripts/run_zjet_skim.py \
    --config configs/zjet_omnifold_skim_local.json
```
→ `outputs/skims/<dataset>/part-*.parquet`.

## 2. LPC (HTCondor via lpcjobqueue, apptainer)
```bash
ssh cmslpc303 && cd ~/nobackup
# one-time: bootstrap the coffea container + ./shell launcher
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh                         # pins a coffea-dask image, writes ./shell
# enter the container (coffea 2025/2026, py3.12) and run:
./shell coffeateam/coffea-dask-almalinux9:2025.12.0-py3.12
export X509_USER_PROXY=$HOME/x509up_u$(id -u)   # proxy must be valid (voms-proxy-init)
python scripts/run_zjet_skim.py --config configs/zjet_omnifold_skim_lpc.json --max-files 5  # smoke test
python scripts/run_zjet_skim.py --config configs/zjet_omnifold_skim_lpc.json                # full
python scripts/merge_skims.py --indir ~/nobackup/omnifold_skims --outdir ~/nobackup/omnifold_skims_merged
```
Workers write per-chunk Parquet directly into `outdir` on `~/nobackup`
(NFS-visible from LPC condor nodes), so the run never needs a large merge.
Per-era splitting is optional (just add/remove `samples` entries) — not required
for memory since output is flat.

## 3. coffea-casa
Same as LPC but `--config configs/zjet_omnifold_skim_casa.json` from a casa
notebook/terminal (`outdir` on casa-writable storage).

## 4. Copy skims here & unfold
```bash
scp -r cmslpc303:nobackup/omnifold_skims_merged/ /Users/aritra/Projects/omnifold/data/skims/
cd /Users/aritra/Projects/omnifold
.venv/bin/python -m omnifold_zjet.run_zjet_omnifold \
    --sim data/skims/DYJetsToLL_pythia_UL18.parquet \
    --data data/skims/DYJetsToLL_herwig_UL18.parquet \
    --mode gen_closure --features pt eta mass phi --iterations 4 --out outputs/zjet_omnifold
```
Modes: `self_closure` (split one MC), `gen_closure` (Pythia→Herwig),
`data` (real-data target).
