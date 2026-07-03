#!/usr/bin/env python
"""Run the GEN-only Sherpa jet-mass ntuple over ALL Sherpa files on coffea-casa.

Uses the repo's existing casa plumbing (ensure_client -> CoffeaCasaCluster,
upload_package_if_casa, make_runner -> Dask executor) with the minimal
SherpaGenProcessor. Output = per-event .npz ntuple (w, mll, jetpt, ptz, mu, mg,
ru, rg), weighted by genWeight -- fine enough to derive a (pt, rho/mass) reweight,
and the SAME schema as the Herwig ev_*.npz files.

    # on a coffea-casa terminal, from the repo root, in the analysis env:
    python tools/arc_diagnostics/run_sherpa_gen_casa.py
    # quick smoke over 1 file first:
    python tools/arc_diagnostics/run_sherpa_gen_casa.py --test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import smp_jetmass_run2.notebook_utils as nb
from smp_jetmass_run2.zjet_sherpa_gen_processor import SherpaGenProcessor

SHERPA_TXT = "sherpa_RunIISummer16NanoAODv7_inclusive.txt"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true", help="1 file smoke test")
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--prepend", default="root://xcache/",
                    help="xrootd prefix for casa (default root://xcache/)")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "sherpa_gen_casa.npz"))
    args = ap.parse_args(argv)

    paths = nb.get_analysis_paths(REPO_ROOT)
    fileset = nb.build_fileset_from_txts(
        [SHERPA_TXT], paths.samples_mc_dir, args.prepend, split_ht=False)
    if args.test:
        k = next(iter(fileset))
        fileset = {k: fileset[k][:1]}
    n = sum(len(v) for v in fileset.values())
    print(f"Sherpa gen run: {n} file(s) over datasets {list(fileset)}", flush=True)

    client = nb.ensure_client(casa=True, test=args.test, useDefault=False,
                              executor_mode="dask-casa")
    nb.upload_package_if_casa(client, casa=True)

    run = nb.make_runner(use_dask=True, client=client,
                         chunksize=args.chunksize,
                         maxchunks=1 if args.test else None,
                         skipbadfiles=True)
    t0 = time.time()
    out = run(fileset, SherpaGenProcessor(), treename="Events")
    print(f"Done in {nb.format_time(time.time() - t0)}", flush=True)

    cols = {k: v.value for k, v in out.items()}   # column_accumulator -> np.array
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **cols)
    n = len(cols["jetpt"])
    print(f"Saved {args.out}  ({n:,} selected events)", flush=True)
    pt = cols["jetpt"]
    for lo, hi in [(200, 290), (290, 400), (400, 1e9)]:
        m = (pt >= lo) & (pt < hi)
        print(f"  pt[{lo},{hi}): n={int(m.sum())}", flush=True)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
