#!/usr/bin/env python
"""Run the GEN-only Sherpa jet-mass shape over ALL Sherpa files on coffea-casa.

Uses the repo's existing casa plumbing (ensure_client -> CoffeaCasaCluster,
upload_package_if_casa, make_runner -> Dask executor) with the minimal
SherpaGenProcessor. Output = pickle of {rho_g, rho_u, mass_g} hist.Hist,
weighted by genWeight (a normalized SHAPE; absolute yields are meaningless for
this pre-UL sample).

    # on a coffea-casa terminal, from the repo root, in the analysis env:
    python tools/arc_diagnostics/run_sherpa_gen_casa.py
    # quick smoke over 1 file first:
    python tools/arc_diagnostics/run_sherpa_gen_casa.py --test
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

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
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "sherpa_gen_casa.pkl"))
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(out, fh)
    print(f"Saved {args.out}", flush=True)
    for key, h in out.items():
        print(f"  {key}: sumw={h.sum().value:.3g}, entries={int(h.sum().value>0)}", flush=True)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
