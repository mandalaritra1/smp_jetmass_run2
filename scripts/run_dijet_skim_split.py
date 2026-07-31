#!/usr/bin/env python
"""Run a hadronic rho_skim over the QCD HT bins as one submit per bin, on a
single long-lived Dask client.

Nothing here is dijet-specific despite the name -- `--bins` is matched against
the config's dataset names, so the same script drives the trijet skim and the
per-era data runs (`--bins Run2018A Run2018B Run2018C Run2018D`).

Why split at all: the coffea Runner gathers the whole accumulator back into the
CLIENT before pickling it, and the coffea-casa submit pod is cgroup-capped at
4 cores / 8 GiB (`nproc` reports the node's 112, not the quota). An all-in-one
run over the full UL18 madgraphMLM QCD therefore dies at the final reduce no
matter how much memory the workers get. One bin per submit keeps the client
peak ~8x smaller.

Why not a bash loop: a new process per bin means a new cluster per bin, so the
worker pool restarts from zero every time. Here the client is created once and
passed into run_from_config (which reuses and leaves open any client it is
given), so the pool stays hot across bins; only the accumulator is freed
between them.

    python scripts/run_dijet_skim_split.py --config configs/dijet_mg_pythia8_2018_rho_skim.json
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import smp_jetmass_run2.notebook_utils as nbutils  # noqa: E402

log = functools.partial(print, flush=True)

#### High HT first: fewest files, most events surviving pT > 200, so the useful
#### statistics land early and a run stopped partway is still usable.
HT_BINS = [
    "HT2000toInf", "HT1500to2000", "HT1000to1500", "HT700to1000",
    "HT500to700", "HT300to500", "HT200to300", "HT100to200",
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--bins", nargs="*", default=HT_BINS,
                   help="HT bin substrings to run (default: all, high HT first)")
    p.add_argument("--worker-memory", default=None,
                   help="override per-worker memory (default: the config's)")
    p.add_argument("--chunksize", type=int, default=None)
    args = p.parse_args(argv)

    base = nbutils.validate_analysis_config(json.loads(args.config.read_text()))
    if args.worker_memory:
        base["worker_memory"] = args.worker_memory
    if args.chunksize:
        base["chunksize"] = args.chunksize

    #### One client for the whole loop -- this is the hot pool.
    client = nbutils.ensure_client(
        casa=base.get("casa", False),
        test=base.get("test", False),
        useDefault=base.get("useDefault", False),
        executor_mode=base.get("executor_mode"),
        worker_memory=base.get("worker_memory"),
        n_workers=base.get("n_workers"),
    )
    #### run_from_config only ships the package zip to the workers when it owns
    #### the client, so a caller that supplies one must do it here or every task
    #### dies with ModuleNotFoundError: smp_jetmass_run2. upload_file registers
    #### a worker plugin, so workers added later by adapt() get it too.
    nbutils.upload_package_if_casa(client, casa=base.get("casa", False))
    log(f"[split] client up: {client}")
    log(f"[split] dashboard: {getattr(client, 'dashboard_link', '?')}")

    done, failed = [], []
    t0 = time.time()
    for i, b in enumerate(args.bins, 1):
        cfg = dict(base, dataset_filter=b)
        log(f"\n{'=' * 70}\n[split] ({i}/{len(args.bins)}) {b}  "
            f"elapsed {nbutils.format_time(time.time() - t0)}\n{'=' * 70}")
        try:
            outputs, out = nbutils.run_from_config(
                cfg, client=client, repo_root=REPO_ROOT, log=log)
        except Exception as exc:              # one bad bin must not kill the rest
            log(f"[split] BIN_FAIL {b}: {type(exc).__name__}: {exc}")
            failed.append(b)
            continue
        finally:
            #### drop the accumulator before the next bin so the 8 GiB client
            #### never holds two bins at once
            out = None
            gc.collect()

        for src in map(Path, outputs):
            dst = src.with_name(f"{src.stem}_{b}{src.suffix}")
            shutil.move(str(src), str(dst))
            log(f"[split] BIN_OK {b} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
            done.append(str(dst))
        gc.collect()

    log(f"\n[split] done in {nbutils.format_time(time.time() - t0)}: "
        f"{len(done)} ok, {len(failed)} failed {failed}")
    for d in done:
        log(d)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
