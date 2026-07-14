#!/usr/bin/env python
"""Run the ZJetOmniFoldSkimmer across local / coffea-casa / LPC executors.

The skim writes one Parquet file per chunk (memory-flat) under ``outdir`` (or,
in ``accumulator`` mode for small debug runs, one merged Parquet per dataset).

    python scripts/run_zjet_skim.py --config configs/zjet_omnifold_skim_local.json
    python scripts/run_zjet_skim.py --config configs/zjet_omnifold_skim_lpc.json \
        --max-files 5            # quick remote smoke test

Config keys (JSON):
    executor_mode : "futures" | "dask-local" | "dask-casa" | "dask-lpc"
    redirector    : "local" | "lpc" | "casa"   (prepended to /store/... paths)
    outdir        : output directory for parquet
    output_mode   : "parquet" | "accumulator"
    pt_min        : leading-jet pT floor (GeV)
    chunksize, maxchunks, workers
    treereduce    : false (default) streams chunk outputs to the client as they
                    finish (StreamingDaskExecutor, no worker-side merge spike);
                    true restores coffea DaskExecutor's worker tree reduction
    samples       : list of {dataset, files:[...]} or {dataset, filelist: "path.txt"}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _quiet_logs():
    """Silence the dask/distributed log flood so skims.log only shows skim progress."""
    try:
        import dask
        dask.config.set({
            "distributed.logging.distributed": "error",
            "distributed.logging.distributed.client": "error",
            "distributed.logging.bokeh": "error",
            "distributed.admin.system-monitor.disk": False,
        })
    except Exception:
        pass
    for name in ("distributed", "distributed.scheduler", "distributed.worker",
                 "distributed.nanny", "distributed.core", "distributed.batched",
                 "distributed.comm", "distributed.deploy", "distributed.utils_perf",
                 "distributed.active_memory_manager", "dask_jobqueue", "coffea_casa"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # The dashboard (bokeh/tornado) logs "Token is expired" at ERROR level when a
    # stale dashboard tab reconnects -> need CRITICAL to silence those.
    for name in ("tornado", "tornado.application", "tornado.general",
                 "tornado.access", "bokeh", "bokeh.server", "bokeh.server.views.ws"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

import awkward as ak  # noqa: E402
import numpy as np  # noqa: E402
from coffea import processor  # noqa: E402
from coffea.nanoevents import NanoAODSchema  # noqa: E402

from smp_jetmass_run2.streaming_executor import StreamingDaskExecutor  # noqa: E402
from smp_jetmass_run2.zjet_omnifold_skimmer import (  # noqa: E402
    ZJetOmniFoldSkimmer, to_ak_record)

REDIRECTOR_PREPENDS = {
    "local": "",
    "lpc": "root://cmsxrootd.fnal.gov/",
    "casa": "root://xcache/",
}


def _resolve_files(entry, redirector, max_files):
    """Return the list of (possibly redirector-prefixed) file URLs for a sample."""
    prepend = REDIRECTOR_PREPENDS.get(redirector, "")
    if "files" in entry:
        files = list(entry["files"])
    elif "filelist" in entry:
        path = entry["filelist"]
        if not os.path.isabs(path):
            path = str(REPO_ROOT / path)
        files = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    else:
        raise ValueError(f"sample {entry!r} needs 'files' or 'filelist'")

    urls = []
    for f in files:
        if f.startswith("root://") or os.path.isabs(f) and os.path.exists(f):
            urls.append(f)
        elif f.startswith("/store/"):
            urls.append(prepend.rstrip("/") + "/" + f)
        else:
            urls.append(f)
    if max_files:
        urls = urls[:max_files]
    return urls


def _build_fileset(cfg, max_files, only=None):
    redirector = cfg.get("redirector", "local")
    fileset = {}
    only = set(only) if only else None
    for entry in cfg["samples"]:
        if only and entry["dataset"] not in only:
            continue
        urls = _resolve_files(entry, redirector, max_files)
        fileset[entry["dataset"]] = urls
    if only and not fileset:
        raise SystemExit(f"--only {only} matched no dataset in config")
    return fileset


def _sched_banner(client):
    """Print the scheduler address prominently so workers can be killed manually."""
    try:
        addr = client.scheduler.address
    except Exception:
        addr = getattr(client, "scheduler_info", lambda: {})().get("address", "?")
    print("=" * 60, flush=True)
    print(f"[skim] SCHEDULER ADDRESS: {addr}", flush=True)
    print(f"[skim] dashboard: {getattr(client, 'dashboard_link', '?')}", flush=True)
    print("[skim] to kill workers:  Client(\"%s\", security=<casa Security>)"
          ".retire_workers([...], close_workers=True)" % addr, flush=True)
    print("=" * 60, flush=True)


def _dask_executor(cfg, client, **dask_kwargs):
    """Streaming (accumulate-as-completed) executor by default; the
    ``treereduce`` config key / --treereduce flag restores coffea's
    DaskExecutor with its worker-side tree reduction. Streaming keeps the
    merge on the client -- (running total + one chunk) peak -- so many-axis
    histogram skims can't OOM small workers at the end of the run. The client
    must still hold the full accumulated output either way.
    """
    if cfg.get("treereduce", False):
        return processor.DaskExecutor(client=client, **dask_kwargs)
    dask_kwargs.pop("treereduction", None)
    return StreamingDaskExecutor(client=client, **dask_kwargs)


def _make_executor(cfg):
    _quiet_logs()
    mode = cfg.get("executor_mode", "futures")
    if mode == "iterative":
        return processor.IterativeExecutor(compression=None), None
    if mode == "futures":
        return processor.FuturesExecutor(workers=cfg.get("workers", 4), compression=None), None

    if mode == "dask-local":
        from distributed import Client, LocalCluster

        cluster = LocalCluster(n_workers=cfg.get("workers", 4), threads_per_worker=1)
        client = Client(cluster)
        _sched_banner(client)
        return _dask_executor(cfg, client, retries=6), client

    if mode == "dask-casa":
        import shutil
        from coffea_casa import CoffeaCasaCluster
        from distributed import Client

        cluster = CoffeaCasaCluster(cores=1, memory=cfg.get("worker_memory", "6 GiB"))
        cluster.adapt(minimum=cfg.get("min_workers", 1), maximum=cfg.get("max_workers", 100))
        client = Client(cluster)
        # Ship the user package so ZJetOmniFoldSkimmer is importable on every
        # (current and future) casa worker -- matches notebook_utils' casa path.
        pkg_dir = REPO_ROOT / "smp_jetmass_run2"
        zip_path = "/tmp/smp_jetmass_run2_omnifold.zip"
        shutil.make_archive(zip_path[:-4], "zip", str(pkg_dir.parent), pkg_dir.name)
        client.upload_file(zip_path)
        _quiet_logs()
        _sched_banner(client)
        return _dask_executor(cfg, client, retries=6, treereduction=8), client

    if mode == "dask-lpc":
        import shutil
        from lpcjobqueue import LPCCondorCluster
        from distributed import Client

        # Ship the user package to workers: zip it, transfer it, and upload_file
        # so it lands on sys.path of every (current and future) worker.
        pkg_dir = REPO_ROOT / "smp_jetmass_run2"
        zip_path = "/tmp/smp_jetmass_run2_omnifold.zip"
        shutil.make_archive(zip_path[:-4], "zip", str(pkg_dir.parent), pkg_dir.name)

        cluster = LPCCondorCluster(
            memory=cfg.get("worker_memory", "6GB"),
            transfer_input_files=[zip_path],
            scheduler_options={"dashboard_address": ":%d" % cfg.get("dashboard_port", 8787)},
        )
        n_fixed = cfg.get("n_workers")
        if n_fixed:
            cluster.scale(n_fixed)
        else:
            cluster.adapt(minimum=cfg.get("min_workers", 1),
                          maximum=cfg.get("max_workers", 100))
        client = Client(cluster)
        client.upload_file(zip_path)
        _quiet_logs()
        _sched_banner(client)
        return _dask_executor(cfg, client, retries=6, treereduction=8,
                              status=False), client

    raise ValueError(f"unknown executor_mode: {mode}")


def _write_accumulator_parquet(out, outdir):
    """In accumulator mode, write one merged Parquet per dataset."""
    written = []
    for dataset, payload in out.items():
        cols = payload.get("columns")
        if not cols:
            continue
        arrays = {k: acc.value for k, acc in cols.items()}
        os.makedirs(os.path.join(outdir, dataset), exist_ok=True)
        path = os.path.join(outdir, dataset, "merged.parquet")
        ak.to_parquet(to_ak_record(arrays), path)
        written.append(path)
    return written


def main(argv=None):
    _quiet_logs()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--max-files", type=int, default=None,
                    help="Cap files per sample (quick smoke tests).")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only run these exact dataset name(s), e.g. "
                         "--only nlo_ptz_100To250_UL18NanoAODv9 (one ptZ bin/era).")
    ap.add_argument("--output-mode", choices=["parquet", "accumulator"], default=None,
                    help="Override config output_mode.")
    ap.add_argument("--executor-mode", default=None, help="Override config executor_mode.")
    ap.add_argument("--n-workers", type=int, default=None,
                    help="Fixed worker count (overrides adaptive scaling).")
    ap.add_argument("--treereduce", action="store_true",
                    help="Merge chunk outputs on the workers (coffea DaskExecutor "
                         "tree reduction) instead of streaming them to the client.")
    args = ap.parse_args(argv)

    cfg = json.loads(args.config.read_text())
    if args.output_mode:
        cfg["output_mode"] = args.output_mode
    if args.executor_mode:
        cfg["executor_mode"] = args.executor_mode
    if args.n_workers:
        cfg["n_workers"] = args.n_workers
    if args.treereduce:
        cfg["treereduce"] = True
    outdir = cfg.get("outdir", "outputs/skims")
    if not os.path.isabs(outdir):
        outdir = str(REPO_ROOT / outdir)
    output_mode = cfg.get("output_mode", "parquet")

    fileset = _build_fileset(cfg, args.max_files, only=args.only)
    n_files = sum(len(v) for v in fileset.values())
    print(f"[skim] datasets={list(fileset)}  files={n_files}  mode={output_mode}  "
          f"executor={cfg.get('executor_mode','futures')}  outdir={outdir}", flush=True)

    proc = ZJetOmniFoldSkimmer(
        output_mode=output_mode,
        outdir=outdir if output_mode == "parquet" else None,
        pt_min=cfg.get("pt_min", 150.0),
    )
    executor, client = _make_executor(cfg)
    runner = processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=cfg.get("chunksize", 50000),
        maxchunks=cfg.get("maxchunks", None),
        skipbadfiles=cfg.get("skipbadfiles", False),
        xrootdtimeout=cfg.get("xrootdtimeout", 120),
    )

    out = runner(fileset, treename="Events", processor_instance=proc)

    print("---SKIM-SUMMARY---", flush=True)
    total_files = []
    for dataset, payload in out.items():
        print(f"  {dataset}: read={payload['n_read']} selected={payload['n_selected']} "
              f"matched={payload['n_matched']} chunkfiles={len(payload.get('files', []))}",
              flush=True)
        total_files += payload.get("files", [])

    if output_mode == "accumulator":
        total_files = _write_accumulator_parquet(out, outdir)

    print("---OUTPUT-FILES---", flush=True)
    for p in total_files:
        print(p, flush=True)

    if client is not None:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
