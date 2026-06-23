#!/usr/bin/env python
"""Probe whether LPC dask (condor) workers can write to the user's ~/nobackup.

Spins up a tiny LPCCondorCluster, runs a write-probe on each worker, and reports
where it landed. If workers cannot write/see ~/nobackup, the skim must use
``output_mode="accumulator"`` (gather to the login-node client) rather than
per-chunk Parquet writes.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

# resolved target of ~/nobackup (readlink -f ~/nobackup)
NOBACKUP = os.environ.get("NOBACKUP_REAL", "/uscms_data/d3/amandal2")


def _probe():
    host = socket.gethostname()
    info = {
        "host": host,
        "cwd": os.getcwd(),
        "nobackup_exists": os.path.isdir(NOBACKUP),
        "nobackup_writable": os.access(NOBACKUP, os.W_OK),
        "wrote": None,
        "err": None,
    }
    try:
        d = os.path.join(NOBACKUP, "omnifold_worker_probe")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"probe_{host}_{os.getpid()}.txt")
        with open(p, "w") as f:
            f.write(f"hello from {host}\n")
        info["wrote"] = p
    except Exception as exc:  # noqa: BLE001
        info["err"] = repr(exc)
    return info


def main():
    from distributed import Client
    from lpcjobqueue import LPCCondorCluster

    cluster = LPCCondorCluster(memory="4GB")
    cluster.scale(2)
    client = Client(cluster)
    print(f"[probe] dashboard: {cluster.dashboard_link}", flush=True)
    print("[probe] waiting for >=1 worker (timeout 300s)...", flush=True)
    client.wait_for_workers(1, timeout=300)

    results = client.run(_probe)
    print("---WORKER-PROBE---", flush=True)
    any_ok = False
    for worker, r in results.items():
        ok = bool(r.get("wrote"))
        any_ok = any_ok or ok
        print(f"  {worker} host={r['host']} exists={r['nobackup_exists']} "
              f"writable={r['nobackup_writable']} wrote={r['wrote']} err={r['err']}",
              flush=True)
    print(f"[probe] VERDICT: workers_can_write_nobackup={any_ok}", flush=True)
    client.close()
    cluster.close()


if __name__ == "__main__":
    sys.exit(main())
