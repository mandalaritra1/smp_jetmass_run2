#!/usr/bin/env python
"""Recompute the `corrections.sumw_qcd_mg` denominators from the NanoAOD files.

WHAT THE NUMBER IS (re-derived 2026-08-07 -- it was undocumented, and the
obvious guess is wrong):

    sumw = sum over EVERY event of EVERY file in the dataset of
           Events/LHEWeight_originalXWGTUP

It is NOT `Runs/genEventSumw`. For the UL madgraphMLM QCD_HT NanoAODv9 samples
`genWeight` is identically 1.0, so `genEventSumw` equals the event count -- but
`dijet_processor`/`trijet_processor` fill with

    weights = events.LHEWeight.originalXWGTUP   # falls back to genWeight

whenever the LHEWeight collection exists, which it does for all of these. The
xs*lumi*1000/sumw factor applied in `postprocess` therefore has to be divided
by the sum of exactly that branch. Verified: the stored 2016APV / 2016 / 2017
`QCD_HT2000toInf` entries are reproduced to a relative 1e-9 by this script (the
last digits differ because the historical values were accumulated in float32).

USAGE (needs a valid grid proxy; the CA path is mandatory on macOS):

    voms-proxy-init -voms cms -valid 24:00
    X509_CERT_DIR=$HOME/.globus/certificates \
        python tools/compute_sumw_qcd_mg.py HT100to200 --cache /tmp/sumw.jsonl

`--cache` is a per-file jsonl; rerunning skips files already in it, so an
interrupted run (or an expired proxy) resumes rather than restarts. A dataset
is only reported COMPLETE when every file in the fileset is accounted for --
never paste a PARTIAL number into corrections.py, the normalization has to be
exact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import uproot

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILESET = REPO_ROOT / "samples" / "hadronic" / "fileset_MG_pythia8.json"

#### Global redirector first: it has been the most reliable for these blocks.
REDIRECTORS = [
    "root://cms-xrd-global.cern.ch/",
    "root://cmsxrootd.fnal.gov/",
    "root://xrootd-cms.infn.it/",
]

_lock = threading.Lock()


def era_of(das: str) -> str:
    if "APV" in das:
        return "2016APV"
    if "UL18" in das:
        return "2018"
    if "UL17" in das:
        return "2017"
    return "2016"


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["p"]] = rec
    return cache


def read_one(path: str, cache: dict, cache_file: Path, timeout: int) -> None:
    if path in cache:
        return
    last = None
    for attempt in range(len(REDIRECTORS) * 2):
        red = REDIRECTORS[attempt % len(REDIRECTORS)]
        try:
            with uproot.open(red + path, timeout=timeout) as handle:
                arr = handle["Events"]["LHEWeight_originalXWGTUP"].array(library="np")
            rec = {"p": path, "s": float(np.float64(arr).sum()), "n": int(len(arr))}
            with _lock:
                with cache_file.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                cache[path] = rec
                if len(cache) % 10 == 0:
                    print(f"    {len(cache)} files cached", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - any xrootd/uproot failure retries
            last = exc
    with _lock:
        print(f"  BADFILE {path}: {type(last).__name__}: {str(last)[:120]}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="+",
                    help="HT-bin substrings to process, e.g. HT100to200 HT2000toInf")
    ap.add_argument("--fileset", type=Path, default=DEFAULT_FILESET)
    ap.add_argument("--cache", type=Path, required=True,
                    help="per-file jsonl; reused on rerun so an interrupted run resumes")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args(argv)

    if not os.environ.get("X509_CERT_DIR"):
        print("WARNING: X509_CERT_DIR is unset. On macOS every xrootd read then "
              "fails with 'Auth failed' -- rerun with "
              "X509_CERT_DIR=$HOME/.globus/certificates.", file=sys.stderr)

    cache = load_cache(args.cache)
    top = json.load(args.fileset.open())["QCD_binned"]

    rows = []
    for das, files in top.items():
        if not any(p in das for p in args.patterns):
            continue
        todo = [f for f in files if f not in cache]
        if todo:
            print(f"{das}\n   {len(todo)}/{len(files)} files to read", flush=True)
            with ThreadPoolExecutor(args.threads) as pool:
                list(pool.map(lambda f: read_one(f, cache, args.cache, args.timeout), todo))
        got = [cache[f] for f in files if f in cache]
        total = float(np.float64([r["s"] for r in got]).sum())
        rows.append((era_of(das), das.split("/")[1], len(got), len(files),
                     total, sum(r["n"] for r in got)))

    print("\n#### paste COMPLETE rows into corrections.sumw_qcd_mg")
    for era, name, ngot, nall, total, nev in rows:
        state = "COMPLETE" if ngot == nall else "PARTIAL -- DO NOT USE"
        print(f"{era:8s} {ngot:3d}/{nall:3d} {state:22s} nevents={nev}")
        print(f"    '{name}': {total!r},")
    return 0 if all(r[2] == r[3] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
