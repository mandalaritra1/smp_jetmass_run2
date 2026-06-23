#!/usr/bin/env python
"""Merge per-chunk skim Parquet files into one Parquet per dataset.

The skimmer writes ``{indir}/{dataset}/part-*.parquet`` (one file per chunk).
This concatenates them per dataset, streaming row-group by row-group so peak
memory stays at ~one chunk regardless of total size (no giant in-RAM merge,
no `hadd` -- this is Parquet, not ROOT).

    python scripts/merge_skims.py --indir outputs/skims --outdir outputs/skims_merged
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pyarrow.parquet as pq


def merge_dataset(part_files, out_path):
    writer = None
    n_rows = 0
    for f in sorted(part_files):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches():
            if writer is None:
                writer = pq.ParquetWriter(out_path, batch.schema)
            writer.write_batch(batch)
            n_rows += batch.num_rows
    if writer is not None:
        writer.close()
    return n_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", required=True, help="dir containing {dataset}/part-*.parquet")
    ap.add_argument("--outdir", required=True, help="output dir for {dataset}.parquet")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    datasets = [d for d in sorted(os.listdir(args.indir))
                if os.path.isdir(os.path.join(args.indir, d))]
    if not datasets:
        print(f"[merge] no dataset subdirs under {args.indir}")
        return 1

    for ds in datasets:
        parts = glob.glob(os.path.join(args.indir, ds, "part-*.parquet"))
        if not parts:
            continue
        out_path = os.path.join(args.outdir, f"{ds}.parquet")
        n = merge_dataset(parts, out_path)
        print(f"[merge] {ds}: {len(parts)} parts -> {n} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
