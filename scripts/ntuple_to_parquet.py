#!/usr/bin/env python
"""Convert reco_jet_ntuple pickle(s) from `mass_diagnostic_ntuple` runs to Parquet.

The zjet `mass_diagnostic_ntuple` mode stores a per-event ntuple under
``output["reco_jet_ntuple"]`` as a dict of coffea column_accumulators, with the
weights already scaled by xsec*lumi/sumw in postprocess. This flattens that into
a single Parquet file ready for OmniFold.

    python scripts/ntuple_to_parquet.py outputs/*.pkl --outdir outputs/ntuple_parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _ntuple_columns(obj):
    """Return {name: numpy array} from a reco_jet_ntuple dict-accumulator."""
    cols = {}
    for name, acc in obj.items():
        val = acc.value if hasattr(acc, "value") else np.asarray(acc)
        cols[name] = np.asarray(val)
    return cols


def convert(pkl_path, out_path):
    with open(pkl_path, "rb") as f:
        out = pickle.load(f)
    if "reco_jet_ntuple" not in out:
        raise SystemExit(
            f"{pkl_path}: no 'reco_jet_ntuple' key "
            f"(run with mode='mass_diagnostic_ntuple'). Keys: {list(out)[:8]}"
        )
    cols = _ntuple_columns(out["reco_jet_ntuple"])
    lengths = {k: len(v) for k, v in cols.items()}
    n = next(iter(lengths.values()))
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"{pkl_path}: ragged columns {lengths}")
    # pandas handles mixed string (dataset/channel/systematic) + numeric columns;
    # awkward rejects object dtypes. ak.from_parquet still reads the result fine.
    pd.DataFrame(cols).to_parquet(out_path, index=False)
    return n, list(cols)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pickles", nargs="+", help="pickle file(s) or glob(s)")
    ap.add_argument("--outdir", default="outputs/ntuple_parquet")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    paths = [p for pat in args.pickles for p in glob.glob(pat)] or args.pickles
    for pk in paths:
        out_path = os.path.join(args.outdir, Path(pk).stem + ".parquet")
        n, names = convert(pk, out_path)
        print(f"{pk}: {n} rows, {len(names)} cols -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
