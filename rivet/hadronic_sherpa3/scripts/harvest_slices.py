#!/usr/bin/env python3
"""Combine chunk ntuples into slice-normalized (pT, rho_g) histograms.

The normalization rule from CAMPAIGN_PLAN.md section 9:

    H_{s,k} = sigma_s * (sum_{c in s} sum_{events in k} w)
                      / (sum_{c in s} sum_{all events} w)
    H_k     = sum_s H_{s,k}

Each slice is ONE physical cross section and its chunks are partitions of that
cross section.  The two ways to get this wrong are to give every chunk the full
slice cross section and add them (which multiplies the yield by the chunk
count), and to merge already-finalized per-chunk histograms.  Both are avoided
by accumulating raw weight sums here and applying sigma_s exactly once.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

RHO_EDGES = np.array(
    [-10, -5, -4, -3.4, -2.85, -2.25, -1.8, -1.5,
     -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0],
    dtype=float,
)
DIJET_PT_EDGES = np.array(
    [200, 290, 400, 480, 570, 680, 760, 820, 1500], dtype=float
)
TRIJET_PT_EDGES = np.array([185, 200, 290, 400, 1500], dtype=float)

# Sherpa RE-ESTIMATES the cross section from each generated sample, so chunks
# sharing one integration grid do NOT agree exactly: two 200-event chunks on
# the same grid gave 314.9 +- 22.3 pb and 284.8 pb, a 10.6% spread that is
# under one sigma.  Consistency is therefore a statistical test on the quoted
# errors, not an equality test.  5 sigma is the failure threshold so that a
# 90-chunk campaign does not trip on ordinary Gaussian tails.
MAX_XSEC_PULL = 5.0


def footer(text: str, name: str, default: float | None = None) -> float:
    match = re.search(rf"^# {re.escape(name)}\s+(.+)$", text, re.MULTILINE)
    if not match:
        if default is not None:
            return default
        raise SystemExit(f"missing footer {name!r}")
    return float(match.group(1))


def load_chunk(ntuple: Path) -> dict:
    text = ntuple.read_text()
    rows = np.loadtxt(ntuple.open(), comments="#", ndmin=2)
    return {
        "path": ntuple,
        "pt": rows[:, 2] if len(rows) else np.empty(0),
        "rho": rows[:, 6] if len(rows) else np.empty(0),
        "weight": rows[:, 7] if len(rows) else np.empty(0),
        "cross_section_pb": footer(text, "cross_section_pb"),
        "cross_section_error_pb": footer(text, "cross_section_error_pb",
                                        default=0.0),
        "sumw": footer(text, "sumw"),
        "raw_sumw": footer(text, "raw_sumw"),
        "sumw2": footer(text, "sumw2"),
        "generated_events": footer(text, "generated_events"),
    }


def harvest_slice(chunks: list[dict], pt_edges: np.ndarray) -> dict:
    """Accumulate one slice.  sigma_s is applied once, at the end."""
    sigmas = np.array([c["cross_section_pb"] for c in chunks])
    counts = np.array([c["generated_events"] for c in chunks], dtype=float)
    errors = np.array([c["cross_section_error_pb"] for c in chunks])

    # Event-count weighting, not inverse-variance: for weighted MC the quoted
    # error correlates with the estimate itself, so inverse-variance weighting
    # biases the mean low.  With equal-size chunks this is the plain mean.
    sigma_s = float((sigmas * counts).sum() / counts.sum())

    pulls = np.abs(sigmas - sigma_s) / np.where(errors > 0, errors, np.inf)
    max_pull = float(np.nanmax(pulls)) if len(chunks) > 1 else 0.0
    if np.isfinite(max_pull) and max_pull > MAX_XSEC_PULL:
        worst = chunks[int(np.nanargmax(pulls))]["path"]
        raise SystemExit(
            f"FATAL: chunk cross sections are inconsistent (max pull "
            f"{max_pull:.1f} sigma > {MAX_XSEC_PULL}), worst: {worst}.  Chunks "
            "of one configuration must share an integration grid; check "
            "grid_in_sha256 in the manifests before combining."
        )
    relative_spread = float(sigmas.std() / sigma_s) if sigma_s else 0.0

    shape = (len(pt_edges) - 1, len(RHO_EDGES) - 1)
    numerator = np.zeros(shape)
    numerator_sq = np.zeros(shape)
    denominator = 0.0
    for c in chunks:
        # Raw weight sums only; no per-chunk normalization happens here.
        h, _, _ = np.histogram2d(
            c["pt"], c["rho"], bins=[pt_edges, RHO_EDGES], weights=c["weight"]
        )
        h2, _, _ = np.histogram2d(
            c["pt"], c["rho"], bins=[pt_edges, RHO_EDGES],
            weights=c["weight"] ** 2,
        )
        numerator += h
        numerator_sq += h2
        denominator += c["raw_sumw"]

    if denominator == 0.0:
        raise SystemExit("FATAL: zero total weight for slice")

    values = sigma_s * numerator / denominator
    # Poisson-like error from the weight sums, propagated through the same
    # normalization; N_eff per bin is the honest statistics measure.
    errors = sigma_s * np.sqrt(numerator_sq) / denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(numerator_sq > 0, numerator**2 / numerator_sq, 0.0)
    return {
        "cross_section_pb": sigma_s,
        "cross_section_max_pull": max_pull,
        "cross_section_relative_spread": relative_spread,
        "sum_weights": denominator,
        "values": values,
        "errors": errors,
        "n_eff": n_eff,
        "chunks": len(chunks),
        "generated_events": sum(c["generated_events"] for c in chunks),
    }


def _n_eff_report(n_eff: np.ndarray) -> dict:
    """N_eff summary over the rho_g >= -3 study region, populated rows only."""
    study = n_eff[:, RHO_EDGES[:-1] >= -3.0]
    populated = study[n_eff.sum(axis=1) > 0]
    if not populated.size:
        return {"min_n_eff_rho_ge_-3": None, "empty_study_cells": int(study.size)}
    return {
        "min_n_eff_rho_ge_-3": float(populated.min()),
        "median_n_eff_rho_ge_-3": float(np.median(populated)),
        "empty_study_cells": int((populated == 0).sum()),
        "populated_pt_rows": int((n_eff.sum(axis=1) > 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+",
                        help="unpacked chunk run directories of ONE slice")
    parser.add_argument("--channel", choices=("dijet", "trijet"), default="dijet")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pt_edges = DIJET_PT_EDGES if args.channel == "dijet" else TRIJET_PT_EDGES
    ntuples = [d / f"pairsplit_{args.channel}.txt" for d in args.run_dirs]
    missing = [str(p) for p in ntuples if not p.is_file()]
    if missing:
        raise SystemExit(f"missing ntuples: {missing}")

    result = harvest_slice([load_chunk(p) for p in ntuples], pt_edges)
    np.savez(
        args.output,
        values=result["values"],
        errors=result["errors"],
        n_eff=result["n_eff"],
        pt_edges=pt_edges,
        rho_edges=RHO_EDGES,
    )
    summary = {
        "channel": args.channel,
        "chunks": result["chunks"],
        "generated_events": result["generated_events"],
        "cross_section_pb": result["cross_section_pb"],
        "cross_section_max_pull": result["cross_section_max_pull"],
        "cross_section_relative_spread": result["cross_section_relative_spread"],
        "sum_weights": result["sum_weights"],
        "integral_pb": float(result["values"].sum()),
        # Restricted to pT rows this slice actually populates: a single slice
        # leaves every other row empty, so a global minimum is always 0 and
        # says nothing about whether the N_eff targets are met.
        **_n_eff_report(result["n_eff"]),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
