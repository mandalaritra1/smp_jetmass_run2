#!/usr/bin/env python3
"""Validate Rivet event diagnostics against the authoritative pair_split cuts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_COLUMNS = (
    "event_index njet pt1 pt2 pt3 y1 y2 y3 phi1 phi2 phi3 "
    "dphi12 dphi13 dphi23 pair_boundary pass_dijet pass_trijet weight"
).split()

CANONICAL_COLUMNS = {name: i for i, name in enumerate(EXPECTED_COLUMNS)}


def read_columns(path: Path) -> dict[str, int]:
    """Map ntuple column name -> index from the routine's own header line.

    Hardcoded indices silently decayed into an off-by-one once the routine
    started writing pair_boundary, so the layout is read from the file.
    """
    with path.open() as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            names = line.lstrip("#").split()
            if names and names[0] == "event_index":
                if names != EXPECTED_COLUMNS:
                    raise SystemExit(
                        f"FATAL: unexpected ntuple layout in {path}\n"
                        f"  found:    {' '.join(names)}\n"
                        f"  expected: {' '.join(EXPECTED_COLUMNS)}"
                    )
                return {name: i for i, name in enumerate(names)}
    raise SystemExit(f"FATAL: no column header found in {path}")


def expected_flags(
    rows: np.ndarray, col: dict[str, int] = CANONICAL_COLUMNS
) -> tuple[np.ndarray, np.ndarray]:
    pt1, pt2, pt3 = (rows[:, col[n]] for n in ("pt1", "pt2", "pt3"))
    y1, y2, y3 = (rows[:, col[n]] for n in ("y1", "y2", "y3"))
    dphi12, dphi13, dphi23 = (
        rows[:, col[n]] for n in ("dphi12", "dphi13", "dphi23")
    )
    boundary = np.pi - 200.0 / (0.5 * (pt1 + pt2))
    dijet = (
        (pt1 > 200.0)
        & (pt2 > 200.0)
        & (np.abs(y1) < 2.4)
        & (np.abs(y2) < 2.4)
        & (dphi12 > boundary)
    )
    trijet = (
        (rows[:, col["njet"]] >= 3)
        & (pt1 > 200.0)
        & (pt2 > 200.0)
        & (pt3 > 185.0)
        & (np.abs(y1) < 2.4)
        & (np.abs(y2) < 2.4)
        & (np.abs(y3) < 2.4)
        & (np.minimum.reduce((dphi12, dphi13, dphi23)) > 1.0)
        & (dphi12 < boundary)
    )
    return dijet, trijet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_ntuple", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    col = read_columns(args.event_ntuple)
    rows = np.loadtxt(args.event_ntuple, comments="#", ndmin=2)
    expected_dijet, expected_trijet = expected_flags(rows, col)
    stored_dijet = rows[:, col["pass_dijet"]].astype(bool)
    stored_trijet = rows[:, col["pass_trijet"]].astype(bool)
    result = {
        "events_with_two_jets": int(len(rows)),
        "dijet_selected": int(expected_dijet.sum()),
        "trijet_selected": int(expected_trijet.sum()),
        "overlap": int((expected_dijet & expected_trijet).sum()),
        "dijet_mismatches": int((expected_dijet != stored_dijet).sum()),
        "trijet_mismatches": int((expected_trijet != stored_trijet).sum()),
        "status": "pass",
    }
    if result["overlap"] or result["dijet_mismatches"] or result["trijet_mismatches"]:
        result["status"] = "fail"
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
