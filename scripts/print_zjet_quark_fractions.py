#!/usr/bin/env python
"""Print Z+jet GEN leading-jet quark fractions per analysis jet-pT bin.

Reads validation-mode processor pickles (the same ``pt_flavor_jet0_gen``
histogram used by plot_zjet_gen_flavor_pt_y.py) and prints the quark fraction
f_q = quark / (quark + gluon) in the reported analysis pT intervals
(200-290, 290-400, >400 GeV) plus the inclusive >200 GeV row.

Made for the SMP-25-010 approval question on quantifying "quark-enriched"
(quark fraction per pT bin, Pythia vs Herwig spread).

    python3 scripts/print_zjet_quark_fractions.py \
        Pythia=outputs/validation_pythia_all.pkl \
        Herwig=outputs/validation_herwig_all.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

QUARK_CODE = 1
GLUON_CODE = 2
OTHER_CODE = 0

PT_WINDOWS = [(200.0, 290.0), (290.0, 400.0), (400.0, np.inf), (200.0, np.inf)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="LABEL=PATH",
        help="One or more labeled validation pickles, e.g. Pythia=outputs/validation_pythia_all.pkl",
    )
    return parser.parse_args()


def load_pt_flavor(path: Path):
    with path.open("rb") as handle:
        out = pickle.load(handle)
    if "pt_flavor_jet0_gen" not in out:
        raise KeyError(
            f"{path}: no pt_flavor_jet0_gen histogram. "
            "Re-run the zjet processor in validation mode (configs/zjet_*_validation.json)."
        )
    h = out["pt_flavor_jet0_gen"]
    if "syst" in h.axes.name:
        try:
            h = h[{"syst": "nominal"}]
        except Exception:
            pass
    return h.project("pt", "n")


def window_sums(h2, lo: float, hi: float):
    """Sum (value, variance) over fine pt bins covering [lo, hi); hi=inf includes overflow."""
    edges = h2.axes["pt"].edges
    vals = h2.values(flow=True)
    variances = h2.variances(flow=True)
    if variances is None:
        variances = vals
    # row i+1 covers [edges[i], edges[i+1]); row 0 underflow, row -1 overflow
    rows = [i + 1 for i in range(len(edges) - 1) if edges[i] >= lo and edges[i + 1] <= hi]
    if np.isinf(hi):
        rows.append(vals.shape[0] - 1)
    n_axis = h2.axes["n"]
    counts = {}
    for name, code in (("quark", QUARK_CODE), ("gluon", GLUON_CODE), ("other", OTHER_CODE)):
        col = n_axis.index(code) + 1
        counts[name] = (
            float(vals[rows, col].sum()),
            float(variances[rows, col].sum()),
        )
    return counts


def fraction_with_error(num, den_other):
    """f = a/(a+b) with weighted-count error propagation; num/den_other are (sum, var)."""
    a, var_a = num
    b, var_b = den_other
    total = a + b
    if total <= 0:
        return float("nan"), float("nan")
    f = a / total
    err = np.sqrt(b * b * var_a + a * a * var_b) / (total * total)
    return f, err


def main() -> int:
    args = parse_args()
    results = {}
    for spec in args.inputs:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"Bad input spec '{spec}': expected LABEL=PATH")
        results[label] = load_pt_flavor(Path(path))

    header = f"{'pT window [GeV]':<18}" + "".join(
        f"{label + ' f_q [%]':>20}{label + ' other [%]':>18}" for label in results
    )
    print(header)
    print("-" * len(header))
    for lo, hi in PT_WINDOWS:
        window = f"{lo:.0f}-{'inf' if np.isinf(hi) else f'{hi:.0f}'}"
        row = f"{window:<18}"
        for h2 in results.values():
            counts = window_sums(h2, lo, hi)
            f_q, f_q_err = fraction_with_error(counts["quark"], counts["gluon"])
            n_other = counts["other"][0]
            n_all = counts["quark"][0] + counts["gluon"][0] + n_other
            other_frac = 100.0 * n_other / n_all if n_all > 0 else float("nan")
            row += f"{100.0 * f_q:>13.1f} ± {100.0 * f_q_err:<4.1f}{other_frac:>18.1f}"
        print(row)

    if len(results) > 1:
        labels = list(results)
        print()
        print(f"Spread ({' vs '.join(labels)}) in f_q, per window:")
        for lo, hi in PT_WINDOWS:
            window = f"{lo:.0f}-{'inf' if np.isinf(hi) else f'{hi:.0f}'}"
            fqs = []
            for h2 in results.values():
                counts = window_sums(h2, lo, hi)
                fqs.append(fraction_with_error(counts["quark"], counts["gluon"])[0])
            print(f"  {window:<14} max-min = {100.0 * (max(fqs) - min(fqs)):.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
