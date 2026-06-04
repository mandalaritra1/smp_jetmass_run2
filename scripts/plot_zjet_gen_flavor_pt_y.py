#!/usr/bin/env python
"""Plot Z+jet GEN leading-jet quark/gluon flavor splits in pT and rapidity.

Default input assumes a 2016 validation-mode Pythia run:

    python3 scripts/plot_zjet_gen_flavor_pt_y.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path

import hist as histlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smp_jetmass_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "smp_jetmass_cache"))

import matplotlib.pyplot as plt
import numpy as np

try:
    import mplhep as hep
except ImportError:
    hep = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "outputs" / "validation_pythia_2016.pkl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "plots" / "zjet_gen_flavor_pt_y_2016.pdf"

QUARK_CODE = 1
GLUON_CODE = 2
QUARK_COLOR = "#3f90da"
GLUON_COLOR = "#ffa90e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Processor pickle to read. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Plot path to write. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional exact dataset-axis category. By default all datasets are summed.",
    )
    parser.add_argument(
        "--pt-xmax",
        type=float,
        default=500.0,
        help="Upper x-axis limit for the pT panel.",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=(-2.4, 2.4),
        metavar=("YMIN", "YMAX"),
        help="Rapidity x-axis range for the y panel.",
    )
    return parser.parse_args()


def load_output(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def select_dataset(h, dataset: str | None):
    if dataset is None:
        return h
    if "dataset" not in h.axes.name:
        return h
    return h[{"dataset": dataset}]


def project_flavor(h, observable_axis: str):
    h2 = h.project(observable_axis, "n")
    quark_hist = h2[{"n": histlib.loc(QUARK_CODE)}].project(observable_axis)
    gluon_hist = h2[{"n": histlib.loc(GLUON_CODE)}].project(observable_axis)
    return quark_hist, gluon_hist


def hist_arrays(h):
    axis = h.axes[0]
    return axis.edges, np.asarray(h.values(flow=False), dtype=float)


def plot_flavor_stack(ax, h, observable_axis: str, xlabel: str, xlim: tuple[float, float] | None):
    quark_hist, gluon_hist = project_flavor(h, observable_axis)
    edges, quark = hist_arrays(quark_hist)
    _, gluon = hist_arrays(gluon_hist)

    total_qg = quark.sum() + gluon.sum()
    quark_frac = 100.0 * quark.sum() / total_qg if total_qg > 0 else 0.0
    gluon_frac = 100.0 * gluon.sum() / total_qg if total_qg > 0 else 0.0

    widths = np.diff(edges)
    quark_bars = ax.bar(
        edges[:-1],
        quark,
        width=widths,
        align="edge",
        color=QUARK_COLOR,
        label=f"Quark initiated ({quark_frac:.1f}%)",
    )
    gluon_bars = ax.bar(
        edges[:-1],
        gluon,
        width=widths,
        align="edge",
        bottom=quark,
        color=GLUON_COLOR,
        label=f"Gluon initiated ({gluon_frac:.1f}%)",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend([gluon_bars, quark_bars], [gluon_bars.get_label(), quark_bars.get_label()])


def add_cms_labels(axes):
    if hep is None:
        return
    for ax in axes:
        hep.cms.label("Simulation Internal", data=False, com=13, loc=0, ax=ax)


def main() -> int:
    args = parse_args()

    if hep is not None:
        plt.style.use(hep.style.CMS)

    out = load_output(args.input)
    required_hists = {
        "pt_flavor_jet0_gen": ("pt", r"Leading GEN jet $p_T$ [GeV]", (0.0, args.pt_xmax)),
        "y_flavor_jet0_gen": ("y", "Leading GEN jet rapidity", tuple(args.y_range)),
    }
    missing = [name for name in required_hists if name not in out]
    if missing:
        raise KeyError(
            "Missing histogram(s): "
            + ", ".join(missing)
            + ". Re-run zjet validation mode with the updated processor."
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    for ax, (hist_name, (axis_name, xlabel, xlim)) in zip(axes, required_hists.items()):
        h = select_dataset(out[hist_name], args.dataset)
        plot_flavor_stack(ax, h, axis_name, xlabel, xlim)

    add_cms_labels(axes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
