#!/usr/bin/env python
"""Plot GEN jet mass and mpt split by quark/gluon initiating flavor."""
from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smp_jetmass_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "smp_jetmass_cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import mplhep as hep
except ImportError:
    hep = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "outputs" / "validation_pythia_2016.pkl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "plots" / "zjet_gen_flavor_mass_mpt_2016.pdf"

FLAVOR_COLORS = {
    "quark": "#3f90da",
    "gluon": "#ffa90e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional exact dataset-axis category. By default all datasets are summed.",
    )
    return parser.parse_args()


def load_output(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def select_dataset(h, dataset: str | None):
    if dataset is None or "dataset" not in h.axes.name:
        return h
    return h[{"dataset": dataset}]


def flavor_arrays(h, observable_axis: str):
    projected = h.project(observable_axis, "parton_flavor")
    arrays = {}
    edges = projected.axes[observable_axis].edges
    for flavor in FLAVOR_COLORS:
        arrays[flavor] = np.asarray(
            projected[{"parton_flavor": flavor}].project(observable_axis).values(),
            dtype=float,
        )
    return edges, arrays


def plot_flavor_stack(ax, h, observable_axis: str, xlabel: str):
    edges, arrays = flavor_arrays(h, observable_axis)
    quark = arrays["quark"]
    gluon = arrays["gluon"]

    total_qg = quark.sum() + gluon.sum()
    fractions = {
        flavor: 100.0 * values.sum() / total_qg if total_qg > 0 else 0.0
        for flavor, values in arrays.items()
    }

    widths = np.diff(edges)
    quark_bars = ax.bar(
        edges[:-1],
        quark,
        width=widths,
        align="edge",
        color=FLAVOR_COLORS["quark"],
        label=f"Quark initiated ({fractions['quark']:.1f}%)",
    )
    gluon_bars = ax.bar(
        edges[:-1],
        gluon,
        width=widths,
        align="edge",
        bottom=quark,
        color=FLAVOR_COLORS["gluon"],
        label=f"Gluon initiated ({fractions['gluon']:.1f}%)",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.legend(
        [gluon_bars, quark_bars],
        [gluon_bars.get_label(), quark_bars.get_label()],
    )


def add_cms_labels(axes):
    if hep is None:
        return
    hep.cms.label(
        "Internal",
        data=False,
        com=13,
        loc=0,
        ax=axes[0],
        fontsize=16,
    )


def main() -> int:
    args = parse_args()
    if hep is not None:
        plt.style.use(
            [
                hep.style.CMS,
                {
                    "font.size": 18,
                    "axes.labelsize": 20,
                    "legend.fontsize": 15,
                },
            ]
        )

    out = load_output(args.input)
    required_hists = {
        "mass_flavor_jet0_gen": ("mass", r"Leading GEN jet mass [GeV]"),
        "mpt_flavor_jet0_gen": (
            "mpt_gen",
            r"$2\log_{10}(m_{\mathrm{GEN}}/(p_{T,\mathrm{GEN}}R))$, $R=0.8$",
        ),
    }
    missing = [name for name in required_hists if name not in out]
    if missing:
        raise KeyError(
            "Missing histogram(s): "
            + ", ".join(missing)
            + ". Re-run zjet validation mode with the updated processor."
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    for ax, (hist_name, (axis_name, xlabel)) in zip(axes, required_hists.items()):
        plot_flavor_stack(
            ax,
            select_dataset(out[hist_name], args.dataset),
            axis_name,
            xlabel,
        )

    add_cms_labels(axes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
