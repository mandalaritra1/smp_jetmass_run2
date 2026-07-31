#!/usr/bin/env python
"""Plot the bottom-line data/MC disagreement before and after unfolding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


METRICS = (
    ("median_absolute", r"Median $|R-1|$"),
    ("rms", r"RMS $(R-1)$"),
    ("maximum_absolute", r"Maximum $|R-1|$"),
)
CHANNELS = ("dijet", "trijet")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/hadronic_rho_crosschecks.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/figs/hadronic_rho_fullstat/"
            "data_mc_disagreement_before_after.png"
        ),
    )
    args = parser.parse_args(argv)

    audit = json.loads(args.input.read_text())
    hep.style.use(hep.style.CMS)
    fig, ax = plt.subplots(layout="constrained")

    colors = {"before": "#5790fc", "after": "#f89c20"}
    markers = {"before": "o", "after": "s"}
    offsets = {"before": -0.12, "after": 0.12}
    channel_centers = {"dijet": 1.0, "trijet": 5.0}

    for stage, label in (
        ("before", "Before unfolding"),
        ("after", "After unfolding"),
    ):
        xs = []
        ys = []
        for channel in CHANNELS:
            core = audit[channel]["data_mc_before_after_unfolding"][
                f"{stage}_unfolding"
            ]["core"]
            center = channel_centers[channel]
            for metric_index, (metric, _) in enumerate(METRICS):
                xs.append(center + metric_index - 1 + offsets[stage])
                ys.append(core[metric])
        ax.plot(
            xs,
            ys,
            linestyle="none",
            marker=markers[stage],
            markersize=11,
            color=colors[stage],
            label=label,
        )

    tick_positions = []
    tick_labels = []
    for channel in CHANNELS:
        center = channel_centers[channel]
        for metric_index, (_, label) in enumerate(METRICS):
            tick_positions.append(center + metric_index - 1)
            tick_labels.append(label)
        comparison = audit[channel]["data_mc_before_after_unfolding"]
        ax.text(
            center,
            0.97,
            (
                f"{channel.capitalize()}\n"
                f"larger in {comparison['core_bins_with_larger_absolute_disagreement']}"
                f"/{comparison['n_core_bins']} core bins"
            ),
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=19,
        )

    ax.set_xticks(tick_positions, tick_labels, rotation=20, ha="right")
    ax.axvline(3.0, color="#9c9ca1", linewidth=1.5, alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Normalized data/MC shape disagreement")
    ax.set_xlim(-0.65, 6.65)
    visible_maximum = max(line.get_ydata().max() for line in ax.lines[:2])
    ax.set_ylim(0.0, 1.55 * visible_maximum)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.84), ncol=2)
    hep.cms.label(
        "Preliminary",
        data=True,
        loc=0,
        rlabel="2018 hadronic (13 TeV)",
        ax=ax,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
