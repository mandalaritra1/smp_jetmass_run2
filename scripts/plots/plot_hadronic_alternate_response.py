#!/usr/bin/env python3
"""Summarize the Pythia-vs-Herwig hadronic rho response stress test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


METRICS = (
    ("particle_level_generator_shape_change", "Particle-level\nshape"),
    ("conditional_response_total_variation", "Conditional\nresponse"),
    ("unfolded_data_shape_change", "Same-data\nunfolded shape"),
    ("response_mc_stat_relative_uncertainty", "Herwig response\nMC statistics"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dijet", type=Path, required=True)
    parser.add_argument("--trijet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open() as source:
        return json.load(source)


def core_interval(result: dict, key: str) -> tuple[float, float]:
    entry = result[key]
    if key == "response_mc_stat_relative_uncertainty":
        entry = entry["alternate"]
    core = entry["core"]
    return 100.0 * core["median"], 100.0 * core["maximum"]


def main() -> None:
    args = parse_args()
    results = {
        "Dijet": load_summary(args.dijet),
        "Trijet": load_summary(args.trijet),
    }

    plt.style.use(hep.style.CMS)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    colors = {"Dijet": "#3f7f93", "Trijet": "#d97732"}

    base_positions = np.arange(len(METRICS))
    offsets = {"Dijet": -0.13, "Trijet": 0.13}
    for channel, result in results.items():
        medians = []
        maxima = []
        for key, _ in METRICS:
            median, maximum = core_interval(result, key)
            medians.append(median)
            maxima.append(maximum)

        positions = base_positions + offsets[channel]
        axis.hlines(
            positions,
            medians,
            maxima,
            color=colors[channel],
            linewidth=5,
            alpha=0.45,
            zorder=1,
        )
        axis.scatter(
            medians,
            positions,
            s=85,
            color=colors[channel],
            edgecolor="white",
            linewidth=0.9,
            zorder=2,
            label=channel,
        )
        axis.scatter(
            maxima,
            positions,
            s=65,
            marker="|",
            linewidth=2.4,
            color=colors[channel],
            zorder=3,
        )
        for y, (median, maximum) in enumerate(zip(medians, maxima, strict=True)):
            axis.text(
                maximum + 0.8,
                y + offsets[channel],
                f"{median:.1f}–{maximum:.1f}%",
                va="center",
                fontsize=10.5,
                color=colors[channel],
            )

    axis.set_xlabel("Absolute relative effect in normalized core bins (%)")
    axis.set_xlim(0.0, 58.0)
    axis.set_xticks(np.arange(0.0, 51.0, 10.0))
    axis.grid(axis="x", color="0.86", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.invert_yaxis()
    axis.set_yticks(
        base_positions,
        [label for _, label in METRICS],
    )
    axis.legend(loc="lower right", frameon=False, ncol=2)
    hep.cms.label(
        "Work in progress",
        data=True,
        lumi=59.7,
        year="2018",
        com=13,
        ax=axis,
        fontsize=14,
    )
    axis.set_title(
        "Herwig response stress test: median to maximum across normalized core bins",
        fontsize=14,
        pad=28,
    )
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.14, top=0.78)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
