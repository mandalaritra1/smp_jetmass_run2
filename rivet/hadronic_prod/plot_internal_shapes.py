#!/usr/bin/env python3
"""Plot groomed-rho shapes with a variation/CP5 ratio panel."""

import argparse
import datetime
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_internal_variations import paired_bootstrap, pooled_shape


hep.style.use(hep.style.CMS)

CONFIGS = ("cp5", "cr1", "cr2", "fraghard", "fragsoft")
COLORS = {
    "cp5": "black",
    "cr1": "#5790fc",
    "cr2": "#f89c20",
    "fraghard": "#e42536",
    "fragsoft": "#964a8b",
}
LABELS = {
    "cp5": "CP5 (reference)",
    "cr1": "CR: QCD-inspired",
    "cr2": "CR: gluon-move",
    "fraghard": "Fragmentation: hard",
    "fragsoft": "Fragmentation: soft",
}
LINESTYLES = {
    "cp5": "-",
    "cr1": "--",
    "cr2": "-.",
    "fraghard": (0, (5, 1)),
    "fragsoft": (0, (2, 1)),
}
PT_BINS = ((200, 290), (290, 400), (400, None))


def cp5_bootstrap_error(data, pt_low, nboot, seed):
    bins = data["chunk_bins"].astype(str)
    groups = [np.flatnonzero(bins == bin_name) for bin_name in np.unique(bins)]
    rng = np.random.default_rng(seed + pt_low)
    replicas = []
    for _ in range(nboot):
        sampled = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        replicas.append(pooled_shape(data, "cp5", pt_low, sampled))
    return np.nanstd(np.asarray(replicas), axis=0, ddof=1)


def git_describe(script_dir):
    result = subprocess.run(
        ["git", "describe", "--tags", "--always", "--dirty"],
        cwd=script_dir, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() or "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("--channel", choices=("dijet", "trijet"), required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--min-count", type=int, default=50)
    args = parser.parse_args()

    data = np.load(args.input)
    input_channel = str(data["channel"])
    if input_channel != args.channel:
        raise ValueError(
            f"input contains channel={input_channel}, but --channel={args.channel}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    rho_edges = data["rho_edges"]
    centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    widths = np.diff(rho_edges)
    first = int(np.searchsorted(rho_edges, -4.0))
    display_edges = rho_edges[first:]
    repo_version = git_describe(os.path.dirname(os.path.abspath(__file__)))
    campaign = str(data["campaign"])
    all_indices = np.arange(len(data["chunk_bins"]))

    for pt_low, pt_high in PT_BINS:
        shapes = {
            config: pooled_shape(data, config, pt_low, all_indices)
            for config in CONFIGS
        }
        cp5_error = cp5_bootstrap_error(data, pt_low, args.bootstrap, args.seed)
        ratios, ratio_errors = paired_bootstrap(
            data, pt_low, args.bootstrap, args.seed
        )
        densities = {config: shapes[config] / widths for config in CONFIGS}
        cp5_density_error = cp5_error / widths

        fig, (ax, ratio_ax) = plt.subplots(
            2, 1, sharex=True, height_ratios=(3.2, 1.25),
            layout="constrained",
        )
        cp5_density = densities["cp5"]
        lower = np.clip(cp5_density - cp5_density_error, 0, None)
        upper = cp5_density + cp5_density_error
        ax.fill_between(
            display_edges,
            np.append(lower[first:], lower[-1]),
            np.append(upper[first:], upper[-1]),
            step="post", color="0.75", alpha=0.55, linewidth=0,
            label="CP5 bootstrap unc.",
        )
        for config in CONFIGS:
            ax.stairs(
                densities[config][first:], display_edges,
                color=COLORS[config], linewidth=2.4,
                linestyle=LINESTYLES[config], label=LABELS[config],
            )

        visible_max = max(
            np.nanmax(density[first:]) for density in densities.values()
        )
        visible_max = max(visible_max, np.nanmax(upper[first:]))
        ax.set_ylim(0, 1.85 * visible_max)
        ax.set_xlim(display_edges[0], display_edges[-1])
        ax.set_ylabel(r"$(1/N)\,dN/d\log_{10}(\rho_{g}^{2})$")
        ax.grid(alpha=0.25)
        hep.cms.label("Preliminary", data=False, loc=2, ax=ax, rlabel="13 TeV")

        cms_text = [text for text in ax.texts if text.get_text() == "CMS"]
        anchor_x = cms_text[0].get_position()[0] if cms_text else 0.03
        pt_symbol = r"p_{T,3}" if args.channel == "trijet" else r"p_T"
        pt_text = (
            rf"${pt_symbol} \geq {pt_low}\,\mathrm{{GeV}}$"
            if pt_high is None
            else rf"${pt_low} \leq {pt_symbol} < {pt_high}\,\mathrm{{GeV}}$"
        )
        ax.text(
            anchor_x, 0.79, f"{args.channel.capitalize()}\n{pt_text}",
            transform=ax.transAxes, ha="left", va="top", fontsize=20,
        )
        ax.legend(
            loc="upper right", bbox_to_anchor=(0.98, 0.82), ncol=2,
            fontsize=15.5, frameon=True, facecolor="white",
            edgecolor="none", framealpha=1.0,
        )

        raw_cp5 = data[f"cp5_pt{pt_low}_n"]
        ratio_keep = (centers >= -4.0) & (raw_cp5 >= args.min_count)
        ratio_ax.axhline(1.0, color="black", linewidth=1.3)
        ratio_lows = []
        ratio_highs = []
        for config in CONFIGS[1:]:
            ratio = ratios[config]
            error = ratio_errors[config]
            valid = ratio_keep & np.isfinite(ratio) & np.isfinite(error)
            ratio_ax.errorbar(
                centers[valid], ratio[valid], yerr=error[valid],
                color=COLORS[config], marker="o", markersize=4.8,
                linewidth=1.6, linestyle=LINESTYLES[config],
                elinewidth=1.2, capsize=0,
            )
            ratio_lows.extend((ratio[valid] - error[valid]).tolist())
            ratio_highs.extend((ratio[valid] + error[valid]).tolist())

        ratio_low = min(ratio_lows + [1.0])
        ratio_high = max(ratio_highs + [1.0])
        ratio_span = max(ratio_high - ratio_low, 0.12)
        ratio_ax.set_ylim(
            ratio_low - 0.16 * ratio_span,
            ratio_high + 0.16 * ratio_span,
        )
        ratio_ax.set_xlabel(r"$\log_{10}(\rho_{g}^{2})$")
        ratio_ax.set_ylabel("Variation / CP5")
        ratio_ax.grid(alpha=0.25)

        fig.get_layout_engine().set(rect=(0, 0.025, 1, 0.97))
        fig.text(
            0.99, 0.003,
            f"{datetime.date.today().isoformat()} | smp_jetmass_run2 {repo_version} | inputs: {campaign}",
            ha="right", va="bottom", fontsize=7, color="0.45", family="monospace",
        )
        output = os.path.join(
            args.output_dir,
            f"internal_shapes_{args.channel}_rho_g_pt{pt_low}.png",
        )
        fig.savefig(output, dpi=140)
        plt.close(fig)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
