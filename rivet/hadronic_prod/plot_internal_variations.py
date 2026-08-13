#!/usr/bin/env python3
"""Plot paired-bootstrap CP5/CR/fragmentation groomed-rho ratios."""

import argparse
import datetime
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


hep.style.use(hep.style.CMS)

CONFIGS = ("cr1", "cr2", "fraghard", "fragsoft")
COLORS = {
    "cr1": "#5790fc",
    "cr2": "#f89c20",
    "fraghard": "#e42536",
    "fragsoft": "#964a8b",
}
LABELS = {
    "cr1": "CR: QCD-inspired",
    "cr2": "CR: gluon-move",
    "fraghard": "Fragmentation: hard",
    "fragsoft": "Fragmentation: soft",
}
PT_BINS = ((200, 290), (290, 400), (400, None))


def pooled_shape(data, config, pt_low, sampled_indices):
    hist = np.zeros(len(data["rho_edges"]) - 1)
    bins = data["chunk_bins"].astype(str)
    chunk_h = data[f"{config}_pt{pt_low}_chunk_h"]
    denominator = data[f"{config}_chunk_denominator"]
    for bin_name in np.unique(bins):
        indices = sampled_indices[bins[sampled_indices] == bin_name]
        denom = denominator[indices].sum()
        if denom > 0:
            hist += chunk_h[indices].sum(axis=0) / denom
    total = hist.sum()
    return hist / total if total > 0 else np.full_like(hist, np.nan)


def paired_bootstrap(data, pt_low, nboot, seed):
    bins = data["chunk_bins"].astype(str)
    groups = [np.flatnonzero(bins == bin_name) for bin_name in np.unique(bins)]
    nominal_indices = np.arange(len(bins))
    nominal_cp5 = pooled_shape(data, "cp5", pt_low, nominal_indices)
    nominal = {}
    for config in CONFIGS:
        varied = pooled_shape(data, config, pt_low, nominal_indices)
        nominal[config] = np.divide(
            varied, nominal_cp5,
            out=np.full_like(varied, np.nan), where=nominal_cp5 > 0,
        )
    replicas = {config: [] for config in CONFIGS}
    rng = np.random.default_rng(seed + pt_low)
    for _ in range(nboot):
        sampled = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        cp5 = pooled_shape(data, "cp5", pt_low, sampled)
        for config in CONFIGS:
            varied = pooled_shape(data, config, pt_low, sampled)
            replicas[config].append(np.divide(
                varied, cp5,
                out=np.full_like(varied, np.nan), where=cp5 > 0,
            ))
    errors = {}
    for config in CONFIGS:
        values = np.asarray(replicas[config])
        error = np.full(values.shape[1], np.nan)
        for index in range(values.shape[1]):
            finite = values[:, index][np.isfinite(values[:, index])]
            if len(finite) > 1:
                error[index] = np.std(finite, ddof=1)
        errors[config] = error
    return nominal, errors


def git_describe(script_dir):
    result = subprocess.run(
        ["git", "describe", "--tags", "--always", "--dirty"],
        cwd=script_dir,
        capture_output=True,
        text=True,
        check=False,
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
    display = centers >= -4.0
    repo_version = git_describe(os.path.dirname(os.path.abspath(__file__)))
    campaign = str(data["campaign"])

    for pt_low, pt_high in PT_BINS:
        ratios, errors = paired_bootstrap(data, pt_low, args.bootstrap, args.seed)
        raw_cp5 = data[f"cp5_pt{pt_low}_n"]
        keep = display & (raw_cp5 >= args.min_count)
        if not np.any(keep):
            continue

        fig, ax = plt.subplots(layout="constrained")
        ax.axhline(1.0, color="black", linewidth=1.4)
        lows = []
        highs = []
        for config in CONFIGS:
            ratio = ratios[config]
            error = errors[config]
            valid = keep & np.isfinite(ratio) & np.isfinite(error)
            ax.errorbar(
                centers[valid], ratio[valid], yerr=error[valid],
                color=COLORS[config], marker="o", markersize=5.5,
                linewidth=1.8, elinewidth=1.4, capsize=0,
                label=LABELS[config],
            )
            lows.extend((ratio[valid] - error[valid]).tolist())
            highs.extend((ratio[valid] + error[valid]).tolist())

        low = min(lows + [1.0])
        high = max(highs + [1.0])
        span = max(high - low, 0.12)
        ax.set_ylim(low - 0.12 * span, high + 0.48 * span)
        ax.set_xlim(rho_edges[np.flatnonzero(display)[0]], rho_edges[-1])
        ax.set_xlabel(r"$\log_{10}(\rho_{g}^{2})$")
        ax.set_ylabel("Normalized variation / CP5")
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
            fontsize=17, frameon=True,
            facecolor="white", edgecolor="none", framealpha=1.0,
        )

        fig.get_layout_engine().set(rect=(0, 0.025, 1, 0.96))
        fig.text(
            0.99, 0.003,
            f"{datetime.date.today().isoformat()} | smp_jetmass_run2 {repo_version} | inputs: {campaign}",
            ha="right", va="bottom", fontsize=7, color="0.45", family="monospace",
        )
        output = os.path.join(
            args.output_dir,
            f"internal_variations_{args.channel}_rho_g_pt{pt_low}.png",
        )
        fig.savefig(output, dpi=140)
        plt.close(fig)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
