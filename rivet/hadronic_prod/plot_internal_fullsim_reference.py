#!/usr/bin/env python3
"""Build FullSim-anchored hadronization and shower modelling shapes.

The official UL18 MG+MLM+CP5 NanoAODSIM truth is the nominal shape. Internal
Pythia variations are transferred with variation/standalone-CP5 ratios. Vincia
is transferred separately with the MG+MESS+Vincia / matched MG+MLM+CP5 ratio
from the MESS campaign. Keeping the two denominators explicit avoids folding a
standalone-versus-FullSim closure difference into the modelling variations.
"""

import argparse
import datetime
import os
import pickle
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

INTERNAL_CONFIGS = ("cr1", "cr2", "fraghard", "fragsoft")
VARIATIONS = INTERNAL_CONFIGS + ("vincia",)
COLORS = {
    "cp5": "#666666",
    "cr1": "#5790fc",
    "cr2": "#f89c20",
    "fraghard": "#e42536",
    "fragsoft": "#964a8b",
    "vincia": "#009E73",
}
LABELS = {
    "cp5": "Standalone CP5",
    "cr1": "CR: QCD-inspired",
    "cr2": "CR: gluon-move",
    "fraghard": "Fragmentation: hard",
    "fragsoft": "Fragmentation: soft",
    "vincia": "MESS+Vincia",
}
LINESTYLES = {
    "cp5": (0, (1, 1)),
    "cr1": "--",
    "cr2": "-.",
    "fraghard": (0, (5, 1)),
    "fragsoft": (0, (2, 1)),
    "vincia": "-",
}
PT_BINS = ((200, 290), (290, 400), (400, None))


def bootstrap_shape_error(data, config, pt_low, nboot, seed):
    """HT-stratified chunk bootstrap for one standalone normalized shape."""
    bins = data["chunk_bins"].astype(str)
    groups = [np.flatnonzero(bins == name) for name in np.unique(bins)]
    config_offset = ("cp5",) + INTERNAL_CONFIGS
    rng = np.random.default_rng(
        seed + pt_low + 1000 * config_offset.index(config)
    )
    replicas = []
    for _ in range(nboot):
        sampled = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        replicas.append(pooled_shape(data, config, pt_low, sampled))
    return np.nanstd(np.asarray(replicas), axis=0, ddof=1)


def fullsim_shape(path, pt_low, pt_high, expected_rho_edges):
    """Read the processor's normalized FullSim CP5 generator-level shape."""
    with open(path, "rb") as handle:
        output = pickle.load(handle)
    hist = output["ptjet_rhojet_g_gen"]
    selected = hist[{"systematic": "nominal"}][{"dataset": sum}]
    pt_edges = np.asarray(selected.axes["ptgen"].edges)
    rho_edges = np.asarray(selected.axes["mpt_gen"].edges)
    if not np.array_equal(rho_edges, expected_rho_edges):
        raise ValueError(
            f"FullSim rho edges {rho_edges} do not match standalone "
            f"edges {expected_rho_edges}"
        )
    values = selected.values()
    variances = selected.variances()
    pt_mask = pt_edges[:-1] >= pt_low
    if pt_high is not None:
        pt_mask &= pt_edges[1:] <= pt_high
    spectrum = values[pt_mask].sum(axis=0)
    variance = variances[pt_mask].sum(axis=0)
    total = spectrum.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"invalid FullSim total for pT bin starting at {pt_low}")
    shape = spectrum / total
    # This diagonal approximation is sufficient for the tiny FullSim MC-stat
    # display band; the final transfer will retain the histogram covariance.
    error = np.sqrt(np.clip(variance, 0, None)) / total
    return shape, error


def npz_shape(sample, pt_low, expected_rho_edges):
    """Read a normalized shape and diagonal error from a harvested NPZ."""
    rho_edges = np.asarray(sample["rho_edges"])
    if not np.array_equal(rho_edges, expected_rho_edges):
        raise ValueError(
            f"MESS rho edges {rho_edges} do not match standalone "
            f"edges {expected_rho_edges}"
        )
    spectrum = np.asarray(sample[f"pt{pt_low}_h"], dtype=float)
    error = np.asarray(sample[f"pt{pt_low}_e"], dtype=float)
    total = spectrum.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"invalid harvested total for pT bin {pt_low}")
    return spectrum / total, error / total


def independent_ratio(numerator, numerator_error, denominator, denominator_error):
    """Ratio with diagonal uncertainty for statistically independent samples."""
    ratio = np.divide(
        numerator, denominator,
        out=np.full_like(numerator, np.nan), where=denominator > 0,
    )
    relative_num = np.divide(
        numerator_error, numerator,
        out=np.full_like(numerator_error, np.nan), where=numerator > 0,
    )
    relative_den = np.divide(
        denominator_error, denominator,
        out=np.full_like(denominator_error, np.nan), where=denominator > 0,
    )
    ratio_error = ratio * np.hypot(relative_num, relative_den)
    return ratio, ratio_error


def git_describe(script_dir):
    result = subprocess.run(
        ["git", "describe", "--tags", "--always", "--dirty"],
        cwd=script_dir, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() or "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("standalone")
    parser.add_argument("fullsim")
    parser.add_argument("mess_vincia")
    parser.add_argument("mess_cp5")
    parser.add_argument("output_dir")
    parser.add_argument("--channel", choices=("dijet", "trijet"), required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--min-count", type=int, default=50)
    args = parser.parse_args()

    data = np.load(args.standalone)
    mess_vincia = np.load(args.mess_vincia)
    mess_cp5 = np.load(args.mess_cp5)
    input_channel = str(data["channel"])
    if input_channel != args.channel:
        raise ValueError(
            f"standalone input contains channel={input_channel}, "
            f"but --channel={args.channel}"
        )
    for label, sample in (("MESS+Vincia", mess_vincia), ("MESS CP5", mess_cp5)):
        sample_channel = str(sample["channel"])
        if sample_channel != args.channel:
            raise ValueError(
                f"{label} input contains channel={sample_channel}, "
                f"but --channel={args.channel}"
            )
    os.makedirs(args.output_dir, exist_ok=True)
    rho_edges = data["rho_edges"]
    centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    widths = np.diff(rho_edges)
    first = int(np.searchsorted(rho_edges, -4.0))
    display_edges = rho_edges[first:]
    all_indices = np.arange(len(data["chunk_bins"]))
    repo_version = git_describe(os.path.dirname(os.path.abspath(__file__)))
    campaign = str(data["campaign"])
    summary = {
        "channel": np.array(args.channel),
        "campaign": np.array(campaign),
        "manifest_count": data["manifest_count"],
        "rho_edges": rho_edges,
        "pt_edges": np.array([200.0, 290.0, 400.0, 13000.0]),
        "fullsim_path": np.array(os.path.abspath(args.fullsim)),
        "mess_vincia_path": np.array(os.path.abspath(args.mess_vincia)),
        "mess_cp5_path": np.array(os.path.abspath(args.mess_cp5)),
        "prescription": np.array(
            "FullSim_CP5 * variation/standalone_CP5 for internal variations; "
            "FullSim_CP5 * MESS_Vincia/MESS_campaign_CP5 for Vincia"
        ),
    }

    for pt_low, pt_high in PT_BINS:
        fullsim, fullsim_error = fullsim_shape(
            args.fullsim, pt_low, pt_high, rho_edges
        )
        standalone_cp5 = pooled_shape(data, "cp5", pt_low, all_indices)
        standalone_cp5_error = bootstrap_shape_error(
            data, "cp5", pt_low, args.bootstrap, args.seed
        )
        internal_ratios, internal_ratio_errors = paired_bootstrap(
            data, pt_low, args.bootstrap, args.seed
        )
        mess_vincia_shape, mess_vincia_error = npz_shape(
            mess_vincia, pt_low, rho_edges
        )
        mess_cp5_shape, mess_cp5_error = npz_shape(
            mess_cp5, pt_low, rho_edges
        )
        vincia_ratio, vincia_ratio_error = independent_ratio(
            mess_vincia_shape, mess_vincia_error,
            mess_cp5_shape, mess_cp5_error,
        )
        ratios = dict(internal_ratios)
        ratio_errors = dict(internal_ratio_errors)
        ratios["vincia"] = vincia_ratio
        ratio_errors["vincia"] = vincia_ratio_error
        anchored_shapes = {
            config: fullsim * ratios[config] for config in VARIATIONS
        }
        anchored_densities = {
            config: shape / widths for config, shape in anchored_shapes.items()
        }
        standalone_cp5_density = standalone_cp5 / widths
        fullsim_density = fullsim / widths
        fullsim_density_error = fullsim_error / widths

        fig, (ax, ratio_ax) = plt.subplots(
            2, 1, sharex=True, height_ratios=(3.2, 1.25),
            layout="constrained",
        )
        lower = np.clip(fullsim_density - fullsim_density_error, 0, None)
        upper = fullsim_density + fullsim_density_error
        ax.fill_between(
            display_edges,
            np.append(lower[first:], lower[-1]),
            np.append(upper[first:], upper[-1]),
            step="post", color="0.75", alpha=0.55, linewidth=0,
            label="FullSim stat.",
        )
        ax.stairs(
            fullsim_density[first:], display_edges, color="black",
            linewidth=2.8, label="FullSim CP5 truth",
        )
        ax.stairs(
            standalone_cp5_density[first:], display_edges,
            color=COLORS["cp5"], linewidth=1.8,
            linestyle=LINESTYLES["cp5"], label="Standalone CP5 closure",
        )
        for config in VARIATIONS:
            ax.stairs(
                anchored_densities[config][first:], display_edges,
                color=COLORS[config], linewidth=2.2,
                linestyle=LINESTYLES[config], label=LABELS[config],
            )

        visible_max = max(
            np.nanmax(fullsim_density[first:]),
            np.nanmax(standalone_cp5_density[first:]),
            *(np.nanmax(density[first:])
              for density in anchored_densities.values()),
        )
        ax.set_ylim(0, 1.95 * visible_max)
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
            anchor_x, 0.79,
            f"{args.channel.capitalize()}\n{pt_text}",
            transform=ax.transAxes, ha="left", va="top", fontsize=18,
        )
        if args.channel == "dijet" and pt_low < 400:
            ax.text(
                anchor_x, 0.68, "Vincia: low-HT statistics limited",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=13.5, color="#e42536",
            )
        ax.legend(
            loc="upper right", bbox_to_anchor=(0.98, 0.82), ncol=2,
            fontsize=13.3, frameon=True, facecolor="white",
            edgecolor="none", framealpha=1.0,
        )

        ratio_ax.axhline(1.0, color="black", linewidth=1.3)
        fullsim_relative_error = np.divide(
            fullsim_error, fullsim,
            out=np.zeros_like(fullsim_error), where=fullsim > 0,
        )
        ratio_ax.fill_between(
            display_edges,
            np.append((1.0 - fullsim_relative_error)[first:],
                      1.0 - fullsim_relative_error[-1]),
            np.append((1.0 + fullsim_relative_error)[first:],
                      1.0 + fullsim_relative_error[-1]),
            step="post", color="0.75", alpha=0.55, linewidth=0,
        )
        ratio_lows = []
        ratio_highs = []
        closure_ratio = np.divide(
            standalone_cp5, fullsim,
            out=np.full_like(fullsim, np.nan), where=fullsim > 0,
        )
        closure_error = np.divide(
            standalone_cp5_error, fullsim,
            out=np.full_like(fullsim, np.nan), where=fullsim > 0,
        )
        closure_count = data[f"cp5_pt{pt_low}_n"]
        closure_valid = (
            (centers >= -4.0) & (closure_count >= args.min_count)
            & np.isfinite(closure_ratio) & np.isfinite(closure_error)
        )
        ratio_ax.errorbar(
            centers[closure_valid], closure_ratio[closure_valid],
            yerr=closure_error[closure_valid], color=COLORS["cp5"],
            marker="o", markersize=3.8, linewidth=1.2,
            linestyle=LINESTYLES["cp5"], elinewidth=1.0, capsize=0,
        )
        ratio_lows.extend(
            (closure_ratio[closure_valid] - closure_error[closure_valid]).tolist()
        )
        ratio_highs.extend(
            (closure_ratio[closure_valid] + closure_error[closure_valid]).tolist()
        )
        for config in VARIATIONS:
            ratio = ratios[config]
            ratio_error = ratio_errors[config]
            if config == "vincia":
                raw_count = np.minimum(
                    mess_vincia[f"pt{pt_low}_n"], mess_cp5[f"pt{pt_low}_n"]
                )
            else:
                raw_count = np.minimum(
                    data[f"cp5_pt{pt_low}_n"], data[f"{config}_pt{pt_low}_n"]
                )
            valid = (
                (centers >= -4.0) & (raw_count >= args.min_count)
                & np.isfinite(ratio) & np.isfinite(ratio_error)
            )
            ratio_ax.errorbar(
                centers[valid], ratio[valid], yerr=ratio_error[valid],
                color=COLORS[config], marker="o", markersize=4.6,
                linewidth=1.5, linestyle=LINESTYLES[config],
                elinewidth=1.1, capsize=0,
            )
            ratio_lows.extend((ratio[valid] - ratio_error[valid]).tolist())
            ratio_highs.extend((ratio[valid] + ratio_error[valid]).tolist())
            summary[f"{config}_pt{pt_low}_anchored_shape"] = anchored_shapes[config]
            summary[f"{config}_pt{pt_low}_transfer_ratio"] = ratio
            summary[f"{config}_pt{pt_low}_transfer_ratio_error"] = ratio_error

        summary[f"fullsim_pt{pt_low}_shape"] = fullsim
        summary[f"fullsim_pt{pt_low}_error"] = fullsim_error
        summary[f"standalone_cp5_pt{pt_low}_shape"] = standalone_cp5
        summary[f"standalone_cp5_pt{pt_low}_bootstrap_error"] = standalone_cp5_error
        summary[f"standalone_cp5_pt{pt_low}_ratio_to_fullsim"] = closure_ratio
        summary[f"standalone_cp5_pt{pt_low}_ratio_error"] = closure_error
        summary[f"mess_cp5_pt{pt_low}_shape"] = mess_cp5_shape
        summary[f"mess_cp5_pt{pt_low}_error"] = mess_cp5_error
        summary[f"mess_vincia_pt{pt_low}_shape"] = mess_vincia_shape
        summary[f"mess_vincia_pt{pt_low}_error"] = mess_vincia_error
        ratio_low = min(ratio_lows + [1.0])
        ratio_high = max(ratio_highs + [1.0])
        ratio_span = max(ratio_high - ratio_low, 0.12)
        ratio_ax.set_ylim(
            ratio_low - 0.14 * ratio_span,
            ratio_high + 0.14 * ratio_span,
        )
        ratio_ax.set_xlabel(r"$\log_{10}(\rho_{g}^{2})$")
        ratio_ax.set_ylabel("Variation / CP5")
        ratio_ax.grid(alpha=0.25)

        fig.get_layout_engine().set(rect=(0.015, 0.025, 0.995, 0.95))
        fig.text(
            0.99, 0.003,
            f"{datetime.date.today().isoformat()} | smp_jetmass_run2 "
            f"{repo_version} | inputs: {campaign} ({int(data['manifest_count'])}/1000); "
            "MESS final 1200/1200; FullSim UL18 MG+MLM+CP5",
            ha="right", va="bottom", fontsize=6.7,
            color="0.45", family="monospace",
        )
        output = os.path.join(
            args.output_dir,
            f"internal_fullsimref_{args.channel}_rho_g_pt{pt_low}.png",
        )
        fig.savefig(output, dpi=140)
        plt.close(fig)
        print(f"wrote {output}")

    summary_path = os.path.join(
        args.output_dir, f"internal_fullsimref_{args.channel}.npz"
    )
    np.savez(summary_path, **summary)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
