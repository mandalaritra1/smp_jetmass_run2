"""Plot CP5 versus MESS+Vincia groomed-rho shapes and campaign statistics."""

import argparse
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use(hep.style.CMS)

BLUE = "#5790fc"
ORANGE = "#f89c20"


def stamp(fig, inputs):
    try:
        version = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
    except Exception:
        version = "unknown"
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.97))
    fig.text(0.99, 0.003, f"2026-08-02 | smp_jetmass_run2 {version} | inputs: {inputs}",
             ha="right", va="bottom", fontsize=7, color="0.45", family="monospace")


def density(hist, error, widths):
    total = hist.sum()
    return hist / total / widths, error / total / widths


def shape_plots(cp5, mess, channel, output_dir, tag):
    edges = cp5["rho_edges"]
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pt_edges = cp5["pt_edges"]
    outputs = []
    for index, low in enumerate(pt_edges[:-1]):
        low = int(low)
        high = int(pt_edges[index + 1])
        cp5_y, cp5_e = density(cp5[f"pt{low}_h"], cp5[f"pt{low}_e"], widths)
        mess_y, mess_e = density(mess[f"pt{low}_h"], mess[f"pt{low}_e"], widths)

        fig, (ax, ratio_ax) = plt.subplots(
            2, 1, sharex=True, height_ratios=(3, 1), layout="constrained"
        )
        ax.stairs(cp5_y, edges, color=BLUE, linewidth=2.2, label="MG+MLM+CP5")
        ax.errorbar(centers, cp5_y, yerr=cp5_e, fmt="none", color=BLUE, alpha=0.65)
        ax.stairs(mess_y, edges, color=ORANGE, linewidth=2.2,
                  label="MG+MESS+Vincia")
        ax.errorbar(centers, mess_y, yerr=mess_e, fmt="none", color=ORANGE, alpha=0.75)
        ax.set_ylabel(r"$1/\sigma\;\mathrm{d}\sigma/\mathrm{d}\rho_g$")
        ax.legend(loc="upper right", frameon=True)
        visible_max = max(np.max(cp5_y + cp5_e), np.max(mess_y + mess_e))
        ax.set_ylim(0, 1.55 * visible_max)
        ax.grid(alpha=0.22)
        high_label = r"\infty" if high > 9999 else str(high)
        hep.cms.label("Preliminary", data=False, loc=2, ax=ax,
                      rlabel=rf"${low} < p_{{T}} < {high_label}$ GeV")
        display_tag = tag.replace("_", " ")
        ax.text(0.03, 0.72, f"{channel} • particle level • {display_tag}",
                transform=ax.transAxes, fontsize=17, ha="left", va="top")
        if channel == "dijet" and low < 400:
            ax.text(0.03, 0.65, "Low-HT MESS statistics incomplete",
                    transform=ax.transAxes, fontsize=16, color="#e42536",
                    ha="left", va="top")

        valid = cp5_y > 0
        ratio = np.full_like(cp5_y, np.nan)
        ratio_error = np.full_like(cp5_y, np.nan)
        ratio[valid] = mess_y[valid] / cp5_y[valid]
        positive = valid & (mess_y > 0)
        ratio_error[positive] = ratio[positive] * np.hypot(
            mess_e[positive] / mess_y[positive], cp5_e[positive] / cp5_y[positive]
        )
        ratio_ax.axhline(1.0, color="0.35", linewidth=1.4)
        ratio_ax.errorbar(centers[valid], ratio[valid], yerr=ratio_error[valid],
                          fmt="o", color=ORANGE, markersize=5)
        ratio_ax.set_ylabel("MESS / CP5")
        ratio_ax.set_xlabel(r"$\rho_g = 2\log_{10}(m_g/(p_T R))$")
        finite = np.isfinite(ratio_error)
        if finite.any():
            lower = np.nanmin(ratio[finite] - ratio_error[finite])
            upper = np.nanmax(ratio[finite] + ratio_error[finite])
            pad = max(0.15, 0.12 * (upper - lower))
            ratio_ax.set_ylim(max(0, lower - pad), upper + pad)
        ratio_ax.grid(alpha=0.25)
        stamp(fig, f"{tag}; CP5 matched-gridpack pool")
        path = os.path.join(output_dir, f"{channel}_rho_g_pt{low}_{tag}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)
    return outputs


def statistics_plots(cp5, mess, output_dir, tag):
    def ht_low_edge(name):
        return int(name.replace("QCD_HT", "").split("to")[0])

    ht_names = sorted((str(value) for value in mess["ht_bins"]), key=ht_low_edge)
    original_names = [str(value) for value in mess["ht_bins"]]
    indices = [original_names.index(name) for name in ht_names]
    ht_labels = [name.replace("QCD_HT", "").replace("to", "–").replace("Inf", "∞")
                 for name in ht_names]
    efficiency = (mess["ht_analyzed"] / mess["ht_requested"])[indices]
    fig, ax = plt.subplots(layout="constrained")
    ax.bar(np.arange(len(ht_labels)), efficiency, color=ORANGE)
    ax.set_xticks(np.arange(len(ht_labels)), ht_labels, rotation=25, ha="right")
    ax.set_ylabel("Nonzero MESS acceptance")
    ax.set_ylim(0, 1.35 * efficiency.max())
    ax.grid(axis="y", alpha=0.25)
    hep.cms.label("Preliminary", data=False, loc=2, ax=ax, rlabel="")
    ax.text(0.97, 0.96, "Particle level", transform=ax.transAxes,
            fontsize=17, ha="right", va="top")
    ax.text(0.97, 0.88, tag.replace("_", " "), transform=ax.transAxes,
            fontsize=17, ha="right")
    stamp(fig, tag)
    efficiency_path = os.path.join(output_dir, f"mess_acceptance_{tag}.png")
    fig.savefig(efficiency_path, dpi=150)
    plt.close(fig)

    pt_edges = cp5["pt_edges"]
    labels, cp5_neff, mess_neff = [], [], []
    for index, low_value in enumerate(pt_edges[:-1]):
        low = int(low_value)
        high = int(pt_edges[index + 1])
        labels.append(f"{low}–{'∞' if high > 9999 else high}")
        for sample, target in ((cp5, cp5_neff), (mess, mess_neff)):
            hist, error = sample[f"pt{low}_h"], sample[f"pt{low}_e"]
            target.append(hist.sum() ** 2 / np.sum(error ** 2))
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(layout="constrained")
    ax.bar(positions - 0.18, cp5_neff, 0.36, color=BLUE, label="MG+MLM+CP5")
    ax.bar(positions + 0.18, mess_neff, 0.36, color=ORANGE, label="MG+MESS+Vincia")
    ax.set_xticks(positions, labels)
    ax.set_xlabel(r"Jet $p_T$ [GeV]")
    ax.set_ylabel("Effective entries")
    ax.set_yscale("log")
    ax.set_ylim(top=2.2 * max(max(cp5_neff), max(mess_neff)))
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.25)
    hep.cms.label("Preliminary", data=False, loc=2, ax=ax, rlabel="")
    stamp(fig, tag)
    neff_path = os.path.join(output_dir, f"effective_statistics_{tag}.png")
    fig.savefig(neff_path, dpi=150)
    plt.close(fig)
    return [efficiency_path, neff_path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp5-trijet", required=True)
    parser.add_argument("--mess-trijet", required=True)
    parser.add_argument("--cp5-dijet", required=True)
    parser.add_argument("--mess-dijet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cp5_trijet = np.load(args.cp5_trijet)
    mess_trijet = np.load(args.mess_trijet)
    cp5_dijet = np.load(args.cp5_dijet)
    mess_dijet = np.load(args.mess_dijet)
    outputs = []
    outputs += shape_plots(cp5_trijet, mess_trijet, "trijet", args.output_dir, args.tag)
    outputs += shape_plots(cp5_dijet, mess_dijet, "dijet", args.output_dir, args.tag)
    outputs += statistics_plots(cp5_trijet, mess_trijet, args.output_dir, args.tag)
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
