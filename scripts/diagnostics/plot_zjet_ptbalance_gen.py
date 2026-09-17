"""Gen-level Z/jet pT-balance comparison, HERWIG vs PYTHIA (2026-09-17).

Reads the `validation` mode pickles written with the ptfrac_gen/ptasym_gen/
dphi_gen histograms (see zjet_processor, commit cf0aad5) and draws, per gen
Z-pT bin, the normalized pT(jet)/pT(Z) distribution for both generators, plus
summary panels of the mean/median/68% width and the topology-cut efficiency.

    python scripts/diagnostics/plot_zjet_ptbalance_gen.py \
        --herwig /tmp/ptbal/validation_herwig_2018.pkl \
        --pythia /tmp/ptbal/validation_pythia_2018.pkl \
        --era 2018 --outdir outputs/plots/ptbalance_gen

"pre" is the kinematic gen selection only (two gen leptons, one gen AK8 jet,
pT(Z) > 90, 71 < mZ < 111).  "post" adds the analysis topology cuts
(|pT asymmetry| < 0.3, dphi > 1.57), which are cuts on the balance itself.
"""
from __future__ import annotations

import argparse, datetime, pickle, subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use(hep.style.CMS)

PUB_LABEL_FS, PUB_TICK_FS, PUB_LEGEND_FS, PUB_ANNOT_FS = 30, 26, 20, 22
COLORS = {"pythia": "#5790fc", "herwig": "#e42536"}
LABELS = {"pythia": "Pythia8 (CP5)", "herwig": "Herwig7 (CH3)"}


def stamp(fig, inputs, enabled=True):
    if not enabled:
        return
    try:
        ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:
        ver = "unknown"
    try:
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
    except Exception:
        pass
    fig.text(0.99, 0.002,
             f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  inputs: {inputs}",
             ha="right", va="bottom", fontsize=7, color="0.45", family="monospace")


def slice_1d(hists, name, ptz_index):
    """(density, density error, bin centers, edges, raw sumw) for one pT(Z) slice."""
    h = hists[name][{"dataset": sum, "systematic": sum}]
    values = h.values()[ptz_index]
    variances = h.variances()[ptz_index]
    edges = h.axes[-1].edges
    widths = np.diff(edges)
    total = values.sum()
    if total <= 0:
        return None
    density = values / widths / total
    # shape-only error: the per-bin MC error scaled by the same normalization
    error = np.sqrt(np.clip(variances, 0.0, None)) / widths / total
    return density, error, 0.5 * (edges[:-1] + edges[1:]), edges, total


def moments(hists, name, ptz_index):
    """mean, median, half 68% interval of the distribution in one pT(Z) slice."""
    h = hists[name][{"dataset": sum, "systematic": sum}]
    values = h.values()[ptz_index]
    edges = h.axes[-1].edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = values.sum()
    if total <= 0:
        return np.nan, np.nan, np.nan
    mean = float((values * centers).sum() / total)
    cdf = np.cumsum(values) / total
    q = lambda p: float(np.interp(p, cdf, centers))
    return mean, q(0.5), 0.5 * (q(0.84) - q(0.16))


def ptz_labels(hists):
    axis = hists["ptfrac_gen_pre"].axes["ptz"]
    out = []
    for lo, hi in zip(axis.edges[:-1], axis.edges[1:]):
        out.append(rf"${lo:.0f} < p_{{T,Z}} < {hi:.0f}$ GeV" if hi < 1e4
                   else rf"$p_{{T,Z}} > {lo:.0f}$ GeV")
    return out


def panel(data, name, stage, ptz_index, ptz_label, era, outdir, xlabel, no_stamp):
    fig, ax = plt.subplots(layout="constrained")
    ymax = 0.0
    for gen in ("pythia", "herwig"):
        got = slice_1d(data[gen], f"{name}_{stage}", ptz_index)
        if got is None:
            continue
        density, error, centers, edges, total = got
        ax.stairs(density, edges, color=COLORS[gen], lw=2.5,
                  ls="dotted" if gen == "pythia" else "dashdot",
                  label=LABELS[gen], baseline=None)
        ax.errorbar(centers, density, yerr=error, fmt="none",
                    ecolor=COLORS[gen], elinewidth=1.2, capsize=2)
        ymax = max(ymax, float((density + error).max()))
    hep.cms.label("", data=False, loc=2, ax=ax, rlabel=f"{era} (13 TeV)")
    ax.set_xlabel(xlabel, fontsize=PUB_LABEL_FS)
    ax.set_ylabel("Normalized", fontsize=PUB_LABEL_FS)
    ax.tick_params(labelsize=PUB_TICK_FS)
    ax.set_ylim(0, ymax * 1.55)
    note = ptz_label + ("\nkinematic selection" if stage == "pre" else "\n+ topology cuts")
    cms_x = next((t.get_position()[0] for t in ax.texts if t.get_text() == "CMS"), 0.045)
    ax.text(cms_x, 0.72, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=PUB_ANNOT_FS)
    leg = ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS, frameon=False)
    stamp(fig, f"validation {era} gen pT balance", enabled=not no_stamp)
    out = Path(outdir) / f"{name}_{stage}_{era}_ptz{ptz_index}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def summary(data, era, outdir, no_stamp):
    """Mean/median + 68% width of pT(jet)/pT(Z), and the topology-cut efficiency."""
    labels = ptz_labels(data["pythia"])
    centers, table = [], {}
    axis = data["pythia"]["ptfrac_gen_pre"].axes["ptz"]
    for lo, hi in zip(axis.edges[:-1], axis.edges[1:]):
        centers.append(0.5 * (lo + min(hi, 600.0)))
    for gen in ("pythia", "herwig"):
        rows = [moments(data[gen], "ptfrac_gen_pre", i) for i in range(len(labels))]
        eff = []
        for i in range(len(labels)):
            pre = data[gen]["ptfrac_gen_pre"][{"dataset": sum, "systematic": sum}].values()[i].sum()
            post = data[gen]["ptfrac_gen_post"][{"dataset": sum, "systematic": sum}].values()[i].sum()
            eff.append(post / pre if pre > 0 else np.nan)
        table[gen] = (np.array(rows), np.array(eff))

    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(31, 10.6))
    x = np.array(centers)
    for ax, col, ylab in zip(
            axes, (0, 2, None),
            (r"$\langle p_{T}^{jet}/p_{T}^{Z}\rangle$",
             r"68% half-width of $p_{T}^{jet}/p_{T}^{Z}$",
             "Topology-cut efficiency")):
        for gen in ("pythia", "herwig"):
            rows, eff = table[gen]
            y = eff if col is None else rows[:, col]
            ax.plot(x, y, marker="o" if gen == "pythia" else "s", ms=10, lw=2.5,
                    color=COLORS[gen], ls="dotted" if gen == "pythia" else "dashdot",
                    label=LABELS[gen])
        hep.cms.label("", data=False, loc=2, ax=ax, rlabel=f"{era} (13 TeV)")
        ax.set_xlabel(r"$p_{T,Z}^{GEN}$ (GeV)", fontsize=PUB_LABEL_FS)
        ax.set_ylabel(ylab, fontsize=PUB_LABEL_FS)
        ax.tick_params(labelsize=PUB_TICK_FS)
        ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS, frameon=False)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, lo + (hi - lo) * 1.45)
    stamp(fig, f"validation {era} gen pT balance", enabled=not no_stamp)
    out = Path(outdir) / f"ptbalance_summary_{era}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out, table, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--herwig", required=True, type=Path)
    ap.add_argument("--pythia", required=True, type=Path)
    ap.add_argument("--era", default="2018")
    ap.add_argument("--outdir", default="outputs/plots/ptbalance_gen", type=Path)
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = {"herwig": pickle.load(open(args.herwig, "rb")),
            "pythia": pickle.load(open(args.pythia, "rb"))}
    labels = ptz_labels(data["pythia"])
    made = []
    for i, lab in enumerate(labels):
        for stage in ("pre", "post"):
            made.append(panel(data, "ptfrac_gen", stage, i, lab, args.era, args.outdir,
                              r"$p_{T}^{jet}/p_{T}^{Z}$ (GEN)", args.no_stamp))
        made.append(panel(data, "ptasym_gen", "pre", i, lab, args.era, args.outdir,
                          r"$(p_{T}^{Z}-p_{T}^{jet})/(p_{T}^{Z}+p_{T}^{jet})$ (GEN)",
                          args.no_stamp))
        made.append(panel(data, "dphi_gen", "pre", i, lab, args.era, args.outdir,
                          r"$\Delta\phi(Z,\,jet)$ (GEN)", args.no_stamp))
    out, table, labels = summary(data, args.era, args.outdir, args.no_stamp)
    made.append(out)

    print(f"\n{'pT(Z) bin':24s} {'generator':10s} {'mean':>7s} {'median':>7s} {'width68':>8s} {'topo eff':>9s}")
    for gen in ("pythia", "herwig"):
        rows, eff = table[gen]
        for i, lab in enumerate(labels):
            plain = lab.replace("$", "").replace("\\", "").replace("_{T,Z}", "T,Z")
            print(f"{plain:24s} {gen:10s} {rows[i,0]:7.3f} {rows[i,1]:7.3f} {rows[i,2]:8.3f} {eff[i]:9.3f}")
    print(f"\n{len(made)} figures in {args.outdir}")


if __name__ == "__main__":
    main()
