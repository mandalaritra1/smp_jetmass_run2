"""Dijet: is the Herwig7 groomed-mass pT drift made by the MLM merging?

Four generator-level samples on the production dijet axes, 2018:

* MLM + Pythia8 CP5   (CMS QCD_HT*_madgraphMLM-pythia8, dijet_mc)
* MLM + Herwig7 CH3   (CMS QCD_HT*_madgraphMLM-herwig7, dijet_herwig)
* Flat Pythia8 CP5    (CMS QCD_Pt-15to7000_Flat2018_pythia8, STANDALONE shower, no MLM)
* Flat Herwig7 CH3    (CMS QCD_Pt-15to7000_Flat_herwig7, STANDALONE shower, no MLM)

The number that decides it is the median mass ratio Herwig7/Pythia8 per jet-pT
bin, 10^(d<log10 rho^2>/2): for the MLM pair it falls from 1.18 to 0.56 over
the nine dijet pT bins.  If the flat (unmerged) pair is flat in pT the drift is
the merging; if it also falls it is the shower or the tune.  Also drawn:
Flat/MLM per shower (what the merging does to each), the median slope of every
sample relative to its own 200-290 GeV bin, and the per-pT-bin shapes with a
ratio to MLM + Pythia8.  --mode ungroomed repeats it for the ungroomed mass.

    python scripts/diagnostics/plot_dijet_groomed_flat_vs_mlm.py [--mode groomed]
        [--mlm-eras 2018|all] [--flat-herwig PATH] [--flat-pythia PATH]
"""
from __future__ import annotations

import argparse, datetime, pickle, subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mplhep as hep
import numpy as np

hep.style.use(hep.style.CMS)
PUB_LABEL_FS, PUB_TICK_FS, PUB_LEGEND_FS, PUB_ANNOT_FS = 30, 26, 20, 22
HADRONIC_ERAS = ("2016APV", "2016", "2017", "2018")
PT_LABELS = ["185–200", "200–290", "290–400", "400–480", "480–570", "570–680",
             "680–760", "760–820", "> 820"]
STYLE = {  # key: (label, colour, marker, linestyle, linewidth)
    "mlm_pythia":  ("MG5 MLM + Pythia8 CP5 (CMS)", "#5790fc", "o", ":", 3.0),
    "mlm_herwig":  ("MG5 MLM + Herwig7 CH3 (CMS)", "#e42536", "s", "-.", 3.0),
    "flat_pythia": ("Flat Pythia8 CP5, no merging (CMS)", "#1f4e9a", "D", "-", 2.5),
    "flat_herwig": ("Flat Herwig7 CH3, no merging (CMS)", "#8b0000", "^", "-", 2.5),
}
MODE_KEY = {"groomed": ("ptjet_rhojet_g_gen", "soft-drop groomed"),
            "ungroomed": ("ptjet_rhojet_u_gen", "ungroomed")}


def stamp(fig, enabled=True):
    if not enabled:
        return
    try:
        ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:
        ver = "unknown"
    try:
        fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.965))
    except Exception:
        pass
    # scale the stamp with the figure width so a 1x2 grid keeps it legible
    fig.text(0.99, 0.006, f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  "
             "inputs: dijet/rho/original gen (mc, herwig, flat_herwig, flat_pythia)",
             ha="right", va="bottom", fontsize=7 * fig.get_figwidth() / 10.0,
             color="0.45", family="monospace")


def cms_x(ax, default=0.045):
    return next((t.get_position()[0] for t in ax.texts if t.get_text() == "CMS"), default)


def project(h):
    names = [a.name for a in h.axes]
    index = {}
    if "dataset" in names:
        index["dataset"] = sum
    if "systematic" in names:
        index["systematic"] = "nominal" if "nominal" in list(h.axes["systematic"]) else sum
    return h[index] if index else h


def load_sum(paths, key):
    """(values[pt, rho], variances, rho_edges, pt_edges) summed over the given pickles."""
    values = variances = rho_edges = pt_edges = None
    for path in paths:
        h = project(pickle.load(open(path, "rb"))[key])
        v, var = np.asarray(h.values(), float), np.asarray(h.variances(), float)
        values = v.copy() if values is None else values + v
        variances = var.copy() if variances is None else variances + var
        e_rho, e_pt = np.asarray(h.axes["mpt_gen"].edges), np.asarray(h.axes["ptgen"].edges)
        if rho_edges is not None and not (np.allclose(e_rho, rho_edges) and np.allclose(e_pt, pt_edges)):
            raise SystemExit(f"axis mismatch in {path}")
        rho_edges, pt_edges = e_rho, e_pt
    return values, variances, rho_edges, pt_edges


def sources(inputs, mlm_eras, flat_herwig, flat_pythia):
    root = Path(inputs) / "dijet/rho/original"
    eras = HADRONIC_ERAS if mlm_eras == "all" else (mlm_eras,)
    return {
        "mlm_pythia": [root / era / "dijet_mc" / f"minimal_rho_dijet_mg_pythia8_{era}.pkl" for era in eras],
        "mlm_herwig": [root / era / "dijet_herwig" / f"minimal_rho_dijet_herwig_{era}.pkl" for era in eras],
        "flat_herwig": [Path(flat_herwig) if flat_herwig else
                        root / "2018/dijet_flat_herwig/minimal_rho_dijet_flat_herwig_2018.pkl"],
        "flat_pythia": [Path(flat_pythia) if flat_pythia else
                        root / "2018/dijet_flat_pythia/minimal_rho_dijet_flat_pythia8_2018.pkl"],
    }


def median_stats(v, var, centers):
    """(median, its error, n_eff) of one pT row on the kept rho bins."""
    total = v.sum()
    if total <= 0:
        return np.nan, np.nan, 0.0
    cdf = np.cumsum(v) / total
    q = lambda p: float(np.interp(p, cdf, centers))
    n_eff = total ** 2 / max(var.sum(), 1e-300)
    err = 1.25 * 0.5 * (q(0.84) - q(0.16)) / np.sqrt(max(n_eff, 1.0))
    return q(0.5), err, n_eff


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", type=Path, default=Path.home() / "Projects/unfold/inputs")
    ap.add_argument("--mlm-eras", default="2018", choices=("2018", "2017", "2016", "2016APV", "all"),
                    help="era(s) of the MLM production pickles; the flat samples are 2018 only")
    ap.add_argument("--flat-herwig", type=Path, default=None, help="override the flat Herwig7 pickle path")
    ap.add_argument("--flat-pythia", type=Path, default=None, help="override the flat Pythia8 pickle path")
    ap.add_argument("--mode", default="groomed", choices=tuple(MODE_KEY))
    ap.add_argument("--rho-floor", type=float, default=-5.0)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/plots/dijet_flat_vs_mlm"))
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    key, mode_text = MODE_KEY[args.mode]
    era_text = "Run 2" if args.mlm_eras == "all" else args.mlm_eras
    rlabel = f"{era_text} (13 TeV)" if era_text == "2018" else "2018 flat, %s MLM (13 TeV)" % era_text

    data = {k: load_sum(p, key) for k, p in sources(args.inputs, args.mlm_eras,
                                                     args.flat_herwig, args.flat_pythia).items()}
    _, _, rho_edges, pt_edges = data["mlm_pythia"]
    for k, (_, _, e_rho, e_pt) in data.items():
        if not (np.allclose(e_rho, rho_edges) and np.allclose(e_pt, pt_edges)):
            raise SystemExit(f"{k}: axes differ from mlm_pythia")
    centers_all = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    keep = centers_all > args.rho_floor
    centers = centers_all[keep]
    npt = len(pt_edges) - 1
    x = np.array([0.5 * (lo + min(hi, 1000.0)) for lo, hi in zip(pt_edges[:-1], pt_edges[1:])])

    #### medians, errors and effective statistics per pT bin and sample
    med = {k: np.full(npt, np.nan) for k in data}
    err = {k: np.full(npt, np.nan) for k in data}
    neff = {k: np.zeros(npt) for k in data}
    for k, (v, var, _, _) in data.items():
        for i in range(npt):
            med[k][i], err[k][i], neff[k][i] = median_stats(v[i][keep], var[i][keep], centers)

    def ratio(num, den):
        d = med[num] - med[den]
        r = 10 ** (d / 2)
        s = np.sqrt(err[num] ** 2 + err[den] ** 2)
        return r, r * np.log(10) / 2 * s

    pairs = {
        "flat":      ("Flat Herwig7 / Flat Pythia8 (no merging)", "flat_herwig", "flat_pythia", "#8b0000", "^", "-"),
        "mlm":       ("MLM Herwig7 / MLM Pythia8 (CMS central)", "mlm_herwig", "mlm_pythia", "#e42536", "s", "-."),
        "herwig_fm": ("Flat Herwig7 / MLM Herwig7", "flat_herwig", "mlm_herwig", "#e42536", "^", "-"),
        "pythia_fm": ("Flat Pythia8 / MLM Pythia8", "flat_pythia", "mlm_pythia", "#5790fc", "D", "-"),
    }
    rat = {k: ratio(a, b) for k, (_, a, b, *_r) in pairs.items()}

    #### the table
    print(f"\n{mode_text} mass, gen level, MLM eras = {era_text}, rho floor {args.rho_floor}")
    print(f"{'pT bin':>9s} | " + " ".join(f"{STYLE[k][0].split(' (')[0][:22]:>22s}" for k in data)
          + " | " + " ".join(f"{pairs[p][0].split(' (')[0][:26]:>26s}" for p in pairs))
    print(f"{'':>9s} | " + " ".join(f"{'median  (n_eff)':>22s}" for _ in data) + " | " + " ".join(f"{'mass ratio':>26s}" for _ in pairs))
    for i in range(npt):
        row = f"{PT_LABELS[i]:>9s} | " + " ".join(
            f"{med[k][i]:8.3f}±{err[k][i]:.3f} ({neff[k][i]:8.0f})" for k in data)
        row += " | " + " ".join(f"{rat[p][0][i]:12.3f} ± {rat[p][1][i]:<9.3f}" for p in pairs)
        print(row)
    print("slope (median relative to 200-290, last bin):",
          {k: round(float(med[k][-1] - med[k][1]), 3) for k in data})

    def finish(ax, ylabel, note, loc="upper right"):
        hep.cms.label("", data=False, loc=2, ax=ax, rlabel=rlabel)
        ax.set_xlabel(r"$p_{T}^{jet}$ (GeV)", fontsize=PUB_LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=PUB_LABEL_FS - 4)
        ax.tick_params(labelsize=PUB_TICK_FS)
        ax.set_xlim(170, 950)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.02 * (hi - lo), lo + (hi - lo) * 1.55)
        # enough labelled ticks that the region below unity can be read off
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.text(cms_x(ax), 0.74, note, transform=ax.transAxes, ha="left", va="top", fontsize=PUB_ANNOT_FS)
        ax.legend(loc=loc, fontsize=PUB_LEGEND_FS - 2, frameon=False, borderaxespad=1.0)

    #### figure 1: median mass ratio Herwig7/Pythia8, flat pair next to MLM pair,
    #### and Flat/MLM per shower
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(21, 10.6))
    for ax, keys in zip(axes, (("flat", "mlm"), ("herwig_fm", "pythia_fm"))):
        for p in keys:
            label, _, _, color, marker, ls = pairs[p]
            r, e = rat[p]
            ax.errorbar(x, r, yerr=e, marker=marker, ls=ls, lw=2.5, ms=11, color=color,
                        mfc="white" if p.endswith("_fm") or p == "flat" else color, mew=2.0,
                        capsize=4, label=label)
        ax.axhline(1.0, color="gray", ls=":", lw=1.5)
    finish(axes[0], "median mass ratio Herwig7 / Pythia8", f"Dijet, {mode_text}, GEN\nsame shower pair, with and without MLM")
    finish(axes[1], "median mass ratio flat / MLM", f"Dijet, {mode_text}, GEN\nwhat the merging does to each shower")
    stamp(fig, enabled=not args.no_stamp)
    out = args.outdir / f"dijet_{args.mode}_mass_ratio_flat_vs_mlm.png"
    fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
    print(f"wrote {out}")

    #### figure 2: the slope of every sample relative to its own 200-290 bin
    fig, ax = plt.subplots(layout="constrained")
    for k in ("mlm_pythia", "mlm_herwig", "flat_pythia", "flat_herwig"):
        label, color, marker, ls, lw = STYLE[k]
        rel = med[k] - med[k][1]
        ax.errorbar(x, rel, yerr=np.sqrt(err[k] ** 2 + err[k][1] ** 2), marker=marker, ls=ls, lw=lw,
                    ms=11, color=color, mfc="white" if k.startswith("flat") else color, mew=2.0,
                    capsize=3, label=label)
    ax.axhline(0.0, color="gray", ls=":", lw=1.2)
    finish(ax, r"median $\log_{10}(\rho^2)$ relative to 200–290 GeV",
           f"Dijet, {mode_text}, GEN\nshape per bin; tune offsets removed")
    stamp(fig, enabled=not args.no_stamp)
    out = args.outdir / f"dijet_{args.mode}_slope_flat_vs_mlm.png"
    fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
    print(f"wrote {out}")

    #### figure 3: per-pT-bin shapes with the ratio to MLM + Pythia8, one panel each
    widths = np.diff(rho_edges)[keep]
    e_keep = np.concatenate(([rho_edges[:-1][keep][0]], rho_edges[1:][keep]))
    for i in range(npt):
        fig = plt.figure(layout="constrained", figsize=(10, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=(3, 1.2))
        ax, rax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        shapes = {}
        for k, (v, var, _, _) in data.items():
            vv, vvar = v[i][keep], var[i][keep]
            tot = vv.sum()
            if tot <= 0:
                continue
            shapes[k] = (vv / tot / widths, np.sqrt(vvar) / tot / widths)
        ref = shapes.get("mlm_pythia")
        ymax = 0.0
        for k, (y, ye) in shapes.items():
            label, color, marker, ls, lw = STYLE[k]
            hep.histplot(y, e_keep, ax=ax, histtype="step", color=color, ls=ls, lw=lw, label=label)
            ax.errorbar(centers, y, yerr=ye, fmt="none", ecolor=color, elinewidth=1.2, capsize=2)
            ymax = max(ymax, float(np.nanmax(y + ye)))
            if ref is not None and k != "mlm_pythia":
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(ref[0] > 0, y / ref[0], np.nan)
                    re = np.where(ref[0] > 0, r * np.sqrt((ye / np.where(y > 0, y, 1)) ** 2
                                                          + (ref[1] / ref[0]) ** 2), np.nan)
                hep.histplot(r, e_keep, ax=rax, histtype="step", color=color, ls=ls, lw=lw)
                rax.errorbar(centers, r, yerr=re, fmt="none", ecolor=color, elinewidth=1.2, capsize=2)
        hep.cms.label("", data=False, loc=2, ax=ax, rlabel=rlabel)
        ax.text(cms_x(ax), 0.74, f"Dijet, {mode_text}, GEN\n{PT_LABELS[i]} GeV",
                transform=ax.transAxes, ha="left", va="top", fontsize=PUB_ANNOT_FS)
        ax.set_ylim(0, ymax * 1.6)
        ax.set_ylabel(r"$1/N\; dN/d\log_{10}(\rho^2)$", fontsize=PUB_LABEL_FS - 4)
        ax.tick_params(labelsize=PUB_TICK_FS, labelbottom=False)
        ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS - 3, frameon=False)
        rax.axhline(1.0, color="gray", ls=":", lw=1.2)
        rax.set_xlabel(r"$\log_{10}(\rho^2)$, " + mode_text, fontsize=PUB_LABEL_FS - 4)
        rax.set_ylabel("ratio to MLM + Pythia8", fontsize=PUB_LABEL_FS - 8)
        rax.tick_params(labelsize=PUB_TICK_FS - 4)
        rlo, rhi = rax.get_ylim()
        rax.set_ylim(max(0.0, rlo), rhi)   # data-driven: never clip a ratio point
        rax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        for a in (ax, rax):
            a.set_xlim(e_keep[0], e_keep[-1])
        fig.align_ylabels([ax, rax])
        stamp(fig, enabled=not args.no_stamp)
        out = args.outdir / f"dijet_{args.mode}_shape_pt{i}.png"
        fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
