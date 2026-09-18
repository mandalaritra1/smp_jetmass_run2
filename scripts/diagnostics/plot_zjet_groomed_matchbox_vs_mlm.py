"""Z+jet: does the NLO Matchbox Herwig7 DY sample drift in pT like MLM + Herwig7?

Three generator-level samples on the production Z+jet axes (2018 unless
--mlm-eras all):

* MLM + Pythia8 CP5        (CMS DYJetsToLL madgraphMLM-pythia8, the nominal)
* MLM + Herwig7 CH3        (CMS DYJetsToLL madgraphMLM-herwig7, the JetMET systematic sample)
* Matchbox NLO Herwig7 CH3 (CMS DYToLL_NLO_5FS_TuneCH3_13TeV_matchbox_herwig7:
                            MG5 ME inside Herwig's Matchbox, no external LHE, no MLM)

The number that decides it is the median mass ratio Herwig7/Pythia8 per jet-pT
bin, 10^(d<log10 rho^2>/2).  MLM + Herwig7 falls with pT (the drift); if the
Matchbox sample is flat in pT against MLM + Pythia8 the drift is the MLM
merging step of the CMS Herwig7 configuration, as the dijet flat-pT test found
(plot_dijet_groomed_flat_vs_mlm.py).  Also drawn: Matchbox / MLM Herwig7, the
median slope of every sample relative to its own 200-290 GeV bin, and the
per-pT-bin shapes with a ratio to MLM + Pythia8.  --mode ungroomed repeats it
for the ungroomed mass.

The Matchbox pickle carries the 2x-refined rho axis (rho_refine 2); it is
rebinned onto the production edges here.

    python scripts/diagnostics/plot_zjet_groomed_matchbox_vs_mlm.py [--mode groomed]
        [--mlm-eras 2018|all] [--matchbox PATH] [--inputs ~/cernbox/unfold_inputs]
"""
from __future__ import annotations

import argparse, datetime, pickle, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
import mplhep as hep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_dijet_groomed_flat_vs_mlm import cms_x, median_stats  # noqa: E402

hep.style.use(hep.style.CMS)
PUB_LABEL_FS, PUB_TICK_FS, PUB_LEGEND_FS, PUB_ANNOT_FS = 30, 26, 20, 22
PT_LABELS = ["185–200", "200–290", "290–400", "> 400"]
ZJET_ERA_CATS = {"2018": ("herwig_UL18NanoAODv9", "pythia_UL18NanoAODv9")}
STYLE = {  # key: (label, colour, marker, linestyle, linewidth)
    "mlm_pythia": ("MG5 MLM + Pythia8 CP5", "#5790fc", "o", ":", 3.0),
    "mlm_herwig": ("MG5 MLM + Herwig7 CH3", "#e42536", "s", "-.", 3.0),
    "matchbox":   ("Matchbox NLO + Herwig7 CH3, no MLM", "#8b0000", "^", "-", 2.5),
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
    fig.text(0.985, 0.012, f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  "
             "inputs: zjet/rho/original gen (pythia, herwig, herwig_matchbox)",
             ha="right", va="bottom", fontsize=8 * fig.get_figwidth() / 10.0,
             color="0.45", family="monospace")


def select(h, dataset_cats):
    """Sum the requested dataset categories (all if None), take the nominal systematic."""
    names = [a.name for a in h.axes]
    index = {}
    if "dataset" in names:
        cats = list(h.axes["dataset"])
        if dataset_cats is None:
            index["dataset"] = sum
        else:
            missing = [c for c in dataset_cats if c not in cats]
            if missing:
                raise SystemExit(f"dataset categories {missing} not in {cats}")
            index["dataset"] = [cats.index(c) for c in dataset_cats]
    if "systematic" in names:
        syst = list(h.axes["systematic"])
        if "nominal" not in syst:
            raise SystemExit(f"no 'nominal' systematic category in {syst[:5]}...")
        index["systematic"] = "nominal"
    h = h[index] if index else h
    if "dataset" in [a.name for a in h.axes]:      # list selection keeps the axis
        h = h[{"dataset": sum}]
    return h


def load(path, key, dataset_cats=None):
    """(values[pt, rho], variances, rho_edges, pt_edges)."""
    h = select(pickle.load(open(path, "rb"))[key], dataset_cats)
    return (np.asarray(h.values(), float), np.asarray(h.variances(), float),
            np.asarray(h.axes["mpt_gen"].edges), np.asarray(h.axes["ptgen"].edges))


def rebin_rho(values, variances, edges, target_edges):
    """Merge refined rho bins onto target_edges (every target edge must be a refined edge)."""
    idx = []
    for e in target_edges:
        hits = np.where(np.isclose(edges, e))[0]
        if len(hits) != 1:
            raise SystemExit(f"target edge {e} is not a refined edge; cannot rebin")
        idx.append(int(hits[0]))
    v = np.stack([values[:, a:b].sum(axis=1) for a, b in zip(idx[:-1], idx[1:])], axis=1)
    var = np.stack([variances[:, a:b].sum(axis=1) for a, b in zip(idx[:-1], idx[1:])], axis=1)
    return v, var


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", type=Path, default=Path.home() / "cernbox/unfold_inputs")
    ap.add_argument("--mlm-eras", default="2018", choices=("2018", "all"),
                    help="MLM pickles: the 2018 category of the merged files, or their Run-2 sum")
    ap.add_argument("--matchbox", type=Path, default=None, help="override the Matchbox pickle path")
    ap.add_argument("--mode", default="groomed", choices=tuple(MODE_KEY))
    ap.add_argument("--rho-floor", type=float, default=-5.0)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/plots/zjet_matchbox_vs_mlm"))
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    key, mode_text = MODE_KEY[args.mode]
    root = args.inputs / "zjet/rho/original"
    era_text = "2018" if args.mlm_eras == "2018" else "Run 2"
    rlabel = "2018 (13 TeV)" if args.mlm_eras == "2018" else "2018 Matchbox, Run 2 MLM (13 TeV)"

    if args.mlm_eras == "2018":
        herwig_cats, pythia_cats = [[c] for c in ZJET_ERA_CATS["2018"]]
        pythia_path = root / "pythia_2018.pkl"
    else:
        herwig_cats = pythia_cats = None
        pythia_path = root / "pythia_all.pkl"
    matchbox_path = args.matchbox or (
        root / "2018/herwig_matchbox/minimal_rho_zjet_herwig_matchbox_2018.pkl")
    data = {
        "mlm_pythia": load(pythia_path, key, pythia_cats),
        "mlm_herwig": load(root / "herwig_all.pkl", key, herwig_cats),
        "matchbox": load(matchbox_path, key, None),
    }
    _, _, rho_edges, pt_edges = data["mlm_pythia"]
    v, var, e_rho, e_pt = data["matchbox"]
    if not np.allclose(e_pt, pt_edges):
        raise SystemExit("matchbox: pT axis differs from the production")
    if len(e_rho) != len(rho_edges) or not np.allclose(e_rho, rho_edges):
        v, var = rebin_rho(v, var, e_rho, rho_edges)
        data["matchbox"] = (v, var, rho_edges, pt_edges)
        print(f"matchbox rho axis rebinned {len(e_rho) - 1} -> {len(rho_edges) - 1} bins")
    for k, (_, _, e_rho, e_pt) in data.items():
        if not (np.allclose(e_rho, rho_edges) and np.allclose(e_pt, pt_edges)):
            raise SystemExit(f"{k}: axes differ from mlm_pythia")
    centers_all = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    keep = centers_all > args.rho_floor
    centers = centers_all[keep]
    npt = len(pt_edges) - 1
    x = np.array([0.5 * (lo + min(hi, 600.0)) for lo, hi in zip(pt_edges[:-1], pt_edges[1:])])

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
        "matchbox": ("Matchbox Herwig7 / MLM Pythia8 (no merging)", "matchbox", "mlm_pythia", "#8b0000", "^", "-"),
        "mlm":      ("MLM Herwig7 / MLM Pythia8 (CMS central)", "mlm_herwig", "mlm_pythia", "#e42536", "s", "-."),
        "herwig":   ("Matchbox Herwig7 / MLM Herwig7", "matchbox", "mlm_herwig", "#e42536", "^", "-"),
    }
    rat = {k: ratio(a, b) for k, (_, a, b, *_r) in pairs.items()}

    #### the table
    print(f"\n{mode_text} mass, gen level, Z+jet, MLM eras = {era_text}, rho floor {args.rho_floor}")
    print(f"{'pT bin':>9s} | " + " ".join(f"{STYLE[k][0][:24]:>24s}" for k in data)
          + " | " + " ".join(f"{pairs[p][0].split(' (')[0][:30]:>30s}" for p in pairs))
    print(f"{'':>9s} | " + " ".join(f"{'median  (n_eff)':>24s}" for _ in data) + " | " + " ".join(f"{'mass ratio':>30s}" for _ in pairs))
    for i in range(npt):
        row = f"{PT_LABELS[i]:>9s} | " + " ".join(
            f"{med[k][i]:8.3f}±{err[k][i]:.3f} ({neff[k][i]:9.0f})" for k in data)
        row += " | " + " ".join(f"{rat[p][0][i]:14.3f} ± {rat[p][1][i]:<11.3f}" for p in pairs)
        print(row)
    print("slope (median relative to 200-290, last bin):",
          {k: round(float(med[k][-1] - med[k][1]), 3) for k in data})

    def finish(ax, ylabel, note, loc="upper right", headroom=1.5, note_at="top"):
        hep.cms.label("", data=False, loc=2, ax=ax, rlabel=rlabel)
        ax.set_xlabel(r"$p_{T}^{jet}$ (GeV)", fontsize=PUB_LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=PUB_LABEL_FS - 4)
        ax.tick_params(labelsize=PUB_TICK_FS)
        ax.set_xlim(170, 540)
        ax.set_xticks([200, 300, 400, 500])
        ax.set_xticklabels(["200", "300", "400", "> 400"])
        lo, hi = ax.get_ylim()
        step = next((st for st in (0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0) if (hi - lo) / st <= 9), 10.0)
        bottom = np.floor(lo / step) * step
        ax.set_ylim(bottom, bottom + (hi - bottom) * headroom)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if note_at == "top":
            ax.text(cms_x(ax), 0.74, note, transform=ax.transAxes, ha="left", va="top", fontsize=PUB_ANNOT_FS)
        else:  # bottom-left: the slope curves fall to the right, so this corner is empty
            ax.text(cms_x(ax), 0.04, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=PUB_ANNOT_FS)
        ax.legend(loc=loc, fontsize=PUB_LEGEND_FS - 2, frameon=False, borderaxespad=1.0)

    #### figure 1: median mass ratio Herwig7/Pythia8, Matchbox next to the MLM pair,
    #### and Matchbox / MLM Herwig7
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(21, 10.6))
    for ax, keys in zip(axes, (("matchbox", "mlm"), ("herwig",))):
        for p in keys:
            label, _, _, color, marker, ls = pairs[p]
            r, e = rat[p]
            ax.errorbar(x, r, yerr=e, marker=marker, ls=ls, lw=2.5, ms=11, color=color,
                        mfc="white" if p != "mlm" else color, mew=2.0, capsize=4, label=label)
        ax.axhline(1.0, color="gray", ls=":", lw=1.5)
    finish(axes[0], "median mass ratio Herwig7 / Pythia8",
           f"Z+jet, {mode_text}, GEN\nHerwig7 with and without MLM")
    finish(axes[1], "median mass ratio Matchbox / MLM",
           f"Z+jet, {mode_text}, GEN\nwhat the merging does to Herwig7")
    stamp(fig, enabled=not args.no_stamp)
    out = args.outdir / f"zjet_{args.mode}_mass_ratio_matchbox_vs_mlm.png"
    fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
    print(f"wrote {out}")

    #### figure 2: the slope of every sample relative to its own 200-290 bin
    fig, ax = plt.subplots(layout="constrained")
    for k in ("mlm_pythia", "mlm_herwig", "matchbox"):
        label, color, marker, ls, lw = STYLE[k]
        rel = med[k] - med[k][1]
        ax.errorbar(x, rel, yerr=np.sqrt(err[k] ** 2 + err[k][1] ** 2), marker=marker, ls=ls, lw=lw,
                    ms=11, color=color, mfc="white" if k == "matchbox" else color, mew=2.0,
                    capsize=3, label=label)
    ax.axhline(0.0, color="gray", ls=":", lw=1.2)
    finish(ax, r"median $\log_{10}(\rho^2)$ relative to 200–290 GeV",
           f"Z+jet, {mode_text}, GEN\nshape per bin; tune offsets removed", headroom=1.4, note_at="bottom")
    stamp(fig, enabled=not args.no_stamp)
    out = args.outdir / f"zjet_{args.mode}_slope_matchbox_vs_mlm.png"
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
        ratio_tops = []
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
                ratio_tops.append((r, re))
        hep.cms.label("", data=False, loc=2, ax=ax, rlabel=rlabel)
        ax.text(cms_x(ax), 0.74, f"Z+jet, {mode_text}, GEN\n{PT_LABELS[i]} GeV",
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
        # data-driven: bracket every ratio point AND its error bar (sparse tail bins)
        tops = [np.nanmax(r + re) for r, re in ratio_tops] if ratio_tops else []
        top = max([rhi] + [t * 1.05 for t in tops if np.isfinite(t)])
        rax.set_ylim(max(0.0, rlo), top)
        rax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        for a in (ax, rax):
            a.set_xlim(e_keep[0], e_keep[-1])
        fig.align_ylabels([ax, rax])
        stamp(fig, enabled=not args.no_stamp)
        out = args.outdir / f"zjet_{args.mode}_shape_pt{i}.png"
        fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
