"""Detector-level jet mass in GeV, groomed and ungroomed: data vs Pythia8 vs Herwig7.

The rho comparisons show the same thing on a log axis at fixed pT; this is the
mass spectrum itself, integrated over the analysis pT range, so a reader sees
the shift in GeV.  Two panels per channel (soft-drop groomed, ungroomed), unit
area each, with Data/MC ratio panels.

    python scripts/diagnostics/plot_reco_mass_data_generators.py --channel dijet
    python scripts/diagnostics/plot_reco_mass_data_generators.py --channel zjet \
        --zjet-pythia /tmp/ptbal/validation_pythia_2018.pkl

Z+jet: `m_g_vs_m_u_reco` (5 GeV bins to 200 GeV) from the production data and
Herwig pickles and from a validation-mode Pythia pickle (the production Pythia
pickle carries only rho).  Dijet/trijet: `msoftdrop_jet1`/`mass_jet1`
(`_jet3` for trijet), all four eras summed.
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
HADRONIC_ERAS = ("2016APV", "2016", "2017", "2018")
CHANNEL_LABEL = {"zjet": "Z+jet", "dijet": "Dijet", "trijet": "Trijet"}
STYLE = {
    "data":   ("Data",          "k",       "-"),
    "pythia": ("Pythia8 (CP5)", "#5790fc", ":"),
    "herwig": ("Herwig7 (CH3)", "#e42536", "-."),
}
MASS_MAX = {"zjet": 200.0, "dijet": 400.0, "trijet": 400.0}


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


def nominal(h):
    names = [a.name for a in h.axes]
    index = {}
    for name in ("dataset", "channel"):
        if name in names:
            index[name] = sum
    if "systematic" in names:
        index["systematic"] = "nominal" if "nominal" in list(h.axes["systematic"]) else sum
    return h[index] if index else h


def load(channel, inputs_root, zjet_pythia):
    """{mode: {source: (values, variances, edges)}}."""
    root = Path(inputs_root) / channel / "rho" / "original"
    out = {"groomed": {}, "ungroomed": {}}
    if channel == "zjet":
        paths = {"data": [root / "data_all.pkl"], "herwig": [root / "herwig_all.pkl"],
                 "pythia": [Path(zjet_pythia)]}
        for source, files in paths.items():
            h = nominal(pickle.load(open(files[0], "rb"))["m_g_vs_m_u_reco"])
            for mode, axis in (("groomed", "m_g_reco"), ("ungroomed", "m_u_reco")):
                proj = h.project(axis)
                out[mode][source] = (np.asarray(proj.values()), np.asarray(proj.variances()),
                                     proj.axes[0].edges)
        return out
    jet = "jet3" if channel == "trijet" else "jet1"
    keys = {"groomed": f"msoftdrop_{jet}", "ungroomed": f"mass_{jet}"}
    folders = {"data": (f"{channel}_data", "data"), "pythia": (f"{channel}_mc", "mg_pythia8"),
               "herwig": (f"{channel}_herwig", "herwig")}
    for source, (folder, stem) in folders.items():
        for mode, key in keys.items():
            values = variances = edges = None
            for era in HADRONIC_ERAS:
                h = nominal(pickle.load(open(
                    root / era / folder / f"minimal_rho_{channel}_{stem}_{era}.pkl", "rb"))[key])
                v, var = np.asarray(h.values()), np.asarray(h.variances())
                values = v.copy() if values is None else values + v
                variances = var.copy() if variances is None else variances + var
                edges = h.axes[0].edges
            out[mode][source] = (values, variances, edges)
    return out


def density(values, variances, edges):
    widths = np.diff(edges)
    total = values.sum()
    if total <= 0:
        return None, None
    return values / widths / total, np.sqrt(np.clip(variances, 0.0, None)) / widths / total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--channel", default="dijet", choices=("zjet", "dijet", "trijet"))
    ap.add_argument("--inputs", type=Path, default=Path.home() / "Projects/unfold/inputs")
    ap.add_argument("--zjet-pythia", type=Path, default=Path("/tmp/ptbal/validation_pythia_2018.pkl"))
    ap.add_argument("--outdir", type=Path, default=Path("outputs/plots/reco_mass_data"))
    ap.add_argument("--lumi", default=r"138 fb$^{-1}$")
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = load(args.channel, args.inputs, args.zjet_pythia)
    fig, axes = plt.subplots(2, 2, sharex="col", layout="constrained",
                             figsize=(21, 11.5),
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})
    print(f"{args.channel}: median mass [GeV], reco, pT-integrated")
    for col, mode in enumerate(("groomed", "ungroomed")):
        ax, rax = axes[0, col], axes[1, col]
        edges_full = data[mode]["pythia"][2]
        keep = edges_full[1:] <= MASS_MAX[args.channel] + 1e-9
        if mode == "ungroomed":
            keep &= edges_full[:-1] >= 5.0 - 1e-9   # m < 5 GeV is unphysical here
        first = int(np.argmax(keep))
        edges = edges_full[first: first + int(keep.sum()) + 1]
        centers = 0.5 * (edges[:-1] + edges[1:])
        curves, ymax, medians = {}, 0.0, {}
        for source, (label, color, style) in STYLE.items():
            values, variances, _ = data[mode][source]
            values, variances = values[keep], variances[keep]
            dens, err = density(values, variances, edges)
            if dens is None:
                continue
            curves[source] = (dens, err)
            cdf = np.cumsum(values) / values.sum()
            medians[source] = float(np.interp(0.5, cdf, centers))
            if source == "data":
                ax.errorbar(centers, dens, yerr=err, xerr=np.diff(edges) / 2, fmt="o",
                            color=color, ms=6, label=label)
            else:
                ax.stairs(dens, edges, color=color, lw=2.5, ls=style, label=label, baseline=None)
            ymax = max(ymax, float((dens + err).max()))
        print(f"  {mode:10s} " + "  ".join(f"{s}={m:6.1f}" for s, m in medians.items()))
        hep.cms.label("Preliminary", data=True, loc=2, ax=ax, rlabel=f"{args.lumi} (13 TeV)")
        ax.set_ylabel(r"$(1/N)\,dN/dm$ (GeV$^{-1}$)", fontsize=PUB_LABEL_FS)
        ax.tick_params(labelsize=PUB_TICK_FS)
        # Log axis: the tail is the point.  Floor just under the smallest
        # drawn density; two decades of headroom keep the peak clear of the
        # CMS block (top-left) and the legend (top-right).
        positive = np.concatenate([d[d > 0] for d, _ in curves.values()])
        ax.set_yscale("log")
        ax.set_ylim(positive.min() / 3.0, ymax * 100.0)
        # Channel/mode text under the legend, on the right, where the falling
        # spectrum leaves the frame empty; the left is the peak.
        ax.text(0.97, 0.70, f"{CHANNEL_LABEL[args.channel]}, {mode}, RECO\n$p_T > 200$ GeV",
                transform=ax.transAxes, ha="right", va="top", fontsize=PUB_ANNOT_FS)
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: labels[i] != "Data")
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc="upper right", fontsize=PUB_LEGEND_FS, frameon=False)

        # Fixed [0, 3] on both panels so they read side by side; bins where
        # either input carries > 30% error are dropped (the tail is
        # statistics-starved); a point beyond the range is marked with a
        # triangle at the frame in the series colour rather than silently cut.
        top, bottom = 3.0, 0.0
        dd, ed = curves["data"]
        rel_d = np.divide(ed, dd, out=np.zeros_like(dd), where=dd > 0)
        for source in ("pythia", "herwig"):
            ds, es = curves[source]
            ratio = np.divide(dd, ds, out=np.full_like(dd, np.nan), where=ds > 0)
            rel_s = np.divide(es, ds, out=np.zeros_like(ds), where=ds > 0)
            err = np.abs(ratio) * np.sqrt(rel_d ** 2 + rel_s ** 2)
            usable = np.isfinite(ratio) & (rel_s < 0.3) & (rel_d < 0.3) & (dd > 0)
            inside = usable & (ratio - err >= bottom) & (ratio + err <= top)
            color, style = STYLE[source][1], STYLE[source][2]
            rax.stairs(np.where(usable, np.clip(ratio, bottom, top), np.nan), edges,
                       color=color, lw=2.5, ls=style, baseline=None)
            rax.errorbar(centers[inside], ratio[inside], yerr=err[inside], fmt="none",
                         ecolor=color, elinewidth=1.2, capsize=2)
            over = usable & ~inside
            if over.any():
                high = over & (ratio + err > top)
                low = over & (ratio - err < bottom)
                rax.plot(centers[high], np.full(high.sum(), top - 0.05 * (top - bottom)),
                         marker="^", ls="none", color=color, ms=7, clip_on=True)
                rax.plot(centers[low], np.full(low.sum(), bottom + 0.05 * (top - bottom)),
                         marker="v", ls="none", color=color, ms=7, clip_on=True)
        rax.axhline(1.0, color="gray", ls="--", lw=1)
        rax.set_ylim(bottom, top)
        rax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
        rax.set_ylabel("Data / MC", fontsize=PUB_LABEL_FS - 4)
        rax.set_xlabel(("Soft-drop " if mode == "groomed" else "Ungroomed ") + "jet mass (GeV)",
                       fontsize=PUB_LABEL_FS)
        rax.tick_params(labelsize=PUB_TICK_FS)
    stamp(fig, f"{args.channel}/rho/original reco", enabled=not args.no_stamp)
    out = args.outdir / f"reco_mass_{args.channel}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
