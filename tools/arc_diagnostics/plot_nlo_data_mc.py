#!/usr/bin/env python
"""Reco-level data/MC rho (dimensionless jet mass) comparison with the NLO DY overlay.

ARC SMP-25-010 (L237 / Modelling#3) asks whether using NLO DY instead of LO improves
the reco-level groomed/ungroomed jet-mass data/MC agreement (AN Figs 32-33). This
overlays, per pT bin and for groomed + ungroomed:
    Data  vs  LO (madgraphMLM pythia8)  vs  NLO (amcatnloFXFX, PtZ-stitched)  vs  Herwig
area-normalized per pT bin (the measurement is normalized, so the shape is the point),
with a ratio-to-data panel.

Inputs (pkls from `minimal_rho` runs, 2018):
  ~/Downloads/minimal_rho_nlo_ptz_2018.pkl     dataset MC_UL18NanoAODv9  (coarse rho)
  ~/Downloads/minimal_rho_pythia_2018.pkl      dataset pythia_UL18...    (coarse rho)
  ~/Downloads/minimal_rho_fine_data_2018.pkl   EGamma+SingleMuon         (fine rho)
  ~/Downloads/minimal_rho_fine_herwig_2018.pkl herwig_UL18...            (fine rho)
The coarse 24-bin rho edges are an exact subset of the fine 96-bin edges, so the fine
data/herwig are rebinned down to the coarse axis. ptreco = [0,200,290,400,13000] in all.
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep

hep.style.use(hep.style.CMS)

D = os.path.expanduser("~/Downloads")
OUT = os.path.expanduser("~/Projects/smp_jetmass_run2/review/figs/nlo_data_mc")
os.makedirs(OUT, exist_ok=True)
LUMI = "59.8 fb$^{-1}$ (2018, 13 TeV)"

# (label, file, dataset(s), color, is_data)
SRC = {
    "data":   ("Data",        "minimal_rho_fine_data_2018.pkl",
               ["EGamma_UL2018", "SingleMuon_UL2018"], "black", True),
    "lo":     ("LO (MG+Py8)",  "minimal_rho_pythia_2018.pkl",
               ["pythia_UL18NanoAODv9"], "#e42536", False),
    "nlo":    ("NLO (FxFx)",   "minimal_rho_nlo_ptz_2018.pkl",
               ["MC_UL18NanoAODv9"], "#3f90da", False),
    "herwig": ("Herwig7",      "minimal_rho_fine_herwig_2018.pkl",
               ["herwig_UL18NanoAODv9"], "#92dadd", False),
}
PT_BINS = [(1, "200 < $p_{T}$ < 290 GeV", "pt200_290"),
           (2, "290 < $p_{T}$ < 400 GeV", "pt290_400"),
           (3, "$p_{T}$ > 400 GeV",       "pt400inf")]
GROOM = {"u": ("Ungroomed", "ptjet_rhojet_u_reco"),
         "g": ("Groomed",   "ptjet_rhojet_g_reco")}


def load(f):
    return pickle.load(open(os.path.join(D, f), "rb"))


def rebin_to(coarse_edges, fine_edges, fine_vals, fine_var):
    """Sum fine bins into the coarse edges (which must be a subset of the fine edges)."""
    fe = np.round(fine_edges, 4)
    idx = np.searchsorted(fe, np.round(coarse_edges, 4))
    out_v = np.add.reduceat(fine_vals, idx[:-1])
    out_e = np.add.reduceat(fine_var, idx[:-1])
    return out_v, out_e


def get_rho(key, hname, ptbin_idx, coarse_edges):
    """Return (values, variances) on the coarse rho axis for one pT bin."""
    label, f, datasets, color, is_data = SRC[key]
    o = load(f)
    H = o[hname][{"systematic": "nominal"}]
    vals = var = None
    for ds in datasets:
        h = H[{"dataset": ds, "ptreco": ptbin_idx}]
        v, e = h.values(), h.variances()
        vals = v if vals is None else vals + v
        var = e if var is None else var + e
    fine_edges = H.axes["mpt_reco"].edges
    if len(vals) != len(coarse_edges) - 1:               # fine -> rebin to coarse
        vals, var = rebin_to(coarse_edges, fine_edges, vals, var)
    return vals, var


def density(vals, var, edges, normalize=True):
    """Differential yield dN/drho. If normalize, divide by the integral (unit area)."""
    w = np.diff(edges)
    cx = 0.5 * (edges[:-1] + edges[1:])
    norm = vals.sum() if normalize else 1.0
    if norm <= 0:
        return np.zeros_like(vals), np.zeros_like(vals), cx, w
    return vals / w / norm, np.sqrt(var) / w / norm, cx, w


def main(normalize=True):
    nlo = load("minimal_rho_nlo_ptz_2018.pkl")
    coarse_edges = nlo["ptjet_rhojet_g_reco"].axes["mpt_reco"].edges
    outdir = OUT if normalize else OUT + "_abs"
    os.makedirs(outdir, exist_ok=True)
    ylab = r"$(1/N)\,dN/d\rho$" if normalize else r"$dN/d\rho$  (events)"
    # Herwig is normalized to its model-uncertainty xs (5.036 pb) -> only meaningful
    # area-normalized; drop it from the absolute (lumi-scaled) yield comparison.
    mc_keys = ["lo", "nlo", "herwig"] if normalize else ["lo", "nlo"]

    for gkey, (gname, hname) in GROOM.items():
        for ptidx, ptlabel, pttag in PT_BINS:
            curves = {}
            for key in ["data"] + mc_keys:
                v, e = get_rho(key, hname, ptidx, coarse_edges)
                curves[key] = density(v, e, coarse_edges, normalize=normalize)

            fig = plt.figure(figsize=(10, 11))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.07)
            ax = fig.add_subplot(gs[0])
            rax = fig.add_subplot(gs[1], sharex=ax)

            d, derr, cx, _ = curves["data"]
            ax.errorbar(cx, d, yerr=derr, fmt="o", color="black", ms=7,
                        label=SRC["data"][0], zorder=5)
            for key in mc_keys:
                y, yerr, cx2, _ = curves[key]
                ax.step(coarse_edges, np.append(y, y[-1]), where="post",
                        color=SRC[key][3], lw=2.2, label=SRC[key][0])
                # ratio MC/data
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(d > 0, y / d, np.nan)
                    rerr = np.where(d > 0, yerr / d, np.nan)
                rax.step(coarse_edges, np.append(r, r[-1]), where="post",
                         color=SRC[key][3], lw=2.2)
                rax.fill_between(coarse_edges, np.append(r - rerr, (r - rerr)[-1]),
                                 np.append(r + rerr, (r + rerr)[-1]), step="post",
                                 color=SRC[key][3], alpha=0.15)
            # data ref band in ratio
            with np.errstate(divide="ignore", invalid="ignore"):
                drel = np.where(d > 0, derr / d, 0.0)
            rax.fill_between(coarse_edges, np.append(1 - drel, (1 - drel)[-1]),
                             np.append(1 + drel, (1 + drel)[-1]), step="post",
                             color="black", alpha=0.18, label="Data unc.")
            rax.axhline(1.0, color="black", lw=1, ls="--")

            allmax = max(np.nanmax(c[0]) for c in curves.values())
            ax.set_ylim(0, allmax * 1.5)
            ax.set_ylabel(ylab)
            ax.legend(loc="upper left", fontsize=18, frameon=False,
                      title=f"{gname}", title_fontsize=18)
            hep.cms.label("Preliminary", data=True, loc=0, rlabel=LUMI, ax=ax)
            ax.text(0.97, 0.92, ptlabel, ha="right", va="top", transform=ax.transAxes,
                    fontsize=19)
            plt.setp(ax.get_xticklabels(), visible=False)

            rax.set_ylim(0.4, 1.6)
            rax.set_ylabel("MC / Data", fontsize=18)
            rax.set_xlabel(r"$\rho = 2\,\log_{10}(m/(p_{T}R))$")
            rax.set_xlim(-7, 0)

            out = os.path.join(outdir, f"nlo_data_mc_{gkey}_{pttag}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print("wrote", out)


if __name__ == "__main__":
    import sys
    if "--abs" in sys.argv:
        main(normalize=False)
    elif "--both" in sys.argv:
        main(normalize=True); main(normalize=False)
    else:
        main(normalize=True)
