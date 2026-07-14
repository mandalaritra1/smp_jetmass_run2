#!/usr/bin/env python
"""Reco-level data/MC jet-MASS comparison with the NLO DY overlay (groomed + ungroomed).

The mass analog of plot_nlo_data_mc.py (which does rho). ARC SMP-25-010
(L237 / Modelling#3) asks whether NLO DY instead of LO improves the reco-level
groomed/ungroomed jet-mass data/MC agreement. Per pT bin, for groomed (soft-drop)
and ungroomed mass, overlays:
    Data  vs  LO (madgraphMLM pythia8)  vs  NLO (amcatnloFXFX, PtZ-stitched)  vs  Herwig
area-normalized per pT bin (the measurement is normalized -> shape is the point),
with a ratio-to-data panel.

Inputs (2018):
  Data / LO / Herwig come from the per-jet reco ntuples (mass=ungroomed,
  msoftdrop=groomed, weight); NLO comes from the binned `ptjet_mjet_{g,u}_reco`
  histograms in the minimal (mass) NLO run.
    ~/Downloads/mass_diagnostic_ntuple_data_2018.pkl    reco_jet_ntuple (data)
    ~/Downloads/mass_diagnostic_ntuple_pythia_2018.pkl  reco_jet_ntuple (LO)
    ~/Downloads/mass_diagnostic_ntuple_herwig_2018.pkl  reco_jet_ntuple (Herwig)
    ~/Downloads/minimal_nlo_ptz_2018.pkl                ptjet_mjet_{g,u}_reco (NLO)
The NLO histogram's fine mreco axis (integer-GeV up to 200) nests the 5-GeV display
grid, so NLO is rebinned down with np.add.reduceat; the ntuples are histogrammed
straight onto the display grid. ptreco = [0,200,290,400,13000] in both.
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep

hep.style.use(hep.style.CMS)

D = os.path.expanduser("~/Downloads")
OUT = os.path.expanduser("~/Projects/smp_jetmass_run2/review/figs/nlo_data_mc_mass")
LUMI = "59.8 fb$^{-1}$ (2018, 13 TeV)"

# Per-jet reco ntuples: (label, file, column-for-groomed-key, color)
NT = {
    "data":   ("Data",        "mass_diagnostic_ntuple_data_2018.pkl",   "black"),
    "lo":     ("LO (MG+Py8)",  "mass_diagnostic_ntuple_pythia_2018.pkl", "#e42536"),
    "herwig": ("Herwig7",      "mass_diagnostic_ntuple_herwig_2018.pkl", "#92dadd"),
}
NLO_FILE = "minimal_nlo_ptz_2018.pkl"
NLO_LABEL, NLO_COLOR = "NLO (FxFx)", "#3f90da"

PT_BINS = [(1, 200, 290, "200 < $p_{T}$ < 290 GeV", "pt200_290"),
           (2, 290, 400, "290 < $p_{T}$ < 400 GeV", "pt290_400"),
           (3, 400, 1e9, "$p_{T}$ > 400 GeV",       "pt400inf")]
# (key, pretty, ntuple-column, NLO-hist, display upper edge [GeV])
GROOM = {"g": ("Groomed (soft drop)", "msoftdrop", "ptjet_mjet_g_reco", 150.0),
         "u": ("Ungroomed",           "mass",      "ptjet_mjet_u_reco", 200.0)}

_cache = {}


def ntuple(f):
    if f not in _cache:
        o = pickle.load(open(os.path.join(D, f), "rb"))["reco_jet_ntuple"]
        cols = {}
        for k in ("pt", "mass", "msoftdrop", "weight"):
            v = o[k]
            cols[k] = np.asarray(v.value if hasattr(v, "value") else v)
        _cache[f] = cols
    return _cache[f]


def nt_hist(f, col, ptlo, pthi, edges):
    """Weighted (values, variances) of `col` in the pT window, on display edges."""
    c = ntuple(f)
    sel = (c["pt"] >= ptlo) & (c["pt"] < pthi)
    x, w = c[col][sel], c["weight"][sel]
    vals, _ = np.histogram(x, bins=edges, weights=w)
    var, _ = np.histogram(x, bins=edges, weights=w * w)
    return vals, var


def nlo_hist(hname, ptidx, edges, fine_edges):
    """NLO (values, variances): sum channels, select pT bin, rebin fine->display."""
    o = pickle.load(open(os.path.join(D, NLO_FILE), "rb"))
    H = o[hname][{"systematic": "nominal", "dataset": sum}]
    H = H[{"ptreco": ptidx, "channel": sum}]
    v, e = H.values(), H.variances()
    fe = np.round(fine_edges, 4)
    idx = np.searchsorted(fe, np.round(edges, 4))
    return np.add.reduceat(v, idx[:-1]), np.add.reduceat(e, idx[:-1])


def density(vals, var, edges, normalize=True):
    w = np.diff(edges)
    norm = vals.sum() if normalize else 1.0
    if norm <= 0:
        return np.zeros_like(vals), np.zeros_like(vals)
    return vals / w / norm, np.sqrt(var) / w / norm


def main(normalize=True):
    outdir = OUT if normalize else OUT + "_abs"
    os.makedirs(outdir, exist_ok=True)
    nlo = pickle.load(open(os.path.join(D, NLO_FILE), "rb"))
    fine_edges = nlo["ptjet_mjet_g_reco"].axes["mreco"].edges
    ylab = r"$(1/N)\,dN/dm$  [1/GeV]" if normalize else r"$dN/dm$  [events/GeV]"
    mc_keys = ["lo", "herwig"] if normalize else ["lo"]   # Herwig xs only meaningful normalized

    for gkey, (gname, col, hname, mtop) in GROOM.items():
        edges = np.arange(0.0, mtop + 5.0, 5.0)            # 5-GeV display grid (subset of fine)
        cx = 0.5 * (edges[:-1] + edges[1:])
        for ptidx, ptlo, pthi, ptlabel, pttag in PT_BINS:
            curves = {}
            dv, de = nt_hist(NT["data"][1], col, ptlo, pthi, edges)
            curves["data"] = density(dv, de, edges, normalize)
            for key in mc_keys:
                mv, mvar = nt_hist(NT[key][1], col, ptlo, pthi, edges)
                curves[key] = density(mv, mvar, edges, normalize)
            nv, nvar = nlo_hist(hname, ptidx, edges, fine_edges)
            curves["nlo"] = density(nv, nvar, edges, normalize)

            fig = plt.figure(figsize=(10, 11))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.07)
            ax = fig.add_subplot(gs[0]); rax = fig.add_subplot(gs[1], sharex=ax)

            d, derr = curves["data"]
            ax.errorbar(cx, d, yerr=derr, fmt="o", color="black", ms=7,
                        label=NT["data"][0], zorder=5)
            order = (["lo", "herwig", "nlo"] if normalize else ["lo", "nlo"])
            colmap = {"lo": NT["lo"][2], "herwig": NT["herwig"][2], "nlo": NLO_COLOR}
            labmap = {"lo": NT["lo"][0], "herwig": NT["herwig"][0], "nlo": NLO_LABEL}
            for key in order:
                y, yerr = curves[key]
                ax.step(edges, np.append(y, y[-1]), where="post",
                        color=colmap[key], lw=2.2, label=labmap[key])
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(d > 0, y / d, np.nan)
                    rerr = np.where(d > 0, yerr / d, np.nan)
                rax.step(edges, np.append(r, r[-1]), where="post", color=colmap[key], lw=2.2)
                rax.fill_between(edges, np.append(r - rerr, (r - rerr)[-1]),
                                 np.append(r + rerr, (r + rerr)[-1]), step="post",
                                 color=colmap[key], alpha=0.15)
            with np.errstate(divide="ignore", invalid="ignore"):
                drel = np.where(d > 0, derr / d, 0.0)
            rax.fill_between(edges, np.append(1 - drel, (1 - drel)[-1]),
                             np.append(1 + drel, (1 + drel)[-1]), step="post",
                             color="black", alpha=0.18, label="Data unc.")
            rax.axhline(1.0, color="black", lw=1, ls="--")

            allmax = max(np.nanmax(c[0]) for c in curves.values())
            ax.set_ylim(0, allmax * 1.5)
            ax.set_ylabel(ylab)
            ax.legend(loc="upper right", fontsize=18, frameon=False,
                      title=gname, title_fontsize=18)
            hep.cms.label("Preliminary", data=True, loc=0, rlabel=LUMI, ax=ax)
            ax.text(0.97, 0.62, ptlabel, ha="right", va="top", transform=ax.transAxes,
                    fontsize=19)
            plt.setp(ax.get_xticklabels(), visible=False)

            rax.set_ylim(0.4, 1.6)
            rax.set_ylabel("MC / Data", fontsize=18)
            xname = "soft-drop mass" if gkey == "g" else "mass"
            rax.set_xlabel(rf"AK8 jet {xname} $m$ [GeV]")
            rax.set_xlim(0, mtop)

            out = os.path.join(outdir, f"nlo_mass_{gkey}_{pttag}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
            print("wrote", out)


if __name__ == "__main__":
    import sys
    if "--abs" in sys.argv:
        main(normalize=False)
    elif "--both" in sys.argv:
        main(normalize=True); main(normalize=False)
    else:
        main(normalize=True)
