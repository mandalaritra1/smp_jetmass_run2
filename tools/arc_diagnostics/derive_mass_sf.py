#!/usr/bin/env python
"""Prototype: gen->reco mass scale factor SF(pT, m) to center the response on the diagonal.

Step 1-2 of the diagonalization proposal. Per matched jet (passes_both) we form the
response ratio r = m_reco / m_gen (groomed = msoftdrop, ungroomed = mass) and profile
it, RECO-INDEXED so it is applicable to data:
        SF(m_reco, pT_reco) = profile of r  in (reco-pT bin, reco-mass bin)
Corrected reco mass would then be  m_reco / SF(m_reco).  Two estimators are shown:
median (robust to the soft-drop low-mass tail) and an iterative +-2 sigma Gaussian
core. Pythia (LO, nominal) vs Herwig gives the model spread = the SF systematic.

Inputs (2018 per-jet reco ntuples):
  ~/Downloads/mass_diagnostic_ntuple_pythia_2018.pkl
  ~/Downloads/mass_diagnostic_ntuple_herwig_2018.pkl
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

D = os.path.expanduser("~/Downloads")
OUT = os.path.expanduser("~/Projects/smp_jetmass_run2/review/figs/nlo_deck")
os.makedirs(OUT, exist_ok=True)
LUMI = "(2018, 13 TeV)"
MINJETS = 50                                            # min matched jets / bin

SRC = {"lo":     ("Pythia8 (LO)", "mass_diagnostic_ntuple_pythia_2018.pkl", "#e42536"),
       "herwig": ("Herwig7",      "mass_diagnostic_ntuple_herwig_2018.pkl", "#3f90da")}
PT_BINS = [(200, 290, "200 < $p_{T}$ < 290 GeV", "pt200_290"),
           (290, 400, "290 < $p_{T}$ < 400 GeV", "pt290_400"),
           (400, 1e9, "$p_{T}$ > 400 GeV",       "pt400inf")]
# (key, pretty, reco-col, gen-col, display upper edge)
GROOM = {"g": ("Groomed (soft drop)", "msoftdrop", "gen_msoftdrop", 150.0),
         "u": ("Ungroomed",           "mass",      "gen_mass",      200.0)}

_cache = {}


def cols(f):
    if f not in _cache:
        o = pickle.load(open(os.path.join(D, f), "rb"))["reco_jet_ntuple"]
        d = {}
        for k in ("pt", "mass", "msoftdrop", "gen_mass", "gen_msoftdrop",
                  "weight", "passes_both"):
            v = o[k]
            d[k] = np.asarray(v.value if hasattr(v, "value") else v)
        _cache[f] = d
    return _cache[f]


def gauss_core(r, w, n_iter=3):
    """Iterative +-2 sigma weighted mean (Gaussian-core estimator)."""
    m = np.ones(len(r), bool)
    mu = np.nan
    for _ in range(n_iter):
        if m.sum() < 5:
            break
        mu = np.average(r[m], weights=w[m])
        sd = np.sqrt(np.average((r[m] - mu) ** 2, weights=w[m]))
        if sd == 0:
            break
        m = np.abs(r - mu) < 2 * sd
    return mu


def profile(f, reco_col, gen_col, ptlo, pthi, edges):
    """Return (median, gauss-core) SF per reco-mass bin for one pT window."""
    c = cols(f)
    sel = (c["passes_both"].astype(bool) & (c["pt"] >= ptlo) & (c["pt"] < pthi)
           & (c[gen_col] > 0) & (c[reco_col] > 0))
    mreco, r, w = c[reco_col][sel], c[reco_col][sel] / c[gen_col][sel], c["weight"][sel]
    idx = np.digitize(mreco, edges) - 1
    med = np.full(len(edges) - 1, np.nan)
    core = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < MINJETS:
            continue
        med[b] = np.median(r[m])
        core[b] = gauss_core(r[m], w[m])
    return med, core


def main():
    print(f"{'grooming':10s} {'pT bin':16s} {'SF (median, all-mass)':>22s}   model")
    for gkey, (gname, rcol, gcol, mtop) in GROOM.items():
        edges = np.arange(0.0, mtop + 5.0, 5.0)
        cx = 0.5 * (edges[:-1] + edges[1:])
        for ptlo, pthi, ptlabel, pttag in PT_BINS:
            fig, ax = plt.subplots()
            for key in ("lo", "herwig"):
                label, f, color = SRC[key]
                med, core = profile(f, rcol, gcol, ptlo, pthi, edges)
                ax.plot(cx, med, "-", color=color, lw=2.4, label=f"{label} (median)")
                if key == "lo":
                    ax.plot(cx, core, "--", color=color, lw=1.8, alpha=0.8,
                            label=f"{label} (Gauss core)")
                # quick all-mass number
                c = cols(f)
                s = (c["passes_both"].astype(bool) & (c["pt"] >= ptlo) & (c["pt"] < pthi)
                     & (c[gcol] > 0) & (c[rcol] > 0))
                allsf = np.median(c[rcol][s] / c[gcol][s])
                print(f"{gname:10s} {ptlabel:16s} {allsf:22.3f}   {label}")

            ax.axhline(1.0, color="gray", ls=":", lw=1.5)
            ax.set_ylim(0.6, 1.3)
            ax.set_xlim(0, mtop)
            ax.set_xlabel(rf"reco {'soft-drop ' if gkey=='g' else ''}mass $m_{{reco}}$ [GeV]")
            ax.set_ylabel(r"SF $= \langle m_{reco}/m_{gen}\rangle$")
            ax.legend(loc="lower right", fontsize=15, frameon=False,
                      title=gname, title_fontsize=16)
            hep.cms.label("Preliminary", data=False, loc=0, rlabel=LUMI, ax=ax)
            ax.text(0.04, 0.94, ptlabel, ha="left", va="top", transform=ax.transAxes,
                    fontsize=18)
            out = os.path.join(OUT, f"mass_sf_{gkey}_{pttag}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
            print("  wrote", out)


if __name__ == "__main__":
    main()
