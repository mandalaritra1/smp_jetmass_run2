#!/usr/bin/env python3
"""Groomed rho response P(rho_reco | rho_gen) vs a groomed-mass floor (NLO skims).

rho = 2*log10(m/(pt*R)), R=0.8 -- the actual measured observable. Because rho ~ log(m),
the multiplicative mass scale becomes an additive shift and the relative resolution a
~constant additive width, so the rho response band is more uniform than the mass one.

The groomed-mass veto m_g > thr maps to a pt-dependent FLOOR in rho_gen (applied
event-by-event). For each jet-pT bin we show P(rho_reco|rho_gen) at m_g > {0,2,5,10} GeV;
each gen column is unit-normalized (colour = P), dashed = y=x, white verticals = the
analysis gen-rho bin edges. Mean purity/stability on the analysis 12-bin gen-rho grid
are printed for every cut. Cut applied to BOTH gen and reco groomed mass.

  ~/Projects/GluonJetMass/.venv/bin/python tools/arc_diagnostics/nlo_groomed_response_masscut_rho.py
"""
from __future__ import annotations
import os, sys, glob, re
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

hep.style.use(hep.style.CMS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlo_groomed_response_masscut import load, matrix          # reuse loader + matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figs")
R = 0.8
PT_BINS = [(200, 290, "$200<p_T<290$"), (290, 400, "$290<p_T<400$"), (400, 1e9, "$p_T>400$")]
CUTS_TABLE = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0]
CUTS_PLOT = [0.0, 2.0, 5.0, 10.0]
RHO_LO, RHO_HI = -7.0, 0.0
FINE = np.arange(RHO_LO, RHO_HI + 1e-6, 0.1)                   # visual grid
COARSE = np.array([-6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.])  # analysis gen-rho


def rho(m, pt):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(m / (pt * R))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    c = load()
    rg = rho(c["gmg"], c["gpt"]); rr = rho(c["rmg"], c["rpt"])
    print(f"pooled matched gen jets: {len(c['gpt']):,d}\n")

    print(f"{'pT bin':14s} {'cut':>10s}  {'purity':>7s} {'stab':>6s} {'diag':>6s} {'kept':>6s}")
    for plo, phi, lab in PT_BINS:
        base = ((c["gpt"] >= plo) & (c["gpt"] < phi) & (c["rpt"] >= plo) & (c["rpt"] < phi)
                & np.isfinite(rg) & np.isfinite(rr))
        ntot = c["w"][base].sum()
        for thr in CUTS_TABLE:
            sel = base & (c["gmg"] > thr) & (c["rmg"] > thr)
            _, pu, st, dg = matrix(rg[sel], rr[sel], c["w"][sel], COARSE)
            print(f"{lab:14s} {('m_g>'+format(thr,'g')):>10s}  {pu:7.3f} {st:6.3f} {dg:6.3f} "
                  f"{c['w'][sel].sum()/ntot:6.1%}")
        print()

    nr, nc = len(PT_BINS), len(CUTS_PLOT)
    fig, axes = plt.subplots(nr, nc, figsize=(6.2 * nc, 6.0 * nr), layout="constrained")
    pcm = None
    for i, (plo, phi, lab) in enumerate(PT_BINS):
        base = ((c["gpt"] >= plo) & (c["gpt"] < phi) & (c["rpt"] >= plo) & (c["rpt"] < phi)
                & np.isfinite(rg) & np.isfinite(rr))
        for j, thr in enumerate(CUTS_PLOT):
            ax = axes[i, j]
            sel = base & (c["gmg"] > thr) & (c["rmg"] > thr)
            P, pu, st, dg = matrix(rg[sel], rr[sel], c["w"][sel], FINE)
            pcm = ax.pcolormesh(FINE, FINE, P, cmap="cividis",
                                norm=mcolors.LogNorm(vmin=1e-3, vmax=1.0))
            ax.plot([RHO_LO, RHO_HI], [RHO_LO, RHO_HI], color="#e42536", lw=1.4, ls="--")
            for e in COARSE:
                ax.axvline(e, color="white", lw=0.5, alpha=0.3)
            ax.set_xlim(RHO_LO, RHO_HI); ax.set_ylim(RHO_LO, RHO_HI); ax.set_aspect("equal")
            cuttxt = "no cut" if thr == 0 else f"$m_g>{thr:g}$ GeV"
            ax.text(0.05, 0.95, f"{lab}\n{cuttxt}\npurity {pu:.2f}", ha="left", va="top",
                    transform=ax.transAxes, fontsize=13,
                    bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))
            if i == nr - 1:
                ax.set_xlabel(r"gen $\log_{10}(\rho_g^2)$")
            if j == 0:
                ax.set_ylabel(r"reco $\log_{10}(\rho_g^2)$")
            if i == 0:
                hep.cms.label("", data=False, loc=0, ax=ax, rlabel="NLO")
    cb = fig.colorbar(pcm, ax=axes, pad=0.01, shrink=0.85)
    cb.set_label(r"$P(\rho_{reco}\,|\,\rho_{gen})$  (groomed)")
    out = os.path.join(OUTDIR, "nlo_groomed_response_masscut_rho.png")
    fig.savefig(out, dpi=105); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
