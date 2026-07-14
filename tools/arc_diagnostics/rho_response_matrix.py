#!/usr/bin/env python
"""Rho probability matrix P(rho_reco | rho_gen) per pT bin, groomed + ungroomed (Pythia).

The rho analog of mass_response_matrix.py.  rho = 2*log10(m/(pt*0.8)) (R=0.8), the
actual measured observable.  Because rho ~ log(m), the multiplicative mass scale
(JMS) becomes a constant ADDITIVE shift in rho, and the relative mass resolution
becomes a roughly constant additive width -> the rho response band is more uniform
than the mass one.  Fine rho grid for the visual; purity is also reported on the
analysis 12-bin gen-rho grid for a like-for-like comparison with mass.

Input: ~/Downloads/mass_diagnostic_ntuple_pythia_2018.pkl
       ~/Downloads/minimal_rho_nlo_ptz_2018.pkl  (analysis gen-rho edges)
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

hep.style.use(hep.style.CMS)

D = os.path.expanduser("~/Downloads")
OUT = os.path.expanduser("~/Projects/smp_jetmass_run2/review/figs/nlo_deck")
NT = "mass_diagnostic_ntuple_pythia_2018.pkl"
R = 0.8
RHO_LO, RHO_HI = -7.0, 0.0
FINE = np.arange(RHO_LO, RHO_HI + 0.1, 0.1)
PT_BINS = [(200, 290, "$200 < p_{T} < 290$ GeV", "pt1"),
           (290, 400, "$290 < p_{T} < 400$ GeV", "pt2"),
           (400, 1e9, "$p_{T} > 400$ GeV",       "pt3")]
GROOM = {"g": ("groomed", "msoftdrop", "gen_msoftdrop"),
         "u": ("ungroomed", "mass",     "gen_mass")}


def cols():
    o = pickle.load(open(os.path.join(D, NT), "rb"))["reco_jet_ntuple"]
    return {k: np.asarray(o[k].value if hasattr(o[k], "value") else o[k])
            for k in ("pt", "gen_pt", "msoftdrop", "gen_msoftdrop",
                      "mass", "gen_mass", "weight", "passes_both")}


def rho(m, pt):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(m / (pt * R))


def mean_purity(gen, reco, w, edges):
    nb = len(edges) - 1
    gi = np.digitize(gen, edges) - 1; ri = np.digitize(reco, edges) - 1
    ok = (gi >= 0) & (gi < nb) & (ri >= 0) & (ri < nb)
    M = np.zeros((nb, nb)); np.add.at(M, (gi[ok], ri[ok]), w[ok])
    d = np.diag(M); cs = M.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(cs > 0, d / cs, np.nan)
    return np.nansum(p * cs) / np.nansum(cs)


def main():
    gen_rho_edges = pickle.load(open(os.path.join(D, "minimal_rho_nlo_ptz_2018.pkl"),
                                "rb"))["response_matrix_rho_g"].axes["mpt_gen"].edges
    gen_rho_edges = gen_rho_edges[gen_rho_edges >= -6.0]    # drop -10 underflow
    c = cols()
    print(f"{'grooming':10s} {'pT bin':16s}  mean purity (analysis 12-bin gen-rho grid)")

    for gkey, (gname, rcol, gcol) in GROOM.items():
        for plo, phi, ptlabel, pttag in PT_BINS:
            sel = (c["passes_both"].astype(bool) & (c[gcol] > 0) & (c[rcol] > 0)
                   & (c["pt"] >= plo) & (c["pt"] < phi)
                   & (c["gen_pt"] >= plo) & (c["gen_pt"] < phi))
            rg = rho(c[gcol][sel], c["gen_pt"][sel])
            rr = rho(c[rcol][sel], c["pt"][sel])
            w = c["weight"][sel]
            ok = np.isfinite(rg) & np.isfinite(rr)
            rg, rr, w = rg[ok], rr[ok], w[ok]
            print(f"{gname:10s} {ptlabel:16s}  {mean_purity(rg, rr, w, gen_rho_edges):.3f}")

            nb = len(FINE) - 1
            gi = np.digitize(rg, FINE) - 1; ri = np.digitize(rr, FINE) - 1
            m = (gi >= 0) & (gi < nb) & (ri >= 0) & (ri < nb)
            M = np.zeros((nb, nb))                          # M[reco, gen]
            np.add.at(M, (ri[m], gi[m]), w[m])
            cs = M.sum(0, keepdims=True)
            P = np.divide(M, cs, out=np.zeros_like(M), where=cs > 0)

            fig, ax = plt.subplots(layout="constrained")
            pcm = ax.pcolormesh(FINE, FINE, P, cmap="cividis",
                                norm=mcolors.LogNorm(vmin=1e-3, vmax=1.0))
            ax.plot([RHO_LO, RHO_HI], [RHO_LO, RHO_HI], color="#e42536", lw=1.6, ls="--")
            for e in gen_rho_edges:                          # analysis gen-bin guides
                ax.axvline(e, color="white", lw=0.5, alpha=0.35)
            cb = fig.colorbar(pcm, ax=ax, pad=0.02)
            cb.set_label(r"$P(\rho_{reco}\,|\,\rho_{gen})$")
            ax.set_xlim(RHO_LO, RHO_HI); ax.set_ylim(RHO_LO, RHO_HI)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"gen {gname} $\rho_{{gen}}$")
            ax.set_ylabel(rf"reco {gname} $\rho_{{reco}}$")
            hep.cms.label("", data=False, loc=0, rlabel="(2018, 13 TeV)", ax=ax)
            ax.text(0.05, 0.90, ptlabel + f"\n{gname}", ha="left", va="top",
                    transform=ax.transAxes, fontsize=16,
                    bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))
            out = os.path.join(OUT, f"rho_response_{gkey}_{pttag}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
