#!/usr/bin/env python
"""Why the mass SF can't diagonalize: purity vs mass-bin width, before/after SF (Pythia).

The closure (mass_sf_closure.py) showed purity/stability barely move under the SF.
This isolates the reason: purity is set by mass RESOLUTION (bin width), not by the
few-percent scale the SF removes. We sweep a uniform mass-bin width and plot the
event-weighted mean purity before (dashed) and after (solid) the reco-indexed SF,
per pT bin. Diagonalization only arrives when bins approach the resolution
(~20-30 GeV), and the SF is at best a few-point add-on there.

Input: ~/Downloads/mass_diagnostic_ntuple_pythia_2018.pkl
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

D = os.path.expanduser("~/Downloads")
OUT = os.path.expanduser("~/Projects/smp_jetmass_run2/review/figs/nlo_deck")
NT = "mass_diagnostic_ntuple_pythia_2018.pkl"
WIDTHS = np.array([2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40])
PT_BINS = [(200, 290, "200-290", "#e42536"),
           (290, 400, "290-400", "#f89c20"),
           (400, 1e9, ">400",     "#3f90da")]
GROOM = {"g": ("Groomed (soft drop)", "msoftdrop", "gen_msoftdrop", 150.0),
         "u": ("Ungroomed",           "mass",      "gen_mass",      200.0)}


def cols():
    o = pickle.load(open(os.path.join(D, NT), "rb"))["reco_jet_ntuple"]
    return {k: np.asarray(o[k].value if hasattr(o[k], "value") else o[k])
            for k in ("pt", "gen_pt", "msoftdrop", "gen_msoftdrop",
                      "mass", "gen_mass", "weight", "passes_both")}


def sf_apply(mreco, mgen):
    grid = np.arange(0.0, 202.0, 2.0); cx = 0.5 * (grid[:-1] + grid[1:])
    r = mreco / mgen; idx = np.digitize(mreco, grid) - 1
    sf = np.full(len(cx), np.nan)
    for b in range(len(cx)):
        m = idx == b
        if m.sum() >= 50:
            sf[b] = np.median(r[m])
    g = np.isfinite(sf)
    return np.interp(mreco, cx[g], sf[g], left=sf[g][0], right=sf[g][-1])


def mean_purity(gen, reco, w, edges, lo, hi):
    nb = len(edges) - 1
    gi = np.digitize(gen, edges) - 1; ri = np.digitize(reco, edges) - 1
    ok = (gi >= 0) & (gi < nb) & (ri >= 0) & (ri < nb)
    M = np.zeros((nb, nb)); np.add.at(M, (gi[ok], ri[ok]), w[ok])
    d = np.diag(M); cs = M.sum(0); cx = 0.5 * (edges[:-1] + edges[1:]); bk = (cx >= lo) & (cx <= hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(cs > 0, d / cs, np.nan)
    return np.nansum((p * cs)[bk]) / np.nansum(cs[bk])


def main():
    c = cols()
    fig, axes = plt.subplots(1, 2, figsize=(21, 10.6))
    for ax, (gkey, (gname, rcol, gcol, hi)) in zip(axes, GROOM.items()):
        for plo, phi, pl, col in PT_BINS:
            s = (c["passes_both"].astype(bool) & (c[gcol] > 0) & (c[rcol] > 0)
                 & (c["pt"] >= plo) & (c["pt"] < phi)
                 & (c["gen_pt"] >= plo) & (c["gen_pt"] < phi))
            gen, reco, w = c[gcol][s], c[rcol][s], c["weight"][s]
            corr = reco / sf_apply(reco, gen)
            p0 = [mean_purity(gen, reco, w, np.arange(0, hi + bw, bw), 10, hi) for bw in WIDTHS]
            p1 = [mean_purity(gen, corr, w, np.arange(0, hi + bw, bw), 10, hi) for bw in WIDTHS]
            ax.plot(WIDTHS, p0, "--", color=col, lw=2.0, alpha=0.7)
            ax.plot(WIDTHS, p1, "-o", color=col, lw=2.4, ms=6,
                    label=rf"$p_T$ {pl} GeV")
        ax.axvline(2, color="gray", ls=":", lw=1.5)
        ax.text(2.4, 0.05, "analysis grid (2 GeV)", rotation=90, va="bottom",
                ha="left", fontsize=14, color="gray")
        ax.axhline(0.5, color="gray", ls=":", lw=1.0)
        ax.set_ylim(0, 1.0); ax.set_xlim(0, 40)
        ax.set_xlabel("mass-bin width [GeV]")
        ax.set_ylabel("mean purity")
        ax.legend(loc="lower right", fontsize=17, frameon=False,
                  title=f"{gname}\n dashed: before SF · solid: after SF",
                  title_fontsize=15)
        hep.cms.label("Simulation", data=False, loc=0, rlabel="(2018, 13 TeV)", ax=ax)
    fig.savefig(os.path.join(OUT, "mass_sf_purity_vs_binwidth.png"),
                dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote mass_sf_purity_vs_binwidth.png")


if __name__ == "__main__":
    main()
