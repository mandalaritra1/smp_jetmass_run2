#!/usr/bin/env python
"""Adaptive rho binning: start at width 0.1, merge adjacent bins until purity & stability >= 50%.

Recipe (per pT bin, groomed & ungroomed, Pythia 2018):
  1. fine rho grid, width 0.1
  2. build the migration M[gen, reco] on that grid
  3. greedily accumulate adjacent fine bins left->right; close a merged bin once BOTH
     purity = N(gen&reco in bin)/N(reco in bin)  and
     stability = N(gen&reco in bin)/N(gen in bin)  reach >= 0.5
  4. a trailing sub-threshold group is merged back into the previous bin
Reports the resulting edges, bin count, and achieved purity/stability, vs the
current 12-bin analysis grid. rho = 2*log10(m/(pt*0.8)).

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
R = 0.8
THR = 0.50
FINE = np.round(np.arange(-7.0, 0.0 + 0.1, 0.1), 4)
PT_BINS = [(200, 290, "200-290", "#e42536"),
           (290, 400, "290-400", "#f89c20"),
           (400, 1e9, ">400",     "#3f90da")]
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


def fine_migration(rg, rr, w):
    nb = len(FINE) - 1
    gi = np.digitize(rg, FINE) - 1; ri = np.digitize(rr, FINE) - 1
    ok = (gi >= 0) & (gi < nb) & (ri >= 0) & (ri < nb)
    M = np.zeros((nb, nb)); np.add.at(M, (gi[ok], ri[ok]), w[ok])
    return M                                                 # M[gen, reco]


def ps(M, a, b):
    diag = M[a:b, a:b].sum()
    nreco = M[:, a:b].sum(); ngen = M[a:b, :].sum()
    p = diag / nreco if nreco > 0 else 0.0
    s = diag / ngen if ngen > 0 else 0.0
    return p, s


def adaptive_edges(M):
    """Greedy left->right accumulation to the purity/stability floor."""
    nb = M.shape[0]
    bounds = [0]
    start = 0
    while start < nb:
        end = start + 1
        while end <= nb:
            p, s = ps(M, start, end)
            if p >= THR and s >= THR:
                break
            end += 1
        end = min(end, nb)
        bounds.append(end)
        start = end
    # merge a trailing sub-threshold group into the previous bin
    if len(bounds) > 2:
        p, s = ps(M, bounds[-2], bounds[-1])
        if p < THR or s < THR:
            bounds.pop(-2)
    return bounds


def main():
    gen12 = pickle.load(open(os.path.join(D, "minimal_rho_nlo_ptz_2018.pkl"), "rb"))[
        "response_matrix_rho_g"].axes["mpt_gen"].edges
    c = cols()
    fig, axes = plt.subplots(1, 2, figsize=(21, 10.6))

    for ax, (gkey, (gname, rcol, gcol)) in zip(axes, GROOM.items()):
        print(f"\n===== {gname} rho  (start 0.1, floor {THR:.0%}) =====")
        for plo, phi, pl, col in PT_BINS:
            sel = (c["passes_both"].astype(bool) & (c[gcol] > 0) & (c[rcol] > 0)
                   & (c["pt"] >= plo) & (c["pt"] < phi)
                   & (c["gen_pt"] >= plo) & (c["gen_pt"] < phi))
            rg = rho(c[gcol][sel], c["gen_pt"][sel])
            rr = rho(c[rcol][sel], c["pt"][sel])
            w = c["weight"][sel]
            ok = np.isfinite(rg) & np.isfinite(rr)
            M = fine_migration(rg[ok], rr[ok], w[ok])
            bounds = adaptive_edges(M)
            edges = FINE[bounds]
            nb = len(edges) - 1
            # achieved purity/stability per merged bin (over populated range)
            pp, ss, ww = [], [], []
            for a, b in zip(bounds[:-1], bounds[1:]):
                p, s = ps(M, a, b); pp.append(p); ss.append(s); ww.append(M[:, a:b].sum())
            pp, ss, ww = np.array(pp), np.array(ss), np.array(ww)
            mask = ww > 0
            print(f"  pT {pl:7s}: {nb:2d} bins   edges = "
                  f"[{', '.join(f'{e:.1f}' for e in edges)}]")
            print(f"              mean purity {np.average(pp[mask],weights=ww[mask]):.2f}"
                  f"  stability {np.average(ss[mask],weights=ww[mask]):.2f}"
                  f"  (widths {np.round(np.diff(edges),1).tolist()})")
            # step plot of the achieved purity on the merged grid
            ax.step(edges, np.append(pp, pp[-1]), where="post", color=col, lw=2.4,
                    label=f"$p_T$ {pl} GeV  ({nb} bins)")
        ax.axhline(THR, color="gray", ls=":", lw=1.5)
        ax.set_ylim(0, 1.0); ax.set_xlim(-6, 0)
        ax.set_xlabel(rf"{gname} $\rho$")
        ax.set_ylabel("purity (merged bin)")
        ax.legend(loc="lower right", fontsize=16, frameon=False,
                  title=f"{gname} — adaptive (floor {THR:.0%})", title_fontsize=15)
        hep.cms.label("", data=False, loc=0, rlabel="(2018, 13 TeV)", ax=ax)
    fig.savefig(os.path.join(OUT, "rho_adaptive_binning.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\ncurrent analysis gen-rho grid: {len(gen12)-1} bins "
          f"[{', '.join(f'{e:.1f}' for e in gen12)}]")
    print("wrote rho_adaptive_binning.png")


if __name__ == "__main__":
    main()
