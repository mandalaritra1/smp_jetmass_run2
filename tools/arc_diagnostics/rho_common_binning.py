#!/usr/bin/env python
"""Common rho binning across pT bins for 2D (pT x rho) unfolding.

For a rectangular 2D grid the rho edges must be shared by every pT bin. We greedily
accumulate the fine (0.1) rho bins and close a merged bin only when BOTH purity and
stability reach >= 50% in ALL THREE pT bins simultaneously -> the merge is driven by
the worst (lowest-pT, coarsest-resolution) bin, giving one common binning valid
everywhere. The extreme low-rho and high-rho ends are the tail bins, free to be
merged/trimmed further. rho = 2*log10(m/(pt*0.8)).

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
FINE = np.round(np.arange(-6.0, 0.0 + 0.1, 0.1), 4)
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
    return M


def ps(M, a, b):
    diag = M[a:b, a:b].sum()
    nreco = M[:, a:b].sum(); ngen = M[a:b, :].sum()
    return (diag / nreco if nreco > 0 else 0.0,
            diag / ngen if ngen > 0 else 0.0)


def common_edges(Ms):
    """Greedy: close a bin when min over pT bins of (purity, stability) >= THR."""
    nb = Ms[0].shape[0]
    bounds = [0]; start = 0
    while start < nb:
        end = start + 1
        while end <= nb:
            vals = [ps(M, start, end) for M in Ms]
            if all(p >= THR and s >= THR for p, s in vals):
                break
            end += 1
        end = min(end, nb)
        bounds.append(end); start = end
    if len(bounds) > 2:                                  # merge sub-threshold tail back
        vals = [ps(M, bounds[-2], bounds[-1]) for M in Ms]
        if any(p < THR or s < THR for p, s in vals):
            bounds.pop(-2)
    return bounds


def main():
    c = cols()
    fig, axes = plt.subplots(1, 2, figsize=(21, 10.6))
    for ax, (gkey, (gname, rcol, gcol)) in zip(axes, GROOM.items()):
        Ms, masks = [], []
        for plo, phi, *_ in PT_BINS:
            sel = (c["passes_both"].astype(bool) & (c[gcol] > 0) & (c[rcol] > 0)
                   & (c["pt"] >= plo) & (c["pt"] < phi)
                   & (c["gen_pt"] >= plo) & (c["gen_pt"] < phi))
            rg = rho(c[gcol][sel], c["gen_pt"][sel]); rr = rho(c[rcol][sel], c["pt"][sel])
            w = c["weight"][sel]; ok = np.isfinite(rg) & np.isfinite(rr)
            Ms.append(fine_migration(rg[ok], rr[ok], w[ok]))
        bounds = common_edges(Ms)
        edges = FINE[bounds]; nb = len(edges) - 1
        print(f"\n===== {gname} rho — COMMON binning ({nb} bins, floor {THR:.0%} in every pT) =====")
        print(f"  edges = [{', '.join(f'{e:.1f}' for e in edges)}]")
        print(f"  widths = {np.round(np.diff(edges),1).tolist()}")
        for (plo, phi, pl, col), M in zip(PT_BINS, Ms):
            pp, ss, ww = [], [], []
            for a, b in zip(bounds[:-1], bounds[1:]):
                p, s = ps(M, a, b); pp.append(p); ss.append(s); ww.append(M[:, a:b].sum())
            pp, ss, ww = np.array(pp), np.array(ss), np.array(ww); m = ww > 0
            print(f"    pT {pl:7s}: mean purity {np.average(pp[m],weights=ww[m]):.2f}"
                  f"  stability {np.average(ss[m],weights=ww[m]):.2f}"
                  f"  min purity {pp[m].min():.2f}")
            ax.step(edges, np.append(pp, pp[-1]), where="post", color=col, lw=2.4,
                    label=f"$p_T$ {pl} GeV")
        for e in edges:
            ax.axvline(e, color="gray", lw=0.5, alpha=0.4)
        ax.axhline(THR, color="gray", ls=":", lw=1.5)
        ax.set_ylim(0, 1.0); ax.set_xlim(-6, 0)
        ax.set_xlabel(rf"{gname} $\rho$"); ax.set_ylabel("purity (common bin)")
        ax.legend(loc="lower right", fontsize=16, frameon=False,
                  title=f"{gname} — common ({nb} bins)", title_fontsize=15)
        hep.cms.label("", data=False, loc=0, rlabel="(2018, 13 TeV)", ax=ax)
    fig.savefig(os.path.join(OUT, "rho_common_binning.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("\nwrote rho_common_binning.png")


if __name__ == "__main__":
    main()
