#!/usr/bin/env python3
"""Nested (overlapping-edge) groomed-rho binning across pT bins, m_g > 2 GeV (NLO skims).

Each pT bin may have a different number of rho bins, but ALL edges are drawn from one
common lattice so coarser bins are a strict subset of finer ones -- the bin edges
overlap and the three binnings can be shown together (2D / overlay).

Construction:
  1. master grid = adaptive edges of the FINEST pT bin (>400) on the 0.1 fine grid
     (greedy-merge to purity & stability >= 50%, floor m_g>2).
  2. for every pT bin, build the migration ON THE MASTER grid and greedy-merge ADJACENT
     master bins to the floor -> result is a subset of master edges (nesting guaranteed).

  ~/Projects/GluonJetMass/.venv/bin/python tools/arc_diagnostics/nlo_rho_nested_binning.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlo_groomed_response_masscut import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figs")
R, THR, FLOOR = 0.8, 0.50, 2.0
FINE = np.round(np.arange(-7.0, 0.0 + 0.1, 0.1), 4)
PT_BINS = [(200, 290, "200$-$290 GeV", "#e42536"),
           (290, 400, "290$-$400 GeV", "#f89c20"),
           (400, 1e9, "$p_T>$400 GeV", "#3f90da")]


def rho(m, pt):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(m / (pt * R))


def migration(edges, rg, rr, w):
    nb = len(edges) - 1
    gi = np.digitize(rg, edges) - 1; ri = np.digitize(rr, edges) - 1
    ok = (gi >= 0) & (gi < nb) & (ri >= 0) & (ri < nb)
    M = np.zeros((nb, nb)); np.add.at(M, (gi[ok], ri[ok]), w[ok])
    return M                                                  # M[gen, reco]


def ps(M, a, b):
    diag = M[a:b, a:b].sum(); nreco = M[:, a:b].sum(); ngen = M[a:b, :].sum()
    return (diag / nreco if nreco > 0 else 0.0), (diag / ngen if ngen > 0 else 0.0)


def merge_to_floor(M):
    """Greedy left->right merge of adjacent grid bins to purity & stability >= THR.
    Returns boundary indices into the grid the migration was built on."""
    nb = M.shape[0]; bounds = [0]; start = 0
    while start < nb:
        end = start + 1
        while end <= nb:
            p, s = ps(M, start, end)
            if p >= THR and s >= THR:
                break
            end += 1
        end = min(end, nb); bounds.append(end); start = end
    if len(bounds) > 2:
        p, s = ps(M, bounds[-2], bounds[-1])
        if p < THR or s < THR:
            bounds.pop(-2)
    return bounds


def sel_for(c, rg, rr, plo, phi):
    return ((c["gpt"] >= plo) & (c["gpt"] < phi) & (c["rpt"] >= plo) & (c["rpt"] < phi)
            & np.isfinite(rg) & np.isfinite(rr) & (c["gmg"] > FLOOR) & (c["rmg"] > FLOOR))


def achieved(M, bounds):
    pp, ss, ww = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        p, s = ps(M, a, b); pp.append(p); ss.append(s); ww.append(M[:, a:b].sum())
    return np.array(pp), np.array(ss), np.array(ww)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    c = load()
    rg = rho(c["gmg"], c["gpt"]); rr = rho(c["rmg"], c["rpt"])

    # 1. master grid from the finest pT bin (>400) on the fine 0.1 lattice
    plo, phi = PT_BINS[-1][0], PT_BINS[-1][1]
    s = sel_for(c, rg, rr, plo, phi)
    master = FINE[merge_to_floor(migration(FINE, rg[s], rr[s], c["w"][s]))]
    print(f"master grid ({len(master)-1} bins, from >400): "
          f"[{', '.join(f'{e:.1f}' for e in master)}]\n")

    # 2. each pT bin -> subset of master
    results = {}
    for plo, phi, lab, col in PT_BINS:
        s = sel_for(c, rg, rr, plo, phi)
        M = migration(master, rg[s], rr[s], c["w"][s])
        bounds = merge_to_floor(M)
        edges = master[bounds]
        pp, ss, ww = achieved(M, bounds); m = ww > 0
        results[lab] = (edges, col)
        # verify nesting
        nested = set(np.round(edges, 4)).issubset(set(np.round(master, 4)))
        print(f"pT {lab:16s}: {len(edges)-1:2d} bins  meanP {np.average(pp[m],weights=ww[m]):.2f}"
              f"  meanS {np.average(ss[m],weights=ww[m]):.2f}  nested={nested}")
        print(f"     edges = [{', '.join(f'{e:.1f}' for e in edges)}]")

    # ---- figure: the three binnings drawn together on the common rho axis ----
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    for y, (plo, phi, lab, col) in enumerate(PT_BINS):
        edges, _ = results[lab]
        ax.hlines(y, edges[0], edges[-1], color=col, lw=3, alpha=0.5)
        ax.vlines(edges, y - 0.32, y + 0.32, color=col, lw=2.4)
        ax.text(0.02, y + 0.40, f"{lab}  ({len(edges)-1} bins)", color=col, fontsize=15,
                ha="left", va="bottom")
    # guide lines at every master edge to show alignment
    for e in master:
        ax.axvline(e, color="0.7", lw=0.6, ls=":", zorder=0)
    ax.set_yticks([]); ax.set_ylim(-0.7, len(PT_BINS) - 0.2)
    ax.set_xlim(master[1] - 0.3, 0.05)          # start near the floor, not the -7 catch-all
    ax.set_xlabel(r"groomed $\log_{10}(\rho_g^2)$")
    hep.cms.label("", data=False, loc=0, ax=ax, rlabel="NLO  ($m_g>2$ GeV)")
    fig.savefig(os.path.join(OUTDIR, "nlo_rho_nested_binning.png"), dpi=120)
    plt.close(fig)
    print("\nwrote nlo_rho_nested_binning.png")


if __name__ == "__main__":
    main()
