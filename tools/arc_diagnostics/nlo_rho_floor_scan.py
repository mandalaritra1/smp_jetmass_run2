#!/usr/bin/env python3
"""Confirm the groomed-mass floor for the rho measurement, and re-derive the adaptive
groomed-rho binning with it folded in (NLO skims).

(1) Scan m_g floor 0..12 GeV: groomed-rho purity / diagonal-fraction / kept-fraction
    per pT bin (analysis coarse gen-rho grid). Marks the chosen 2 GeV.
(2) Adaptive groomed-rho binning (start 0.1, greedy-merge to purity & stability >= 50%)
    with NO cut vs m_g > 2 GeV -- edges, bin count, achieved purity/stability.

  ~/Projects/GluonJetMass/.venv/bin/python tools/arc_diagnostics/nlo_rho_floor_scan.py
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
from nlo_groomed_response_masscut import load, matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figs")
R = 0.8
FLOOR = 2.0
THR = 0.50
PT_BINS = [(200, 290, "200$-$290 GeV", "#e42536"),
           (290, 400, "290$-$400 GeV", "#f89c20"),
           (400, 1e9, "$p_T>$400 GeV", "#3f90da")]
SCAN = np.round(np.arange(0.0, 12.01, 0.5), 2)
COARSE = np.array([-6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.])
FINE = np.round(np.arange(-7.0, 0.0 + 0.1, 0.1), 4)


def rho(m, pt):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(m / (pt * R))


def fine_migration(rg, rr, w):
    nb = len(FINE) - 1
    gi = np.digitize(rg, FINE) - 1; ri = np.digitize(rr, FINE) - 1
    ok = (gi >= 0) & (gi < nb) & (ri >= 0) & (ri < nb)
    M = np.zeros((nb, nb)); np.add.at(M, (gi[ok], ri[ok]), w[ok])
    return M                                                  # M[gen, reco]


def ps(M, a, b):
    diag = M[a:b, a:b].sum(); nreco = M[:, a:b].sum(); ngen = M[a:b, :].sum()
    return (diag / nreco if nreco > 0 else 0.0), (diag / ngen if ngen > 0 else 0.0)


def adaptive_edges(M):
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


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    c = load()
    rg = rho(c["gmg"], c["gpt"]); rr = rho(c["rmg"], c["rpt"])

    # ---------- (1) floor scan ----------
    fig, (axp, axk) = plt.subplots(1, 2, figsize=(21, 10.6), layout="constrained")
    for plo, phi, lab, col in PT_BINS:
        base = ((c["gpt"] >= plo) & (c["gpt"] < phi) & (c["rpt"] >= plo) & (c["rpt"] < phi)
                & np.isfinite(rg) & np.isfinite(rr))
        ntot = c["w"][base].sum()
        pur, dia, kep = [], [], []
        for thr in SCAN:
            sel = base & (c["gmg"] > thr) & (c["rmg"] > thr)
            _, pu, st, dg = matrix(rg[sel], rr[sel], c["w"][sel], COARSE)
            pur.append(pu); dia.append(dg); kep.append(c["w"][sel].sum() / ntot)
        axp.plot(SCAN, pur, color=col, lw=2.6, label=f"{lab}  (purity)")
        axp.plot(SCAN, dia, color=col, lw=2.0, ls="--", alpha=0.8)
        axk.plot(SCAN, kep, color=col, lw=2.6, label=lab)
    for ax in (axp, axk):
        ax.axvline(FLOOR, color="black", lw=1.6, ls=":")
        ax.set_xlabel(r"groomed-mass floor $m_g$ [GeV]"); ax.set_xlim(0, 12)
        hep.cms.label("", data=False, loc=0, ax=ax, rlabel="NLO")
    axp.set_ylabel("groomed-$\\rho$ purity (solid) / diag-frac (dashed)")
    axp.set_ylim(0.4, 0.85); axp.grid(alpha=0.25)
    axp.legend(loc="lower right", fontsize=15, frameon=False)
    axp.text(FLOOR + 0.2, 0.42, "2 GeV", fontsize=14)
    axk.set_ylabel("fraction of jets kept"); axk.set_ylim(0, 1.02); axk.grid(alpha=0.25)
    axk.legend(loc="lower left", fontsize=16, frameon=False)
    fig.savefig(os.path.join(OUTDIR, "nlo_rho_floor_scan.png"), dpi=110)
    plt.close(fig)
    print("wrote nlo_rho_floor_scan.png")

    # ---------- (2) adaptive binning, no-cut vs m_g>2 ----------
    for tag, thr in [("no cut", 0.0), (f"m_g>{FLOOR:g} GeV", FLOOR)]:
        print(f"\n===== groomed-rho adaptive binning  ({tag}, floor {THR:.0%}) =====")
        for plo, phi, lab, _ in PT_BINS:
            sel = ((c["gpt"] >= plo) & (c["gpt"] < phi) & (c["rpt"] >= plo) & (c["rpt"] < phi)
                   & np.isfinite(rg) & np.isfinite(rr) & (c["gmg"] > thr) & (c["rmg"] > thr))
            M = fine_migration(rg[sel], rr[sel], c["w"][sel])
            bounds = adaptive_edges(M); edges = FINE[bounds]
            pp, ss, ww = [], [], []
            for a, b in zip(bounds[:-1], bounds[1:]):
                p, s = ps(M, a, b); pp.append(p); ss.append(s); ww.append(M[:, a:b].sum())
            pp, ss, ww = np.array(pp), np.array(ss), np.array(ww); m = ww > 0
            print(f"  pT {lab:16s}: {len(edges)-1:2d} bins  "
                  f"meanP {np.average(pp[m],weights=ww[m]):.2f} meanS {np.average(ss[m],weights=ww[m]):.2f}")
            print(f"       edges = [{', '.join(f'{e:.1f}' for e in edges)}]")


if __name__ == "__main__":
    main()
