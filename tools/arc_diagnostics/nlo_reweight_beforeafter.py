#!/usr/bin/env python3
"""Before/after of the NLO->Herwig gen reweight: single-pass (naive) vs iterated.

For each observable (groomed, ungroomed) and pT bin, plots the closure ratio
(NLO x rw) / Herwig for two reweights:
  BEFORE = one-pass Herwig/NLO density ratio (interpolated) -- fails at the sparse
           floor edge / fast-varying regions (tens-of-% non-closure),
  AFTER  = the same ratio ITERATED to convergence and anchored only on well-populated
           bins -> closes to <1%.

  ~/Projects/GluonJetMass/.venv/bin/python tools/arc_diagnostics/nlo_reweight_beforeafter.py
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

SP = ("/private/tmp/claude-501/-Users-aritra-Projects-smp-jetmass-run2/"
      "5475621c-e50a-4ac9-9c83-0b954dcd7fe5/scratchpad/herwig_gen_shapes.npz")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
R, FLOOR = 0.8, 2.0
PT = [(200, 290, "200$-$290 GeV"), (290, 400, "290$-$400 GeV"), (400, 1e9, "$p_T>$400 GeV")]
OBS = {"g": ("groomed", "gmu_unused"), "u": ("ungroomed", "")}


def rho(m, pt):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(m / (pt * R))


def single_pass(rg, w, edges, hw):
    """Naive one-pass Herwig/NLO density ratio, interpolated."""
    nlo, _ = np.histogram(rg, bins=edges, weights=w); bw = np.diff(edges)
    nd = nlo / max(nlo.sum(), 1e-30) / bw
    hd = hw / max(hw.sum(), 1e-30) / bw
    ratio = np.where(nd > 0, hd / nd, np.nan)
    cx = 0.5 * (edges[:-1] + edges[1:]); g = np.isfinite(ratio)
    return lambda x: np.interp(x, cx[g], ratio[g], left=ratio[g][0], right=ratio[g][-1])


def iterated(rg, w, edges, hw, niter=6):
    bw = np.diff(edges); cx = 0.5 * (edges[:-1] + edges[1:])
    nlo0, _ = np.histogram(rg, bins=edges, weights=w)
    pop = nlo0 > 0.02 * nlo0.max()
    hd = np.where(pop, hw, 0.0); hd = hd / max(hd.sum(), 1e-30) / bw
    rw = np.ones_like(cx)
    mk = lambda n: (lambda x: np.interp(x, cx[pop], n[pop], left=n[pop][0], right=n[pop][-1]))
    for _ in range(niter):
        cur, _ = np.histogram(rg, bins=edges, weights=w * mk(rw)(rg))
        cd = np.where(pop, cur, 0.0); cd = cd / max(cd.sum(), 1e-30) / bw
        with np.errstate(divide="ignore", invalid="ignore"):
            rw = np.clip(np.where((cd > 0) & pop, rw * hd / cd, rw), 0.2, 5.0)
    return mk(rw), pop


def closure(rg, w, edges, hw, rwf, pop):
    bw = np.diff(edges)
    rew, _ = np.histogram(rg, bins=edges, weights=w * rwf(rg))
    rd = np.where(pop, rew, 0.0); rd = rd / max((rd * bw).sum(), 1e-30)
    hd = np.where(pop, hw, 0.0); hd = hd / max((hd * bw).sum(), 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((hd > 0) & pop, rd / hd, np.nan)


def main():
    z = np.load(SP); c = load()
    for obs in ("g", "u"):
        gm = c["gmg"] if obs == "g" else c["gmu"]
        edges = z[f"edges_{obs}"]; cx = 0.5 * (edges[:-1] + edges[1:])
        fig, axes = plt.subplots(2, 3, figsize=(31, 14), layout="constrained", sharex=True)
        for ip, (plo, phi, lab) in enumerate(PT):
            sel = (c["gpt"] >= plo) & (c["gpt"] < phi) & (c["gmg"] > FLOOR)
            rg = rho(gm[sel], c["gpt"][sel]); w = c["w"][sel]
            ok = np.isfinite(rg) & (rg >= edges[0]) & (rg <= edges[-1]); rg, w = rg[ok], w[ok]
            hw = z[f"hw_{obs}_pt{ip+1}"]
            nlo0, _ = np.histogram(rg, bins=edges, weights=w); pop = nlo0 > 0.02 * nlo0.max()

            rwf_b = single_pass(rg, w, edges, hw)
            rwf_a, pop_a = iterated(rg, w, edges, hw)
            cb = closure(rg, w, edges, hw, rwf_b, pop)
            ca = closure(rg, w, edges, hw, rwf_a, pop_a)
            for row, (cl, ttl, col) in enumerate([(cb, "BEFORE (single pass)", "#e42536"),
                                                  (ca, "AFTER (iterated)", "#3f90da")]):
                ax = axes[row, ip]
                ax.step(cx, cl, where="mid", color=col, lw=2.6)
                ax.axhline(1.0, color="black", lw=1.2)
                ax.set_ylim(0.4, 1.6); ax.set_xlim(edges[0] + 0.5, 0); ax.grid(alpha=0.25)
                mc = np.nanmax(np.abs(cl[np.isfinite(cl)] - 1)) * 100
                ax.text(0.04, 0.9, f"{ttl}\nmax non-closure {mc:.1f}%",
                        transform=ax.transAxes, va="top", fontsize=15)
                if row == 0:
                    hep.cms.label("", data=False, loc=0, ax=ax,
                                  rlabel=f"{OBS[obs][0]}  {lab}")
                if ip == 0:
                    ax.set_ylabel(r"(NLO$\times$rw) / Herwig")
                if row == 1:
                    ax.set_xlabel(rf"gen $\log_{{10}}(\rho_{obs}^2)$")
        out = os.path.join(OUTDIR, f"nlo_reweight_beforeafter_{obs}.png")
        fig.savefig(out, dpi=60); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
