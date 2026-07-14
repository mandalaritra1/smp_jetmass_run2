#!/usr/bin/env python3
"""Validation of the NLO->Herwig gen reweighting used for the model uncertainty.

Per pT bin (groomed), shows:
  row 1  normalized gen densities: NLO (nominal), Herwig (target), NLO x rw (reweighted)
  row 2  closure ratio  (NLO x rw) / Herwig   -- should sit on 1
  row 3  the reweight function rw(rho) = Herwig_density / NLO_density (what each event gets)

The reweight is the continuous (interpolated) Herwig/NLO density ratio, applied per event
by gen rho; only the region NLO actually populates (above the m_g>2 floor) is comparable,
so densities are area-normalized over the populated bins.

  ~/Projects/GluonJetMass/.venv/bin/python tools/arc_diagnostics/nlo_reweight_validation.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep

hep.style.use(hep.style.CMS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlo_groomed_response_masscut import load            # no-ROOT loader

SP = ("/private/tmp/claude-501/-Users-aritra-Projects-smp-jetmass-run2/"
      "5475621c-e50a-4ac9-9c83-0b954dcd7fe5/scratchpad/herwig_gen_shapes.npz")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
R, FLOOR = 0.8, 2.0
PT = [(200, 290, "200$-$290 GeV"), (290, 400, "290$-$400 GeV"), (400, 1e9, "$p_T>$400 GeV")]


def rho(m, pt):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(m / (pt * R))


def reweight_fn(rg, w, edges, hw_counts, niter=6):
    """Continuous NLO->Herwig gen reweight, ITERATED to closure.

    A single Herwig/NLO density ratio (interpolated) doesn't close: where the ratio
    varies fast and NLO is sparse, re-histogramming NLO*rw != Herwig. So we iterate --
    apply the current reweight, measure the residual Herwig/current ratio, fold it into
    the node values, repeat. Converges in a few passes; the result is a continuous rw(rho)
    whose re-histogram matches Herwig at the node (0.5) resolution."""
    bw = np.diff(edges); cx = 0.5 * (edges[:-1] + edges[1:])
    nlo0, _ = np.histogram(rg, bins=edges, weights=w)
    # anchor only on WELL-populated bins; the sharply-cut floor edge (NLO~0, Herwig smooth)
    # is not a fair reweight target -> exclude it and flat-extrapolate the reweight there.
    pop = nlo0 > 0.02 * nlo0.max()
    hd = np.where(pop, hw_counts, 0.0); hd = hd / max(hd.sum(), 1e-30) / bw   # target, on pop
    rw_nodes = np.ones_like(cx)

    def make(nodes):
        return lambda x: np.interp(x, cx[pop], nodes[pop],
                                   left=nodes[pop][0], right=nodes[pop][-1])
    for _ in range(niter):
        cur, _ = np.histogram(rg, bins=edges, weights=w * make(rw_nodes)(rg))
        cd = np.where(pop, cur, 0.0); cd = cd / max(cd.sum(), 1e-30) / bw
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where((cd > 0) & pop, hd / cd, 1.0)
        rw_nodes = np.clip(np.where(pop, rw_nodes * corr, rw_nodes), 0.2, 5.0)
    return make(rw_nodes), cx, np.where(pop, rw_nodes, np.nan)


def density(counts, edges, mask):
    bw = np.diff(edges); area = np.sum(counts[mask] * bw[mask])
    return np.where(mask, counts / max(area, 1e-30), np.nan)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    z = np.load(SP); c = load()
    edges = z["edges_g"]; cx = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=(31, 16), layout="constrained")
    gs = gridspec.GridSpec(3, 3, height_ratios=[3, 1.1, 1.4], figure=fig)
    for ip, (plo, phi, lab) in enumerate(PT):
        sel = ((c["gpt"] >= plo) & (c["gpt"] < phi) & (c["gmg"] > FLOOR))
        rg = rho(c["gmg"][sel], c["gpt"][sel]); w = c["w"][sel]
        ok = np.isfinite(rg) & (rg >= edges[0]) & (rg <= edges[-1])
        rg, w = rg[ok], w[ok]
        hw = z[f"hw_g_pt{ip+1}"]

        rwf, rcx, rratio = reweight_fn(rg, w, edges, hw)
        wr = w * rwf(rg)
        nlo_c, _ = np.histogram(rg, bins=edges, weights=w)
        rew_c, _ = np.histogram(rg, bins=edges, weights=wr)
        pop = nlo_c > 0.02 * nlo_c.max()                # well-populated bins (reweight anchors)

        nd = density(nlo_c, edges, pop)
        hd = density(hw, edges, pop)
        rd = density(rew_c, edges, pop)

        ax = fig.add_subplot(gs[0, ip]); rax = fig.add_subplot(gs[1, ip], sharex=ax)
        wax = fig.add_subplot(gs[2, ip], sharex=ax)
        ax.step(cx, nd, where="mid", color="black", lw=2.6, label="NLO (nominal)")
        ax.step(cx, hd, where="mid", color="#e42536", lw=2.6, label="Herwig (target)")
        ax.step(cx, rd, where="mid", color="#3f90da", lw=2.2, ls="--", label=r"NLO $\times\,$rw")
        ax.set_ylabel("normalized density"); ax.grid(alpha=0.25)
        ax.set_ylim(0, np.nanmax([nd, hd, rd]) * 1.4)
        hep.cms.label("", data=False, loc=0, ax=ax, rlabel=lab)
        ax.legend(loc="upper left", fontsize=14, frameon=False,
                  title=f"groomed gen  ({lab})")
        plt.setp(ax.get_xticklabels(), visible=False)

        with np.errstate(divide="ignore", invalid="ignore"):
            clos = np.where((hd > 0) & pop, rd / hd, np.nan)
        rax.step(cx, clos, where="mid", color="#3f90da", lw=2.4)
        rax.axhline(1.0, color="#e42536", lw=1.4)
        rax.set_ylim(0.9, 1.1); rax.set_ylabel(r"$\frac{\mathrm{NLO\times rw}}{\mathrm{Herwig}}$")
        rax.grid(alpha=0.25); plt.setp(rax.get_xticklabels(), visible=False)
        mc = np.nanmax(np.abs(clos[np.isfinite(clos)] - 1)) * 100
        rax.text(0.04, 0.85, f"max non-closure {mc:.1f}%", transform=rax.transAxes, fontsize=13)

        xfine = np.linspace(edges[0], 0, 400)
        wax.plot(xfine, rwf(xfine), color="#7a21dd", lw=2.4)
        wax.plot(rcx[pop], rratio[pop], "o", color="#7a21dd", ms=7)
        wax.axhline(1.0, color="gray", ls=":", lw=1.2)
        wax.set_ylabel(r"rw$(\rho)$ = Hw/NLO"); wax.set_xlabel(r"gen $\log_{10}(\rho_g^2)$")
        wax.grid(alpha=0.25); wax.set_xlim(edges[0] + 0.5, 0); wax.set_ylim(0, 2.2)

    out = os.path.join(OUTDIR, "nlo_reweight_validation_g.png")
    fig.savefig(out, dpi=100); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
