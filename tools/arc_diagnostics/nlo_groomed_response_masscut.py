#!/usr/bin/env python3
"""Groomed-mass response P(m_reco | m_gen) vs a groomed-mass floor (NLO skims).

The mass_response_matrix.py analog, rebuilt on the new per-event NLO ptZ skims (the
old Downloads ntuple is gone). For each jet-pT bin we show the groomed soft-drop mass
response and how a groomed-mass veto m_g > {0,2,5,10} GeV cleans up the smeared
low-mass corner. Each gen column is normalized to unit sum, so colour = P(m_reco|m_gen);
the dashed line is y=x. pT is held on its diagonal block (gen & reco pT in the same
analysis bin) so this is the pure mass sub-response.

The cut is applied to BOTH gen and reco groomed mass (a clean fiducial+reco region).
Mean purity / stability on the analysis coarse gen-mass grid are printed for every cut.

  ~/Projects/GluonJetMass/.venv/bin/python tools/arc_diagnostics/nlo_groomed_response_masscut.py
"""
from __future__ import annotations
import os, glob, re
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

hep.style.use(hep.style.CMS)

SKIM = "/Users/aritra/Projects/unfold/inputs/zjet/nlo_skims/nlo_ptz_skims"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figs")
XS = {"100To250": 97.2, "250To400": 3.701, "400To650": 0.5086, "650ToInf": 0.04728}
PT_BINS = [(200, 290, "$200<p_T<290$"), (290, 400, "$290<p_T<400$"), (400, 1e9, "$p_T>400$")]
CUTS_TABLE = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0]          # printed purity/stability
CUTS_PLOT = [0.0, 2.0, 5.0, 10.0]                     # columns shown
MTOP = 150.0
FINE = np.arange(0.0, MTOP + 1e-6, 2.5)              # visual grid
COARSE = np.array([0, 10, 20, 30, 50, 70, 90, 110, 130, 150.])   # analysis-style gen grid


def load():
    """Pool NLO skims -> dict of reco/gen groomed mass, pt, weight. Schema-resilient."""
    R = {k: [] for k in ("rpt", "rmg", "rmu", "gpt", "gmg", "gmu", "w")}
    for f in sorted(glob.glob(SKIM + "/*/merged.parquet")):
        ds = os.path.basename(os.path.dirname(f))
        xs = XS[re.search(r"ptz_(\d+To\w+?)_UL", ds).group(1)]
        try:
            a = ak.from_parquet(f)
        except Exception as e:
            print(f"  !! SKIP {ds}: unreadable ({type(e).__name__})"); continue
        F = a.fields
        if "passes_both" in F:
            pb = ak.to_numpy(a.passes_both); gw = ak.to_numpy(a.weight)
            rpt = a.pt; rmg = a.msoftdrop
        elif "matched" in F:
            print(f"  ** {ds}: legacy schema (mm-only)")
            pb = ak.to_numpy(a.matched); gw = ak.to_numpy(a.genWeight)
            rpt = a.reco_pt; rmg = a.reco_msoftdrop
        else:
            print(f"  !! SKIP {ds}: unknown schema"); continue
        rmu = a.reco_mass if "reco_mass" in F else a.mass
        rpt = ak.to_numpy(ak.fill_none(rpt, np.nan)); rmg = ak.to_numpy(ak.fill_none(rmg, np.nan))
        rmu = ak.to_numpy(ak.fill_none(rmu, np.nan))
        gpt = ak.to_numpy(ak.fill_none(a.gen_pt, np.nan)); gmg = ak.to_numpy(ak.fill_none(a.gen_msoftdrop, np.nan))
        gmu = ak.to_numpy(ak.fill_none(a.gen_mass, np.nan))
        sel = pb & np.isfinite(gpt) & np.isfinite(rpt)
        norm = xs / np.sum(np.abs(gw[sel]))
        R["rpt"].append(rpt[sel]); R["rmg"].append(rmg[sel]); R["rmu"].append(rmu[sel])
        R["gpt"].append(gpt[sel]); R["gmg"].append(gmg[sel]); R["gmu"].append(gmu[sel])
        R["w"].append(np.sign(gw[sel]) * np.abs(gw[sel]) * norm)
    return {k: np.concatenate(v) for k, v in R.items()}


def matrix(gen, reco, w, edges):
    """M[reco, gen]; column-normalized P, bin-averaged purity & stability, diag frac.
    purity_i  = d_i/reco_marginal_i  = P(gen==i | reco==i)
    stability_i = d_i/gen_marginal_i = P(reco==i | gen==i)
    (bin-averaged over populated bins; diag = global same-bin fraction)."""
    nb = len(edges) - 1
    gi = np.digitize(gen, edges) - 1; ri = np.digitize(reco, edges) - 1
    ok = (gi >= 0) & (gi < nb) & (ri >= 0) & (ri < nb)
    M = np.zeros((nb, nb)); np.add.at(M, (ri[ok], gi[ok]), w[ok])
    d = np.diag(M)
    cs = M.sum(0); rs = M.sum(1)                      # gen marginal, reco marginal
    with np.errstate(divide="ignore", invalid="ignore"):
        pur_i = np.where(rs > 0, d / rs, np.nan)
        sta_i = np.where(cs > 0, d / cs, np.nan)
        P = np.divide(M, cs[None, :], out=np.zeros_like(M), where=cs[None, :] > 0)
    diag = d.sum() / M.sum() if M.sum() > 0 else np.nan
    return P, np.nanmean(pur_i), np.nanmean(sta_i), diag


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    c = load()
    print(f"pooled matched gen jets: {len(c['gpt']):,d}\n")

    # ---- purity / stability table (groomed, analysis coarse grid) ----
    print(f"{'pT bin':14s} {'cut':>10s}  {'purity':>7s} {'stab':>6s} {'diag':>6s} {'kept':>6s}")
    for plo, phi, lab in PT_BINS:
        base = ((c["gpt"] >= plo) & (c["gpt"] < phi) & (c["rpt"] >= plo) & (c["rpt"] < phi)
                & (c["gmg"] > 0) & (c["rmg"] > 0))
        ntot = c["w"][base].sum()
        for thr in CUTS_TABLE:
            sel = base & (c["gmg"] > thr) & (c["rmg"] > thr)
            _, pu, st, dg = matrix(c["gmg"][sel], c["rmg"][sel], c["w"][sel], COARSE)
            print(f"{lab:14s} {('m_g>'+format(thr,'g')):>10s}  {pu:7.3f} {st:6.3f} {dg:6.3f} "
                  f"{c['w'][sel].sum()/ntot:6.1%}")
        print()

    # ---- response-matrix grid: pT (rows) x cut (cols) ----
    nr, nc = len(PT_BINS), len(CUTS_PLOT)
    fig, axes = plt.subplots(nr, nc, figsize=(6.2 * nc, 6.0 * nr), layout="constrained")
    pcm = None
    for i, (plo, phi, lab) in enumerate(PT_BINS):
        base = ((c["gpt"] >= plo) & (c["gpt"] < phi) & (c["rpt"] >= plo) & (c["rpt"] < phi)
                & (c["gmg"] > 0) & (c["rmg"] > 0))
        for j, thr in enumerate(CUTS_PLOT):
            ax = axes[i, j]
            sel = base & (c["gmg"] > thr) & (c["rmg"] > thr)
            P, pu, st, dg = matrix(c["gmg"][sel], c["rmg"][sel], c["w"][sel], FINE)
            pcm = ax.pcolormesh(FINE, FINE, P, cmap="cividis",
                                norm=mcolors.LogNorm(vmin=1e-3, vmax=1.0))
            ax.plot([0, MTOP], [0, MTOP], color="#e42536", lw=1.4, ls="--")
            if thr > 0:
                ax.axvline(thr, color="white", lw=1.0, alpha=0.6)
                ax.axhline(thr, color="white", lw=1.0, alpha=0.6)
            ax.set_xlim(0, MTOP); ax.set_ylim(0, MTOP); ax.set_aspect("equal")
            cuttxt = "no cut" if thr == 0 else f"$m_g>{thr:g}$ GeV"
            ax.text(0.05, 0.95, f"{lab}\n{cuttxt}\npurity {pu:.2f}", ha="left", va="top",
                    transform=ax.transAxes, fontsize=13,
                    bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))
            if i == nr - 1:
                ax.set_xlabel(r"gen $m_{gen}$ [GeV]")
            if j == 0:
                ax.set_ylabel(r"reco $m_{reco}$ [GeV]")
            if i == 0:
                hep.cms.label("", data=False, loc=0, ax=ax, rlabel="NLO")
    cb = fig.colorbar(pcm, ax=axes, pad=0.01, shrink=0.85)
    cb.set_label(r"$P(m_{reco}\,|\,m_{gen})$  (groomed)")
    out = os.path.join(OUTDIR, "nlo_groomed_response_masscut.png")
    fig.savefig(out, dpi=105); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
