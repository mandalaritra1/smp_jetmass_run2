#!/usr/bin/env python3
"""Effect of a groomed-mass floor on the NLO rho = 2*log10(m/(pt*R)) distribution.

Gen/particle level, NLO DY (the per-event ptZ skims we produced). For each jet-pT
bin we plot the nominal rho distribution and the same distribution after vetoing
jets with groomed (soft-drop) mass below m_g = {0.1, 1, 2, 5, 10} GeV. Because the
cut is a fixed mass while rho = 2*log10(m_g/(pt*0.8)), a mass floor maps to a
pt-dependent rho floor -- applied here event-by-event (exact), not as a single line.

Two observables:
  rho_g : groomed   -- the cut acts directly on its own axis
  rho_u : ungroomed -- shows how a groomed-mass veto would sculpt the ungroomed spectrum

xs-stitch: |genWeight| is ~constant per amcatnlo sample, so each ptZ slice is
weighted sign(gw) * xs_slice / sum|gw|_slice (eras pooled); shapes are area-normalized
per pT bin so the absolute lumi/Sigma_gen normalization cancels.

  ~/Projects/GluonJetMass/.venv/bin/python tools/arc_diagnostics/nlo_masscut_rho.py
"""
from __future__ import annotations
import os, glob, re
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep

hep.style.use(hep.style.CMS)

SKIM = "/Users/aritra/Projects/unfold/inputs/zjet/nlo_skims/nlo_ptz_skims"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figs")
R = 0.8

XS = {"100To250": 97.2, "250To400": 3.701, "400To650": 0.5086, "650ToInf": 0.04728}  # pb (XSDB)
PT = [(200., 290., "200$-$290 GeV"), (290., 400., "290$-$400 GeV"), (400., np.inf, "$p_T$ > 400 GeV")]
CUTS = [0.1, 1.0, 2.0, 5.0, 10.0]
EDGES = np.arange(-8.0, 0.0001, 0.2)
CTR = 0.5 * (EDGES[:-1] + EDGES[1:])
XLAB = {"g": r"$\log_{10}(\rho_g^2)$", "u": r"$\log_{10}(\rho_u^2)$"}
TITLE = {"g": "groomed (soft-drop)", "u": "ungroomed"}
CMAP = plt.cm.viridis(np.linspace(0.15, 0.9, len(CUTS)))


def load():
    """Pool all skims -> (gen_pt, gen_mass_u, gen_mass_g, weight). Gen-matched, gen_pt>200.
    Resilient to (a) corrupt/truncated parquet (skipped) and (b) the older skimmer
    schema (reco_*/genWeight/matched) which is name-mapped."""
    gpt, gmu, gmg, w = [], [], [], []
    for f in sorted(glob.glob(SKIM + "/*/merged.parquet")):
        ds = os.path.basename(os.path.dirname(f))
        m = re.search(r"ptz_(\d+To\w+?)_UL", ds)
        xs = XS[m.group(1)]
        try:
            a = ak.from_parquet(f)
        except Exception as e:
            print(f"  !! SKIP {ds}: unreadable ({type(e).__name__})")
            continue
        if "passes_both" in a.fields:                       # current schema
            pb = ak.to_numpy(a.passes_both); gw = ak.to_numpy(a.weight)
        elif "matched" in a.fields:                         # legacy schema (mm-only)
            print(f"  ** {ds}: legacy schema (mm-only) -- name-mapped")
            pb = ak.to_numpy(a.matched); gw = ak.to_numpy(a.genWeight)
        else:
            print(f"  !! SKIP {ds}: unknown schema {a.fields}")
            continue
        pt = ak.to_numpy(ak.fill_none(a.gen_pt, np.nan))
        mu = ak.to_numpy(ak.fill_none(a.gen_mass, np.nan))
        mg = ak.to_numpy(ak.fill_none(a.gen_msoftdrop, np.nan))
        sel = pb & np.isfinite(pt) & (pt > 200.0)
        sgn = np.sign(gw[sel])
        norm = xs / np.sum(np.abs(gw[sel]))          # slice -> total |w| = xs
        gpt.append(pt[sel]); gmu.append(mu[sel]); gmg.append(mg[sel])
        w.append(sgn * np.abs(gw[sel]) * norm)
    return (np.concatenate(gpt), np.concatenate(gmu),
            np.concatenate(gmg), np.concatenate(w))


def rho(mass, pt):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(mass / (pt * R))      # -inf where mass<=0/nan


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    gpt, gmu, gmg, w = load()
    print(f"pooled gen jets (gen_pt>200, matched): {len(gpt):,d}")

    for obs in ("g", "u"):
        mass = gmg if obs == "g" else gmu            # the plotted observable's mass
        x = rho(mass, gpt)
        fig = plt.figure(figsize=(31, 11), layout="constrained")
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1], figure=fig)
        print(f"\n=== rho_{obs} ({TITLE[obs]}): fraction of jets RETAINED by m_g floor ===")

        for col, (lo, hi, ptlab) in enumerate(PT):
            ax = fig.add_subplot(gs[0, col]); rax = fig.add_subplot(gs[1, col], sharex=ax)
            inpt = (gpt >= lo) & (gpt < hi)
            wtot = np.sum(w[inpt])                    # nominal yield (denominator)
            h0, _ = np.histogram(x[inpt], bins=EDGES, weights=w[inpt])
            norm = np.sum(h0 * np.diff(EDGES))        # area-normalize to nominal
            if norm <= 0:
                continue
            ax.fill_between(CTR, h0 / norm, step="mid", color="0.8", lw=0, zorder=0)
            ax.step(CTR, h0 / norm, where="mid", color="black", lw=2.6, label="no cut")

            fracs = []
            for thr, c in zip(CUTS, CMAP):
                keep = inpt & np.isfinite(gmg) & (gmg > thr)   # cut ALWAYS on groomed mass
                hk, _ = np.histogram(x[keep], bins=EDGES, weights=w[keep])
                frac = np.sum(w[keep]) / wtot
                fracs.append(frac)
                ax.step(CTR, hk / norm, where="mid", color=c, lw=2.0,
                        label=f"$m_g>{thr:g}$ GeV  ({frac:.0%})")
                with np.errstate(divide="ignore", invalid="ignore"):
                    rr = np.where(h0 > 0, hk / h0, np.nan)
                rax.step(CTR, rr, where="mid", color=c, lw=2.0)

            hep.cms.label("", data=False, loc=0, ax=ax, rlabel=ptlab)
            ax.set_ylabel("normalized density"); ax.grid(alpha=0.25); ax.margins(x=0.02)
            ax.set_ylim(0, (h0 / norm).max() * 1.5)
            ax.legend(loc="upper left", framealpha=0.92, fontsize=15,
                      title=f"NLO  $\\rho_{obs}$ {TITLE[obs]}")
            plt.setp(ax.get_xticklabels(), visible=False)
            rax.axhline(1.0, color="black", lw=1.0)
            rax.set_ylim(0, 1.1); rax.set_ylabel("kept / nominal")
            rax.set_xlabel(XLAB[obs]); rax.grid(alpha=0.25); rax.set_xlim(EDGES[0], 0.0)

            print(f"  pt {ptlab:14s}  " + "  ".join(
                f"m_g>{t:g}:{fr:5.1%}" for t, fr in zip(CUTS, fracs)))

        out = os.path.join(OUTDIR, f"nlo_masscut_rho_{obs}.png")
        fig.savefig(out, dpi=110); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
