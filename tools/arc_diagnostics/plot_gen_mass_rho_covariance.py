#!/usr/bin/env python3
"""Groomed<->ungroomed gen-level correlation matrices (ARC Scope#3).

Runs the zjet processor in `mass_cov` mode on a DY MC file, then turns the joint
gen (groomed)x(ungroomed) histograms into the cross-correlation matrices for both
mass and rho. For a gen joint hist H (filled from the same events), the statistical
covariance between groomed bin a and ungroomed bin b is the shared cell's sumw2:

    Cov(N^g_a, N^u_b) = sumw2[a,b]
    Var^g_a = sum_b sumw2[a,b],  Var^u_b = sum_a sumw2[a,b]
    Corr(a,b) = sumw2[a,b] / sqrt(Var^g_a * Var^u_b)

Default input is the local HT400-600 UL18 DY test file (a single HT slice -> a
TEMPORARY illustration, not the production covariance).
"""
import argparse
import os
import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False
from smp_jetmass_run2.zjet_processor import QJetMassProcessor

DEF_FILE = "/Users/aritra/Projects/omnifold/data/nanoaod/DYJetsToLL_HT400to600_UL18_test.root"
# synthetic store/mc path so the processor's filename parser is happy
FAKE = ("/eos/cms/store/mc/RunIISummer20UL18NanoAODv9/"
        "DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/x.root")


def cross_corr(h):
    """(corr, var_g, var_u) cross block from a joint hist with sumw2 cells."""
    h2 = h[{"dataset": sum, "channel": sum, "ptgen": sum, "systematic": sum}]
    sw2 = h2.variances()                 # Cov(N^g_a, N^u_b) = sumw2[a,b]
    var_g = sw2.sum(axis=1)              # groomed marginal variance (rows)
    var_u = sw2.sum(axis=0)              # ungroomed marginal variance (cols)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = sw2 / np.sqrt(np.outer(var_g, var_u))
    return corr, var_g, var_u, h2.axes


def panel(ax, corr, var_g, var_u, axes, title, unit):
    g_edges = axes[0].edges
    u_edges = axes[1].edges
    # keep only populated bins for a readable matrix
    gkeep = np.where(var_g > 0)[0]
    ukeep = np.where(var_u > 0)[0]
    sub = corr[np.ix_(gkeep, ukeep)]
    im = ax.imshow(sub, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ukeep)))
    ax.set_yticks(range(len(gkeep)))
    ax.set_xticklabels([f"{u_edges[i]:.0f}" if unit == "GeV" else f"{u_edges[i]:.1f}" for i in ukeep],
                       rotation=90, fontsize=7)
    ax.set_yticklabels([f"{g_edges[i]:.0f}" if unit == "GeV" else f"{g_edges[i]:.1f}" for i in gkeep],
                       fontsize=7)
    ax.set_xlabel(f"ungroomed bin low edge [{unit}]")
    ax.set_ylabel(f"groomed bin low edge [{unit}]")
    ax.set_title(title, fontsize=11)
    for (j, i), val in np.ndenumerate(sub):
        if np.isfinite(val):
            ax.text(i, j, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.6 else "black", fontsize=6)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEF_FILE)
    ap.add_argument("--nmax", type=int, default=0, help="limit events (0 = all)")
    ap.add_argument("--out", default="review/figs/gen_groomed_ungroomed_correlation_HT400to600.png")
    args = ap.parse_args()

    ev = NanoEventsFactory.from_root(
        {args.file: "Events"}, schemaclass=NanoAODSchema, mode="eager",
        metadata={"dataset": "DYJets_pythia_UL18NanoAODv9", "filename": FAKE},
    ).events()
    if args.nmax:
        ev = ev[:args.nmax]
    print(f"events: {len(ev)}")
    out = QJetMassProcessor(do_gen=True, mode="mass_cov",
                            systematics=["nominal"], jet_systematics=["nominal"]).process(ev)

    cm, vg_m, vu_m, ax_m = cross_corr(out["ptjet_mjet_g_vs_u_gen"])
    cr, vg_r, vu_r, ax_r = cross_corr(out["ptjet_rhojet_g_vs_u_gen"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6))
    im1 = panel(a1, cm, vg_m, vu_m, ax_m, "Gen mass: groomed vs ungroomed corr.", "GeV")
    im2 = panel(a2, cr, vg_r, vu_r, ax_r, r"Gen rho: groomed vs ungroomed corr.", "")
    fig.colorbar(im2, ax=a2, fraction=0.046, label="correlation")
    fig.colorbar(im1, ax=a1, fraction=0.046, label="correlation")
    fig.suptitle("Gen-level groomed<->ungroomed statistical correlation "
                 "(DY HT400-600, UL18, single slice - TEMPORARY)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"mass max off-diag corr: {np.nanmax(cm - np.eye(*cm.shape)[:cm.shape[0], :cm.shape[1]]):.3f}; "
          f"rho max off-diag corr: {np.nanmax(cr - np.eye(*cr.shape)[:cr.shape[0], :cr.shape[1]]):.3f}")


if __name__ == "__main__":
    main()
