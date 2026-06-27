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
import mplhep as hep

hep.style.use("CMS")

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False
from smp_jetmass_run2.zjet_processor import QJetMassProcessor

DEF_FILE = "/Users/aritra/Projects/omnifold/data/nanoaod/DYJetsToLL_HT400to600_UL18_test.root"
# synthetic store/mc path so the processor's filename parser is happy
FAKE = ("/eos/cms/store/mc/RunIISummer20UL18NanoAODv9/"
        "DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/x.root")


def cross_corr(h):
    """(corr, var_g, var_u, axes) cross block from a joint hist with sumw2 cells.

    Axis order is [dataset, channel, pt(gen|reco), groomed_obs, ungroomed_obs,
    systematic]; we sum over everything except the two observable axes (3, 4).
    """
    drop = {a.name for i, a in enumerate(h.axes) if i not in (3, 4)}
    h2 = h[{n: sum for n in drop}]
    sw2 = h2.variances()                 # Cov(N^g_a, N^u_b) = sumw2[a,b]
    var_g = sw2.sum(axis=1)              # groomed marginal variance (rows)
    var_u = sw2.sum(axis=0)              # ungroomed marginal variance (cols)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = sw2 / np.sqrt(np.outer(var_g, var_u))
    return corr, var_g, var_u, h2.axes


def panel(ax, corr, var_g, var_u, axes, obs_label, unit, level=""):
    g_edges = axes[0].edges
    u_edges = axes[1].edges
    # keep only populated bins for a readable matrix
    gkeep = np.where(var_g > 0)[0]
    ukeep = np.where(var_u > 0)[0]
    sub = corr[np.ix_(gkeep, ukeep)]
    im = ax.imshow(sub, origin="lower", aspect="auto", cmap="cividis", vmin=0, vmax=1)
    fmt = (lambda x: f"{x:.0f}") if unit == "GeV" else (lambda x: f"{x:.1f}")
    ax.set_xticks(range(len(ukeep)))
    ax.set_yticks(range(len(gkeep)))
    ax.set_xticklabels([fmt(u_edges[i]) for i in ukeep], rotation=90, fontsize=12)
    ax.set_yticklabels([fmt(g_edges[i]) for i in gkeep], fontsize=12)
    xunit = f" [{unit}]" if unit else ""
    ax.set_xlabel(rf"ungroomed {obs_label}{xunit}", fontsize=18)
    ax.set_ylabel(rf"groomed {obs_label}{xunit}", fontsize=18)
    for (j, i), val in np.ndenumerate(sub):
        if np.isfinite(val):
            ax.text(i, j, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.5 else "black", fontsize=8)
    if level:
        ax.text(0.03, 0.94, level, transform=ax.transAxes, fontsize=16,
                fontweight="bold", color="white",
                bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
    hep.cms.label("Private Work", data=False, com=13, ax=ax, fontsize=15)
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

    # rows = GEN / RECO, cols = mass / rho
    specs = [
        ("ptjet_mjet_g_vs_u_gen",   "GEN",  r"$m$",                  "GeV"),
        ("ptjet_rhojet_g_vs_u_gen", "GEN",  r"$\log_{10}(\rho^2)$", ""),
        ("ptjet_mjet_g_vs_u_reco",  "RECO", r"$m$",                  "GeV"),
        ("ptjet_rhojet_g_vs_u_reco","RECO", r"$\log_{10}(\rho^2)$", ""),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    for ax, (hname, level, obs, unit) in zip(axes.flat, specs):
        corr, vg, vu, hax = cross_corr(out[hname])
        im = panel(ax, corr, vg, vu, hax, obs, unit, level=level)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("statistical correlation", fontsize=15)
        cb.ax.tick_params(labelsize=11)
        offdiag = np.nanmax(corr - np.eye(*corr.shape)[:corr.shape[0], :corr.shape[1]])
        print(f"{hname}: max off-diag corr {offdiag:.3f}")
    fig.text(0.5, 0.004,
             "groomed$\\leftrightarrow$ungroomed statistical correlation  ·  "
             "DY HT400-600 UL18 (single slice, TEMPORARY)", ha="center", fontsize=14)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
