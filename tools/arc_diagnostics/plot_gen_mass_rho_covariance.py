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


def flat_cross_corr(h):
    """Flattened (pT x observable) groomed-vs-ungroomed correlation matrix.

    Axis order is [dataset, channel, pt, groomed_obs, ungroomed_obs, systematic];
    sum over dataset/channel/systematic, keep pt(2), groomed(3), ungroomed(4).
    The combined observable is (pT bin, obs bin) flattened. Because each event has
    ONE jet pT (shared by groomed & ungroomed), the same-event covariance is
    block-diagonal in pT: Cov[(pt_i,a),(pt_j,b)] = delta_ij * sumw2[pt_i,a,b].
    """
    drop = {a.name for i, a in enumerate(h.axes) if i not in (2, 3, 4)}
    h2 = h[{n: sum for n in drop}]
    pt_ax, g_ax, u_ax = h2.axes
    sw2 = h2.variances()                      # (npt, ng, nu)
    pt_edges = np.asarray(pt_ax.edges)
    npt, ng, nu = sw2.shape
    # measurement pT bins (low edge >= 200) that are populated
    pt_keep = [i for i in range(npt) if pt_edges[i] >= 200 and sw2[i].sum() > 0]
    sub = sw2[pt_keep]                        # (npk, ng, nu)
    g_keep = [a for a in range(ng) if sub[:, a, :].sum() > 0]
    u_keep = [b for b in range(nu) if sub[:, :, b].sum() > 0]
    G = [(pi, a) for pi in pt_keep for a in g_keep]
    U = [(pj, b) for pj in pt_keep for b in u_keep]
    C = np.zeros((len(G), len(U)))
    for gi, (pi, a) in enumerate(G):
        for uj, (pj, b) in enumerate(U):
            if pi == pj:
                C[gi, uj] = sw2[pi, a, b]
    VG = np.array([sw2[pi, a, :].sum() for (pi, a) in G])
    VU = np.array([sw2[pj, :, b].sum() for (pj, b) in U])
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = C / np.sqrt(np.outer(VG, VU))
    return dict(corr=corr, pt_keep=pt_keep, g_keep=g_keep, u_keep=u_keep,
                pt_edges=pt_edges, g_edges=np.asarray(g_ax.edges), u_edges=np.asarray(u_ax.edges))


def panel(ax, r, obs_label, unit, level=""):
    corr, pt_keep, g_keep, u_keep = r["corr"], r["pt_keep"], r["g_keep"], r["u_keep"]
    ng, nu, npk = len(g_keep), len(u_keep), len(pt_keep)
    im = ax.imshow(corr, origin="lower", aspect="auto", cmap="cividis", vmin=0, vmax=1)
    fmt = (lambda x: f"{x:.0f}") if unit == "GeV" else (lambda x: f"{x:.1f}")
    # pT-block dividers + value annotations
    for k in range(1, npk):
        ax.axvline(k * nu - 0.5, color="white", lw=2.5)
        ax.axhline(k * ng - 0.5, color="white", lw=2.5)
    for (j, i), v in np.ndenumerate(corr):
        if np.isfinite(v) and v > 0.005:
            ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=6)
    # within-block observable ticks (low edge), repeated per pT block
    ax.set_xticks(range(nu * npk))
    ax.set_yticks(range(ng * npk))
    ax.set_xticklabels([fmt(r["u_edges"][b]) for _ in range(npk) for b in u_keep], rotation=90, fontsize=6)
    ax.set_yticklabels([fmt(r["g_edges"][a]) for _ in range(npk) for a in g_keep], fontsize=6)
    # pT-range labels centered under each x block (below the obs tick labels)
    for k, pi in enumerate(pt_keep):
        lo, hi = r["pt_edges"][pi], r["pt_edges"][pi + 1]
        lab = f"{lo:.0f}-{hi:.0f}" if hi < 1e4 else rf"$>{lo:.0f}$"
        ax.text((k + 0.5) * nu - 0.5, -0.16, rf"$p_T$ {lab}", ha="center", va="top",
                fontsize=10, fontweight="bold", transform=ax.get_xaxis_transform())
    xunit = f", {unit}" if unit else ""
    ax.set_xlabel(rf"ungroomed: $p_T$ block $\otimes$ {obs_label}{xunit}", fontsize=13)
    ax.set_ylabel(rf"groomed: $p_T$ block $\otimes$ {obs_label}{xunit}", fontsize=13)
    if level:
        ax.text(0.02, 0.97, level, transform=ax.transAxes, fontsize=15, fontweight="bold",
                color="white", va="top", bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
    hep.cms.label("Private Work", data=False, com=13, ax=ax, fontsize=11)
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
    fig, axes = plt.subplots(2, 2, figsize=(19, 16))
    for ax, (hname, level, obs, unit) in zip(axes.flat, specs):
        r = flat_cross_corr(out[hname])
        im = panel(ax, r, obs, unit, level=level)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("statistical correlation", fontsize=15)
        cb.ax.tick_params(labelsize=11)
        print(f"{hname}: {len(r['pt_keep'])} pT blocks, max corr {np.nanmax(r['corr']):.3f}")
    fig.text(0.5, 0.004,
             "groomed$\\leftrightarrow$ungroomed statistical correlation, "
             "flattened ($p_T$ block $\\otimes$ observable; block-diagonal in $p_T$)  ·  "
             "DY HT400-600 UL18 (single slice, TEMPORARY)", ha="center", fontsize=13)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
