#!/usr/bin/env python3
"""Groomed<->ungroomed gen/reco correlation matrices (ARC Scope#3).

Runs the zjet processor in `mass_cov` mode on a DY MC file, then turns the joint
gen/reco (groomed)x(ungroomed) histograms into correlation matrices. For a joint
hist H (same events), the statistical covariance between groomed bin a and
ungroomed bin b is the shared cell's sumw2:

    Cov(N^g_a, N^u_b) = sumw2[a,b],  Corr = sumw2[a,b]/sqrt(Var^g_a*Var^u_b)

Produces:
  * one groomed-vs-ungroomed mass+rho correlation figure PER pT bin (2x2 gen/reco
    x mass/rho), and
  * one pT-axis figure: the groomed-vs-ungroomed pT correlation (integrated over
    mass/rho) -- diagonal, since groomed & ungroomed share the jet pT, so an event
    is in one pT bin (no same-event cross-pT correlation).

Default input is the local HT400-600 UL18 DY test file (single HT slice -> a
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
FAKE = ("/eos/cms/store/mc/RunIISummer20UL18NanoAODv9/"
        "DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/x.root")

SPECS = [  # hist name, level, observable label, unit
    ("ptjet_mjet_g_vs_u_gen",   "GEN",  r"$m$",                  "GeV"),
    ("ptjet_rhojet_g_vs_u_gen", "GEN",  r"$\log_{10}(\rho^2)$", ""),
    ("ptjet_mjet_g_vs_u_reco",  "RECO", r"$m$",                  "GeV"),
    ("ptjet_rhojet_g_vs_u_reco","RECO", r"$\log_{10}(\rho^2)$", ""),
]


def _proj(h):
    """Sum dataset/channel/systematic; keep [pt, groomed_obs, ungroomed_obs]."""
    drop = {a.name for i, a in enumerate(h.axes) if i not in (2, 3, 4)}
    return h[{n: sum for n in drop}]


def obs_corr_at_pt(h, ipt):
    h2 = _proj(h)
    sw2 = h2.variances()[ipt]                 # (ng, nu)
    var_g, var_u = sw2.sum(axis=1), sw2.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = sw2 / np.sqrt(np.outer(var_g, var_u))
    return corr, var_g, var_u, np.asarray(h2.axes[1].edges), np.asarray(h2.axes[2].edges)


def pt_keep_of(h):
    h2 = _proj(h)
    pe = np.asarray(h2.axes[0].edges)
    tot = h2.variances().sum(axis=(1, 2))
    return [i for i in range(len(tot)) if pe[i] >= 200 and tot[i] > 0], pe


def pt_corr(h):
    """Groomed-pT vs ungroomed-pT correlation (integrated over mass/rho)."""
    h2 = _proj(h)
    tot = h2.variances().sum(axis=(1, 2))     # per-pT total sumw2
    keep, pe = pt_keep_of(h)
    V = tot[keep]
    C = np.diag(V)                            # same event -> diagonal in pT
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = C / np.sqrt(np.outer(V, V))
    return corr, keep, pe


def panel_obs(ax, corr, var_g, var_u, g_edges, u_edges, obs, unit, level):
    gkeep = np.where(var_g > 0)[0]
    ukeep = np.where(var_u > 0)[0]
    sub = corr[np.ix_(gkeep, ukeep)]
    im = ax.imshow(sub, origin="lower", aspect="auto", cmap="cividis", vmin=0, vmax=1)
    fmt = (lambda x: f"{x:.0f}") if unit == "GeV" else (lambda x: f"{x:.1f}")
    ax.set_xticks(range(len(ukeep)));  ax.set_yticks(range(len(gkeep)))
    ax.set_xticklabels([fmt(u_edges[i]) for i in ukeep], rotation=90, fontsize=11)
    ax.set_yticklabels([fmt(g_edges[i]) for i in gkeep], fontsize=11)
    xunit = f" [{unit}]" if unit else ""
    ax.set_xlabel(rf"ungroomed {obs}{xunit}", fontsize=16)
    ax.set_ylabel(rf"groomed {obs}{xunit}", fontsize=16)
    for (j, i), v in np.ndenumerate(sub):
        if np.isfinite(v):
            ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=8)
    ax.text(0.03, 0.95, level, transform=ax.transAxes, fontsize=15, fontweight="bold",
            color="white", va="top", bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
    hep.cms.label("Private Work", data=False, com=13, ax=ax, fontsize=13)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEF_FILE)
    ap.add_argument("--nmax", type=int, default=0)
    ap.add_argument("--outdir", default="review/figs")
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
    os.makedirs(args.outdir, exist_ok=True)

    # pT binning (from the gen rho hist; measurement bins, low edge >= 200)
    pt_keep, pt_edges = pt_keep_of(out["ptjet_rhojet_g_vs_u_gen"])

    # ---- one mass+rho correlation figure PER pT bin (2x2 gen/reco x mass/rho) ----
    for ipt in pt_keep:
        lo, hi = pt_edges[ipt], pt_edges[ipt + 1]
        rng = f"{lo:.0f}-{hi:.0f} GeV" if hi < 1e4 else rf"$p_T>{lo:.0f}$ GeV"
        tag = f"pt{lo:.0f}to{hi:.0f}" if hi < 1e4 else f"ptgt{lo:.0f}"
        fig, axes = plt.subplots(2, 2, figsize=(15, 14))
        for ax, (hname, level, obs, unit) in zip(axes.flat, SPECS):
            corr, vg, vu, ge, ue = obs_corr_at_pt(out[hname], ipt)
            im = panel_obs(ax, corr, vg, vu, ge, ue, obs, unit, level)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("statistical correlation", fontsize=14); cb.ax.tick_params(labelsize=11)
        fig.suptitle(f"groomed$\\leftrightarrow$ungroomed correlation, "
                     rf"$p_T$ {rng}  ·  DY HT400-600 UL18 (TEMPORARY)", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fpath = os.path.join(args.outdir, f"gru_corr_{tag}.png")
        fig.savefig(fpath, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {fpath}")

    # ---- one pT-axis figure: groomed-pT vs ungroomed-pT correlation (gen, reco) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (hname, level) in zip(axes, [("ptjet_rhojet_g_vs_u_gen", "GEN"),
                                         ("ptjet_rhojet_g_vs_u_reco", "RECO")]):
        corr, keep, pe = pt_corr(out[hname])
        im = ax.imshow(corr, origin="lower", aspect="auto", cmap="cividis", vmin=0, vmax=1)
        labs = [f"{pe[i]:.0f}-{pe[i+1]:.0f}" if pe[i+1] < 1e4 else f">{pe[i]:.0f}" for i in keep]
        ax.set_xticks(range(len(keep))); ax.set_yticks(range(len(keep)))
        ax.set_xticklabels(labs, rotation=45, fontsize=12); ax.set_yticklabels(labs, fontsize=12)
        ax.set_xlabel(r"ungroomed $p_T$ bin [GeV]", fontsize=16)
        ax.set_ylabel(r"groomed $p_T$ bin [GeV]", fontsize=16)
        for (j, i), v in np.ndenumerate(corr):
            if np.isfinite(v):
                ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=11)
        ax.text(0.03, 0.95, level, transform=ax.transAxes, fontsize=15, fontweight="bold",
                color="white", va="top", bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
        hep.cms.label("Private Work", data=False, com=13, ax=ax, fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("statistical correlation", fontsize=14)
    fig.suptitle("groomed$\\leftrightarrow$ungroomed $p_T$ correlation (diagonal: pT is "
                 "shared, bins independent)  ·  DY HT400-600 UL18 (TEMPORARY)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fpath = os.path.join(args.outdir, "gru_corr_ptaxis.png")
    fig.savefig(fpath, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {fpath}")


if __name__ == "__main__":
    main()
