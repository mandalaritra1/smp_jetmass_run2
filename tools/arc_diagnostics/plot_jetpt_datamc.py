#!/usr/bin/env python
"""Jet-pT (ptreco) data/MC shape comparison: Data vs LO vs NLO (2018).

The AK8 jet-pT spectrum across the analysis pT bins, from the minimal_rho reco
hists (project ptjet_rhojet_g_reco over rho). Normalized to the per-event
fraction in each pT bin so the shape (not the DY normalization) is compared.
Answers: does the NLO DY reproduce the measured jet-pT spectrum?
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep

hep.style.use(hep.style.CMS)
D = os.path.expanduser("~/Downloads")
OUT = os.path.expanduser("~/Projects/smp_jetmass_run2/review/figs/nlo_deck")
LABELS = ["200-290", "290-400", ">400"]


def jetpt(f, dss):
    o = pickle.load(open(os.path.join(D, f), "rb"))
    H = o["ptjet_rhojet_g_reco"][{"systematic": "nominal"}]
    if "dataset" in [a.name for a in H.axes]:
        H = H[{"dataset": sum}] if dss is None else sum(H[{"dataset": d}] for d in dss)
    return H.project("ptreco").values()[1:]            # drop the 0-200 underflow


def main():
    data = jetpt("minimal_rho_fine_data_2018.pkl", ["EGamma_UL2018", "SingleMuon_UL2018"])
    lo = jetpt("minimal_rho_pythia_2018.pkl", ["pythia_UL18NanoAODv9"])
    nlo = jetpt("minimal_rho_nlo_ptz_2018.pkl", None)
    d, l, n = [a / a.sum() for a in (data, lo, nlo)]
    derr = np.sqrt(data) / data.sum()                  # data stat on the fraction

    x = np.arange(3)
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.07)
    ax = fig.add_subplot(gs[0]); rax = fig.add_subplot(gs[1], sharex=ax)

    ax.bar(x - 0.22, l, 0.22, color="#e42536", label="LO (MG+Py8)", alpha=0.85)
    ax.bar(x + 0.00, n, 0.22, color="#3f90da", label="NLO (FxFx)", alpha=0.85)
    ax.errorbar(x + 0.22, d, yerr=derr, fmt="o", color="black", ms=9, label="Data")

    for off, mc, col in [(-0.22, l, "#e42536"), (0.22, n, "#3f90da")]:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d > 0, mc / d, np.nan)
            re = np.where(d > 0, mc / d * derr / d, np.nan)
        rax.errorbar(x, r, yerr=re, fmt="o", color=col, ms=8)
    rax.axhline(1.0, color="gray", ls="--")

    ax.set_ylim(0, max(d.max(), n.max(), l.max()) * 1.35)
    ax.set_ylabel("Fraction of events / pT bin")
    ax.legend(loc="upper right", fontsize=18, frameon=False)
    hep.cms.label("Preliminary", data=True, loc=0, rlabel="59.8 fb$^{-1}$ (2018, 13 TeV)", ax=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    rax.set_ylim(0.9, 1.15)
    rax.set_ylabel("MC / Data", fontsize=18)
    rax.set_xticks(x); rax.set_xticklabels(LABELS)
    rax.set_xlabel(r"AK8 jet $p_T$ [GeV]")

    out = os.path.join(OUT, "jetpt_datamc.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
