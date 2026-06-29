#!/usr/bin/env python
"""Z mass & Z pT reco-level data/MC validation, with the NLO DY overlay.

Validates the event selection / lepton calibration (Z-mass peak) and the recoil
spectrum (Z-pT) for data vs LO DY, NLO DY (+ small backgrounds). All-era (Run 2).

Inputs:
  Data   : unfold/inputs/zjet/validation/validation_data.pkl          pt_Z, mass_Z
  LO DY  : unfold/inputs/zjet/validation/validation_pythia_<era>.pkl   pt_Z, mass_Z
  Bkg    : unfold/inputs/zjet/validation/validation_{backgrounds,st}_all.pkl
  NLO DY : ~/Downloads/minimal_rho_nlo_ptz_all.pkl   ptz_mz_reco (2D mass x pt -> project)
All hists are already xs*lumi normalized; data has unit weights.
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep

hep.style.use(hep.style.CMS)

V = "/Users/aritra/Projects/unfold/inputs/zjet/validation"
DL = os.path.expanduser("~/Downloads")
OUT = os.path.expanduser("~/Projects/smp_jetmass_run2/review/figs/z_validation")
os.makedirs(OUT, exist_ok=True)
LUMI = "138 fb$^{-1}$ (13 TeV)"
ERAS = ["2016", "2016APV", "2017", "2018"]


def load(path):
    return pickle.load(open(path, "rb"))


def proj1d(h, axis):
    """Sum over dataset (+ nominal systematic if present), return (vals, var, edges)."""
    if "systematic" in [a.name for a in h.axes]:
        h = h[{"systematic": "nominal"}]
    h = h[{"dataset": sum}].project(axis)
    return h.values(), h.variances(), h.axes[axis].edges


def add(a, b):
    return b if a is None else (a[0] + b[0], a[1] + b[1], a[2])


def get(obs):
    """obs in {'mass','pt'}. Return dict of (vals,var,edges) for data/lo/nlo/bkg."""
    vname = {"mass": "mass_Z", "pt": "pt_Z"}[obs]
    out = {}
    out["data"] = proj1d(load(f"{V}/validation_data.pkl")[vname], obs)
    lo = None
    for e in ERAS:
        lo = add(lo, proj1d(load(f"{V}/validation_pythia_{e}.pkl")[vname], obs))
    out["lo"] = lo
    bkg = proj1d(load(f"{V}/validation_backgrounds_all.pkl")[vname], obs)
    bkg = add(bkg, proj1d(load(f"{V}/validation_st_all.pkl")[vname], obs))
    out["bkg"] = bkg
    # NLO: project the 2D ptz_mz_reco onto the requested axis
    out["nlo"] = proj1d(load(f"{DL}/minimal_rho_nlo_ptz_all.pkl")["ptz_mz_reco"], obs)
    return out


def panel(obs, logy=False, xlim=None, normalize=False):
    c = get(obs)
    edges = c["data"][2]
    w = np.diff(edges)
    cx = 0.5 * (edges[:-1] + edges[1:])

    d, dv = c["data"][0], c["data"][1]
    bkg = c["bkg"][0]
    lo = c["lo"][0] + bkg          # DY(LO) + backgrounds
    nlo = c["nlo"][0] + bkg        # DY(NLO) + backgrounds

    if normalize:                  # area-normalize each curve to unit integral
        dv = dv / d.sum() ** 2
        d = d / d.sum()
        lo = lo / lo.sum()
        nlo = nlo / nlo.sum()

    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.07)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)

    ax.errorbar(cx, d / w, yerr=np.sqrt(dv) / w, fmt="o", color="black", ms=6,
                label="Data", zorder=5)
    for y, col, lab in [(lo, "#e42536", "LO (MG+Py8) + bkg"),
                        (nlo, "#3f90da", "NLO (FxFx) + bkg")]:
        ax.step(edges, np.append(y, y[-1]) / np.append(w, w[-1]), where="post",
                color=col, lw=2.2, label=lab)
    if not normalize:
        ax.step(edges, np.append(bkg, bkg[-1]) / np.append(w, w[-1]), where="post",
                color="grey", lw=1.3, ls=":", label="Backgrounds")

    for y, col in [(lo, "#e42536"), (nlo, "#3f90da")]:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d > 0, y / d, np.nan)
        rax.step(edges, np.append(r, r[-1]), where="post", color=col, lw=2.2)
    with np.errstate(divide="ignore", invalid="ignore"):
        drel = np.where(d > 0, np.sqrt(dv) / d, 0.0)
    rax.fill_between(edges, np.append(1 - drel, (1 - drel)[-1]),
                     np.append(1 + drel, (1 + drel)[-1]), step="post",
                     color="black", alpha=0.18)
    rax.axhline(1.0, color="black", lw=1, ls="--")

    if logy:
        ax.set_yscale("log")
        top = (nlo / w).max()
        ax.set_ylim(top * 3e-5, top * 3)
    else:
        ax.set_ylim(0, (d / w).max() * 1.55)
    if xlim:
        ax.set_xlim(*xlim); rax.set_xlim(*xlim)
    xlabel = {"mass": r"$m_{\ell\ell}$ [GeV]", "pt": r"$p_{T}^{Z}$ [GeV]"}[obs]
    ax.set_ylabel(r"$(1/N)\,dN/d$GeV" if normalize else "Events / GeV")
    ax.legend(loc="upper right" if obs == "pt" else "upper left",
              fontsize=17, frameon=False)
    hep.cms.label("Preliminary", data=True, loc=0, rlabel=LUMI, ax=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    rax.set_ylim(0.7, 1.3)
    rax.set_ylabel("MC / Data", fontsize=18)
    rax.set_xlabel(xlabel)

    outdir = OUT + ("_norm" if normalize else "_abs")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"z_validation_{obs}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for norm in (False, True):
        panel("mass", logy=False, xlim=(70, 110), normalize=norm)
        panel("pt", logy=True, xlim=(150, 900), normalize=norm)
