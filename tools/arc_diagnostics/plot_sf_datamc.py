#!/usr/bin/env python
"""Reco mass data/MC, BEFORE vs AFTER applying the gen->reco mass SF to the MC.

The SF (reco-indexed median m_reco/m_gen, derived from Pythia matched jets, per pT
bin) is the MC self-calibration toward particle level. Here we apply it to the LO
MC reco mass, m_reco -> m_reco/SF(m_reco), and overlay on data to see whether the
reco-level data/MC agreement improves. Groomed (soft drop) + ungroomed, per pT bin,
area-normalized, with an MC/data ratio panel (nominal solid, SF-corrected dashed).

Inputs (2018 per-jet reco ntuples):
  ~/Downloads/mass_diagnostic_ntuple_data_2018.pkl
  ~/Downloads/mass_diagnostic_ntuple_pythia_2018.pkl
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
LUMI = "59.8 fb$^{-1}$ (2018, 13 TeV)"
DATA = "mass_diagnostic_ntuple_data_2018.pkl"
MC = "mass_diagnostic_ntuple_pythia_2018.pkl"
PT_BINS = [(200, 290, "200 < $p_{T}$ < 290 GeV", "pt1"),
           (290, 400, "290 < $p_{T}$ < 400 GeV", "pt2"),
           (400, 1e9, "$p_{T}$ > 400 GeV",       "pt3")]
GROOM = {"g": ("Groomed (soft drop)", "msoftdrop", "gen_msoftdrop", 150.0),
         "u": ("Ungroomed",           "mass",      "gen_mass",      200.0)}
_cache = {}


def cols(f):
    if f not in _cache:
        o = pickle.load(open(os.path.join(D, f), "rb"))["reco_jet_ntuple"]
        _cache[f] = {k: np.asarray(o[k].value if hasattr(o[k], "value") else o[k])
                     for k in ("pt", "msoftdrop", "gen_msoftdrop", "mass",
                               "gen_mass", "weight", "passes_both")}
    return _cache[f]


def sf_curve(rcol, gcol, plo, phi):
    """Reco-indexed median SF(m_reco) on a 2-GeV grid for one pT bin (from MC)."""
    c = cols(MC)
    s = (c["passes_both"].astype(bool) & (c[gcol] > 0) & (c[rcol] > 0)
         & (c["pt"] >= plo) & (c["pt"] < phi))
    reco, r = c[rcol][s], c[rcol][s] / c[gcol][s]
    grid = np.arange(0.0, 202.0, 2.0); cx = 0.5 * (grid[:-1] + grid[1:])
    idx = np.digitize(reco, grid) - 1
    sf = np.full(len(cx), np.nan)
    for b in range(len(cx)):
        m = idx == b
        if m.sum() >= 50:
            sf[b] = np.median(r[m])
    g = np.isfinite(sf)
    return cx[g], sf[g]


def density(f, rcol, plo, phi, edges, sf_x=None, sf_y=None):
    c = cols(f)
    s = (c["pt"] >= plo) & (c["pt"] < phi)
    x, w = c[rcol][s].copy(), c["weight"][s]
    if sf_x is not None:
        x = x / np.interp(x, sf_x, sf_y, left=sf_y[0], right=sf_y[-1])
    v, _ = np.histogram(x, bins=edges, weights=w)
    var, _ = np.histogram(x, bins=edges, weights=w * w)
    width = np.diff(edges); norm = v.sum()
    if norm <= 0:
        return np.zeros_like(v), np.zeros_like(v)
    return v / width / norm, np.sqrt(var) / width / norm


def main():
    for gkey, (gname, rcol, gcol, mtop) in GROOM.items():
        edges = np.arange(0.0, mtop + 5.0, 5.0)
        cx = 0.5 * (edges[:-1] + edges[1:])
        for plo, phi, ptlabel, pttag in PT_BINS:
            sfx, sfy = sf_curve(rcol, gcol, plo, phi)
            d, derr = density(DATA, rcol, plo, phi, edges)
            m0, _ = density(MC, rcol, plo, phi, edges)
            m1, _ = density(MC, rcol, plo, phi, edges, sfx, sfy)

            fig = plt.figure(figsize=(10, 11))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.07)
            ax = fig.add_subplot(gs[0]); rax = fig.add_subplot(gs[1], sharex=ax)

            ax.errorbar(cx, d, yerr=derr, fmt="o", color="black", ms=7,
                        label="Data", zorder=5)
            ax.step(edges, np.append(m0, m0[-1]), where="post", color="#e42536",
                    lw=2.4, label="LO reco (nominal)")
            ax.step(edges, np.append(m1, m1[-1]), where="post", color="#3f90da",
                    lw=2.4, ls="--", label="LO reco / SF")
            with np.errstate(divide="ignore", invalid="ignore"):
                r0 = np.where(d > 0, m0 / d, np.nan)
                r1 = np.where(d > 0, m1 / d, np.nan)
                drel = np.where(d > 0, derr / d, 0.0)
            rax.step(edges, np.append(r0, r0[-1]), where="post", color="#e42536", lw=2.4)
            rax.step(edges, np.append(r1, r1[-1]), where="post", color="#3f90da",
                     lw=2.4, ls="--")
            rax.fill_between(edges, np.append(1 - drel, (1 - drel)[-1]),
                             np.append(1 + drel, (1 + drel)[-1]), step="post",
                             color="black", alpha=0.18, label="Data unc.")
            rax.axhline(1.0, color="black", lw=1, ls="--")

            ax.set_ylim(0, max(d.max(), m0.max(), m1.max()) * 1.5)
            ax.set_ylabel(r"$(1/N)\,dN/dm$  [1/GeV]")
            ax.legend(loc="upper right", fontsize=17, frameon=False,
                      title=gname, title_fontsize=17)
            hep.cms.label("Preliminary", data=True, loc=0, rlabel=LUMI, ax=ax)
            ax.text(0.97, 0.60, ptlabel, ha="right", va="top", transform=ax.transAxes,
                    fontsize=18)
            plt.setp(ax.get_xticklabels(), visible=False)
            rax.set_ylim(0.4, 1.6); rax.set_xlim(0, mtop)
            rax.set_ylabel("MC / Data", fontsize=18)
            xname = "soft-drop mass" if gkey == "g" else "mass"
            rax.set_xlabel(rf"AK8 jet {xname} $m$ [GeV]")

            out = os.path.join(OUT, f"sf_datamc_{gkey}_{pttag}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
            print("wrote", out)


if __name__ == "__main__":
    main()
