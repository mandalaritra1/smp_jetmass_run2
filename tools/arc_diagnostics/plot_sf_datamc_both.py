#!/usr/bin/env python
"""Reco mass data/MC with the SF applied to BOTH data and MC (the calibration way).

A calibration SF is applied to data and MC alike. Because SF(m_reco) is a
deterministic function of m_reco, applying it to both is a monotonic relabel of the
mass axis common to numerator and denominator -> the data/MC density ratio is
invariant (the Jacobian cancels pointwise). This plots the "after" world (both
corrected) and overlays the MC/data ratio BEFORE vs AFTER, with a chi2/ndf, to show
a common reco-side SF cannot change data/MC agreement. groomed + ungroomed, per pT.

Inputs: ~/Downloads/mass_diagnostic_ntuple_{data,pythia}_2018.pkl
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


def hist(f, rcol, plo, phi, edges, sfx=None, sfy=None):
    c = cols(f)
    s = (c["pt"] >= plo) & (c["pt"] < phi)
    x, w = c[rcol][s].copy(), c["weight"][s]
    if sfx is not None:
        x = x / np.interp(x, sfx, sfy, left=sfy[0], right=sfy[-1])
    v, _ = np.histogram(x, bins=edges, weights=w)
    var, _ = np.histogram(x, bins=edges, weights=w * w)
    return v, var


def norm_density(v, var, edges):
    width = np.diff(edges); n = v.sum()
    if n <= 0:
        return np.zeros_like(v), np.zeros_like(v)
    return v / width / n, np.sqrt(var) / width / n


def chi2_ndf(mc, mcv, da, dav, edges):
    md, _ = norm_density(mc, mcv, edges)
    dd, dde = norm_density(da, dav, edges)
    m = (dd > 0) & (dde > 0)
    return np.sum((md[m] - dd[m]) ** 2 / dde[m] ** 2) / m.sum()


def main():
    for gkey, (gname, rcol, gcol, mtop) in GROOM.items():
        edges = np.arange(0.0, mtop + 5.0, 5.0)
        cx = 0.5 * (edges[:-1] + edges[1:])
        for plo, phi, ptlabel, pttag in PT_BINS:
            sfx, sfy = sf_curve(rcol, gcol, plo, phi)
            # before: nominal;  after: SF applied to BOTH
            d0, d0v = hist(DATA, rcol, plo, phi, edges)
            m0, m0v = hist(MC, rcol, plo, phi, edges)
            d1, d1v = hist(DATA, rcol, plo, phi, edges, sfx, sfy)
            m1, m1v = hist(MC, rcol, plo, phi, edges, sfx, sfy)
            c2b = chi2_ndf(m0, m0v, d0, d0v, edges)
            c2a = chi2_ndf(m1, m1v, d1, d1v, edges)

            dd1, dd1e = norm_density(d1, d1v, edges)
            mm1, _ = norm_density(m1, m1v, edges)
            dd0, dd0e = norm_density(d0, d0v, edges)
            mm0, _ = norm_density(m0, m0v, edges)

            fig = plt.figure(figsize=(10, 11))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.07)
            ax = fig.add_subplot(gs[0]); rax = fig.add_subplot(gs[1], sharex=ax)

            ax.errorbar(cx, dd1, yerr=dd1e, fmt="o", color="black", ms=7,
                        label="Data / SF", zorder=5)
            ax.step(edges, np.append(mm1, mm1[-1]), where="post", color="#3f90da",
                    lw=2.4, label="LO / SF")
            with np.errstate(divide="ignore", invalid="ignore"):
                rb = np.where(dd0 > 0, mm0 / dd0, np.nan)
                ra = np.where(dd1 > 0, mm1 / dd1, np.nan)
            rax.step(edges, np.append(rb, rb[-1]), where="post", color="gray",
                     lw=2.6, label=f"before  ($\\chi^2$/ndf {c2b:.2f})")
            rax.step(edges, np.append(ra, ra[-1]), where="post", color="#3f90da",
                     lw=2.2, ls="--", label=f"after  ($\\chi^2$/ndf {c2a:.2f})")
            rax.axhline(1.0, color="black", lw=1, ls="--")

            ax.set_ylim(0, max(dd1.max(), mm1.max()) * 1.5)
            ax.set_ylabel(r"$(1/N)\,dN/dm$  [1/GeV]")
            ax.legend(loc="upper right", fontsize=17, frameon=False,
                      title=f"{gname} — SF on both", title_fontsize=16)
            hep.cms.label("Preliminary", data=True, loc=0, rlabel=LUMI, ax=ax)
            ax.text(0.97, 0.60, ptlabel, ha="right", va="top", transform=ax.transAxes,
                    fontsize=18)
            plt.setp(ax.get_xticklabels(), visible=False)
            rax.set_ylim(0.4, 1.6); rax.set_xlim(0, mtop)
            rax.set_ylabel("MC / Data", fontsize=18)
            rax.legend(loc="upper left", fontsize=13, frameon=False, ncol=2)
            xname = "soft-drop mass" if gkey == "g" else "mass"
            rax.set_xlabel(rf"AK8 jet {xname} $m$ [GeV]")

            out = os.path.join(OUT, f"sf_datamc_both_{gkey}_{pttag}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
            print(f"wrote {os.path.basename(out)}   chi2/ndf before {c2b:.2f} -> after {c2a:.2f}")


if __name__ == "__main__":
    main()
