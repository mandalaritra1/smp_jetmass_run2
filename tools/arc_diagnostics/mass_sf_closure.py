#!/usr/bin/env python
"""Closure: does the gen->reco mass SF diagonalize the response? (Pythia only)

Step 3-4 of the diagonalization proposal. From Pythia matched jets we derive the
reco-indexed SF(m_reco, pT) = median(m_reco/m_gen), correct reco mass
    m_reco -> m_corr = m_reco / SF(m_reco, pT),
rebuild the mass migration on the analysis gen binning, and compare purity &
stability per gen-mass bin BEFORE vs AFTER. pT is held on its diagonal block (gen
and reco pT in the same analysis bin) so this isolates the MASS response, which is
all the SF touches. The ridge check (median m_reco vs m_gen) is printed too.

    purity_i    = M[i,i] / sum_gen  M[:,i]   (of reco bin i, fraction truly from gen i)
    stability_i = M[i,i] / sum_reco M[i,:]   (of gen bin i, fraction reco'd in bin i)

Input: ~/Downloads/mass_diagnostic_ntuple_pythia_2018.pkl  (per-jet matched ntuple)
       ~/Downloads/minimal_nlo_ptz_2018.pkl                (for the analysis mgen edges)
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
NT = "mass_diagnostic_ntuple_pythia_2018.pkl"
PT_BINS = [(200, 290, "200 < $p_{T}$ < 290 GeV", "pt200_290"),
           (290, 400, "290 < $p_{T}$ < 400 GeV", "pt290_400"),
           (400, 1e9, "$p_{T}$ > 400 GeV",       "pt400inf")]
GROOM = {"g": ("Groomed (soft drop)", "msoftdrop", "gen_msoftdrop", 150.0),
         "u": ("Ungroomed",           "mass",      "gen_mass",      200.0)}

_o = None


def cols():
    global _o
    if _o is None:
        o = pickle.load(open(os.path.join(D, NT), "rb"))["reco_jet_ntuple"]
        _o = {k: np.asarray(o[k].value if hasattr(o[k], "value") else o[k])
              for k in ("pt", "gen_pt", "mass", "msoftdrop", "gen_mass",
                        "gen_msoftdrop", "weight", "passes_both")}
    return _o


def sf_lookup(mreco, mgen, w, apply_at):
    """Reco-indexed median SF on a 2-GeV grid, linearly interpolated to apply_at."""
    grid = np.arange(0.0, 202.0, 2.0)
    cx = 0.5 * (grid[:-1] + grid[1:])
    r = mreco / mgen
    idx = np.digitize(mreco, grid) - 1
    sf = np.full(len(cx), np.nan)
    for b in range(len(cx)):
        m = idx == b
        if m.sum() >= 50:
            sf[b] = np.median(r[m])
    good = np.isfinite(sf)
    return np.interp(apply_at, cx[good], sf[good],
                     left=sf[good][0], right=sf[good][-1])


def migration(gen, reco, w, edges):
    """M[i,j] = weight with gen in bin i, reco in bin j, on common `edges`."""
    gi = np.digitize(gen, edges) - 1
    ri = np.digitize(reco, edges) - 1
    nb = len(edges) - 1
    ok = (gi >= 0) & (gi < nb) & (ri >= 0) & (ri < nb)
    M = np.zeros((nb, nb))
    np.add.at(M, (gi[ok], ri[ok]), w[ok])
    return M


def pur_stab(M):
    diag = np.diag(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.where(M.sum(0) > 0, diag / M.sum(0), np.nan)     # per reco bin
        stab = np.where(M.sum(1) > 0, diag / M.sum(1), np.nan)       # per gen bin
    return purity, stab


def main():
    mgen = pickle.load(open(os.path.join(D, "minimal_nlo_ptz_2018.pkl"), "rb"))[
        "response_matrix_g"].axes["mgen"].edges
    mgen = mgen[mgen <= 200.0]                       # drop the 200-1000 overflow
    cx = 0.5 * (mgen[:-1] + mgen[1:])
    c = cols()
    print(f"{'grooming':10s} {'pT bin':16s}  mean purity   mean stability   (before -> after)")

    for gkey, (gname, rcol, gcol, mtop) in GROOM.items():
        for plo, phi, ptlabel, pttag in PT_BINS:
            sel = (c["passes_both"].astype(bool) & (c[gcol] > 0) & (c[rcol] > 0)
                   & (c["pt"] >= plo) & (c["pt"] < phi)
                   & (c["gen_pt"] >= plo) & (c["gen_pt"] < phi))
            gen, reco, w = c[gcol][sel], c[rcol][sel], c["weight"][sel]
            sf = sf_lookup(reco, gen, w, reco)
            corr = reco / sf

            M0 = migration(gen, reco, w, mgen)
            M1 = migration(gen, corr, w, mgen)
            p0, s0 = pur_stab(M0)
            p1, s1 = pur_stab(M1)

            # event-weighted means over populated bulk (10-mtop GeV)
            bulk = (cx >= 10) & (cx <= mtop)
            wr, wg = M0.sum(0), M0.sum(1)
            mp = lambda v, ww: np.nansum((v * ww)[bulk]) / np.nansum(ww[bulk])
            print(f"{gname:10s} {ptlabel:16s}  "
                  f"{mp(p0, wr):.3f} -> {mp(p1, wr):.3f}    "
                  f"{mp(s0, wg):.3f} -> {mp(s1, wg):.3f}")

            fig = plt.figure(figsize=(10, 11))
            gs = gridspec.GridSpec(2, 1, hspace=0.07)
            axp = fig.add_subplot(gs[0]); axs = fig.add_subplot(gs[1], sharex=axp)
            axp.step(mgen, np.append(p0, p0[-1]), where="post", color="gray",
                     ls="--", lw=2.0, label="before SF")
            axp.step(mgen, np.append(p1, p1[-1]), where="post", color="#e42536",
                     lw=2.4, label="after SF")
            axs.step(mgen, np.append(s0, s0[-1]), where="post", color="gray",
                     ls="--", lw=2.0)
            axs.step(mgen, np.append(s1, s1[-1]), where="post", color="#3f90da", lw=2.4)

            axp.set_ylim(0, 1.05); axs.set_ylim(0, 1.05)
            axp.set_xlim(0, mtop)
            axp.set_ylabel("Purity")
            axs.set_ylabel("Stability")
            axp.legend(loc="lower right", fontsize=16, frameon=False,
                       title=gname, title_fontsize=16)
            hep.cms.label("Preliminary", data=False, loc=0, rlabel="(2018, 13 TeV)", ax=axp)
            axp.text(0.04, 0.94, ptlabel, ha="left", va="top", transform=axp.transAxes,
                     fontsize=18)
            plt.setp(axp.get_xticklabels(), visible=False)
            xname = "gen soft-drop mass" if gkey == "g" else "gen mass"
            axs.set_xlabel(rf"{xname} $m_{{gen}}$ [GeV]")

            out = os.path.join(OUT, f"mass_diag_{gkey}_{pttag}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
