#!/usr/bin/env python
"""Mass probability matrix P(reco | gen) per pT bin, groomed + ungroomed (Pythia 2018).

Visualizes the response we have been characterizing: each gen-mass column is
normalized to unit sum, so the colour is the probability a jet generated at m_gen is
reconstructed at m_reco. A diagonal y=x is drawn. The width of the band AROUND the
diagonal is the mass resolution (the off-diagonal that a JMS/JMR scale factor cannot
remove for data); the on-diagonal value is the per-bin purity.

Built from the per-jet matched ntuple on the analysis gen-mass grid (reco projected
onto the same grid). pT is held on its diagonal block (gen & reco pT in the same
analysis bin) so this is the pure MASS sub-response. One square panel per pT bin.

Input: ~/Downloads/mass_diagnostic_ntuple_pythia_2018.pkl
       ~/Downloads/minimal_nlo_ptz_2018.pkl   (analysis mgen edges)
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

hep.style.use(hep.style.CMS)

D = os.path.expanduser("~/Downloads")
OUT = os.path.expanduser("~/Projects/smp_jetmass_run2/review/figs/nlo_deck")
NT = "mass_diagnostic_ntuple_pythia_2018.pkl"
PT_BINS = [(200, 290, "$200 < p_{T} < 290$ GeV", "pt1"),
           (290, 400, "$290 < p_{T} < 400$ GeV", "pt2"),
           (400, 1e9, "$p_{T} > 400$ GeV",       "pt3")]
GROOM = {"g": ("groomed soft-drop mass", "msoftdrop", "gen_msoftdrop", 150.0),
         "u": ("ungroomed mass",         "mass",      "gen_mass",      200.0)}


def cols():
    o = pickle.load(open(os.path.join(D, NT), "rb"))["reco_jet_ntuple"]
    return {k: np.asarray(o[k].value if hasattr(o[k], "value") else o[k])
            for k in ("pt", "gen_pt", "msoftdrop", "gen_msoftdrop",
                      "mass", "gen_mass", "weight", "passes_both")}


def main():
    mgen = pickle.load(open(os.path.join(D, "minimal_nlo_ptz_2018.pkl"), "rb"))[
        "response_matrix_g"].axes["mgen"].edges
    c = cols()

    for gkey, (gname, rcol, gcol, mtop) in GROOM.items():
        edges = mgen[mgen <= mtop]
        nb = len(edges) - 1
        for plo, phi, ptlabel, pttag in PT_BINS:
            sel = (c["passes_both"].astype(bool) & (c[gcol] > 0) & (c[rcol] > 0)
                   & (c["pt"] >= plo) & (c["pt"] < phi)
                   & (c["gen_pt"] >= plo) & (c["gen_pt"] < phi))
            gen, reco, w = c[gcol][sel], c[rcol][sel], c["weight"][sel]
            gi = np.digitize(gen, edges) - 1
            ri = np.digitize(reco, edges) - 1
            ok = (gi >= 0) & (gi < nb) & (ri >= 0) & (ri < nb)
            M = np.zeros((nb, nb))                       # M[reco, gen]
            np.add.at(M, (ri[ok], gi[ok]), w[ok])
            colsum = M.sum(0, keepdims=True)
            P = np.divide(M, colsum, out=np.zeros_like(M), where=colsum > 0)

            fig, ax = plt.subplots(layout="constrained")
            pcm = ax.pcolormesh(edges, edges, P, cmap="cividis",
                                norm=mcolors.LogNorm(vmin=1e-3, vmax=1.0))
            ax.plot([0, mtop], [0, mtop], color="#e42536", lw=1.6, ls="--")
            cb = fig.colorbar(pcm, ax=ax, pad=0.02)
            cb.set_label(r"$P(m_{reco}\,|\,m_{gen})$")
            ax.set_xlim(0, mtop); ax.set_ylim(0, mtop)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"gen {gname} $m_{{gen}}$ [GeV]")
            ax.set_ylabel(rf"reco {gname} $m_{{reco}}$ [GeV]")
            hep.cms.label("", data=False, loc=0, rlabel="(2018, 13 TeV)", ax=ax)
            ax.text(0.05, 0.88, ptlabel, ha="left", va="top", transform=ax.transAxes,
                    fontsize=17, bbox=dict(boxstyle="round", fc="white", ec="none",
                                           alpha=0.85))

            out = os.path.join(OUT, f"mass_response_{gkey}_{pttag}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
            print("wrote", out)


if __name__ == "__main__":
    main()
