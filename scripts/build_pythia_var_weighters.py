#!/usr/bin/env python3
"""Build 24-bin CR/hadronization weighters (w = variation/nom) from the
pythia_var gen-rho cache, in PtVarWeighter format, alongside the Vincia one.
-> smp_jetmass_run2/corrections/{cr1,cr2,fraghard,fragsoft}_rho_reweight_{groomed,ungroomed}.npz
"""
import pickle
from pathlib import Path
import numpy as np

CACHE = Path("/private/tmp/claude-501/-Users-aritra-Projects-smp-jetmass-run2/844f4080-8bec-48e3-b1b7-5db08922b718/scratchpad/pythia_var_cache.npz")
FINE = Path.home() / "Projects/unfold/inputs/zjet/rho/finebins/minimal_rho_fine_pythia_2018.pkl"
C = Path.home() / "Projects/smp_jetmass_run2/smp_jetmass_run2/corrections"
PT = np.array([200., 290., 400., 13000.])
SOURCES = ["cr1", "cr2", "fraghard", "fragsoft"]

z = np.load(CACHE)
# 24-bin gen edges (48-bin fine axis merged in pairs), same as the Vincia weighter
e48 = np.asarray(pickle.load(open(FINE, "rb"))["ptjet_rhojet_g_gen"].axes[2].edges)
e24 = e48[::2]


def shape(v, mode, pt, edges):
    rho = z[f"{v}_rho_{mode}"]; w = z[f"{v}_weight"]; jp = z[f"{v}_jet_pt"]
    m = (jp >= PT[pt]) & (jp < PT[pt + 1]) & np.isfinite(rho)
    h, _ = np.histogram(rho[m], bins=edges, weights=w[m])
    c, _ = np.histogram(rho[m], bins=edges)
    return h, c


for src in SOURCES:
    for mode, tag in [("g", "groomed"), ("u", "ungroomed")]:
        rg = np.empty(4, dtype=object); wg = np.empty(4, dtype=object)
        rg[0] = np.array([e24[0], e24[-1]]); wg[0] = np.array([1.0, 1.0])
        for pt in range(3):
            hn, _ = shape("nom", mode, pt, e24)
            hv, cv = shape(src, mode, pt, e24)
            nn = hn / hn.sum(); vn = hv / hv.sum()
            w = np.ones_like(nn); good = (nn > 1e-4) & (cv >= 25)
            w[good] = np.clip(vn[good] / nn[good], 0.2, 5.0)
            rg[pt + 1] = np.repeat(e24, 2)[1:-1]; wg[pt + 1] = np.repeat(w, 2)
        fn = C / f"{src}_rho_reweight_{tag}.npz"
        np.savez(fn, pt_edges=np.array([0., 200., 290., 400., 13000.]),
                 rho_grids=rg, w_grids=wg, source=f"{src}_over_nom_pythia_var_24bin")
        print("wrote", fn.name)
