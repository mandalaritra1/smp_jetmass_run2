#!/usr/bin/env python
"""Quark/gluon content of the measured jets, from the minimal_rho productions.

Reads `ptjet_rhojet_g_reco_flav` (matched final_seq jets split by the matched
gen jet's partonFlavour, nominal weights) from a minimal_rho output pickle,
applies the processor's own postprocess (xs*lumi/sumw per dataset -- the saved
CLI pickles are raw), and prints the flavour fractions per reco pt bin, plus
optionally per (pt, rho_g) bin.

    python scripts/qg_content.py outputs/minimal_rho_dijet_mg_pythia8_2018.pkl --channel dijet
    python scripts/qg_content.py ... --channel trijet --rho
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", type=Path)
    ap.add_argument("--channel", choices=["dijet", "trijet"], required=True)
    ap.add_argument("--rho", action="store_true",
                    help="also print gluon fraction per (pt, rho_g) bin")
    args = ap.parse_args(argv)

    out = pickle.load(open(args.pkl, "rb"))
    if args.channel == "dijet":
        from smp_jetmass_run2.dijet_processor import DijetProcessor as P
    else:
        from smp_jetmass_run2.trijet_processor import TrijetProcessor as P
    #### the saved CLI pickles are NOT postprocessed; reuse the processor's own
    #### xs scaling so fractions combine the HT bins correctly
    out = P(do_gen=True, mode="minimal_rho", jet_systematics=["nominal"],
            systematics=[]).postprocess(out)

    h = out["ptjet_rhojet_g_reco_flav"]
    flavs = list(h.axes["partonFlav"])
    hsum = h[{"dataset": sum}]
    pt_edges = hsum.axes["ptreco"].edges

    per_flav = {fl: hsum[{"partonFlav": fl}].project("ptreco").values(flow=True)
                for fl in flavs}
    tot = np.sum(list(per_flav.values()), axis=0)

    meas = "jets 1+2" if args.channel == "dijet" else "jet 3"
    print(f"\n=== {args.channel} ({meas}) flavour fractions per reco pT bin "
          f"(xs-weighted, matched jets) ===")
    print(f"{'pT bin':>14s} " + "".join(f"{fl:>10s}" for fl in flavs) + f" {'wsum':>12s}")
    #### flow=True prepends underflow; bin i in [1, n] maps to edges[i-1:i]
    for i in range(1, len(pt_edges)):
        if tot[i] <= 0:
            continue
        row = "".join(f"{per_flav[fl][i]/tot[i]:>10.3f}" for fl in flavs)
        print(f"{pt_edges[i-1]:>6.0f}-{pt_edges[i]:<7.0f} {row} {tot[i]:>12.4g}")

    if args.rho:
        g2 = hsum[{"partonFlav": "Gluon"}].values(flow=False)
        t2 = sum(hsum[{"partonFlav": fl}].values(flow=False) for fl in flavs)
        rho_edges = hsum.axes["mpt_reco"].edges
        print(f"\n--- gluon fraction per (pt, rho_g) bin ---")
        hdr = " ".join(f"[{rho_edges[j]:.2f},{rho_edges[j+1]:.2f}]"
                       for j in range(len(rho_edges) - 1))
        print(f"{'pT bin':>14s}  rho bins: {hdr}")
        with np.errstate(invalid="ignore"):
            frac = np.where(t2 > 0, g2 / t2, np.nan)
        for i in range(len(pt_edges) - 1):
            if np.all(~np.isfinite(frac[i])):
                continue
            row = " ".join(f"{v:.2f}" if np.isfinite(v) else "  - " for v in frac[i])
            print(f"{pt_edges[i]:>6.0f}-{pt_edges[i+1]:<7.0f} {row}")


if __name__ == "__main__":
    main()
