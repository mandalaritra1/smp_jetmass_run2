#!/usr/bin/env python3
"""ARC Misc#4 — pie/table breakdown of WHERE the unfolding efficiency is lost.

Categorises every gen-fiducial event by its (mutually exclusive) fate and shows
the composition of the ~37% inefficiency. Reuses the validated selection in
make_event_displays.build().
"""
import os
import sys
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling import

import awkward as ak
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from make_event_displays import build, FILE

NanoAODSchema.warn_missing_crossrefs = False
OUT = "review/figs"


def main():
    ev = NanoEventsFactory.from_root({FILE: "Events"}, schemaclass=NanoAODSchema, mode="eager").events()
    ev, sel, z_gen, z_reco = build(ev)

    genfid = ak.to_numpy(sel.all("kinsel_gen", "jet200"))
    is_mm = ak.to_numpy(ak.fill_none(np.abs(ak.firsts(ev.GenDressedLepton.pdgId)) == 13, False))
    okevt = ak.to_numpy(sel.all("npv", "MET"))
    twolep = ak.to_numpy(sel.all("twoReco_leptons"))
    zcut = ak.to_numpy(sel.all("z_ptcut_reco", "z_mcut_reco"))
    jet = ak.to_numpy(sel.all("oneRecoJet"))

    # mutually-exclusive fate among gen-fiducial events (priority order)
    cats = {
        "Passes (matched)":              genfid & okevt & twolep & zcut & jet,
        "MET / npv":                     genfid & ~okevt,
        "Lepton ID/iso — ee":            genfid & okevt & ~twolep & ~is_mm,
        "Lepton ID/iso — μμ":  genfid & okevt & ~twolep & is_mm,
        "Z pT/mass window":              genfid & okevt & twolep & ~zcut,
        "Reco jet":                      genfid & okevt & twolep & zcut & ~jet,
    }
    N = genfid.sum()
    counts = {k: int(v.sum()) for k, v in cats.items()}
    assert sum(counts.values()) == N, (sum(counts.values()), N)

    print(f"gen-fiducial events: {N}")
    print(f"{'category':24s} {'N':>6s} {'% of gen-fid':>13s} {'% of loss':>11s}")
    loss = N - counts["Passes (matched)"]
    for k, c in counts.items():
        pl = "" if k == "Passes (matched)" else f"{100*c/loss:11.1f}"
        print(f"  {k:22s} {c:6d} {100*c/N:12.1f} {pl:>11s}")
    print(f"\nefficiency = {counts['Passes (matched)']/N:.3f}")

    # ---- pie: fate of all gen-fiducial events ----
    labels = list(cats.keys())
    vals = [counts[k] for k in labels]
    colors = ["#4caf50", "#9e9e9e", "#1f77b4", "#d62728", "#ff7f0e", "#9467bd"]
    explode = [0.0, 0.05, 0.08, 0.08, 0.05, 0.05]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    wedges, _ = ax.pie(vals, colors=colors, explode=explode, startangle=90,
                       counterclock=False, wedgeprops=dict(width=0.45, edgecolor="white"))
    eff = counts["Passes (matched)"] / N
    ax.text(0, 0, f"efficiency\n{eff:.0%}", ha="center", va="center", fontsize=18, fontweight="bold")
    leg_labels = [f"{k}  —  {100*counts[k]/N:.1f}%" for k in labels]
    ax.legend(wedges, leg_labels, loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=11, frameon=False)
    ax.set_title("Where the Z+jet unfolding efficiency is lost\n"
                 "(fate of gen-fiducial events; DY MC, 200<$p_T$<$\\infty$)", fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.62, top=0.86, bottom=0.04)
    os.makedirs(OUT, exist_ok=True)
    fout = f"{OUT}/efficiency_loss_pie.png"
    fig.savefig(fout, dpi=140)
    print(f"wrote {fout}")


if __name__ == "__main__":
    main()
