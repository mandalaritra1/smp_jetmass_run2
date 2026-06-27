#!/usr/bin/env python3
"""ARC Misc#4 — break down the zjet unfolding-efficiency miss rate by reco cut.

Efficiency (Figs 40-41) = matched / (matched + misses) = P(reco sel | gen-fiducial).
We reproduce the gen + reco selections on a local DY MC file and report, among
gen-fiducial events, the marginal pass rate of each RECO cut to find what drives
the flat ~0.66 efficiency. Lepton/Z cuts are reproduced exactly (no JEC needed);
the reco-jet leg uses raw FatJet pt (approximate, JEC shifts pt a few %).

Self-validation: the reproduced combined efficiency should land near ~0.66.
"""
import sys
sys.path.insert(0, ".")

import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection

from smp_jetmass_run2.smp_utils import (
    get_z_gen_selection,
    get_z_reco_selection,
    apply_lepton_separation,
    apply_lepton_separation_gen,
)
from smp_jetmass_run2.corrections import MET_filters, HEMVeto, getRapidity

NanoAODSchema.warn_missing_crossrefs = False

FILE = "/Users/aritra/Projects/omnifold/data/nanoaod/DYJetsToLL_HT400to600_UL18_test.root"
IOV = "2018"
PTE, PTM = 40, 29  # lepptcuts [ele, mu]


def frac(mask, denom):
    d = ak.sum(denom)
    return (ak.sum(mask & denom) / d) if d > 0 else float("nan")


def main():
    ev = NanoEventsFactory.from_root(
        {FILE: "Events"}, schemaclass=NanoAODSchema, mode="eager"
    ).events()
    n = len(ev)
    print(f"Loaded {n} events from {FILE.split('/')[-1]}\n")
    sel = PackedSelection()

    # ----------------- GEN selection (faithful) -----------------
    gl = ev.GenDressedLepton
    is_e = np.abs(gl.pdgId) == 11
    is_m = np.abs(gl.pdgId) == 13
    ptcut = (is_e & (gl.pt > PTE)) | (is_m & (gl.pt > PTM))
    etacut = np.abs(gl.eta) < 2.4
    ev = ak.with_field(ev, gl[ptcut & etacut], "GenDressedLepton")

    gj = ak.with_field(ev.GenJetAK8, getRapidity(ev.GenJetAK8), "rapidity")
    ev = ak.with_field(ev, gj, "GenJetAK8")
    ev = ak.with_field(ev, ev.GenJetAK8[(ev.GenJetAK8.pt > 0) & (np.abs(ev.GenJetAK8.rapidity) < 2.4)], "GenJetAK8")
    gjc = apply_lepton_separation_gen(ev.GenJetAK8, ev.GenDressedLepton, dr_cut=0.4)
    ev = ak.with_field(ev, gjc[(gjc.pt > 0) & (np.abs(gjc.rapidity) < 2.4)], "GenJetAK8")

    sel.add("oneGenJet", ak.sum((ev.GenJetAK8.pt > 0) & (np.abs(ev.GenJetAK8.rapidity) < 2.4), axis=1) >= 1)
    sel.add("oneGenJet_pt200", ak.sum((ev.GenJetAK8.pt > 200) & (np.abs(ev.GenJetAK8.rapidity) < 2.4), axis=1) >= 1)

    z_gen = get_z_gen_selection(ev, sel, PTE, PTM, None, None)
    sel.add("z_ptcut_gen", sel.all("twoGen_leptons") & ak.fill_none(z_gen.pt > 90.0, False))
    sel.add("z_mcut_gen", sel.all("twoGen_leptons") & ak.fill_none((z_gen.mass > 71.0) & (z_gen.mass < 111.0), False))

    # gen-fiducial denominator (reported region: gen jet pt>200)
    GEN = sel.all("twoGen_leptons", "oneGenJet_pt200", "z_ptcut_gen", "z_mcut_gen")
    gen_mm = GEN & sel.all("twoGen_mm")
    gen_ee = GEN & sel.all("twoGen_ee")
    print(f"GEN-fiducial events: {ak.sum(GEN)}  (mm={ak.sum(gen_mm)}, ee={ak.sum(gen_ee)})\n")

    # ----------------- RECO object selection (faithful leptons) -----------------
    eeta = np.abs(ev.Electron.eta)
    ev = ak.with_field(ev, ev.Electron[
        (ev.Electron.pt > PTE) & (eeta < 2.4)
        & ((eeta < 1.422) | (eeta > 1.566))
        & (ev.Electron.pfRelIso03_all < 0.25)
        & (ev.Electron.cutBased > 3)
        & (np.abs(ev.Electron.dz) < 0.5) & (np.abs(ev.Electron.dxy) < 0.2)
    ], "Electron")
    ev = ak.with_field(ev, ev.Muon[
        (ev.Muon.pt > PTM) & (np.abs(ev.Muon.eta) < 2.4)
        & (ev.Muon.pfIsoId > 3) & (ev.Muon.tightId == True)
        & (np.abs(ev.Muon.dz) < 0.5) & (np.abs(ev.Muon.dxy) < 0.2)
    ], "Muon")

    z_reco = get_z_reco_selection(ev, sel, PTE, PTM, None, None)
    has_z = sel.all("twoReco_leptons")
    sel.add("z_ptcut_reco", has_z & ak.fill_none(z_reco.pt > 90, False))
    sel.add("z_mcut_reco", has_z & ak.fill_none((z_reco.mass > 71.0) & (z_reco.mass < 111.0), False))

    # ----------------- RECO event/jet cuts -----------------
    sel.add("npv", ev.PV.npvsGood > 0)
    metmask = np.logical_and.reduce(
        np.array([ev.Flag[f] for f in MET_filters[IOV] if f in ev.Flag.fields]), axis=0
    )
    sel.add("MET", metmask)

    # approximate reco jet leg (raw pt, subjets present, lepton-cleaned, jetId, HEM)
    fj = ev.FatJet
    fj = fj[(fj.subJetIdx1 > -1) & (fj.subJetIdx2 > -1)]
    fj = apply_lepton_separation(fj, ev.Muon, ev.Electron, dr_cut=0.4)
    hem = HEMVeto(fj, ev.run)  # data-style boolean; MC year=2018 weight path not used here
    sel.add("oneRecoJet", ak.sum((fj.pt > 200) & (np.abs(fj.eta) < 2.5) & (fj.jetId == 6), axis=1) >= 1)

    # ----------------- cutflow among GEN-fiducial -----------------
    reco_cuts = ["npv", "MET", "twoReco_leptons", "z_ptcut_reco", "z_mcut_reco", "oneRecoJet"]
    print("Marginal P(reco cut | GEN-fiducial)   [all / mm / ee]")
    for c in reco_cuts:
        m = sel.all(c)
        print(f"  {c:18s}: {frac(m, GEN):.3f}   {frac(m, gen_mm):.3f}   {frac(m, gen_ee):.3f}")

    print("\nSequential cutflow (cumulative pass | GEN-fiducial)   [all / mm / ee]")
    cum = GEN
    cum_mm, cum_ee = gen_mm, gen_ee
    for c in reco_cuts:
        m = sel.all(c)
        cum = cum & m
        cum_mm = cum_mm & m
        cum_ee = cum_ee & m
        a = ak.sum(cum) / ak.sum(GEN)
        mm = ak.sum(cum_mm) / ak.sum(gen_mm) if ak.sum(gen_mm) > 0 else float("nan")
        ee = ak.sum(cum_ee) / ak.sum(gen_ee) if ak.sum(gen_ee) > 0 else float("nan")
        print(f"  + {c:18s}: {a:.3f}   {mm:.3f}   {ee:.3f}")

    # combined reco selection (no trigger for MC) ~ efficiency
    allreco = sel.all("npv", "MET", "twoReco_leptons", "z_ptcut_reco", "z_mcut_reco", "oneRecoJet")
    print(f"\n=> Reproduced efficiency P(all reco | GEN) = {frac(allreco, GEN):.3f}"
          f"  (mm={frac(allreco, gen_mm):.3f}, ee={frac(allreco, gen_ee):.3f})")
    print("   [validation: should land near ~0.66 from Figs 40-41]")

    # lepton/Z-only efficiency (drop the jet leg) to isolate the lepton side
    lepz = sel.all("twoReco_leptons", "z_ptcut_reco", "z_mcut_reco")
    print(f"\n   lepton+Z-only  P(twoReco_leptons & zcuts | GEN) = {frac(lepz, GEN):.3f}"
          f"  (mm={frac(lepz, gen_mm):.3f}, ee={frac(lepz, gen_ee):.3f})")
    print(f"   two-reco-leptons only                            = {frac(sel.all('twoReco_leptons'), GEN):.3f}")


if __name__ == "__main__":
    main()
