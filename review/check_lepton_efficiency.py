#!/usr/bin/env python3
"""ARC Misc#4 follow-up — decompose the twoReco_leptons inefficiency.

The unfolding efficiency ~0.66 is driven by twoReco_leptons (ee~0.51, mm~0.79).
Here we find WHICH sub-cut drives it: N-1 (drop one cut at a time) + a reco-pT
threshold scan, per channel, with the gen-fiducial denominator fixed (gen leptons
already pass pt>40 (e) / >29 (mu), |eta|<2.4).

twoReco selection is reimplemented inline as ">=2 leptons passing the cuts, the
leading two opposite-sign" (matches the production ==2 selection closely for DY)
so each cut can be toggled. Baseline should reproduce ee~0.51, mm~0.79.
"""
import sys
sys.path.insert(0, ".")

import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection

from smp_jetmass_run2.smp_utils import get_z_gen_selection, apply_lepton_separation_gen
from smp_jetmass_run2.corrections import getRapidity, MET_filters

NanoAODSchema.warn_missing_crossrefs = False
FILE = "/Users/aritra/Projects/omnifold/data/nanoaod/DYJetsToLL_HT400to600_UL18_test.root"
PTE, PTM = 40, 29


def gen_fiducial(ev):
    sel = PackedSelection()
    gl = ev.GenDressedLepton
    is_e, is_m = np.abs(gl.pdgId) == 11, np.abs(gl.pdgId) == 13
    ptcut = (is_e & (gl.pt > PTE)) | (is_m & (gl.pt > PTM))
    ev = ak.with_field(ev, gl[ptcut & (np.abs(gl.eta) < 2.4)], "GenDressedLepton")
    gj = ak.with_field(ev.GenJetAK8, getRapidity(ev.GenJetAK8), "rapidity")
    ev = ak.with_field(ev, gj[(gj.pt > 0) & (np.abs(gj.rapidity) < 2.4)], "GenJetAK8")
    gjc = apply_lepton_separation_gen(ev.GenJetAK8, ev.GenDressedLepton, dr_cut=0.4)
    ev = ak.with_field(ev, gjc[(gjc.pt > 0) & (np.abs(gjc.rapidity) < 2.4)], "GenJetAK8")
    sel.add("jet200", ak.sum((ev.GenJetAK8.pt > 200) & (np.abs(ev.GenJetAK8.rapidity) < 2.4), axis=1) >= 1)
    z = get_z_gen_selection(ev, sel, PTE, PTM, None, None)
    sel.add("zpt", sel.all("twoGen_leptons") & ak.fill_none(z.pt > 90.0, False))
    sel.add("zm", sel.all("twoGen_leptons") & ak.fill_none((z.mass > 71.0) & (z.mass < 111.0), False))
    GEN = sel.all("twoGen_leptons", "jet200", "zpt", "zm")
    return ev, GEN, GEN & sel.all("twoGen_mm"), GEN & sel.all("twoGen_ee")


def ele_cuts(e, drop=None, pt_thr=PTE, id_thr=3):
    # cutBased: 0 fail,1 veto,2 loose,3 medium,4 tight; id_thr=3 -> tight(>3)
    eta = np.abs(e.eta)
    c = {
        "pt":    e.pt > pt_thr,
        "eta":   eta < 2.4,
        "crack": (eta < 1.422) | (eta > 1.566),
        "iso":   e.pfRelIso03_all < 0.2,
        "id":    e.cutBased > id_thr,
        "dz":    np.abs(e.dz) < 0.5,
        "dxy":   np.abs(e.dxy) < 0.2,
    }
    if drop:
        c.pop(drop)
    m = c[list(c)[0]]
    for k in list(c)[1:]:
        m = m & c[k]
    return m


def mu_cuts(mu, drop=None, pt_thr=PTM, iso_thr=3, tight_id=True):
    # pfIsoId: 1 vloose,2 loose,3 medium,4 tight,...; iso_thr=3 -> tight(>3)
    c = {
        "pt":    mu.pt > pt_thr,
        "eta":   np.abs(mu.eta) < 2.4,
        "iso":   mu.pfIsoId > iso_thr,
        "id":    (mu.tightId == True) if tight_id else (mu.mediumId == True),
        "dz":    np.abs(mu.dz) < 0.5,
        "dxy":   np.abs(mu.dxy) < 0.2,
    }
    if drop:
        c.pop(drop)
    m = c[list(c)[0]]
    for k in list(c)[1:]:
        m = m & c[k]
    return m


def two_lep_pass(lep, mask):
    good = lep[mask]
    n = ak.num(good)
    lead2 = good[:, :2]
    opp = ak.fill_none(ak.sum(lead2.charge, axis=1) == 0, False)
    return (n >= 2) & opp


def eff(mask, denom):
    d = ak.sum(denom)
    return (ak.sum(mask & denom) / d) if d > 0 else float("nan")


def main():
    ev = NanoEventsFactory.from_root({FILE: "Events"}, schemaclass=NanoAODSchema, mode="eager").events()
    ev, GEN, gen_mm, gen_ee = gen_fiducial(ev)
    print(f"gen-fiducial: all={ak.sum(GEN)}  mm={ak.sum(gen_mm)}  ee={ak.sum(gen_ee)}\n")

    el, mu = ev.Electron, ev.Muon

    # ---- baseline ----
    base_ee = two_lep_pass(el, ele_cuts(el))
    base_mm = two_lep_pass(mu, mu_cuts(mu))
    print(f"BASELINE twoReco eff:  mm={eff(base_mm, gen_mm):.3f}   ee={eff(base_ee, gen_ee):.3f}")
    print("(validation vs production: mm~0.79, ee~0.51)\n")

    # ---- N-1: drop one cut at a time ----
    print("N-1 (drop one cut)  ee:")
    for cut in ["pt", "eta", "crack", "iso", "id", "dz", "dxy"]:
        m = two_lep_pass(el, ele_cuts(el, drop=cut))
        print(f"  drop {cut:6s}: ee={eff(m, gen_ee):.3f}   (Delta={eff(m, gen_ee)-eff(base_ee, gen_ee):+.3f})")
    print("N-1 (drop one cut)  mm:")
    for cut in ["pt", "eta", "iso", "id", "dz", "dxy"]:
        m = two_lep_pass(mu, mu_cuts(mu, drop=cut))
        print(f"  drop {cut:6s}: mm={eff(m, gen_mm):.3f}   (Delta={eff(m, gen_mm)-eff(base_mm, gen_mm):+.3f})")

    # ---- pT-threshold scan (reco only; gen denom fixed at >40/>29) ----
    print("\nreco-pT threshold scan  ee (gen e already >40):")
    for thr in [40, 35, 30, 25, 20, 15, 10]:
        m = two_lep_pass(el, ele_cuts(el, pt_thr=thr))
        print(f"  e pt>{thr:2d}: ee={eff(m, gen_ee):.3f}")
    print("reco-pT threshold scan  mm (gen mu already >29):")
    for thr in [29, 25, 20, 15, 10]:
        m = two_lep_pass(mu, mu_cuts(mu, pt_thr=thr))
        print(f"  mu pt>{thr:2d}: mm={eff(m, gen_mm):.3f}")

    # ---- isolate: only reco-found (>=2 leptons, no quality cuts) ----
    # cleanest "reconstructed at all" proxy: >=2 reco leptons with pt>5, |eta|<2.5
    raw_e = (el.pt > 5) & (np.abs(el.eta) < 2.5)
    raw_mu = (mu.pt > 5) & (np.abs(mu.eta) < 2.5)
    print(f"\n'>=2 reco leptons found' (pt>5,|eta|<2.5, no ID/iso):  "
          f"mm={eff(two_lep_pass(mu, raw_mu), gen_mm):.3f}   ee={eff(two_lep_pass(el, raw_e), gen_ee):.3f}")

    # ---- WP loosening scan: full reco efficiency (twoReco + Z pt/mass + jet + MET) ----
    # jet/MET/npv legs (constant across lepton WPs)
    fj = ev.FatJet
    fj = fj[(fj.subJetIdx1 > -1) & (fj.subJetIdx2 > -1)]
    jetleg = ak.sum((fj.pt > 200) & (np.abs(fj.eta) < 2.5) & (fj.jetId == 6), axis=1) >= 1
    metmask = np.logical_and.reduce(
        np.array([ev.Flag[f] for f in MET_filters["2018"] if f in ev.Flag.fields]), axis=0)
    npv = ev.PV.npvsGood > 0
    evt_leg = jetleg & metmask & npv

    def chan_pass(lep, mask):
        good = lep[mask]
        l2 = good[:, :2]
        two = (ak.num(good) >= 2) & ak.fill_none(ak.sum(l2.charge, axis=1) == 0, False)
        z = l2.sum(axis=1)
        return two & ak.fill_none((z.pt > 90) & (z.mass > 71) & (z.mass < 111), False)

    def full_eff(e_id=3, mu_iso=3, mu_tight=True):
        ee_p = chan_pass(el, ele_cuts(el, id_thr=e_id))
        mm_p = chan_pass(mu, mu_cuts(mu, iso_thr=mu_iso, tight_id=mu_tight))
        full = (ee_p | mm_p) & evt_leg
        return eff(full, GEN), eff(mm_p & evt_leg, gen_mm), eff(ee_p & evt_leg, gen_ee)

    print("\nFULL reco efficiency (twoReco + Zpt/mass + jet + MET) under looser WPs")
    print("  scenario                         all     mm      ee")
    scans = [
        ("baseline (e tight, mu isoTight)", dict()),
        ("e MEDIUM id (>2)",                dict(e_id=2)),
        ("e LOOSE id (>1)",                 dict(e_id=1)),
        ("mu iso MEDIUM (>2)",              dict(mu_iso=2)),
        ("mu iso LOOSE (>1)",               dict(mu_iso=1)),
        ("mu id MEDIUM (iso tight)",        dict(mu_tight=False)),
        ("e MED + mu iso MED",              dict(e_id=2, mu_iso=2)),
        ("e MED + mu iso MED + mu medId",   dict(e_id=2, mu_iso=2, mu_tight=False)),
    ]
    for name, kw in scans:
        a, mm, ee = full_eff(**kw)
        print(f"  {name:32s} {a:.3f}   {mm:.3f}   {ee:.3f}")


if __name__ == "__main__":
    main()
