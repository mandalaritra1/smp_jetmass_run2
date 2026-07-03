"""Minimal GEN-ONLY coffea processor for the Z+jet jet-mass shape.

Purpose: get the particle-level groomed/ungroomed jet-mass (rho) shape of an
*alternate generator* (e.g. the pre-UL Sherpa 0-4jet DY sample) at full stats on
coffea-casa, WITHOUT the full zjet_processor. It reproduces zjet_processor's GEN
block byte-for-byte by importing the same smp_utils helpers (get_z_gen_selection,
get_dphi, get_groomed_jet, apply_lepton_separation_gen), so the selection is
identical -- but it skips the reco path and the LHE/PU/PDF/Q2/PS weights (which
crash on samples with no LHE branches, e.g. Sherpa). Event weight = genWeight only,
so the output is a normalized SHAPE (do not read absolute yields).

Run it through the repo's normal casa plumbing:
    client = notebook_utils.ensure_client(casa=True, test=False, useDefault=False,
                                          executor_mode="dask-casa")
    notebook_utils.upload_package_if_casa(client, casa=True)
    run = notebook_utils.make_runner(use_dask=True, client=client,
                                     chunksize=100_000, maxchunks=None)
    out = run(fileset, SherpaGenProcessor(), treename="Events")
See tools/arc_diagnostics/run_sherpa_gen_casa.py for the full driver.
"""
import awkward as ak
import numpy as np
import hist
from coffea import processor
from coffea.analysis_tools import PackedSelection

import smp_jetmass_run2.smp_utils as smp_utils

# Analysis jet-pt bins (same as the ARC modelling figures) + shape axes.
PT_EDGES = [200.0, 290.0, 400.0, 13000.0]
LEP_PT = [40.0, 29.0]  # [ele, mu] == zjet_processor.lepptcuts


def _rapidity(obj, eps=1e-12):
    """Inlined from corrections.getRapidity (avoids importing the reco module)."""
    pt, eta, m = obj.pt, obj.eta, obj.mass
    pz = pt * np.sinh(eta)
    E = np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)
    num = ak.where(E + pz > eps, E + pz, np.nan)
    den = ak.where(E - pz > eps, E - pz, np.nan)
    return 0.5 * np.log(num / den)


class SherpaGenProcessor(processor.ProcessorABC):
    """GEN-only; fills rho_g / rho_u / mass_g histograms weighted by genWeight."""

    def __init__(self):
        ds = hist.axis.StrCategory([], name="dataset", growth=True)
        pt = hist.axis.Variable(PT_EDGES, name="pt", label=r"$p_{T}^{jet}$ [GeV]")
        self._hists = {
            "rho_g": hist.Hist(ds, pt, hist.axis.Regular(32, -9, -1, name="rho_g",
                               label=r"$\rho_g$"), storage=hist.storage.Weight()),
            "rho_u": hist.Hist(ds, pt, hist.axis.Regular(32, -9, -1, name="rho_u",
                               label=r"$\rho_u$"), storage=hist.storage.Weight()),
            "mass_g": hist.Hist(ds, pt, hist.axis.Regular(30, 0, 150, name="mass_g",
                                label=r"$m_{SD}$ [GeV]"), storage=hist.storage.Weight()),
        }

    def process(self, events):
        dataset = events.metadata["dataset"]
        out = {k: h.copy() for k, h in self._hists.items()}
        sel = PackedSelection()

        w = ak.to_numpy(events.genWeight) if "genWeight" in ak.fields(events) \
            else np.ones(len(events))

        # --- lepton pre-filter (zjet_processor 1141-1155) ---
        gdl = events.GenDressedLepton
        is_e = np.abs(gdl.pdgId) == 11
        is_m = np.abs(gdl.pdgId) == 13
        keep = ((is_e & (gdl.pt > LEP_PT[0])) | (is_m & (gdl.pt > LEP_PT[1]))) \
            & (np.abs(gdl.eta) < 2.4)
        events = ak.with_field(events, gdl[keep], "GenDressedLepton")

        # --- gen AK8: rapidity, fiducial, lepton cleaning (1158-1188) ---
        gj = ak.with_field(events.GenJetAK8, _rapidity(events.GenJetAK8), "rapidity")
        events = ak.with_field(events, gj, "GenJetAK8")
        events = ak.with_field(events, events.GenJetAK8[
            (events.GenJetAK8.pt > 0) & (np.abs(events.GenJetAK8.rapidity) < 2.4)],
            "GenJetAK8")
        gjc = smp_utils.apply_lepton_separation_gen(
            events.GenJetAK8, events.GenDressedLepton, dr_cut=0.4)
        events = ak.with_field(events, gjc[
            (gjc.pt > 0) & (np.abs(gjc.rapidity) < 2.4)], "GenJetAK8")

        sel.add("oneGenJet_pt200", ak.sum(
            (events.GenJetAK8.pt > 200) & (np.abs(events.GenJetAK8.rapidity) < 2.4),
            axis=1) >= 1)

        z = smp_utils.get_z_gen_selection(events, sel, LEP_PT[0], LEP_PT[1], None, None)
        sel.add("z_ptcut_gen", sel.all("twoGen_leptons") & ak.fill_none(z.pt > 90.0, False))
        sel.add("z_mcut_gen", sel.all("twoGen_leptons") & ak.fill_none(
            (z.mass > 71.0) & (z.mass < 111.0), False))

        lead_jet, dphi = smp_utils.get_dphi(z, events.GenJetAK8)
        groomed, _ = smp_utils.get_groomed_jet(lead_jet, events.SubGenJetAK8, False)
        asym = np.abs(z.pt - lead_jet.pt) / (z.pt + lead_jet.pt)
        sel.add("dphi", ak.fill_none(dphi > 1.57, False))
        sel.add("asym", ak.fill_none(asym < 0.3, False))

        mask = sel.all("twoGen_leptons", "oneGenJet_pt200", "z_ptcut_gen",
                       "z_mcut_gen", "dphi", "asym")
        mask = ak.to_numpy(ak.fill_none(mask, False))

        pt = ak.to_numpy(ak.fill_none(lead_jet.pt, np.nan))[mask]
        mu = ak.to_numpy(ak.fill_none(lead_jet.mass, np.nan))[mask]
        mg = ak.to_numpy(ak.fill_none(groomed.mass, np.nan))[mask]
        wm = w[mask]

        def rho(m):
            with np.errstate(divide="ignore", invalid="ignore"):
                return 2.0 * np.log10(m / (pt * 0.8))

        gu = np.isfinite(mu) & (mu > 0)
        gg = np.isfinite(mg) & (mg > 0)
        out["rho_u"].fill(dataset=dataset, pt=pt[gu], rho_u=rho(mu)[gu], weight=wm[gu])
        out["rho_g"].fill(dataset=dataset, pt=pt[gg], rho_g=rho(mg)[gg], weight=wm[gg])
        out["mass_g"].fill(dataset=dataset, pt=pt[gg], mass_g=mg[gg], weight=wm[gg])
        return out

    def postprocess(self, accumulator):
        return accumulator
