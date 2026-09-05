"""Exercise real hadronic process() selections across chunk partitions.

Run with PYTHONPATH=. <GluonJetMass-venv>/bin/python tests/test_hadronic_chunk_partition.py.
Synthetic Awkward events use Coffea Lorentz-vector behavior. Only detector
correction/weight providers are deterministic substitutes; selection, matching,
grooming, weight composition, and histogram filling are production code.
Both mass and rho paths check bin contents and sumw2 including flow bins.
This is a control-flow regression, not a calibration or production-impact test.
"""
import contextlib
import importlib
import io
import unittest
from unittest.mock import patch

import awkward as ak
import hist
import numpy as np
from coffea.nanoevents.methods import base, vector

DATASET = "QCD_Pt_300to470_TuneCP5_13TeV_pythia8_RunIISummer20UL17"


def vectors(rows):
    return ak.with_name(ak.Array(rows), "PtEtaPhiMLorentzVector",
                        behavior=vector.behavior)


def make_events(channel, empty_collections=False):
    """Miss, fake, matched and neither-selected events with unequal weights."""
    phis = [0., 3.] if channel == "dijet" else [0., 2., -2.]
    jets = [dict(pt=450.-50.*i, eta=0.1, phi=phi, mass=60.,
                 partonFlavour=21) for i, phi in enumerate(phis)]
    gen = vectors([jets for _ in range(4)])
    reco = vectors([[dict(j, jetId=6, subJetIdx1=2*i,
                         subJetIdx2=2*i+1, msoftdrop=30.)
                     for i, j in enumerate(jets)] for _ in range(4)])
    # Failure is a genuine independent pt-fiducial failure, not a mocked mask.
    gen = ak.with_field(gen, ak.where(ak.Array([False, True, False, True]),
                                     gen.pt * 0.1, gen.pt), "pt")
    reco = ak.with_field(reco, ak.where(ak.Array([True, False, False, True]),
                                       reco.pt * 0.1, reco.pt), "pt")
    reco = ak.with_field(reco, gen, "matched_gen")
    subjets = vectors([[dict(pt=j["pt"] / 2., eta=j["eta"],
                             phi=j["phi"] + offset, mass=10.)
                       for j in jets for offset in (-0.05, 0.05)] for _ in range(4)])
    gensubjets = subjets
    if empty_collections:
        reco = reco[ak.broadcast_arrays(ak.Array([False, True, True, False]), reco.pt)[0]]
        subjets = subjets[ak.broadcast_arrays(ak.Array([False, True, True, False]), subjets.pt)[0]]
        gen = gen[ak.broadcast_arrays(ak.Array([True, False, True, False]), gen.pt)[0]]
        gensubjets = gensubjets[ak.broadcast_arrays(ak.Array([True, False, True, False]), gensubjets.pt)[0]]
        # Retain an option-typed matching field when the gen collection is absent.
        reco = ak.with_field(reco, ak.mask(reco.matched_gen,
            ak.broadcast_arrays(ak.Array([True, False, True, False]), reco.pt)[0]), "matched_gen")
    behavior = {**base.behavior, **vector.behavior}
    events = ak.zip({
        "FatJet": reco, "SubJet": subjets, "GenJetAK8": gen,
        "GenJet": gen, "SubGenJetAK8": gensubjets,
        "Muon": vectors([[dict(pt=10., eta=4., phi=1., mass=.1,
                                mediumId=1, pfIsoId=3)] for _ in range(4)]),
        "PV": ak.Array([dict(npvsGood=1)] * 4),
        "Flag": ak.Array([dict(goodVertices=True)] * 4),
        "run": np.ones(4, dtype=np.int64), "event": np.arange(4),
        "luminosityBlock": np.ones(4, dtype=np.int64),
        "genWeight": np.array([2., 3., 5., -1.]),
    }, depth_limit=1, with_name="NanoEvents", behavior=behavior)
    return ak.with_parameter(events, "metadata", {
        "dataset": DATASET,
        "filename": "/store/mc/RunIISummer20UL17/QCD_Pt_300to470/file.root",
    })


def weight_triplet(events, *args, **kwargs):
    ones = np.ones(len(events))
    return ones, 1.2 * ones, 0.8 * ones


def mass_scale(iov, jets, var="nominal"):
    # Later JMSUp must produce a distinct response mass-bin population.
    factor = 1.5 if var == "up" else 1.
    return ak.with_field(jets, jets.mass * factor, "mass")


def run_processor(channel, events, mode="minimal"):
    module = importlib.import_module(f"smp_jetmass_run2.{channel}_processor")
    cls = getattr(module, "DijetProcessor" if channel == "dijet" else "TrijetProcessor")
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(module, "GetJetCorrections", side_effect=lambda jets, *a, **k: jets))
        stack.enter_context(patch.object(module, "applyjmsSF", side_effect=mass_scale))
        stack.enter_context(patch.object(module, "applyjmrSF", side_effect=lambda iov, jets, **k: jets))
        stack.enter_context(patch.object(module, "get_pu_weights", side_effect=weight_triplet))
        stack.enter_context(patch.object(module, "GetLumiUnc", side_effect=weight_triplet))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        instance = cls(do_gen=True, mode=mode, jet_systematics=["nominal", "JMSUp"],
                       systematics=["puUp"])
        return instance.process(events)


class HadronicChunkPartitionTest(unittest.TestCase):
    def check_partition(self, channel, empty, mode="minimal"):
        events = make_events(channel, empty)
        combined = run_processor(channel, events, mode)
        separated = [run_processor(channel, events[i:i+1], mode) for i in range(len(events))]
        multiplicity = 2 if channel == "dijet" else 1
        observable = "rhojet" if mode == "minimal_rho" else "mjet"
        response = "response_matrix_rho" if mode == "minimal_rho" else "response_matrix"
        for systematic in ("nominal", "puUp", "JMSUp"):
            for suffix in ("u", "g"):
                for family, expected, variance in ((f"ptjet_{observable}_{suffix}_gen", 7., 29.),
                                         (f"ptjet_{observable}_{suffix}_reco", 8., 34.),
                                         (f"{response}_{suffix}", 5., 25.)):
                    with self.subTest(channel=channel, empty=empty, mode=mode,
                                      systematic=systematic, hist=family):
                        selected = {"dataset": hist.sum, "systematic": systematic}
                        actual = combined[family][selected]
                        merged = sum((part[family] for part in separated[1:]),
                                     separated[0][family].copy())[selected]
                        np.testing.assert_allclose(actual.values(flow=True),
                            merged.values(flow=True), rtol=0, atol=1e-12)
                        np.testing.assert_allclose(actual.variances(flow=True),
                            merged.variances(flow=True), rtol=0, atol=1e-12)
                        scale = 1.2 if systematic == "puUp" and not family.endswith("gen") else 1.
                        self.assertAlmostEqual(actual.sum(flow=True).value, expected * multiplicity * scale)
                        self.assertAlmostEqual(actual.sum(flow=True).variance, variance * multiplicity * scale**2)
        nominal_response = combined[f"{response}_u"][{
            "dataset": hist.sum, "systematic": "nominal"}]
        shifted_response = combined[f"{response}_u"][{
            "dataset": hist.sum, "systematic": "JMSUp"}]
        self.assertFalse(np.array_equal(nominal_response.values(flow=True),
                                        shifted_response.values(flow=True)),
                         "The later jet variation must exercise different mass bins")
        self.assertAlmostEqual(combined["sumw"][DATASET], 9.)
        self.assertAlmostEqual(sum(part["sumw"][DATASET] for part in separated), 9.)

    def test_rho_partitions(self):
        for channel in ("dijet", "trijet"):
            for empty in (False, True):
                with self.subTest(channel=channel, empty=empty):
                    self.check_partition(channel, empty, "minimal_rho")

    def test_failed_met_retains_gen(self):
        for channel in ("dijet", "trijet"):
            events = make_events(channel)[2:3]
            events = ak.with_field(events, ak.Array([dict(goodVertices=False)]), "Flag")
            output = run_processor(channel, events)
            multiplicity = 2 if channel == "dijet" else 1
            for systematic in ("nominal", "puUp", "JMSUp"):
                with self.subTest(channel=channel, systematic=systematic):
                    truth = output["ptjet_mjet_u_gen"][{"dataset": hist.sum, "systematic": systematic}]
                    self.assertAlmostEqual(truth.sum(flow=True).value, 5. * multiplicity)
            self.assertEqual(output["ptjet_mjet_u_reco"].sum(flow=True).value, 0.)
            self.assertEqual(output["response_matrix_u"].sum(flow=True).value, 0.)

    def test_dijet_partition(self):
        self.check_partition("dijet", False)

    def test_trijet_partition(self):
        self.check_partition("trijet", False)

    def test_dijet_empty_collections(self):
        self.check_partition("dijet", True)

    def test_trijet_empty_collections(self):
        self.check_partition("trijet", True)


if __name__ == "__main__":
    unittest.main()
