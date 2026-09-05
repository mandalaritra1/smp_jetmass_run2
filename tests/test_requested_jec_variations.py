"""Check that constructing fewer JES variations preserves requested jet values.

Uses the real local NanoAOD fixtures and packaged UL corrections. Run with
PYTHONPATH=. <GluonJetMass-venv>/bin/python tests/test_requested_jec_variations.py.
"""
from pathlib import Path
import unittest
import warnings

import awkward as ak
import numpy as np
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from smp_jetmass_run2.corrections import GetJetCorrections, requested_jes_sources


FIXTURES = Path.home() / "Projects/GluonJetMass/test_files/store"


def corrected_jets(iov, mode="AK8", data=False, sources=None):
    kind = "data" if data else "mc"
    files = sorted((FIXTURES / kind).rglob("*.root"))
    if not files:
        raise FileNotFoundError(f"Stage the verify-processors {kind} fixture under {FIXTURES}")
    events = NanoEventsFactory.from_root(
        {str(files[0]): "Events"}, schemaclass=NanoAODSchema,
        entry_stop=1000, mode="virtual",
    ).events()
    jets = events.FatJet if mode == "AK8" else events.SubJet
    return GetJetCorrections(jets, events, "Run2018A" if data else None,
                             iov, isData=data, mode=mode, uncertainties=sources)


def assert_kinematics_equal(left, right):
    for field in ("pt", "mass"):
        # Keep option masks as well as numerical values in the comparison.
        np.testing.assert_array_equal(ak.to_numpy(ak.num(left[field])),
                                      ak.to_numpy(ak.num(right[field])))
        assert ak.to_list(ak.is_none(left[field], axis=-1)) == ak.to_list(
            ak.is_none(right[field], axis=-1))
        np.testing.assert_array_equal(
            ak.to_numpy(ak.flatten(ak.fill_none(left[field], np.nan))),
            ak.to_numpy(ak.flatten(ak.fill_none(right[field], np.nan))),
        )


class RequestedJECVariationsTest(unittest.TestCase):
    def test_down_only_and_duplicate_sources(self):
        self.assertEqual(requested_jes_sources([
            "nominal", "JERUp", "JMSDown", "JES_RelativeJEREC1Down",
            "JES_AbsoluteScaleUp", "JES_AbsoluteScaleDown",
        ]), ("RelativeJEREC1", "AbsoluteScale"))

    def test_mc_requested_kinematics(self):
        for iov, mode in [(iov, "AK8") for iov in
                          ("2016APV", "2016", "2017", "2018")] + [("2018", "AK4")]:
            with self.subTest(iov=iov, mode=mode):
                full = corrected_jets(iov, mode)
                nominal = corrected_jets(iov, mode, sources=())
                subset = corrected_jets(iov, mode, sources=("RelativeJEREC1",))
                assert_kinematics_equal(full, nominal)
                assert_kinematics_equal(full, subset)
                self.assertFalse(any(name.startswith("JES_") for name in nominal.fields))
                self.assertEqual([name for name in subset.fields if name.startswith("JES_")],
                                 ["JES_RelativeJEREC1"])
                for direction in ("up", "down"):
                    assert_kinematics_equal(full.JER[direction], nominal.JER[direction])
                    assert_kinematics_equal(full.JER[direction], subset.JER[direction])
                    assert_kinematics_equal(full.JES_RelativeJEREC1[direction],
                                           subset.JES_RelativeJEREC1[direction])

    def test_data_kinematics(self):
        assert_kinematics_equal(corrected_jets("2018", data=True),
                               corrected_jets("2018", data=True, sources=()))

    def test_total_uncertainty_without_split_table(self):
        combined = corrected_jets("2018", sources=("jes", "RelativeJEREC1"))
        total_only = corrected_jets("2018", sources=("jes",))
        assert_kinematics_equal(combined, total_only)
        total_fields = [name for name in total_only.fields if name.startswith("JES_")]
        self.assertEqual(len(total_fields), 1)
        for direction in ("up", "down"):
            assert_kinematics_equal(combined[total_fields[0]][direction],
                                   total_only[total_fields[0]][direction])


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    unittest.main()
