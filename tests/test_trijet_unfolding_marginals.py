"""Regression test for the trijet unfolding marginal convention.

The response matrix contains matched events only, but the detector- and
particle-level marginals must be inclusive:

  reco = every event passing recoTot_seq (fakes included)
  gen  = every event passing genTot_seq  (misses included)

``rho_skim`` writes those inclusive selections to flat tables.  This test runs
one local NanoAOD chunk, rehistograms the tables on the production axes, and
requires exact agreement with the nominal groomed-rho marginals.  A
registration/axis-parity test cannot catch this fill-level contract.

Run standalone:
    PYTHONPATH=. <venv>/bin/python tests/test_trijet_unfolding_marginals.py
"""
from pathlib import Path

import hist
import numpy as np
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from smp_jetmass_run2.trijet_processor import TrijetProcessor


LOCAL_MC = Path(
    "/Users/aritra/Projects/GluonJetMass/test_files/store/mc/"
    "RunIISummer20UL18NanoAODv9/"
    "QCD_Pt_170to300_TuneCP5_13TeV_pythia8/NANOAODSIM/"
    "106X_upgrade2018_realistic_v16_L1v1-v1/120000/"
    "5C540F1F-6B0C-1047-B020-539529AB3BB6.root"
)
DATASET = "QCD_Pt_170to300_TuneCP5_13TeV_pythia8_RunIISummer20UL18"
ENTRY_STOP = 50_000
JET_RADIUS = 0.8


def _run_local_chunk():
    if not LOCAL_MC.exists():
        raise FileNotFoundError(
            f"Local regression input is missing: {LOCAL_MC}\n"
            "Use the GluonJetMass NanoAODv9 test-file setup documented in "
            "this repository's AGENTS.md."
        )
    events = NanoEventsFactory.from_root(
        {str(LOCAL_MC): "Events"},
        schemaclass=NanoAODSchema,
        metadata={"dataset": DATASET, "filename": str(LOCAL_MC)},
        entry_stop=ENTRY_STOP,
        mode="eager",
    ).events()
    processor = TrijetProcessor(
        do_gen=True,
        mode="rho_skim",
        jet_systematics=["nominal"],
        systematics=[],
    )
    return processor.process(events)


def _nominal_projection(output, name):
    return output[name][{"dataset": hist.sum, "systematic": "nominal"}]


def _rehistogram_skim(output, table, template):
    skim = output["skim"][table]
    if table == "reco":
        pt_name, mass_name, rho_name = "ptreco", "mreco_g", "mpt_reco"
    elif table == "gen":
        pt_name, mass_name, rho_name = "ptgen", "mgen_g", "mpt_gen"
    else:
        raise ValueError(f"Unsupported skim table: {table}")

    pt = np.asarray(skim[pt_name].value, dtype=float)
    mass = np.asarray(skim[mass_name].value, dtype=float)
    weight = np.asarray(skim["weight"].value, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 2.0 * np.log10(mass / (pt * JET_RADIUS))
    # +/-inf rho values are legitimate under/overflow entries in hist; only NaN
    # is dropped by the production fill.
    valid = ~np.isnan(pt) & ~np.isnan(rho) & ~np.isnan(weight)

    expected = hist.Hist(*template.axes, storage=hist.storage.Weight())
    expected.fill(
        **{pt_name: pt[valid], rho_name: rho[valid]},
        weight=weight[valid],
    )
    return expected


def _assert_weighted_hist_equal(actual, expected, label):
    # The skim intentionally stores float32 kinematics, so exact flow-bin
    # placement at +/-inf can differ from the full-precision processor arrays.
    # Unfolding consumes the finite production bins, which must agree exactly.
    np.testing.assert_allclose(
        actual.values(flow=False),
        expected.values(flow=False),
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"{label} bin contents do not match the inclusive skim",
    )
    np.testing.assert_allclose(
        actual.variances(flow=False),
        expected.variances(flow=False),
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"{label} sumw2 does not match the inclusive skim",
    )


def test_trijet_unfolding_marginals_are_inclusive():
    output = _run_local_chunk()
    for table, hist_name in (
        ("reco", "ptjet_rhojet_g_reco"),
        ("gen", "ptjet_rhojet_g_gen"),
    ):
        actual = _nominal_projection(output, hist_name)
        expected = _rehistogram_skim(output, table, actual)
        _assert_weighted_hist_equal(actual, expected, table)


if __name__ == "__main__":
    test_trijet_unfolding_marginals_are_inclusive()
    print(
        "OK: trijet nominal reco/gen groomed-rho marginals exactly match "
        "the inclusive rho_skim tables."
    )
