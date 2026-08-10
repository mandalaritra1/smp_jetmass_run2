#!/usr/bin/env python
"""Regenerate every MET / HEM / AK4 validation plot from the data + MC
validation-mode pickles. Run after a validation re-run:

    python scripts/make_met_hem_plots.py \
        --data outputs/validation_data_2018.pkl \
        --mc   outputs/validation_pythia_2018.pkl --era 2018

All figures land in outputs/plots/ (PDF, CMS style). Each plot is wrapped so a
missing hist (e.g. an older pickle) skips that one instead of aborting the batch.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smp_jetmass_run2 import plot_utils as pu


def _run(label, fn, *a, **k):
    try:
        fn(*a, **k)
        plt.close("all")
        print(f"  [ok]   {label}")
    except Exception as e:
        plt.close("all")
        print(f"  [skip] {label}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data validation pickle")
    ap.add_argument("--mc", required=True, help="MC validation pickle")
    ap.add_argument("--era", default="2018")
    args = ap.parse_args()

    era = args.era
    data_out = pu.load_out(era, args.data)   # load_out.format() is a no-op w/o {era}
    mc_out = pu.load_out(era, args.mc)
    print(f"loaded data={args.data}  mc={args.mc}  era={era}\nwriting to outputs/plots/ ...")

    # --- MET data/MC: raw, xy-corrected, AK4-HEM-vetoed ---
    for var in ("met_pt", "met_phi", "met_pt_xy", "met_phi_xy",
                "met_pt_ak4veto", "met_phi_ak4veto"):
        _run(f"met_data_mc {var}", pu.plot_met_data_mc, data_out, mc_out, var=var, era=era)

    # --- AK4-in-HEM occupancy + phi ratio ---
    _run("ak4_phi_hem", pu.plot_ak4_phi_hem, data_out, mc_out, era=era)
    _run("ak4_etaphi data", pu.plot_ak4_etaphi, data_out, era=era, data=True)
    _run("ak4_etaphi mc", pu.plot_ak4_etaphi, mc_out, era=era, data=False)

    # --- before/after AK4-HEM veto (points on points, "no change") ---
    for obs in ("met_pt", "met_phi", "mass_jet0"):
        _run(f"veto_compare {obs}", pu.plot_veto_compare, data_out, obs=obs, era=era, data=True)

    # --- MET vs jet pt (resolution justification) ---
    _run("met_vs_jetpt profile", pu.plot_met_vs_jetpt, data_out, mc_out, era=era)
    _run("met_jetpt_2d data", pu.plot_met_jetpt_2d, data_out, era=era, data=True)
    _run("met_jetpt_2d mc", pu.plot_met_jetpt_2d, mc_out, era=era, data=False)

    # --- jet mass insensitive to MET (no-cut justification) ---
    _run("mass_metsplit data", pu.plot_mass_metsplit, data_out, era=era, data=True)
    _run("mass_metsplit mc", pu.plot_mass_metsplit, mc_out, era=era, data=False)

    # --- jet mass data/MC, before and after the AK4-HEM veto ---
    _run("mass_jet0 data/mc", pu.plot_validation_data_mc, data_out, mc_out,
         "mass_jet0", "mass", r"Ungroomed jet mass [GeV]", era=era)
    _run("mass_jet0_ak4veto data/mc", pu.plot_validation_data_mc, data_out, mc_out,
         "mass_jet0_ak4veto", "mass", r"Ungroomed jet mass [GeV], AK4-HEM veto", era=era)

    print("done.")


if __name__ == "__main__":
    main()
