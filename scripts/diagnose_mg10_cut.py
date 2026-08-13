#!/usr/bin/env python
"""Exercise a fixed groomed-mass floor in pair-split dijet rho.

This is a read-only diagnostic over the postprocessed 2018 pair-split skims.
Their stored MC weights already include xs*lumi/sumw, so no additional sample
normalization is applied here.

The study compares the established 2 GeV floor with a candidate 10 GeV floor:

* weighted data and MC retention in every measured-jet pT bin;
* unit-area reco-level rho data/MC plots after each floor;
* joint (pT, rho) purity, stability, out-of-fiducial feed-in and inefficiency.

The fixed mass cut is imposed separately at reco and gen level. In the response
diagnostic, a gen-below-floor jet reconstructed above the floor is therefore
counted as feed-in, not as an ordinary matched signal jet.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_pairsplit import (
    C_BLUE,
    C_ORANGE,
    DATA_LABEL,
    MIN_RAW,
    PT_DIJET,
    cms_label,
    data_mc_ratio,
    draw_ratio,
    panel_text,
    paths_for,
    ptbin_tag,
    ratio_single,
    rho,
    save,
    step,
)

JET_R = 0.8
FLOORS = (2.0, 10.0)
PLOT_RHO_EDGES = np.linspace(-5.0, 0.0, 41)
RESPONSE_RHO_EDGES = np.array(
    [-10.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
)


def bin_index(values, edges):
    index = np.digitize(values, edges) - 1
    return np.where(
        np.isfinite(values) & (index >= 0) & (index < len(edges) - 1),
        index,
        -1,
    )


def mass_floor_rho(mass_floor, pt):
    return 2.0 * np.log10(mass_floor / (pt * JET_R))


def read_table(path, table, columns):
    skim = pickle.load(open(path, "rb"))["skim"][table]
    return {column: np.asarray(skim[column].value) for column in columns}


def accumulate_reco(source):
    shape = (len(PT_DIJET) - 1, len(PLOT_RHO_EDGES) - 1)
    histograms = {
        floor: {
            "sumw": np.zeros(shape),
            "sumw2": np.zeros(shape),
            "raw": np.zeros(shape),
            "yield": np.zeros(len(PT_DIJET) - 1),
        }
        for floor in FLOORS
    }
    for path in paths_for("new", "dijet", source):
        table = read_table(path, "reco", ("ptreco", "mreco_g", "weight"))
        pt = table["ptreco"]
        mass = table["mreco_g"]
        weight = table["weight"]
        pt_bin = bin_index(pt, PT_DIJET)
        rho_value = rho(mass, pt)
        rho_bin = bin_index(rho_value, PLOT_RHO_EDGES)
        for floor in FLOORS:
            selected = (pt_bin >= 0) & np.isfinite(mass) & (mass > floor)
            np.add.at(histograms[floor]["yield"], pt_bin[selected], weight[selected])
            in_hist = selected & (rho_bin >= 0)
            indices = (pt_bin[in_hist], rho_bin[in_hist])
            np.add.at(histograms[floor]["sumw"], indices, weight[in_hist])
            np.add.at(histograms[floor]["sumw2"], indices, weight[in_hist] ** 2)
            np.add.at(histograms[floor]["raw"], indices, 1.0)
        print(f"  {source:>4s} {path.name:<58s} {len(pt):>10d} reco jets")
    return histograms


def accumulate_response(mass_floor):
    shape = (len(PT_DIJET) - 1, len(RESPONSE_RHO_EDGES) - 1)
    totals = {
        key: np.zeros(shape)
        for key in (
            "same",
            "reco_all",
            "gen_all",
            "matched_reco_fid",
            "matched_gen_recofid",
        )
    }

    for path in paths_for("new", "dijet", "mc"):
        matched = read_table(
            path,
            "matched",
            ("ptreco", "mreco_g", "ptgen", "mgen_g", "weight"),
        )
        weight = matched["weight"]
        reco_bin = (
            bin_index(matched["ptreco"], PT_DIJET),
            bin_index(rho(matched["mreco_g"], matched["ptreco"]), RESPONSE_RHO_EDGES),
        )
        gen_bin = (
            bin_index(matched["ptgen"], PT_DIJET),
            bin_index(rho(matched["mgen_g"], matched["ptgen"]), RESPONSE_RHO_EDGES),
        )
        reco_selected = (
            (reco_bin[0] >= 0)
            & (reco_bin[1] >= 0)
            & (matched["mreco_g"] > mass_floor)
        )
        gen_selected = (
            (gen_bin[0] >= 0)
            & (gen_bin[1] >= 0)
            & (matched["mgen_g"] > mass_floor)
        )
        both_selected = reco_selected & gen_selected
        same = (
            both_selected
            & (reco_bin[0] == gen_bin[0])
            & (reco_bin[1] == gen_bin[1])
        )
        np.add.at(
            totals["same"],
            (gen_bin[0][same], gen_bin[1][same]),
            weight[same],
        )
        np.add.at(
            totals["matched_reco_fid"],
            (reco_bin[0][both_selected], reco_bin[1][both_selected]),
            weight[both_selected],
        )
        np.add.at(
            totals["matched_gen_recofid"],
            (gen_bin[0][both_selected], gen_bin[1][both_selected]),
            weight[both_selected],
        )

        reco = read_table(path, "reco", ("ptreco", "mreco_g", "weight"))
        reco_index = (
            bin_index(reco["ptreco"], PT_DIJET),
            bin_index(rho(reco["mreco_g"], reco["ptreco"]), RESPONSE_RHO_EDGES),
        )
        selected = (
            (reco_index[0] >= 0)
            & (reco_index[1] >= 0)
            & (reco["mreco_g"] > mass_floor)
        )
        np.add.at(
            totals["reco_all"],
            (reco_index[0][selected], reco_index[1][selected]),
            reco["weight"][selected],
        )

        gen = read_table(path, "gen", ("ptgen", "mgen_g", "weight"))
        gen_index = (
            bin_index(gen["ptgen"], PT_DIJET),
            bin_index(rho(gen["mgen_g"], gen["ptgen"]), RESPONSE_RHO_EDGES),
        )
        selected = (
            (gen_index[0] >= 0)
            & (gen_index[1] >= 0)
            & (gen["mgen_g"] > mass_floor)
        )
        np.add.at(
            totals["gen_all"],
            (gen_index[0][selected], gen_index[1][selected]),
            gen["weight"][selected],
        )
        print(f"  response m_g>{mass_floor:g}: {path.name}")

    with np.errstate(divide="ignore", invalid="ignore"):
        totals["purity"] = totals["same"] / totals["reco_all"]
        totals["stability"] = totals["same"] / totals["gen_all"]
        totals["feedin"] = 1.0 - totals["matched_reco_fid"] / totals["reco_all"]
        totals["inefficiency"] = 1.0 - totals["matched_gen_recofid"] / totals["gen_all"]
    return totals


def plot_rho(outdir, mc, data, floor, pt_index):
    lo, hi = PT_DIJET[pt_index : pt_index + 2]
    mc_sumw = mc[floor]["sumw"][pt_index]
    mc_sumw2 = mc[floor]["sumw2"][pt_index]
    data_sumw = data[floor]["sumw"][pt_index]
    data_sumw2 = data[floor]["sumw2"][pt_index]
    raw_mc = mc[floor]["raw"][pt_index]
    raw_data = data[floor]["raw"][pt_index]
    scale = data_sumw.sum() / mc_sumw.sum()
    widths = np.diff(PLOT_RHO_EDGES)
    mc_density = mc_sumw * scale / widths
    data_density = data_sumw / widths
    mc_variance = mc_sumw2 * scale**2 / widths**2
    data_variance = data_sumw2 / widths**2

    fig, (main, ratio_axis) = ratio_single()
    step(
        main,
        PLOT_RHO_EDGES,
        mc_density,
        color=C_BLUE,
        lw=2.4,
        label=f"QCD MG+Pythia8 ($\\times${scale:.2f})",
    )
    centers = 0.5 * (PLOT_RHO_EDGES[1:] + PLOT_RHO_EDGES[:-1])
    good = (raw_mc >= MIN_RAW) & (raw_data >= MIN_RAW) & (mc_density > 0)
    main.errorbar(
        centers[good],
        data_density[good],
        yerr=np.sqrt(data_variance[good]),
        fmt="o",
        ms=5,
        color="k",
        lw=1.3,
        label=DATA_LABEL,
    )
    ratio, ratio_error, mc_relative = data_mc_ratio(
        data_density, data_variance, mc_density, mc_variance
    )
    proxies, labels = draw_ratio(
        ratio_axis,
        PLOT_RHO_EDGES,
        [(ratio, ratio_error, good, "o")],
        mc_relative,
        xlim=(-5.0, 0.0),
    )
    if floor == 10.0:
        threshold = sorted((mass_floor_rho(floor, lo), mass_floor_rho(floor, hi)))
        for axis in (main, ratio_axis):
            axis.axvspan(*threshold, color=C_ORANGE, alpha=0.14, lw=0)
        panel_text(
            main,
            f"dijet, ${lo:.0f} < p_T^{{jet}} < {hi:.0f}$ GeV\n"
            f"$m_g > 10$ GeV; threshold: "
            f"${threshold[0]:.2f} < \\rho_{{cut}} < {threshold[1]:.2f}$",
            fontsize=18,
        )
    else:
        panel_text(
            main,
            f"dijet, ${lo:.0f} < p_T^{{jet}} < {hi:.0f}$ GeV\n$m_g > 2$ GeV",
            fontsize=18,
        )
    main.set_ylabel("Weighted jets / unit $\\rho$")
    main.set_xlim(-5.0, 0.0)
    visible_max = max(np.max(mc_density), np.max(data_density[good], initial=0.0))
    main.set_ylim(0.0, visible_max * 1.65)
    main.grid(alpha=0.25)
    cms_label(main, True)
    handles, main_labels = main.get_legend_handles_labels()
    main.legend(handles + proxies, main_labels + labels, loc="upper right", fontsize=16)
    ratio_axis.set_xlabel(r"$\rho_g = 2\log_{10}(m_g/(p_T R))$")
    return save(
        fig,
        outdir,
        f"rho_datamc_mg{floor:g}_{ptbin_tag(lo, hi)}.png",
        stamp_inputs="pair_split 2018 postprocessed skims",
    )


def serializable_matrix(values):
    return np.asarray(values).tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)

    print("Reco spectra (stored weights are already fully scaled):")
    mc = accumulate_reco("mc")
    data = accumulate_reco("data")

    summary = {
        "pt_edges": PT_DIJET.tolist(),
        "rho_edges_plot": PLOT_RHO_EDGES.tolist(),
        "rho_edges_response": RESPONSE_RHO_EDGES.tolist(),
        "retention": [],
        "response": {},
    }
    print("\nWeighted retention relative to m_g>2 GeV")
    print(f"{'pT [GeV]':>16s} {'data':>10s} {'MC':>10s} {'low data':>10s} {'low MC':>10s}")
    for pt_index, (lo, hi) in enumerate(zip(PT_DIJET[:-1], PT_DIJET[1:])):
        data_retention = data[10.0]["yield"][pt_index] / data[2.0]["yield"][pt_index]
        mc_retention = mc[10.0]["yield"][pt_index] / mc[2.0]["yield"][pt_index]
        print(
            f"{lo:6.0f}-{hi:<6.0f} {data_retention:10.4f} {mc_retention:10.4f} "
            f"{1-data_retention:10.4f} {1-mc_retention:10.4f}"
        )
        summary["retention"].append(
            {
                "pt_low": float(lo),
                "pt_high": float(hi),
                "data": float(data_retention),
                "mc": float(mc_retention),
                "data_below_10": float(1.0 - data_retention),
                "mc_below_10": float(1.0 - mc_retention),
                "rho_threshold_low_pt": float(mass_floor_rho(10.0, lo)),
                "rho_threshold_high_pt": float(mass_floor_rho(10.0, hi)),
            }
        )

    for floor in FLOORS:
        for pt_index in range(len(PT_DIJET) - 1):
            plot_rho(args.outdir, mc, data, floor, pt_index)

    for floor in FLOORS:
        response = accumulate_response(floor)
        summary["response"][f"mg{floor:g}"] = {
            key: serializable_matrix(response[key])
            for key in ("purity", "stability", "feedin", "inefficiency")
        }

    shown_bin = int(np.where(PT_DIJET[:-1] == 480.0)[0][0])
    print("\nResponse metrics for the shown 480-570 GeV interval")
    for floor in FLOORS:
        response = summary["response"][f"mg{floor:g}"]
        print(f"\n  m_g > {floor:g} GeV")
        print(f"  {'rho bin':>14s} {'purity':>8s} {'stability':>10s} {'feed-in':>9s} {'ineff':>8s}")
        for rho_index, (lo, hi) in enumerate(
            zip(RESPONSE_RHO_EDGES[:-1], RESPONSE_RHO_EDGES[1:])
        ):
            values = [response[key][shown_bin][rho_index] for key in (
                "purity", "stability", "feedin", "inefficiency"
            )]
            if not np.any(np.isfinite(values)):
                continue
            print(
                f"  [{lo:5.1f},{hi:4.1f}] "
                + " ".join(f"{value:8.3f}" for value in values)
            )

    args.outdir.mkdir(parents=True, exist_ok=True)
    json_path = args.outdir / "mg10_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print("\nwrote", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
