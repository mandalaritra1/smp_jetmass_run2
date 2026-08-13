#!/usr/bin/env python
"""Compare 2018 dijet rho in data, MG+Pythia8, and MG+Herwig7.

All inputs are postprocessed ``rho_skim10`` pair-split skims. Their stored MC
weights already include xs*lumi/sumw, while data carry trigger-prescale
weights. Each generator is normalized independently to the data yield inside
one measured-jet pT bin after the requested groomed-mass floor.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_pairsplit import (  # noqa: E402
    C_BLUE,
    C_ORANGE,
    DATA_LABEL,
    MIN_RAW,
    PT_DIJET,
    cms_label,
    panel_text,
    ptbin_tag,
    ratio_single,
    rho,
    save,
    step,
)

RHO_EDGES = np.linspace(-5.0, 0.0, 41)
FLOORS = (2.0, 10.0)
JET_R = 0.8


def bin_index(values, edges):
    index = np.digitize(values, edges) - 1
    return np.where(
        np.isfinite(values) & (index >= 0) & (index < len(edges) - 1),
        index,
        -1,
    )


def accumulate(paths):
    shape = (len(FLOORS), len(PT_DIJET) - 1, len(RHO_EDGES) - 1)
    sumw = np.zeros(shape)
    sumw2 = np.zeros(shape)
    raw = np.zeros(shape)
    for path in sorted(paths):
        skim = pickle.load(open(path, "rb"))["skim"]["reco"]
        if not skim or not len(skim["ptreco"].value):
            print(f"  {path.name:<60s} EMPTY")
            continue
        pt = np.asarray(skim["ptreco"].value)
        mass = np.asarray(skim["mreco_g"].value)
        weight = np.asarray(skim["weight"].value)
        pt_bin = bin_index(pt, PT_DIJET)
        rho_bin = bin_index(rho(mass, pt), RHO_EDGES)
        for floor_index, floor in enumerate(FLOORS):
            selected = (pt_bin >= 0) & (rho_bin >= 0) & (mass > floor)
            index = (
                np.full(np.count_nonzero(selected), floor_index),
                pt_bin[selected],
                rho_bin[selected],
            )
            np.add.at(sumw, index, weight[selected])
            np.add.at(sumw2, index, weight[selected] ** 2)
            np.add.at(raw, index, 1.0)
        print(f"  {path.name:<60s} {len(pt):>10d} reco jets")
    return {"sumw": sumw, "sumw2": sumw2, "raw": raw}


def normalized(values, variances):
    total = values.sum()
    if total <= 0:
        return np.zeros_like(values), np.zeros_like(variances)
    return values / total, variances / total**2


def ratio_with_stat(data, data_var, mc, mc_var):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = data / mc
        error = np.abs(ratio) * np.sqrt(data_var / data**2 + mc_var / mc**2)
    return ratio, error


def ratio_points(axis, centers, values, errors, good, *, marker, color, label):
    """Draw ratio points and retain off-scale values as edge triangles."""
    lower, upper = 0.5, 1.5
    finite = good & np.isfinite(values) & np.isfinite(errors)
    inside = finite & (values >= lower) & (values <= upper)
    axis.errorbar(
        centers[inside], values[inside], yerr=errors[inside], marker=marker,
        ls="none", ms=5, mfc="none" if marker == "s" else color,
        color=color, lw=1.2, label=label,
    )
    for selected, y, triangle in (
        (finite & (values > upper), upper - 0.035, "^"),
        (finite & (values < lower), lower + 0.035, "v"),
    ):
        if selected.any():
            axis.plot(
                centers[selected], np.full(np.count_nonzero(selected), y),
                marker=triangle, ls="none", ms=9, mfc="none", mec=color,
                mew=1.6,
            )


def mass_floor_rho(mass_floor, pt):
    return 2.0 * np.log10(mass_floor / (pt * JET_R))


def plot_one(outdir, samples, floor_index, pt_index):
    floor = FLOORS[floor_index]
    pt_low, pt_high = PT_DIJET[pt_index : pt_index + 2]
    widths = np.diff(RHO_EDGES)
    centers = 0.5 * (RHO_EDGES[:-1] + RHO_EDGES[1:])

    data, data_var = normalized(
        samples["data"]["sumw"][floor_index, pt_index],
        samples["data"]["sumw2"][floor_index, pt_index],
    )
    pythia, pythia_var = normalized(
        samples["pythia"]["sumw"][floor_index, pt_index],
        samples["pythia"]["sumw2"][floor_index, pt_index],
    )
    herwig, herwig_var = normalized(
        samples["herwig"]["sumw"][floor_index, pt_index],
        samples["herwig"]["sumw2"][floor_index, pt_index],
    )
    data_density, pythia_density, herwig_density = (
        data / widths,
        pythia / widths,
        herwig / widths,
    )

    fig, (main, ratio_axis) = ratio_single()
    step(main, RHO_EDGES, pythia_density, color=C_BLUE, lw=2.4,
         label="QCD MG+Pythia8")
    step(main, RHO_EDGES, herwig_density, color=C_ORANGE, lw=2.4,
         label="QCD MG+Herwig7 (CH3)")
    good_data = samples["data"]["raw"][floor_index, pt_index] >= MIN_RAW
    main.errorbar(
        centers[good_data], data_density[good_data],
        yerr=np.sqrt(data_var[good_data]) / widths[good_data],
        fmt="o", ms=5, color="k", lw=1.3, label=DATA_LABEL,
    )

    pythia_ratio, pythia_error = ratio_with_stat(data, data_var, pythia, pythia_var)
    herwig_ratio, herwig_error = ratio_with_stat(data, data_var, herwig, herwig_var)
    good_pythia = good_data & (
        samples["pythia"]["raw"][floor_index, pt_index] >= MIN_RAW
    ) & np.isfinite(pythia_ratio)
    good_herwig = good_data & (
        samples["herwig"]["raw"][floor_index, pt_index] >= MIN_RAW
    ) & np.isfinite(herwig_ratio)
    ratio_points(
        ratio_axis, centers, pythia_ratio, pythia_error, good_pythia,
        marker="o", color="k", label="Data / Pythia8",
    )
    ratio_points(
        ratio_axis, centers, herwig_ratio, herwig_error, good_herwig,
        marker="s", color=C_ORANGE, label="Data / Herwig7",
    )
    ratio_axis.axhline(1.0, color="k", lw=1.0, ls=":")
    ratio_axis.set_ylim(0.5, 1.5)
    ratio_axis.set_ylabel("Data / MC")
    ratio_axis.set_xlabel(r"$\rho_g = 2\log_{10}(m_g/(p_T R))$")
    ratio_axis.grid(alpha=0.25)
    ratio_axis.legend(loc="upper left", fontsize=13, ncol=2)

    if floor == 10.0:
        threshold = sorted((mass_floor_rho(floor, pt_low), mass_floor_rho(floor, pt_high)))
        for axis in (main, ratio_axis):
            axis.axvspan(*threshold, color="#9c9ca1", alpha=0.14, lw=0)
        floor_text = (
            "$m_g > 10$ GeV; threshold: "
            f"${threshold[0]:.2f} < \\rho_{{cut}} < {threshold[1]:.2f}$"
        )
    else:
        floor_text = "$m_g > 2$ GeV"
    panel_text(
        main,
        f"dijet, ${pt_low:.0f} < p_T^{{jet}} < {pt_high:.0f}$ GeV\n{floor_text}",
        fontsize=18,
    )
    main.set_ylabel(r"Fraction of jets / unit $\rho_g$")
    main.set_xlim(-5.0, 0.0)
    ymax = max(
        np.nanmax(pythia_density), np.nanmax(herwig_density),
        np.nanmax(data_density[good_data], initial=0.0),
    )
    main.set_ylim(0.0, 1.65 * ymax)
    main.grid(alpha=0.25)
    cms_label(main, True)
    main.legend(loc="upper right", fontsize=15)
    return save(
        fig,
        outdir,
        f"rho_datamc_pythia_herwig_mg{floor:g}_{ptbin_tag(pt_low, pt_high)}.png",
        stamp_inputs="pair_split rho_skim10 2018; Pythia8+Herwig7 CH3",
    )


def region_ratio(data, mc, selection):
    data_sum, mc_sum = data[selection].sum(), mc[selection].sum()
    return float(data_sum / mc_sum) if mc_sum > 0 else None


def summarize(samples):
    centers = 0.5 * (RHO_EDGES[:-1] + RHO_EDGES[1:])
    regions = {
        "low": (centers >= -3.5) & (centers < -2.5),
        "shoulder": (centers >= -2.5) & (centers < -1.5),
        "peak": (centers >= -1.3) & (centers < -0.7),
        "core": (centers >= -3.0) & (centers < -0.5),
    }
    out = []
    for floor_index, floor in enumerate(FLOORS):
        for pt_index, (pt_low, pt_high) in enumerate(zip(PT_DIJET[:-1], PT_DIJET[1:])):
            normalized_shapes = {}
            for name in ("data", "pythia", "herwig"):
                normalized_shapes[name], _ = normalized(
                    samples[name]["sumw"][floor_index, pt_index],
                    samples[name]["sumw2"][floor_index, pt_index],
                )
            row = {
                "mass_floor": floor,
                "pt_low": float(pt_low),
                "pt_high": float(pt_high),
            }
            for generator in ("pythia", "herwig"):
                row[f"tv_{generator}"] = float(
                    0.5 * np.abs(normalized_shapes["data"] - normalized_shapes[generator]).sum()
                )
                for region, selection in regions.items():
                    row[f"data_{generator}_{region}"] = region_ratio(
                        normalized_shapes["data"], normalized_shapes[generator], selection
                    )
            out.append(row)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/aritra/cernbox (2)/hadronic_skims_pairsplit/2018"),
    )
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args(argv)

    sources = {
        "data": args.root / "dijet_data",
        "pythia": args.root / "dijet_mc",
        "herwig": args.root / "dijet_herwig",
    }
    samples = {}
    for name, directory in sources.items():
        paths = sorted(directory.glob("*.pkl"))
        if not paths:
            raise SystemExit(f"no {name} skims under {directory}")
        print(f"\n{name}: {len(paths)} files")
        samples[name] = accumulate(paths)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for floor_index in range(len(FLOORS)):
        for pt_index in range(len(PT_DIJET) - 1):
            plot_one(args.outdir, samples, floor_index, pt_index)

    summary = {
        "pt_edges": PT_DIJET.tolist(),
        "rho_edges": RHO_EDGES.tolist(),
        "rows": summarize(samples),
    }
    path = args.outdir / "pythia_herwig_shape_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
