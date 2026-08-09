#!/usr/bin/env python3
"""Render CMS-style Sherpa canary groomed-rho shapes from Rivet ntuples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, "/Users/aritra/.codex/skills/cms-plot-style/scripts")
from reference import stamp  # noqa: E402


RHO_EDGES = np.array(
    [-3.0, -2.85, -2.25, -1.8, -1.5, -1.3, -1.1, -0.9,
     -0.75, -0.65, -0.55, 0.0]
)


def load_rows(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape[1] != 8:
        raise ValueError(f"unexpected ntuple schema in {path}: {rows.shape}")
    return rows


def normalized_histogram(rows: np.ndarray, pt_low: float, pt_high: float):
    selected = (
        (rows[:, 2] >= pt_low)
        & (rows[:, 2] < pt_high)
        & (rows[:, 6] >= RHO_EDGES[0])
        & (rows[:, 6] < RHO_EDGES[-1])
    )
    rho = rows[selected, 6]
    weights = rows[selected, 7]
    values, _ = np.histogram(rho, bins=RHO_EDGES, weights=weights)
    variance, _ = np.histogram(rho, bins=RHO_EDGES, weights=weights**2)
    total = values.sum()
    if total == 0:
        raise ValueError(f"no selected weighted yield for pT [{pt_low}, {pt_high})")
    widths = np.diff(RHO_EDGES)
    density = values / total / widths
    uncertainty = np.sqrt(variance) / abs(total) / widths
    return density, uncertainty, int(selected.sum())


def plot_channel(
    rows: np.ndarray,
    channel: str,
    pt_low: float,
    pt_high: float,
    output: Path,
    repository: Path,
) -> None:
    density, uncertainty, selected_count = normalized_histogram(
        rows, pt_low, pt_high
    )
    centers = 0.5 * (RHO_EDGES[:-1] + RHO_EDGES[1:])
    fig, axis = plt.subplots(layout="constrained")
    axis.stairs(
        density,
        RHO_EDGES,
        baseline=None,
        color="#5790fc",
        linewidth=2.5,
        label="Sherpa 3 Dire + Ahadic",
    )
    axis.errorbar(
        centers,
        density,
        yerr=uncertainty,
        fmt="o",
        color="#5790fc",
        markersize=5,
        linewidth=1.3,
    )
    axis.axvspan(-1.8, -1.3, color="#f89c20", alpha=0.18,
                 label=r"target residual $-1.8<\rho_g<-1.3$")
    axis.set_xlim(-3.0, 0.0)
    visible_max = float(np.nanmax(density + uncertainty))
    axis.set_ylim(0.0, max(1.45 * visible_max, 0.1))
    axis.set_xlabel(r"$\rho_g = 2\log_{10}(m_g/(p_T R))$")
    axis.set_ylabel(r"Normalized jets / unit $\rho_g$")
    axis.grid(alpha=0.22)
    hep.cms.label("Preliminary", data=False, loc=2, ax=axis, rlabel="13 TeV")
    pt_text = (
        rf"{channel}, ${pt_low:.0f}<p_T^{{jet}}<{pt_high:.0f}$ GeV"
        if pt_high < 1500
        else rf"{channel}, $p_T^{{jet}}\geq {pt_low:.0f}$ GeV"
    )
    axis.text(0.05, 0.70, pt_text + f"\ncanary jets: {selected_count}",
              transform=axis.transAxes, fontsize=18, va="top")
    axis.legend(loc="upper right", fontsize=16, framealpha=0.95)
    stamp(fig, inputs="Sherpa pair_split canary", repo="smp_jetmass_run2")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    plot_channel(
        load_rows(args.run_directory / "pairsplit_dijet.txt"),
        "Dijet",
        480.0,
        570.0,
        args.run_directory / "plots/dijet_rho_g_pt480_570.png",
        args.repository,
    )
    plot_channel(
        load_rows(args.run_directory / "pairsplit_trijet.txt"),
        "Trijet",
        400.0,
        1500.0,
        args.run_directory / "plots/trijet_rho_g_pt400_1500.png",
        args.repository,
    )


if __name__ == "__main__":
    hep.style.use(hep.style.CMS)
    main()
