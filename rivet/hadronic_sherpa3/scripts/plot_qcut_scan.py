#!/usr/bin/env python3
"""Qcut merging-scale comparison for the Sherpa 3 pair-split canary.

Emits one square CMS panel per particle-level pT bin: normalized rho_g shapes
for each Qcut, with a ratio panel against the Qcut = 20 GeV baseline and the
+-5% acceptance band from CAMPAIGN_PLAN.md section 5.

Shapes are normalized per pT row, which is what the acceptance criterion is
stated in and what makes the comparison insensitive to the per-sample cross
section scatter (Sherpa re-estimates sigma per chunk; see CANARY_STATUS.md).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

hep.style.use(hep.style.CMS)  # global, once, before any figure

PUB_LABEL_FS, PUB_TICK_FS, PUB_LEGEND_FS, PUB_ANNOT_FS = 30, 26, 20, 22
# Petroff 6-colour CVD-safe scheme; baseline Qcut drawn first.
PETROFF = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
STUDY_RHO_MIN = -3.0
ACCEPTANCE = 0.05  # localized shape distortion tolerated when Qcut changes


def stamp(fig, inputs: str, enabled: bool = True) -> None:
    if not enabled:
        return
    ver = subprocess.run(
        ["git", "describe", "--tags", "--always", "--dirty"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    ).stdout.strip()
    # Constrained layout fills the canvas, so a negative y puts the stamp
    # off-figure entirely. Reserve a bottom band for it instead.
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0, 0.022, 1, 1))
    fig.text(
        0.99, 0.004,
        f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  "
        f"inputs: {inputs}",
        ha="right", va="bottom", fontsize=7, color="0.45", family="monospace",
    )


def normalized_density(values: np.ndarray, edges: np.ndarray):
    """Unit-area shape per pT row, expressed as a density over variable bins."""
    widths = np.diff(edges)
    total = values.sum()
    if total <= 0:
        return np.zeros_like(values), 0.0
    return values / total / widths, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True,
                        metavar="QCUT=FILE.npz",
                        help="repeat per Qcut, e.g. --input 20=harvest_q20.npz")
    parser.add_argument("--baseline", default="20")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--channel", default="dijet")
    parser.add_argument("--slice-label", default="")
    parser.add_argument("--no-stamp", action="store_true")
    args = parser.parse_args()

    data = {}
    for spec in args.input:
        qcut, _, path = spec.partition("=")
        with np.load(path) as f:
            data[qcut] = {k: f[k] for k in f.files}
    if args.baseline not in data:
        raise SystemExit(f"baseline Qcut {args.baseline} not among {list(data)}")

    ref = data[args.baseline]
    pt_edges, rho_edges = ref["pt_edges"], ref["rho_edges"]
    centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    study = rho_edges[:-1] >= STUDY_RHO_MIN
    qcuts = sorted(data, key=float)

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = {"channel": args.channel, "baseline_qcut": args.baseline,
              "acceptance": ACCEPTANCE, "pt_bins": {}}

    for i in range(len(pt_edges) - 1):
        lo, hi = pt_edges[i], pt_edges[i + 1]
        if ref["values"][i].sum() <= 0:
            continue  # pT row this slice does not populate

        # Main panel keeps the ~10x10 CMS footprint; the ratio rides below it.
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(10, 13), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}, layout="constrained",
        )

        ref_shape, _ = normalized_density(ref["values"][i], rho_edges)
        worst = 0.0
        for j, q in enumerate(qcuts):
            shape, _ = normalized_density(data[q]["values"][i], rho_edges)
            is_base = q == args.baseline
            ax.step(rho_edges, np.r_[shape, shape[-1]], where="post",
                    lw=2.4, color=PETROFF[j % len(PETROFF)],
                    ls="-" if is_base else "--",
                    label=rf"$Q_\mathrm{{cut}}$ = {q} GeV"
                          + (" (baseline)" if is_base else ""))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(ref_shape > 0, shape / ref_shape, np.nan)
            rax.step(rho_edges, np.r_[ratio, ratio[-1]], where="post",
                     lw=2.4, color=PETROFF[j % len(PETROFF)],
                     ls="-" if is_base else "--")
            if not is_base:
                dev = np.abs(ratio[study] - 1.0)
                if np.isfinite(dev).any():
                    worst = max(worst, float(np.nanmax(dev)))

        rax.axhspan(1 - ACCEPTANCE, 1 + ACCEPTANCE, color="0.85", zorder=0)
        rax.axhline(1.0, color="0.4", lw=1.2)
        rax.axvline(STUDY_RHO_MIN, color="0.3", lw=1.4, ls=":")
        ax.axvline(STUDY_RHO_MIN, color="0.3", lw=1.4, ls=":")

        hep.cms.label("Preliminary", data=False, loc=2, ax=ax,
                      rlabel="Sherpa 3.0.4 (13 TeV)")
        ax.grid(alpha=0.25)
        ax.margins(x=0.02)
        # Never clip: top from the actual maximum, with legend headroom.
        ymax = max(
            np.nanmax(normalized_density(data[q]["values"][i], rho_edges)[0])
            for q in qcuts
        )
        ax.set_ylim(0, ymax * 1.45)
        ax.set_ylabel("Normalized " + r"$1/N\ \mathrm{d}N/\mathrm{d}\rho_g$",
                      fontsize=PUB_LABEL_FS)
        rax.set_ylabel("Ratio to\nbaseline", fontsize=PUB_LABEL_FS * 0.7)
        rax.set_xlabel(r"$\rho_g = 2\log_{10}(m_g / (p_T \cdot 0.8))$",
                       fontsize=PUB_LABEL_FS)
        rax.set_ylim(0.7, 1.3)
        for a in (ax, rax):
            a.tick_params(labelsize=PUB_TICK_FS)
            a.grid(alpha=0.25)

        anchor = next((t.get_position()[0] for t in ax.texts
                       if t.get_text() == "CMS"), 0.03)
        label = args.slice_label or args.channel
        ax.text(anchor, 0.80,
                f"{label}\n"
                + r"particle-level $p_T$: "
                + f"{lo:.0f}-{hi:.0f} GeV\n"
                + rf"max dev. = {100*worst:.1f}% ($\rho_g \geq -3$)",
                transform=ax.transAxes, fontsize=PUB_ANNOT_FS, va="top")
        ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS, frameon=False)

        stamp(fig, f"qcut scan {args.channel} {lo:.0f}-{hi:.0f}",
              enabled=not args.no_stamp)
        out = args.outdir / f"qcut_{args.channel}_pt{lo:.0f}_{hi:.0f}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)

        report["pt_bins"][f"{lo:.0f}_{hi:.0f}"] = {
            "max_abs_deviation": worst,
            "passes": bool(worst <= ACCEPTANCE),
            "figure": str(out),
        }

    report["status"] = (
        "pass" if all(b["passes"] for b in report["pt_bins"].values()) else "fail"
    )
    (args.outdir / "qcut_acceptance.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
