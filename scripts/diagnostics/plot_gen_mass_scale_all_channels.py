"""Groomed mass-scale ratio vs jet pT, all three channels, with data.

The figure for the generator authors: the median implied mass ratio relative
to Pythia8, 10^(d<rho>/2) from the rho = 2 log10(m/(pT R)) spectra, for
Herwig7 AND for data, on one pT axis per channel.

    python scripts/diagnostics/plot_gen_mass_scale_all_channels.py --level reco

`--level reco` (default) uses the detector-level spectra, so data can be shown
next to the two simulations without unfolding; `--level gen` uses the
generator-level truth (simulation only).
"""
from __future__ import annotations

import argparse, datetime, pickle, subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use(hep.style.CMS)

PUB_LABEL_FS, PUB_TICK_FS, PUB_LEGEND_FS, PUB_ANNOT_FS = 30, 26, 20, 22
HADRONIC_ERAS = ("2016APV", "2016", "2017", "2018")
STYLE = {
    "zjet":   ("Z+jet",  "#5790fc", "o", "-"),
    "dijet":  ("Dijet",  "#e42536", "s", "--"),
    "trijet": ("Trijet", "#964a8b", "^", "-."),
}


def stamp(fig, enabled=True):
    if not enabled:
        return
    try:
        ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:
        ver = "unknown"
    try:
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
    except Exception:
        pass
    fig.text(0.99, 0.002,
             f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  "
             "inputs: <channel>/rho/original gen",
             ha="right", va="bottom", fontsize=7, color="0.45", family="monospace")


def project(hist_obj):
    names = [a.name for a in hist_obj.axes]
    index = {}
    if "dataset" in names:
        index["dataset"] = sum
    if "systematic" in names:
        index["systematic"] = ("nominal" if "nominal" in list(hist_obj.axes["systematic"])
                               else sum)
    return hist_obj[index] if index else hist_obj


SOURCE_FOLDER = {"data": ("{ch}_data", "data"),
                 "pythia": ("{ch}_mc", "mg_pythia8"),
                 "herwig": ("{ch}_herwig", "herwig")}


def load(channel, inputs_root, level="reco"):
    key = "ptjet_rhojet_g_reco" if level == "reco" else "ptjet_rhojet_g_gen"
    axis_rho = "mpt_reco" if level == "reco" else "mpt_gen"
    axis_pt = "ptreco" if level == "reco" else "ptgen"
    sources = ("data", "pythia", "herwig") if level == "reco" else ("pythia", "herwig")
    root = Path(inputs_root) / channel / "rho" / "original"
    out = {}
    for generator in sources:
        if channel == "zjet":
            payloads = [pickle.load(open(root / f"{generator}_all.pkl", "rb"))]
        else:
            folder, stem = SOURCE_FOLDER[generator]
            payloads = [pickle.load(open(
                root / era / folder.format(ch=channel) /
                f"minimal_rho_{channel}_{stem}_{era}.pkl", "rb"))
                for era in HADRONIC_ERAS]
        values = variances = rho_edges = pt_edges = None
        for payload in payloads:
            h = project(payload[key])
            v, var = np.asarray(h.values()), np.asarray(h.variances())
            values = v.copy() if values is None else values + v
            variances = var.copy() if variances is None else variances + var
            rho_edges, pt_edges = h.axes[axis_rho].edges, h.axes[axis_pt].edges
        out[generator] = (values, variances, rho_edges, pt_edges)
    return out


def median_ratio(channel, inputs_root, level="reco", numerator="herwig", rho_floor=-5.0):
    """(pT centers, median mass ratio vs Pythia8, its error) per pT bin."""
    data = load(channel, inputs_root, level)
    _, _, rho_edges, pt_edges = data["pythia"]
    keep = 0.5 * (rho_edges[:-1] + rho_edges[1:]) > rho_floor
    centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])[keep]
    x, y, err = [], [], []
    for i, (lo, hi) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
        medians, spreads = {}, {}
        for generator in (numerator, "pythia"):
            v = data[generator][0][i][keep]
            var = data[generator][1][i][keep]
            total = v.sum()
            if total <= 0:
                break
            cdf = np.cumsum(v) / total
            medians[generator] = float(np.interp(0.5, cdf, centers))
            # median error ~ 1.25 sigma / sqrt(N_eff) on a normal-ish core
            n_eff = total ** 2 / max(var.sum(), 1e-300)
            q = lambda p: float(np.interp(p, cdf, centers))
            spreads[generator] = 1.25 * 0.5 * (q(0.84) - q(0.16)) / np.sqrt(max(n_eff, 1.0))
        if len(medians) < 2:
            continue
        delta = medians[numerator] - medians["pythia"]
        ratio = 10 ** (delta / 2)
        sigma = np.sqrt(spreads[numerator] ** 2 + spreads["pythia"] ** 2)
        x.append(0.5 * (lo + min(hi, 1000.0)))
        y.append(ratio)
        err.append(ratio * np.log(10) / 2 * sigma)
    return np.array(x), np.array(y), np.array(err)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", type=Path, default=Path.home() / "Projects/unfold/inputs")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/plots/reco_rho_data"))
    ap.add_argument("--level", default="reco", choices=("reco", "gen"))
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    #### One panel per channel.  At reco level Data/Pythia8 is drawn next to
    #### Herwig7/Pythia8: the data is what decides whether Herwig's pT trend
    #### is a real feature of nature or a property of the generator.
    channels = list(STYLE)
    series = [("herwig", "Herwig7 / Pythia8", "#e42536", "s", "--")]
    if args.level == "reco":
        # Data first in the legend (CMS convention); drawn last so its markers
        # sit on top where the two series cross.
        series.insert(0, ("data", "Data / Pythia8", "k", "o", "-"))

    fig, axes = plt.subplots(1, len(channels), layout="constrained",
                             figsize=(10 * len(channels) * 1.06, 10 * 1.06))
    axes = np.atleast_1d(axes)
    header = f"{'channel':8s} {'pT (GeV)':>9s} {'HW/PY':>8s} {'err':>6s}"
    print(header + (f" {'data/PY':>8s} {'err':>6s}" if args.level == "reco" else ""))
    for ax, channel in zip(axes, channels):
        rows = {}
        for zorder, (numerator, label, color, marker, style) in enumerate(series):
            x, y, err = median_ratio(channel, args.inputs, args.level, numerator)
            ax.errorbar(x, y, yerr=err, marker=marker, ls=style, lw=2.5, ms=11,
                        color=color, label=label, capsize=4,
                        zorder=5 - zorder)
            rows[numerator] = (x, y, err)
        for i, xi in enumerate(rows["herwig"][0]):
            line = f"{channel:8s} {xi:9.0f} {rows['herwig'][1][i]:8.3f} {rows['herwig'][2][i]:6.3f}"
            if "data" in rows:
                line += f" {rows['data'][1][i]:8.3f} {rows['data'][2][i]:6.3f}"
            print(line)
        ax.axhline(1.0, color="gray", ls=":", lw=1.5)
        hep.cms.label("Preliminary" if args.level == "reco" else "",
                      data=(args.level == "reco"), loc=2, ax=ax,
                      rlabel=(r"138 fb$^{-1}$ (13 TeV)" if args.level == "reco"
                              else "Run 2 (13 TeV)"))
        cms_x = next((t.get_position()[0] for t in ax.texts if t.get_text() == "CMS"), 0.045)
        ax.text(cms_x, 0.70, f"{STYLE[channel][0]}, groomed, {args.level.upper()}",
                transform=ax.transAxes, ha="left", va="top", fontsize=PUB_ANNOT_FS)
        ax.set_xlabel(r"$p_{T}^{jet}$ (GeV)", fontsize=PUB_LABEL_FS)
        ax.set_ylabel("median mass scale relative to Pythia8", fontsize=PUB_LABEL_FS - 4)
        ax.tick_params(labelsize=PUB_TICK_FS)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, lo + (hi - lo) * 1.45)
        ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS, frameon=False)
    stamp(fig, enabled=not args.no_stamp)
    out = args.outdir / f"mass_scale_ratio_all_channels_{args.level}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
