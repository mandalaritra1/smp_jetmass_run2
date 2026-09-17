"""Herwig7/Pythia8 groomed gen mass-scale ratio vs jet pT, all three channels.

One figure for the generator authors: the median implied mass ratio
m(Herwig7)/m(Pythia8) = 10^(d<rho>/2) from the gen rho = 2 log10(m/(pT R))
spectra of Z+jet, dijet and trijet, on a single pT axis.

    python scripts/diagnostics/plot_gen_mass_scale_all_channels.py
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


def load(channel, inputs_root, key="ptjet_rhojet_g_gen"):
    root = Path(inputs_root) / channel / "rho" / "original"
    out = {}
    for generator in ("herwig", "pythia"):
        if channel == "zjet":
            payloads = [pickle.load(open(root / f"{generator}_all.pkl", "rb"))]
        else:
            stem = "herwig" if generator == "herwig" else "mg_pythia8"
            folder = f"{channel}_herwig" if generator == "herwig" else f"{channel}_mc"
            payloads = [pickle.load(open(
                root / era / folder / f"minimal_rho_{channel}_{stem}_{era}.pkl", "rb"))
                for era in HADRONIC_ERAS]
        values = variances = rho_edges = pt_edges = None
        for payload in payloads:
            h = project(payload[key])
            v, var = np.asarray(h.values()), np.asarray(h.variances())
            values = v.copy() if values is None else values + v
            variances = var.copy() if variances is None else variances + var
            rho_edges, pt_edges = h.axes["mpt_gen"].edges, h.axes["ptgen"].edges
        out[generator] = (values, variances, rho_edges, pt_edges)
    return out


def median_ratio(channel, inputs_root, rho_floor=-5.0):
    """(pT centers, upper edges, median mass ratio, its MC-stat error) per pT bin."""
    data = load(channel, inputs_root)
    _, _, rho_edges, pt_edges = data["pythia"]
    keep = 0.5 * (rho_edges[:-1] + rho_edges[1:]) > rho_floor
    centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])[keep]
    x, y, err = [], [], []
    for i, (lo, hi) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
        medians, spreads = {}, {}
        for generator in ("herwig", "pythia"):
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
        delta = medians["herwig"] - medians["pythia"]
        ratio = 10 ** (delta / 2)
        sigma = np.sqrt(spreads["herwig"] ** 2 + spreads["pythia"] ** 2)
        x.append(0.5 * (lo + min(hi, 1000.0)))
        y.append(ratio)
        err.append(ratio * np.log(10) / 2 * sigma)
    return np.array(x), np.array(y), np.array(err)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", type=Path, default=Path.home() / "Projects/unfold/inputs")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/plots/gen_rho_generators"))
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(layout="constrained")
    print(f"{'channel':8s} {'pT (GeV)':>10s} {'m(HW)/m(PY)':>13s} {'err':>8s}")
    # The three channels share the low pT bins; dodge them by a few percent in
    # pT so the markers stay distinguishable where the values nearly coincide.
    for offset, (channel, (label, color, marker, style)) in zip(
            (0.97, 1.0, 1.03), STYLE.items()):
        x, y, err = median_ratio(channel, args.inputs)
        ax.errorbar(x * offset, y, yerr=err, marker=marker, ls=style, lw=2.5, ms=11,
                    color=color, label=label, capsize=4)
        for xi, yi, ei in zip(x, y, err):
            print(f"{channel:8s} {xi:10.0f} {yi:13.3f} {ei:8.3f}")
    ax.axhline(1.0, color="gray", ls=":", lw=1.5)
    hep.cms.label("", data=False, loc=2, ax=ax, rlabel="Run 2 (13 TeV)")
    ax.set_xlabel(r"$p_{T}^{jet}$ (GeV)", fontsize=PUB_LABEL_FS)
    ax.set_ylabel(r"median $m_{\mathrm{Herwig7}} / m_{\mathrm{Pythia8}}$ (groomed, GEN)",
                  fontsize=PUB_LABEL_FS - 4)
    ax.tick_params(labelsize=PUB_TICK_FS)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * 1.45)
    ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS, frameon=False)
    stamp(fig, enabled=not args.no_stamp)
    out = args.outdir / "gen_mass_scale_ratio_all_channels.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
