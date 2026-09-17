"""How the median groomed rho of Z+jet moves with jet pT, per generator setup.

Answers "is the pT drift a Herwig property or an MLM-merging property?".
For every setup the median of log10(rho^2) in the three analysis pT bins is
drawn relative to its own 200-290 GeV value, so tune-level offsets drop out
and only the pT dependence remains.  Sources:

* production (MG5 MLM + shower, central CMS UL NanoAODv9 gen pickles):
  inputs/zjet/rho/original/{pythia,herwig}_all.pkl
* unmerged Rivet runs (rivet/out/): MG5 LO Z+1j LHE showered by Pythia8 and by
  Herwig7 (same LHE, stitched pT slices), standalone Herwig 7.3 CH3 (MEZJet),
  MG5 LO + Pythia8 CP5, Pythia8 internal, Vincia, Sherpa 3
* optionally the unfolded data (Pythia response), --with-data

    python scripts/diagnostics/plot_zjet_groomed_slope_generators.py [--with-data]
"""
from __future__ import annotations

import argparse, datetime, pickle, re, subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use(hep.style.CMS)
PUB_LABEL_FS, PUB_TICK_FS, PUB_LEGEND_FS, PUB_ANNOT_FS = 30, 26, 18, 22
BINS = ("pt200_290", "pt290_400", "pt400_Inf")
X = np.array([245.0, 345.0, 700.0])
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

RIVET = {
    # label, file, colour, marker, linestyle
    "MG5 LO + Herwig7 (same LHE)": ("_norm/mglo_herwig_sliced.yoda", "#e42536", "s", "-"),
    "MG5 LO + Pythia8 (same LHE)": ("_norm/mglo_pythia_sliced.yoda", "#5790fc", "o", "-"),
    "Herwig 7.3 CH3, own ME": ("herwig_ch3.yoda", "#e42536", "D", ":"),
    "MG5 LO + Pythia8 CP5": ("cp5_pythia.yoda", "#5790fc", "D", ":"),
    "Pythia8 internal": ("_norm/pythia_sliced.yoda", "#9c9ca1", "v", ":"),
    "Vincia": ("_norm/vincia_sliced.yoda", "#964a8b", "^", ":"),
    "Sherpa 3": ("_norm/sherpa_sliced.yoda", "#f89c20", "<", ":"),
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
    fig.text(0.99, 0.002, f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  "
             "inputs: zjet/rho/original gen + rivet/out", ha="right", va="bottom",
             fontsize=7, color="0.45", family="monospace")


def read_yoda(path):
    out, cur, edges, vals = {}, None, None, []
    for line in open(path):
        if line.startswith(("BEGIN YODA_ESTIMATE1D", "BEGIN YODA_HISTO1D")):
            cur, edges, vals = line.split()[-1], None, []
        elif line.startswith("Edges(A1):"):
            edges = np.array([float(x) for x in re.findall(NUM, line.split(":", 1)[1])])
        elif line.startswith("END YODA"):
            if cur and edges is not None:
                v = np.array(vals, dtype=float)
                out[cur] = (edges, v[1:-1] if len(v) == len(edges) + 1 else v)
            cur = None
        elif cur and edges is not None and not line.startswith(
                ("#", "Path", "Title", "Type", "---", "ErrorLabels")):
            tok = line.split()
            if tok:
                try:
                    vals.append(float("nan") if tok[0] == "nan" else float(tok[0]))
                except ValueError:
                    pass
    return out


def median(edges, values, floor=-5.0):
    v = np.nan_to_num(np.clip(np.asarray(values, float), 0.0, None))
    c = 0.5 * (edges[:-1] + edges[1:])
    keep = c > floor
    v, c = v[keep], c[keep]
    return float(np.interp(0.5, np.cumsum(v) / v.sum(), c)) if v.sum() > 0 else np.nan


def production_medians(inputs_root):
    out = {}
    for gen in ("pythia", "herwig"):
        h = pickle.load(open(Path(inputs_root) / "zjet/rho/original" / f"{gen}_all.pkl", "rb"))["ptjet_rhojet_g_gen"]
        names = [a.name for a in h.axes]
        index = {}
        if "dataset" in names:
            index["dataset"] = sum
        if "systematic" in names:
            index["systematic"] = "nominal" if "nominal" in list(h.axes["systematic"]) else sum
        h = h[index]
        edges = h.axes["mpt_gen"].edges
        # production pT bins: 185-200, 200-290, 290-400, 400+ -> drop the first
        out[gen] = [median(edges, np.asarray(h.values())[i]) for i in (1, 2, 3)]
    return out


def unfolded_medians(unfold_root):
    d = pickle.load(open(Path(unfold_root) / "outputs/zjet/rho/original/data/unfolded_2d_groomed.pkl", "rb"))
    h = d["unfolded_2dnorm"]
    edges = h.axes[1].edges
    return [median(edges, np.asarray(h.values())[i]) for i in (1, 2, 3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", type=Path, default=Path.home() / "Projects/unfold/inputs")
    ap.add_argument("--unfold-root", type=Path, default=Path.home() / "Projects/unfold")
    ap.add_argument("--rivet-out", type=Path, default=Path(__file__).resolve().parents[2] / "rivet/out")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/plots/gen_rho_generators"))
    ap.add_argument("--with-data", action="store_true")
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    series = []
    prod = production_medians(args.inputs)
    series.append(("MG5 MLM + Pythia8 CP5 (CMS central)", prod["pythia"], "#5790fc", "o", "-", 3.5))
    series.append(("MG5 MLM + Herwig7 CH3 (CMS central)", prod["herwig"], "#e42536", "s", "-", 3.5))
    for label, (fname, color, marker, style) in RIVET.items():
        y = read_yoda(args.rivet_out / fname)
        meds = [median(*y[f"/CMS_ZJET_JETMASS/rho_g_{b}"]) for b in BINS]
        series.append((label, meds, color, marker, style, 2.0))
    if args.with_data:
        series.append(("Data, unfolded (Pythia response)", unfolded_medians(args.unfold_root), "k", "*", "--", 3.0))

    fig, ax = plt.subplots(layout="constrained")
    print(f"{'setup':40s} {'median rho 200-290':>18s} {'290-400':>9s} {'400+':>9s} {'slope':>7s}")
    for label, meds, color, marker, style, lw in series:
        meds = np.asarray(meds, float)
        rel = meds - meds[0]
        # Open markers for the unmerged "same LHE" pair so they read apart from
        # the filled MLM production series of the same colour.
        open_marker = "same LHE" in label
        ax.plot(X, rel, marker=marker, ls=style, lw=lw, ms=11 if lw > 3 else 9, color=color,
                mfc="white" if open_marker else color, mew=2.0, label=label,
                zorder=6 if label.startswith("Data") else 3)
        print(f"{label:40s} {meds[0]:18.3f} {meds[1]:9.3f} {meds[2]:9.3f} {rel[2]:+7.3f}")
    ax.axhline(0.0, color="gray", ls=":", lw=1.2)
    hep.cms.label("Preliminary" if args.with_data else "", data=args.with_data, loc=2, ax=ax,
                  rlabel=r"138 fb$^{-1}$ (13 TeV)" if args.with_data else "13 TeV")
    ax.set_xlabel(r"$p_{T}^{jet}$ (GeV)", fontsize=PUB_LABEL_FS)
    ax.set_ylabel(r"median $\log_{10}(\rho^2)$ relative to 200–290 GeV", fontsize=PUB_LABEL_FS - 4)
    ax.tick_params(labelsize=PUB_TICK_FS)
    ax.set_xlim(200, 800)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.05, hi + 0.55 * (hi - lo))
    # Lower left is empty: every curve starts at 0 and falls to the right.
    cms_x = next((t.get_position()[0] for t in ax.texts if t.get_text() == "CMS"), 0.045)
    ax.text(cms_x, 0.04, "Z+jet, soft-drop groomed, particle level\nshape per bin; tune offsets removed",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=PUB_ANNOT_FS - 2)
    ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS - 3, frameon=False, ncol=1)
    stamp(fig, enabled=not args.no_stamp)
    out = args.outdir / f"gen_groomed_slope_generators{'_with_data' if args.with_data else ''}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
