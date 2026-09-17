"""Groomed rho shapes per pT bin: CMS MLM+Pythia8, CMS MLM+Herwig7, standalone Herwig7.

The three curves the pT-drift question turns on, drawn as shapes (unit area
over the shown bins) in the three Z+jet analysis pT bins, with the ratio to
MLM + Pythia8.  Production gen pickles and the Rivet standalone Herwig 7.3 CH3
yoda are rebinned onto their common edges.  Optionally the unmerged MG5 LO +
Pythia8 (same-LHE pair partner) with --with-lo.

    python scripts/diagnostics/plot_zjet_gen_rho_merged_vs_unmerged.py
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
PUB_LABEL_FS, PUB_TICK_FS, PUB_LEGEND_FS, PUB_ANNOT_FS = 30, 26, 19, 22
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
COMMON_EDGES = np.array([-10, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, 0], dtype=float)
PT_BINS = [("pt200_290", 1, r"$200 < p_T^{jet} < 290$ GeV"),
           ("pt290_400", 2, r"$290 < p_T^{jet} < 400$ GeV"),
           ("pt400_Inf", 3, r"$p_T^{jet} > 400$ GeV")]
STYLE = {
    "mlm_pythia": ("MG5 MLM + Pythia8 CP5 (CMS)", "#5790fc", ":", 3.0),
    "mlm_herwig": ("MG5 MLM + Herwig7 CH3 (CMS)", "#e42536", "-.", 2.5),
    "sa_herwig":  ("Herwig 7.3 CH3, own ME, no merging", "#8b0000", "-", 2.5),
    "lo_pythia":  ("MG5 LO + Pythia8, no merging", "#1f4e9a", "-", 2.0),
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
    out, cur, edges, vals, errs = {}, None, None, [], []
    for line in open(path):
        if line.startswith(("BEGIN YODA_ESTIMATE1D", "BEGIN YODA_HISTO1D")):
            cur, edges, vals, errs = line.split()[-1], None, [], []
        elif line.startswith("Edges(A1):"):
            edges = np.array([float(x) for x in re.findall(NUM, line.split(":", 1)[1])])
        elif line.startswith("END YODA"):
            if cur and edges is not None:
                v, e = np.array(vals, float), np.array(errs, float)
                if len(v) == len(edges) + 1:
                    v, e = v[1:-1], e[1:-1]
                out[cur] = (edges, v, e)
            cur = None
        elif cur and edges is not None and not line.startswith(
                ("#", "Path", "Title", "Type", "---", "ErrorLabels")):
            tok = line.split()
            if tok:
                try:
                    vals.append(float("nan") if tok[0] == "nan" else float(tok[0]))
                    errs.append(float("nan") if len(tok) < 3 or tok[2] == "---" else abs(float(tok[2])))
                except ValueError:
                    pass
    return out


def rebin(edges, values, variances, target):
    """Sum bins of (edges, values) onto `target` edges; every target edge must be a source edge."""
    edges = np.asarray(edges, float)
    idx = [int(np.flatnonzero(np.isclose(edges, t))[0]) for t in target]
    v = np.nan_to_num(np.asarray(values, float))
    var = np.nan_to_num(np.asarray(variances, float))
    return (np.array([v[a:b].sum() for a, b in zip(idx[:-1], idx[1:])]),
            np.array([var[a:b].sum() for a, b in zip(idx[:-1], idx[1:])]))


def production(inputs_root, gen, fine=False):
    path = (Path(inputs_root) / "zjet/model/finebins" / f"minimal_rho_fine_{gen}_2018.pkl" if fine
            else Path(inputs_root) / "zjet/rho/original" / f"{gen}_all.pkl")
    h = pickle.load(open(path, "rb"))["ptjet_rhojet_g_gen"]
    names = [a.name for a in h.axes]
    index = {}
    if "dataset" in names:
        index["dataset"] = sum
    if "systematic" in names:
        index["systematic"] = "nominal" if "nominal" in list(h.axes["systematic"]) else sum
    h = h[index]
    edges = np.asarray(h.axes["mpt_gen"].edges, float)
    out = {}
    for key, i, _ in PT_BINS:
        v, var = np.asarray(h.values())[i], np.asarray(h.variances())[i]
        if fine:
            # keep the fine lattice from -5 up; below it use the coarse buffer bins
            target = np.concatenate((COMMON_EDGES[COMMON_EDGES < -5.0], edges[edges >= -5.0 - 1e-9]))
            out[key] = (target,) + rebin(edges, v, var, target)
        else:
            out[key] = (COMMON_EDGES,) + rebin(edges, v, var, COMMON_EDGES)
    return out


def rivet(path):
    y = read_yoda(path)
    out = {}
    for key, _, _ in PT_BINS:
        edges, v, e = y[f"/CMS_ZJET_JETMASS/rho_g_{key}"]
        widths = np.diff(edges)
        # yoda estimates are dsigma/dx: convert to per-bin content before summing
        out[key] = (COMMON_EDGES,) + rebin(edges, v * widths, (e * widths) ** 2, COMMON_EDGES)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", type=Path, default=Path.home() / "Projects/unfold/inputs")
    ap.add_argument("--rivet-out", type=Path, default=Path(__file__).resolve().parents[2] / "rivet/out")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/plots/gen_rho_generators"))
    ap.add_argument("--rho-floor", type=float, default=-5.0)
    ap.add_argument("--with-lo", action="store_true")
    ap.add_argument("--fine", action="store_true",
                    help="CMS MLM curves from the 0.125-wide 2018 fine pickles; the standalone Herwig stays at 0.5")
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = {"mlm_pythia": production(args.inputs, "pythia", fine=args.fine),
            "mlm_herwig": production(args.inputs, "herwig", fine=args.fine),
            "sa_herwig": rivet(args.rivet_out / "herwig_ch3.yoda")}
    if args.with_lo:
        data["lo_pythia"] = rivet(args.rivet_out / "_norm/mglo_pythia_sliced.yoda")

    def shown(all_edges, v, var):
        """Drop the bins below the floor; return (edges, values, variances)."""
        keep = 0.5 * (all_edges[:-1] + all_edges[1:]) > args.rho_floor
        first = int(np.argmax(keep))
        return all_edges[first:], v[keep], var[keep]

    def to_edges(edges_src, dens_src, target):
        """Piecewise-constant density resampled onto a finer `target` grid (for the ratio)."""
        c = 0.5 * (target[:-1] + target[1:])
        idx = np.clip(np.searchsorted(edges_src, c) - 1, 0, len(dens_src) - 1)
        return dens_src[idx]

    fig, axes = plt.subplots(2, 3, sharex="col", layout="constrained", figsize=(31, 11.5),
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})
    print(f"{'pT bin':12s} {'series':38s} {'median rho':>11s}")
    for col, (key, _, label) in enumerate(PT_BINS):
        ax, rax = axes[0, col], axes[1, col]
        dens, errs, grids, raw = {}, {}, {}, {}
        for series, (name, color, style, lw) in STYLE.items():
            if series not in data:
                continue
            all_edges, v_all, var_all = data[series][key]
            edges, v, var = shown(np.asarray(all_edges, float), v_all, var_all)
            widths = np.diff(edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            total = v.sum()
            d = v / widths / total
            e = np.sqrt(np.clip(var, 0, None)) / widths / total
            dens[series], errs[series], grids[series], raw[series] = d, e, edges, (v, var)
            ax.stairs(d, edges, color=color, ls=style, lw=lw, label=name, baseline=None)
            ax.errorbar(centers, d, yerr=e, fmt="none", ecolor=color, elinewidth=1.0, capsize=2)
            med = float(np.interp(0.5, np.cumsum(v) / total, centers))
            print(f"{key:12s} {name:38s} {med:11.3f}")
        hep.cms.label("", data=False, loc=2, ax=ax, rlabel="13 TeV")
        ax.set_ylabel("Normalized", fontsize=PUB_LABEL_FS)
        ax.tick_params(labelsize=PUB_TICK_FS)
        ymax = max(float((d + e).max()) for d, e in zip(dens.values(), errs.values()))
        ax.set_ylim(0, ymax * 1.75)
        cms_x = next((t.get_position()[0] for t in ax.texts if t.get_text() == "CMS"), 0.045)
        note = f"Z+jet, groomed, particle level\n{label}"
        if args.fine:
            note += "\nCMS curves: 2018, 0.125-wide bins"
        ax.text(cms_x, 0.70, note, transform=ax.transAxes, ha="left", va="top",
                fontsize=PUB_ANNOT_FS - 2)
        ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS - 3, frameon=False)
        # Each ratio on the numerator's own grid: a coarser series gets the
        # reference rebinned onto its bins, so a wide bin is compared with the
        # reference summed over the same interval rather than with a fine slice.
        ref_edges_full = grids["mlm_pythia"]
        ref_v, ref_var = raw["mlm_pythia"]
        for series in dens:
            if series == "mlm_pythia":
                continue
            g = grids[series]
            if len(g) == len(ref_edges_full) and np.allclose(g, ref_edges_full):
                rv, rvar = ref_v, ref_var
            else:
                rv, rvar = rebin(ref_edges_full, ref_v, ref_var, g)
            w = np.diff(g)
            rd = rv / w / rv.sum()
            re_ = np.sqrt(np.clip(rvar, 0, None)) / w / rv.sum()
            d, e = dens[series], errs[series]
            ratio = np.divide(d, rd, out=np.full_like(d, np.nan), where=rd > 0)
            rel = np.divide(e, d, out=np.zeros_like(d), where=d > 0)
            rel_ref = np.divide(re_, rd, out=np.zeros_like(rd), where=rd > 0)
            color, style, lw = STYLE[series][1], STYLE[series][2], STYLE[series][3]
            rax.stairs(ratio, g, color=color, ls=style, lw=lw, baseline=None)
            rax.errorbar(0.5 * (g[:-1] + g[1:]), ratio,
                         yerr=np.abs(ratio) * np.sqrt(rel ** 2 + rel_ref ** 2),
                         fmt="none", ecolor=color, elinewidth=1.0, capsize=2)
        rax.axhline(1.0, color="gray", ls="--", lw=1)
        rax.set_ylim(0.0, 2.5)
        rax.set_yticks([0.5, 1.0, 1.5, 2.0])
        rax.set_ylabel("ratio to MLM+Pythia8", fontsize=PUB_LABEL_FS - 8)
        rax.set_xlabel(r"$\log_{10}(\rho^{2}) = 2\log_{10}(m/(p_{T}R))$", fontsize=PUB_LABEL_FS)
        rax.tick_params(labelsize=PUB_TICK_FS)
    stamp(fig, enabled=not args.no_stamp)
    out = args.outdir / f"gen_rho_merged_vs_unmerged{'_with_lo' if args.with_lo else ''}{'_fine' if args.fine else ''}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
