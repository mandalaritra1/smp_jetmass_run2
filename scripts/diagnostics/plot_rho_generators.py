"""Generator comparisons of the jet-mass observable rho, one script for every sample set.

Replaces (2026-09-18) plot_zjet_gen_rho_generators.py, plot_reco_rho_data_generators.py,
plot_zjet_gen_rho_merged_vs_unmerged.py, plot_zjet_groomed_slope_generators.py,
plot_dijet_groomed_flat_vs_mlm.py and plot_zjet_groomed_matchbox_vs_mlm.py, which were
six copies of the same program: load ``ptjet_rhojet_{g,u}_{gen,reco}`` from the staged
production pickles (or a Rivet yoda), take the median of log10(rho^2) per jet-pT bin
above a floor, and draw

* ``ratio``  -- the implied median mass ratio 10^(dmedian/2) between sample pairs
                versus jet pT (one panel per --pairs group),
* ``slope``  -- every sample's median relative to its own 200-290 GeV bin,
* ``shapes`` -- per-pT-bin unit-area spectra of all samples with a ratio panel to
                --reference (optionally on a mass axis in GeV with --mass-axis).

Samples come from the registry below (``--list`` prints it); the loader resolves
each name for the requested --channel / --level / --mode / --mlm-eras.  Samples
on a finer rho lattice than the reference are rebinned onto it (``--grid
reference``, the default); ``--grid native`` keeps every sample on its own
lattice (the reference is rebinned onto the numerator's grid for each ratio) and
``--grid common`` puts everything on the coarse 11-bin lattice of the Rivet
comparisons.  Ratios with a Rivet sample are only defined in the three
analysis pT bins the Rivet routine books.

    python scripts/diagnostics/plot_rho_generators.py --channel zjet --level gen \\
        --samples mlm_pythia,mlm_herwig,matchbox --pairs matchbox/mlm_pythia,mlm_herwig/mlm_pythia:matchbox/mlm_herwig \\
        --figures ratio,slope,shapes --tag zjet_groomed_matchbox_vs_mlm

    python scripts/diagnostics/plot_rho_generators.py --list
"""
from __future__ import annotations

import argparse, datetime, pickle, re, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
import mplhep as hep
import numpy as np

hep.style.use(hep.style.CMS)
PUB_LABEL_FS, PUB_TICK_FS, PUB_LEGEND_FS, PUB_ANNOT_FS = 30, 26, 20, 22
HADRONIC_ERAS = ("2016APV", "2016", "2017", "2018")
CHANNEL_LABEL = {"zjet": "Z+jet", "dijet": "Dijet", "trijet": "Trijet"}
MODE_TEXT = {"groomed": "soft-drop groomed", "ungroomed": "ungroomed"}
LEVEL = {"gen": ("gen", "ptgen", "mpt_gen", "GEN"), "reco": ("reco", "ptreco", "mpt_reco", "RECO")}
COMMON_EDGES = np.array([-10, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, 0], dtype=float)
RIVET_BINS = {"pt200_290": (200.0, 290.0), "pt290_400": (290.0, 400.0), "pt400_Inf": (400.0, 13000.0)}
JET_R = 0.8
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
LUMI = {"2018": r"59.8 fb$^{-1}$", "all": r"138 fb$^{-1}$"}

# ---------------------------------------------------------------------------
# the sample registry: name -> how to load it and how to draw it
#   kind "prod": the staged production pickles of the channel (pythia / herwig / data),
#                resolved per channel and --mlm-eras
#   kind "pickle": one staged pickle (path under --inputs), dataset axis summed
#   kind "yoda": a Rivet yoda under --rivet-out with the CMS_ZJET_JETMASS rho_{g,u}_<bin> objects
#   kind "unfolded": the unfold repo's unfolded 2D result (groomed only)
SERIES = {
    "mlm_pythia": dict(short="MLM Pythia8", label="MG5 MLM + Pythia8 CP5", color="#5790fc", marker="o", ls=":", lw=3.0, kind="prod", gen="pythia"),
    "mlm_herwig": dict(short="MLM Herwig7", label="MG5 MLM + Herwig7 CH3", color="#e42536", marker="s", ls="-.", lw=3.0, kind="prod", gen="herwig"),
    "data": dict(short="Data", label="Data", color="black", marker="o", ls="-", lw=0.0, kind="prod", gen="data", points=True),
    "flat_pythia": dict(short="Flat Pythia8", label="Flat Pythia8 CP5, no merging", color="#1f4e9a", marker="D", ls="-", lw=2.5, kind="pickle",
                        path="dijet/rho/original/2018/dijet_flat_pythia/minimal_rho_dijet_flat_pythia8_2018.pkl", open=True),
    "flat_herwig": dict(short="Flat Herwig7", label="Flat Herwig7 CH3, no merging", color="#8b0000", marker="^", ls="-", lw=2.5, kind="pickle",
                        path="dijet/rho/original/2018/dijet_flat_herwig/minimal_rho_dijet_flat_herwig_2018.pkl", open=True),
    "matchbox": dict(short="Matchbox Herwig7", label="Matchbox NLO + Herwig7 CH3, no MLM", color="#8b0000", marker="^", ls="-", lw=2.5, kind="pickle",
                     path="zjet/rho/original/2018/herwig_matchbox/minimal_rho_zjet_herwig_matchbox_2018.pkl", open=True),
    # the fine-rho 2018 set produced together on 2026-09-18 (4x rho bins)
    "pythia_fine": dict(short="MLM Pythia8", label="MG5 MLM + Pythia8 CP5", color="#5790fc", marker="o", ls=":", lw=3.0, kind="pickle",
                        path="zjet/rho/fine/2018/minimal_rho_fine_pythia_2018.pkl"),
    "herwig_fine": dict(short="MLM Herwig7", label="MG5 MLM + Herwig7 CH3", color="#e42536", marker="s", ls="-.", lw=3.0, kind="pickle",
                        path="zjet/rho/fine/2018/minimal_rho_fine_herwig_2018.pkl"),
    "matchbox_fine": dict(short="Matchbox Herwig7", label="Matchbox NLO + Herwig7 CH3, no MLM", color="#8b0000", marker="^", ls="-", lw=2.5, kind="pickle",
                          path="zjet/rho/fine/2018/minimal_rho_fine_herwig_matchbox_2018.pkl", open=True),
    # Rivet runs (rivet/out), gen level, Z+jet, three analysis pT bins
    "sa_herwig": dict(short="Herwig 7.3 own ME", label="Herwig 7.3 CH3, own ME, no merging", color="#8b0000", marker="D", ls=":", lw=2.5, kind="yoda", file="herwig_ch3.yoda", open=True),
    "lo_herwig_lhe": dict(short="LO + Herwig7", label="MG5 LO + Herwig7 (same LHE)", color="#e42536", marker="s", ls="-", lw=2.0, kind="yoda", file="_norm/mglo_herwig_sliced.yoda", open=True),
    "lo_pythia_lhe": dict(short="LO + Pythia8", label="MG5 LO + Pythia8 (same LHE)", color="#5790fc", marker="o", ls="-", lw=2.0, kind="yoda", file="_norm/mglo_pythia_sliced.yoda", open=True),
    "cp5_pythia": dict(short="LO + Pythia8 CP5", label="MG5 LO + Pythia8 CP5", color="#5790fc", marker="D", ls=":", lw=2.0, kind="yoda", file="cp5_pythia.yoda"),
    "pythia_internal": dict(short="Pythia8 internal", label="Pythia8 internal", color="#9c9ca1", marker="v", ls=":", lw=2.0, kind="yoda", file="_norm/pythia_sliced.yoda"),
    "vincia": dict(short="Vincia", label="Vincia", color="#964a8b", marker="^", ls=":", lw=2.0, kind="yoda", file="_norm/vincia_sliced.yoda"),
    "sherpa3": dict(short="Sherpa 3", label="Sherpa 3", color="#f89c20", marker="<", ls=":", lw=2.0, kind="yoda", file="_norm/sherpa_sliced.yoda"),
    "data_unfolded": dict(short="Data unfolded", label="Data, unfolded (Pythia response)", color="black", marker="*", ls="--", lw=3.0, kind="unfolded",
                          path="outputs/zjet/rho/original/data/unfolded_2d_groomed.pkl", points=True),
}
PROD_ZJET = {  # gen -> (file, dataset categories) per --mlm-eras
    "pythia": {"all": ("pythia_all.pkl", None), "2018": ("pythia_2018.pkl", ["pythia_UL18NanoAODv9"])},
    "herwig": {"all": ("herwig_all.pkl", None), "2018": ("herwig_all.pkl", ["herwig_UL18NanoAODv9"])},
    "data": {"all": ("data_all.pkl", None), "2018": ("data_all.pkl", ["SingleMuon_UL2018", "EGamma_UL2018"])},
}
PROD_HADRONIC = {"pythia": ("{ch}_mc", "mg_pythia8"), "herwig": ("{ch}_herwig", "herwig"), "data": ("{ch}_data", "data")}


# ---------------------------------------------------------------------------
# loading
def select(h, dataset_cats):
    """Sum the requested dataset categories (all if None), take the nominal systematic."""
    names = [a.name for a in h.axes]
    index = {}
    if "dataset" in names:
        cats = list(h.axes["dataset"])
        if dataset_cats is None:
            index["dataset"] = sum
        else:
            missing = [c for c in dataset_cats if c not in cats]
            if missing:
                raise SystemExit(f"dataset categories {missing} not in {cats}")
            index["dataset"] = [cats.index(c) for c in dataset_cats]
    if "systematic" in names:
        syst = list(h.axes["systematic"])
        index["systematic"] = "nominal" if "nominal" in syst else sum
    h = h[index] if index else h
    if "dataset" in [a.name for a in h.axes]:
        h = h[{"dataset": sum}]
    return h


def from_hist(h, pt_axis, rho_axis):
    """{(lo, hi): (rho_edges, values, variances)} from one projected 2D hist."""
    v, var = np.asarray(h.values(), float), np.asarray(h.variances(), float)
    rho = np.asarray(h.axes[rho_axis].edges, float)
    pt = np.asarray(h.axes[pt_axis].edges, float)
    return {(float(lo), float(hi)): (rho, v[i], var[i]) for i, (lo, hi) in enumerate(zip(pt[:-1], pt[1:]))}


def add_bins(a, b):
    if a is None:
        return b
    out = {}
    for k in a:
        e, v, var = a[k]
        e2, v2, var2 = b[k]
        if not np.allclose(e, e2):
            raise SystemExit("rho edges differ between era pickles")
        out[k] = (e, v + v2, var + var2)
    return out


def load_pickles(paths, key, cats, pt_axis, rho_axis):
    out = None
    for p in paths:
        if not Path(p).exists():
            raise SystemExit(f"missing input {p}")
        h = select(pickle.load(open(p, "rb"))[key], cats)
        out = add_bins(out, from_hist(h, pt_axis, rho_axis))
    return out


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
        elif cur and edges is not None and not line.startswith(("#", "Path", "Title", "Type", "---", "ErrorLabels")):
            tok = line.split()
            if tok:
                try:
                    vals.append(float("nan") if tok[0] == "nan" else float(tok[0]))
                    errs.append(float("nan") if len(tok) < 3 or tok[2] == "---" else abs(float(tok[2])))
                except ValueError:
                    pass
    return out


def load_yoda(path, mode):
    """Rivet estimates are dsigma/dx: converted to per-bin content here."""
    if not Path(path).exists():
        raise SystemExit(f"missing Rivet output {path}")
    y = read_yoda(path)
    g = "g" if mode == "groomed" else "u"
    out = {}
    for name, (lo, hi) in RIVET_BINS.items():
        obj = y.get(f"/CMS_ZJET_JETMASS/rho_{g}_{name}")
        if obj is None:
            continue
        edges, v, e = obj
        w = np.diff(edges)
        out[(lo, hi)] = (np.asarray(edges, float), np.nan_to_num(v * w), np.nan_to_num((e * w) ** 2))
    return out


def load_series(name, args, key, pt_axis, rho_axis):
    spec = SERIES[name]
    kind = spec["kind"]
    if kind == "pickle":
        return load_pickles([args.inputs / spec["path"]], key, None, pt_axis, rho_axis)
    if kind == "prod":
        gen = spec["gen"]
        if args.channel == "zjet":
            f, cats = PROD_ZJET[gen][args.mlm_eras]
            return load_pickles([args.inputs / "zjet/rho/original" / f], key, cats, pt_axis, rho_axis)
        folder, stem = PROD_HADRONIC[gen]
        eras = HADRONIC_ERAS if args.mlm_eras == "all" else (args.mlm_eras,)
        root = args.inputs / args.channel / "rho/original"
        return load_pickles([root / era / folder.format(ch=args.channel) / f"minimal_rho_{args.channel}_{stem}_{era}.pkl"
                             for era in eras], key, None, pt_axis, rho_axis)
    if kind == "yoda":
        if args.level != "gen" or args.channel != "zjet":
            raise SystemExit(f"{name}: Rivet samples exist for zjet gen level only")
        return load_yoda(args.rivet_out / spec["file"], args.mode)
    if kind == "unfolded":
        if args.mode != "groomed":
            raise SystemExit("data_unfolded: groomed only")
        d = pickle.load(open(args.unfold_root / spec["path"], "rb"))["unfolded_2dnorm"]
        pt = np.asarray(d.axes[0].edges, float)
        rho = np.asarray(d.axes[1].edges, float)
        v, var = np.asarray(d.values(), float), np.asarray(d.variances(), float)
        return {(float(lo), float(hi)): (rho, v[i], var[i]) for i, (lo, hi) in enumerate(zip(pt[:-1], pt[1:]))}
    raise SystemExit(f"unknown kind {kind}")


# ---------------------------------------------------------------------------
# grids, medians
def rebin(edges, values, variances, target):
    """Sum bins onto `target`; every target edge must be a source edge."""
    edges = np.asarray(edges, float)
    idx = []
    for t in target:
        hits = np.flatnonzero(np.isclose(edges, t))
        if len(hits) != 1:
            return None
        idx.append(int(hits[0]))
    v, var = np.nan_to_num(np.asarray(values, float)), np.nan_to_num(np.asarray(variances, float))
    return (np.array([v[a:b].sum() for a, b in zip(idx[:-1], idx[1:])]),
            np.array([var[a:b].sum() for a, b in zip(idx[:-1], idx[1:])]))


def onto(series_bins, target_edges):
    out = {}
    for k, (e, v, var) in series_bins.items():
        if len(e) == len(target_edges) and np.allclose(e, target_edges):
            out[k] = (e, v, var)
            continue
        r = rebin(e, v, var, target_edges)
        if r is None:
            raise SystemExit("cannot rebin: target edges are not a subset of the sample's edges")
        out[k] = (np.asarray(target_edges, float), r[0], r[1])
    return out


def above(edges, v, var, floor):
    keep = 0.5 * (edges[:-1] + edges[1:]) > floor
    first = int(np.argmax(keep))
    return edges[first:], v[keep], var[keep]


def median_stats(v, var, centers):
    """(median, its error, n_eff) on the kept rho bins: interpolated CDF median,
    error 1.25 * 0.5 * (q84 - q16) / sqrt(n_eff).  Negative bin contents (unfolded
    data, negative-weight samples in sparse tails) are clipped to zero first."""
    v = np.clip(np.nan_to_num(np.asarray(v, float)), 0.0, None)
    var = np.nan_to_num(np.asarray(var, float))
    total = v.sum()
    if total <= 0:
        return np.nan, np.nan, 0.0
    cdf = np.cumsum(v) / total
    q = lambda p: float(np.interp(p, cdf, centers))
    n_eff = total ** 2 / max(var.sum(), 1e-300)
    err = 1.25 * 0.5 * (q(0.84) - q(0.16)) / np.sqrt(max(n_eff, 1.0))
    return q(0.5), err, n_eff


# ---------------------------------------------------------------------------
# drawing helpers
def stamp(fig, inputs_text, enabled=True):
    if not enabled:
        return
    try:
        ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], capture_output=True, text=True,
                             timeout=10, cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:
        ver = "unknown"
    try:
        fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.965))
    except Exception:
        pass
    fig.text(0.985, 0.012, f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  inputs: {inputs_text}",
             ha="right", va="bottom", fontsize=8 * fig.get_figwidth() / 10.0, color="0.45", family="monospace")


def cms_x(ax, default=0.045):
    return next((t.get_position()[0] for t in ax.texts if t.get_text() == "CMS"), default)


def pt_text(lo, hi):
    return f"{lo:.0f}–{hi:.0f}" if hi < 1e4 else f"> {lo:.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--list", action="store_true", help="print the sample registry and exit")
    ap.add_argument("--channel", default="zjet", choices=tuple(CHANNEL_LABEL))
    ap.add_argument("--level", default="gen", choices=tuple(LEVEL))
    ap.add_argument("--mode", default="groomed", choices=tuple(MODE_TEXT))
    ap.add_argument("--mlm-eras", default="2018", choices=("2018", "2017", "2016", "2016APV", "all"),
                    help="era(s) of the production (prod) samples; zjet supports 2018 and all")
    ap.add_argument("--samples", default="mlm_pythia,mlm_herwig", help="comma-separated registry names, drawn in this order")
    ap.add_argument("--reference", default="mlm_pythia", help="denominator of the shape ratio panels")
    ap.add_argument("--pairs", default="",
                    help="mass-ratio figure: groups separated by ':' of num/den pairs separated by ',', "
                         "e.g. 'matchbox/mlm_pythia,mlm_herwig/mlm_pythia:matchbox/mlm_herwig' (one panel per group)")
    ap.add_argument("--figures", default="ratio,slope,shapes", help="subset of ratio,slope,shapes")
    ap.add_argument("--grid", default="reference", choices=("reference", "native", "common"))
    ap.add_argument("--rho-floor", type=float, default=-5.0)
    ap.add_argument("--mass-axis", action="store_true", help="shape panels on a log mass axis in GeV at the bin's central pT")
    ap.add_argument("--inputs", type=Path, default=Path.home() / "cernbox/unfold_inputs")
    ap.add_argument("--rivet-out", type=Path, default=Path(__file__).resolve().parents[2] / "rivet/out")
    ap.add_argument("--unfold-root", type=Path, default=Path.home() / "Projects/unfold")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/plots/rho_generators"))
    ap.add_argument("--tag", default=None, help="output filename prefix (default: <channel>_<mode>_<level>)")
    ap.add_argument("--rlabel", default=None, help="override the CMS label's right text")
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, s in SERIES.items():
            print(f"{k:16s} {s['kind']:9s} {s['label']}")
        return
    args.outdir.mkdir(parents=True, exist_ok=True)
    lvl, pt_axis, rho_axis, level_text = LEVEL[args.level]
    key = f"ptjet_rhojet_{'g' if args.mode == 'groomed' else 'u'}_{lvl}"
    mode_text = MODE_TEXT[args.mode]
    names = [s.strip() for s in args.samples.split(",") if s.strip()]
    for n in names + [args.reference]:
        if n not in SERIES:
            raise SystemExit(f"unknown sample {n}; --list shows the registry")
    tag = args.tag or f"{args.channel}_{args.mode}_{lvl}"
    has_data = any(SERIES[n].get("points") for n in names)
    if args.rlabel:
        rlabel = args.rlabel
    elif has_data:
        rlabel = f"{LUMI.get(args.mlm_eras, LUMI['all'])} (13 TeV)"
    else:
        rlabel = ("2018 (13 TeV)" if args.mlm_eras == "2018" else "Run 2 (13 TeV)") if args.level == "reco" or all(
            SERIES[n]["kind"] != "yoda" for n in names) else "13 TeV"
    cms_kw = dict(text="Preliminary" if has_data else "", data=has_data, loc=2, rlabel=rlabel)
    inputs_text = f"{args.channel}/rho {lvl} ({', '.join(names)})"

    #### load, then put every sample on the requested grid
    raw = {n: load_series(n, args, key, pt_axis, rho_axis) for n in names}
    if args.reference not in raw:
        raw[args.reference] = load_series(args.reference, args, key, pt_axis, rho_axis)
    pt_bins = list(raw[args.reference].keys())
    if args.grid == "reference":
        data = {}
        for n, bins in raw.items():
            data[n] = {}
            for k in pt_bins:
                if k not in bins:
                    continue
                ref_e = raw[args.reference][k][0]
                data[n][k] = onto({k: bins[k]}, ref_e)[k]
    elif args.grid == "common":
        data = {n: onto({k: b for k, b in bins.items() if k in pt_bins}, COMMON_EDGES) for n, bins in raw.items()}
    else:
        data = {n: {k: b for k, b in bins.items() if k in pt_bins} for n, bins in raw.items()}

    #### medians per pT bin
    med, err, neff = {}, {}, {}
    for n, bins in data.items():
        med[n], err[n], neff[n] = {}, {}, {}
        for k, (e, v, var) in bins.items():
            ek, vk, vark = above(e, v, var, args.rho_floor)
            c = 0.5 * (ek[:-1] + ek[1:])
            med[n][k], err[n][k], neff[n][k] = median_stats(vk, vark, c)

    def ratio(num, den, k):
        if k not in med[num] or k not in med[den] or np.isnan(med[num][k]) or np.isnan(med[den][k]):
            return np.nan, np.nan
        d = med[num][k] - med[den][k]
        r = 10 ** (d / 2)
        return r, r * np.log(10) / 2 * np.sqrt(err[num][k] ** 2 + err[den][k] ** 2)

    groups = [[tuple(p.split("/")) for p in g.split(",") if p.strip()] for g in args.pairs.split(":") if g.strip()]
    for g in groups:
        for a, b in g:
            for n in (a, b):
                if n not in med:
                    raise SystemExit(f"--pairs uses {n}, which is not in --samples/--reference")

    #### the table
    print(f"\n{CHANNEL_LABEL[args.channel]}, {mode_text} mass, {lvl} level, eras {args.mlm_eras}, grid {args.grid}, "
          f"rho floor {args.rho_floor}")
    head = f"{'pT bin':>10s} | " + " ".join(f"{SERIES[n]['label'][:26]:>26s}" for n in data)
    if groups:
        head += " | " + " ".join(f"{a + '/' + b:>24s}" for g in groups for a, b in g)
    print(head)
    print(f"{'':>10s} | " + " ".join(f"{'median ± err (n_eff)':>26s}" for _ in data)
          + (" | " + " ".join(f"{'mass ratio':>24s}" for g in groups for _ in g) if groups else ""))
    for k in pt_bins:
        row = f"{pt_text(*k):>10s} | " + " ".join(
            f"{med[n].get(k, np.nan):8.3f}±{err[n].get(k, np.nan):.3f} ({neff[n].get(k, 0):9.0f})" for n in data)
        if groups:
            row += " | " + " ".join("{:12.3f} ± {:<9.3f}".format(*ratio(a, b, k)) for g in groups for a, b in g)
        print(row)
    ref_bin = next((k for k in pt_bins if np.isclose(k[0], 200.0)), pt_bins[0])
    last = pt_bins[-1]
    print("slope (median in the last bin relative to %s):" % pt_text(*ref_bin),
          {n: round(float(med[n].get(last, np.nan) - med[n].get(ref_bin, np.nan)), 3) for n in data})

    x = np.array([0.5 * (lo + min(hi, 600.0 if args.channel == "zjet" else 1000.0)) for lo, hi in pt_bins])
    xlim = (170, 540) if args.channel == "zjet" else (170, 950)
    figures = [f.strip() for f in args.figures.split(",") if f.strip()]
    many = len(names) > 5      # crowded legend: two columns, smaller type, more headroom
    legend_kw = dict(ncol=2 if many else 1, fontsize=PUB_LEGEND_FS - (5 if many else 2), frameon=False)
    if many:   # a two-column legend spans the top: start it below the CMS block
        legend_kw["bbox_to_anchor"] = (0.99, 0.87)

    def finish(ax, ylabel, note, loc="upper right", headroom=1.5, note_at="top"):
        hep.cms.label(ax=ax, **cms_kw)
        ax.set_xlabel(r"$p_{T}^{jet}$ (GeV)", fontsize=PUB_LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=PUB_LABEL_FS - 4)
        ax.tick_params(labelsize=PUB_TICK_FS)
        ax.set_xlim(*xlim)
        if args.channel == "zjet":
            ax.set_xticks([200, 300, 400, 500])
            ax.set_xticklabels(["200", "300", "400", "> 400"])
        lo, hi = ax.get_ylim()
        step = next((st for st in (0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0) if (hi - lo) / st <= 9), 10.0)
        bottom = np.floor(lo / step) * step
        ax.set_ylim(bottom, bottom + (hi - bottom) * headroom)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        y = 0.74 if note_at == "top" else 0.04
        ax.text(cms_x(ax), y, note, transform=ax.transAxes, ha="left", va="top" if note_at == "top" else "bottom",
                fontsize=PUB_ANNOT_FS)
        ax.legend(loc=loc, borderaxespad=1.0, **legend_kw)

    def style(n):
        s = SERIES[n]
        return s["label"], s["color"], s["marker"], s["ls"], s["lw"], s.get("open", False), s.get("points", False)

    #### figure: median mass ratios of the requested pairs, one panel per group
    if "ratio" in figures and groups:
        ncol = len(groups)
        fig, axes = plt.subplots(1, ncol, layout="constrained", figsize=(10.5 * ncol, 10.6) if ncol > 1 else None)
        axes = np.atleast_1d(axes)
        for ax, g in zip(axes, groups):
            for a, b in g:
                la, color, marker, ls, lw, is_open, pts = style(a)
                rr = np.array([ratio(a, b, k) for k in pt_bins])
                ok = np.isfinite(rr[:, 0])
                ax.errorbar(x[ok], rr[ok, 0], yerr=rr[ok, 1], marker=marker, ls="none" if pts else ls, lw=2.5, ms=11,
                            color=color, mfc="white" if is_open else color, mew=2.0, capsize=4,
                            label=f"{SERIES[a]['short']} / {SERIES[b]['short']}")
            ax.axhline(1.0, color="gray", ls=":", lw=1.5)
            finish(ax, "median mass ratio", f"{CHANNEL_LABEL[args.channel]}, {mode_text}, {level_text}")
        stamp(fig, inputs_text, enabled=not args.no_stamp)
        out = args.outdir / f"{tag}_mass_ratio.png"
        fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
        print(f"wrote {out}")

    #### figure: slope of every sample relative to its own 200-290 bin
    if "slope" in figures:
        fig, ax = plt.subplots(layout="constrained")
        for n in data:
            la, color, marker, ls, lw, is_open, pts = style(n)
            ks = [k for k in pt_bins if k in med[n] and ref_bin in med[n]]
            xs = np.array([x[pt_bins.index(k)] for k in ks])
            rel = np.array([med[n][k] - med[n][ref_bin] for k in ks])
            ye = np.array([np.sqrt(err[n][k] ** 2 + err[n][ref_bin] ** 2) for k in ks])
            ax.errorbar(xs, rel, yerr=ye, marker=marker, ls="none" if pts else ls, lw=lw if lw > 0 else 2.0, ms=11,
                        color=color, mfc="white" if is_open else color, mew=2.0, capsize=3, label=la,
                        zorder=6 if pts else 3)
        ax.axhline(0.0, color="gray", ls=":", lw=1.2)
        finish(ax, r"median $\log_{10}(\rho^2)$ relative to %s GeV" % pt_text(*ref_bin),
               f"{CHANNEL_LABEL[args.channel]}, {mode_text}, {level_text}\nshape per bin; tune offsets removed",
               headroom=1.9 if many else 1.4, note_at="bottom")
        stamp(fig, inputs_text, enabled=not args.no_stamp)
        out = args.outdir / f"{tag}_slope.png"
        fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
        print(f"wrote {out}")

    #### figure: per-pT-bin shapes with the ratio to the reference
    if "shapes" in figures:
        for i, k in enumerate(pt_bins):
            if k not in data[args.reference]:
                continue
            fig = plt.figure(layout="constrained", figsize=(10, 12))
            gs = fig.add_gridspec(2, 1, height_ratios=(3, 1.2))
            ax, rax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
            pt_ref = 0.5 * (k[0] + min(k[1], 1000.0))
            to_x = (lambda e: pt_ref * JET_R * 10 ** (np.asarray(e, float) / 2)) if args.mass_axis else (lambda e: np.asarray(e, float))
            ref_e, ref_v, ref_var = above(*data[args.reference][k], args.rho_floor)
            ymax, tops, drawn = 0.0, [], False
            for n in names:
                if k not in data[n]:
                    continue
                la, color, marker, ls, lw, is_open, pts = style(n)
                e, v, var = above(*data[n][k], args.rho_floor)
                tot = v.sum()
                if tot <= 0:
                    continue
                w = np.diff(e)
                y, ye = v / tot / w, np.sqrt(np.clip(var, 0, None)) / tot / w
                xe = to_x(e)
                c = np.sqrt(xe[:-1] * xe[1:]) if args.mass_axis else 0.5 * (xe[:-1] + xe[1:])
                if pts:
                    ax.errorbar(c, y, yerr=ye, xerr=np.diff(xe) / 2, fmt="o", color=color, ms=7, label=la, zorder=6)
                else:
                    hep.histplot(y, xe, ax=ax, histtype="step", color=color, ls=ls, lw=lw, label=la)
                    ax.errorbar(c, y, yerr=ye, fmt="none", ecolor=color, elinewidth=1.2, capsize=2)
                ymax = max(ymax, float(np.nanmax(y + ye)))
                drawn = True
                if n == args.reference:
                    continue
                # ratio on the numerator's grid; the reference is rebinned onto it if needed
                if len(e) == len(ref_e) and np.allclose(e, ref_e):
                    rv, rvar = ref_v, ref_var
                else:
                    rb = rebin(ref_e, ref_v, ref_var, e)
                    if rb is None:
                        continue
                    rv, rvar = rb
                rtot = rv.sum()
                ry, rye = rv / rtot / w, np.sqrt(np.clip(rvar, 0, None)) / rtot / w
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(ry > 0, y / ry, np.nan)
                    re_ = np.where(ry > 0, np.abs(r) * np.sqrt((ye / np.where(y > 0, y, 1)) ** 2 + (rye / ry) ** 2), np.nan)
                if pts:
                    rax.errorbar(c, r, yerr=re_, fmt="o", color=color, ms=7, capsize=2, zorder=6)
                else:
                    hep.histplot(r, xe, ax=rax, histtype="step", color=color, ls=ls, lw=lw)
                    rax.errorbar(c, r, yerr=re_, fmt="none", ecolor=color, elinewidth=1.2, capsize=2)
                tops.append(np.nanmax(r + re_) if np.isfinite(r + re_).any() else np.nan)
            if not drawn:
                plt.close(fig)
                continue
            hep.cms.label(ax=ax, **cms_kw)
            # with a crowded two-column legend (anchored at 0.87, five rows) the note goes below it
            ax.text(cms_x(ax), 0.50 if many else 0.74,
                    f"{CHANNEL_LABEL[args.channel]}, {mode_text}, {level_text}\n{pt_text(*k)} GeV",
                    transform=ax.transAxes, ha="left", va="top", fontsize=PUB_ANNOT_FS, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0) if many else None)
            if args.mass_axis:
                ax.set_xscale("log"); ax.set_yscale("log"); rax.set_xscale("log")
                ax.set_ylabel(r"$(1/N)\,dN/dm$ (GeV$^{-1}$)", fontsize=PUB_LABEL_FS - 4)
                ax.set_ylim(ymax * 1e-4, ymax * 100.0)
                rax.set_xlabel(("Soft-drop" if args.mode == "groomed" else "Ungroomed") + " jet mass (GeV)", fontsize=PUB_LABEL_FS - 4)
            else:
                ax.set_ylim(0, ymax * (2.2 if many else 1.6))
                ax.set_ylabel(r"$1/N\; dN/d\log_{10}(\rho^2)$", fontsize=PUB_LABEL_FS - 4)
                rax.set_xlabel(r"$\log_{10}(\rho^2)$, " + mode_text, fontsize=PUB_LABEL_FS - 4)
            ax.tick_params(labelsize=PUB_TICK_FS, labelbottom=False)
            ax.legend(loc="upper right", **{**legend_kw, "fontsize": PUB_LEGEND_FS - (6 if many else 3)})
            rax.axhline(1.0, color="gray", ls=":", lw=1.2)
            rax.set_ylabel(f"ratio to {SERIES[args.reference]['label'].split(',')[0]}", fontsize=PUB_LABEL_FS - 8)
            rax.tick_params(labelsize=PUB_TICK_FS - 4)
            rlo, rhi = rax.get_ylim()
            finite = [t for t in tops if np.isfinite(t)]
            rax.set_ylim(max(0.0, rlo), max([rhi] + [t * 1.05 for t in finite]))
            rax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            xe_ref = to_x(ref_e)
            for a in (ax, rax):
                a.set_xlim(xe_ref[0], xe_ref[-1])
            fig.align_ylabels([ax, rax])
            stamp(fig, inputs_text, enabled=not args.no_stamp)
            out = args.outdir / f"{tag}_shape_pt{i}.png"
            fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
