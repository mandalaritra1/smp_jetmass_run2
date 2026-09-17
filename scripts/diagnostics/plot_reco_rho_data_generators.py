"""Detector-level rho: data vs PYTHIA vs HERWIG, all channels (2026-09-17).

Reco-level companion to plot_zjet_gen_rho_generators.py.  Everything here is
measured or simulated at detector level -- no unfolding -- so data can be
overlaid directly on the two generator predictions.  Shapes only: every
spectrum is normalized to unit area in each pT bin, so the (large) HERWIG
normalization offset does not enter.

    python scripts/diagnostics/plot_reco_rho_data_generators.py --channel dijet

Bins where a simulation carries more than 50% MC error are dropped from the
ratio panel.  `--modes` defaults to groomed: the ungroomed mass agrees between
the generators to a few percent.
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
MODES = {"groomed": "ptjet_rhojet_g_reco", "ungroomed": "ptjet_rhojet_u_reco"}
CHANNEL_LABEL = {"zjet": "Z+jet", "dijet": "Dijet", "trijet": "Trijet"}
STYLE = {
    "data":   ("Data",           "k",       "-",  "o"),
    "pythia": ("Pythia8 (CP5)",  "#5790fc", ":",  None),
    "herwig": ("Herwig7 (CH3)",  "#e42536", "-.", None),
}


def stamp(fig, inputs, enabled=True):
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
             f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  inputs: {inputs}",
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


def load(channel, inputs_root, mode):
    """{source: (values, variances)} plus the rho and pT edges, summed over eras."""
    key = MODES[mode]
    root = Path(inputs_root) / channel / "rho" / "original"
    files = {
        "zjet": {s: [root / f"{s}_all.pkl"] for s in ("data", "pythia", "herwig")},
    }.get(channel)
    if files is None:
        files = {}
        for source, folder, stem in (("data", f"{channel}_data", "data"),
                                     ("pythia", f"{channel}_mc", "mg_pythia8"),
                                     ("herwig", f"{channel}_herwig", "herwig")):
            files[source] = [root / era / folder / f"minimal_rho_{channel}_{stem}_{era}.pkl"
                             for era in HADRONIC_ERAS]
    out, rho_edges, pt_edges = {}, None, None
    for source, paths in files.items():
        values = variances = None
        for path in paths:
            h = project(pickle.load(open(path, "rb"))[key])
            v, var = np.asarray(h.values()), np.asarray(h.variances())
            values = v.copy() if values is None else values + v
            variances = var.copy() if variances is None else variances + var
            rho_edges = h.axes["mpt_reco"].edges
            pt_edges = h.axes["ptreco"].edges
        out[source] = (values, variances)
    return out, rho_edges, pt_edges


def density(values, variances, edges):
    widths = np.diff(edges)
    total = values.sum()
    if total <= 0:
        return None, None
    return values / widths / total, np.sqrt(np.clip(variances, 0.0, None)) / widths / total


def pt_label(lo, hi):
    return (rf"${lo:.0f} < p_{{T}}^{{jet}} < {hi:.0f}$ GeV" if hi < 1e4
            else rf"$p_{{T}}^{{jet}} > {lo:.0f}$ GeV")


JET_R = 0.8


def panel(data, mode, pt_index, lo, hi, edges, outdir, channel, lumi, no_stamp,
          mass_axis=False):
    """One pT bin.  With ``mass_axis`` the rho edges are mapped to the
    soft-drop mass in GeV at the bin's central pT, m = pT R 10^(rho/2): the
    same histogram on a log mass axis, density per GeV."""
    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})
    pt_ref = 0.5 * (lo + min(hi, 1000.0))
    if mass_axis:
        edges = pt_ref * JET_R * 10 ** (np.asarray(edges, dtype=float) / 2)
    centers = np.sqrt(edges[:-1] * edges[1:]) if mass_axis else 0.5 * (edges[:-1] + edges[1:])
    curves, ymax = {}, 0.0
    for source, (label, color, style, marker) in STYLE.items():
        values, variances = data[source]
        dens, err = density(values[pt_index], variances[pt_index], edges)
        if dens is None:
            continue
        curves[source] = (dens, err)
        if source == "data":
            ax.errorbar(centers, dens, yerr=err, xerr=np.diff(edges) / 2, fmt=marker,
                        color=color, ms=7, label=label)
        else:
            ax.stairs(dens, edges, color=color, lw=2.5, ls=style, label=label,
                      baseline=None)
            ax.errorbar(centers, dens, yerr=err, fmt="none", ecolor=color,
                        elinewidth=1.2, capsize=2)
        ymax = max(ymax, float((dens + err).max()))

    hep.cms.label("Preliminary", data=True, loc=2, ax=ax, rlabel=f"{lumi} (13 TeV)")
    ax.tick_params(labelsize=PUB_TICK_FS)
    if mass_axis:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$(1/N)\,dN/dm$ (GeV$^{-1}$)", fontsize=PUB_LABEL_FS)
        positive = np.concatenate([d[d > 0] for d, _ in curves.values()])
        ax.set_ylim(positive.min() / 3.0, ymax * 100.0)
        ax.set_xlim(edges[0], edges[-1])
        ticks = [t for t in (2, 5, 10, 20, 50, 100, 200, 500) if edges[0] <= t <= edges[-1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        note = (f"{CHANNEL_LABEL[channel]}, {mode}, RECO\n{pt_label(lo, hi)}\n"
                rf"$m = p_T R\,10^{{\rho/2}}$ at $p_T$ = {pt_ref:.0f} GeV")
        ax.text(0.97, 0.70, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=PUB_ANNOT_FS - 2)
    else:
        ax.set_ylabel("Normalized", fontsize=PUB_LABEL_FS)
        ax.set_ylim(0, ymax * 1.55)
        cms_x = next((t.get_position()[0] for t in ax.texts if t.get_text() == "CMS"), 0.045)
        ax.text(cms_x, 0.66, f"{CHANNEL_LABEL[channel]}, {mode}, RECO\n{pt_label(lo, hi)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=PUB_ANNOT_FS,
                zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=2.0))
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] != "Data")
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper right", fontsize=PUB_LEGEND_FS, frameon=False)

    drawn = []
    if "data" in curves:
        dd, ed = curves["data"]
        rel_d = np.divide(ed, dd, out=np.zeros_like(dd), where=dd > 0)
        for source in ("pythia", "herwig"):
            if source not in curves:
                continue
            ds, es = curves[source]
            ratio = np.divide(dd, ds, out=np.full_like(dd, np.nan), where=ds > 0)
            rel_s = np.divide(es, ds, out=np.zeros_like(ds), where=ds > 0)
            err = np.abs(ratio) * np.sqrt(rel_d ** 2 + rel_s ** 2)
            usable = np.isfinite(ratio) & (rel_s < 0.5) & (rel_d < 0.5) & (dd > 0)
            color, style = STYLE[source][1], STYLE[source][2]
            rax.stairs(np.where(usable, ratio, np.nan), edges, color=color, lw=2.5,
                       ls=style, baseline=None)
            rax.errorbar(centers[usable], ratio[usable], yerr=err[usable], fmt="none",
                         ecolor=color, elinewidth=1.2, capsize=2)
            drawn.extend([(ratio + err)[usable], (ratio - err)[usable]])
    rax.axhline(1.0, color="gray", ls="--", lw=1)
    if drawn:
        flat = np.concatenate(drawn)
        flat = flat[np.isfinite(flat)]
        if flat.size:
            pad = 0.18 * max(flat.max() - flat.min(), 0.2)
            rax.set_ylim(max(0.0, flat.min() - pad), flat.max() + pad)
    rax.set_ylabel("Data / MC", fontsize=PUB_LABEL_FS - 4)
    if mass_axis:
        rax.set_xlabel(("Soft-drop" if mode == "groomed" else "Ungroomed") + " jet mass (GeV)",
                       fontsize=PUB_LABEL_FS)
        rax.set_xscale("log")
        rax.set_xticks(ticks)
        rax.set_xticklabels([str(t) for t in ticks])
    else:
        rax.set_xlabel(r"$\log_{10}(\rho^{2}) = 2\log_{10}(m/(p_{T}R))$, RECO",
                       fontsize=PUB_LABEL_FS)
    rax.tick_params(labelsize=PUB_TICK_FS)

    stamp(fig, f"{channel}/rho/original reco", enabled=not no_stamp)
    out = Path(outdir) / f"reco_{'mass' if mass_axis else 'rho'}_{channel}_{mode}_pt{pt_index}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--channel", default="dijet", choices=("zjet", "dijet", "trijet"))
    ap.add_argument("--inputs", type=Path, default=Path.home() / "Projects/unfold/inputs")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/plots/reco_rho_data"))
    ap.add_argument("--modes", default="groomed")
    ap.add_argument("--rho-floor", type=float, default=-5.0)
    ap.add_argument("--lumi", default=r"138 fb$^{-1}$")
    ap.add_argument("--mass-axis", action="store_true",
                    help="draw each pT bin on a log mass axis in GeV (m = pT R 10^(rho/2) at the bin centre)")
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    made = []
    print(f"{args.channel}: median implied mass scale relative to Pythia8, RECO")
    print(f"{'mode':10s} {'pT bin':14s} {'data/PY':>9s} {'HW/PY':>9s}")
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        raw, rho_edges, pt_edges = load(args.channel, args.inputs, mode)
        keep = 0.5 * (rho_edges[:-1] + rho_edges[1:]) > args.rho_floor
        first = int(np.argmax(keep))
        edges = rho_edges[first:]
        sliced = {s: (v[:, keep], var[:, keep]) for s, (v, var) in raw.items()}
        centers = 0.5 * (edges[:-1] + edges[1:])
        for i, (lo, hi) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
            made.append(panel(sliced, mode, i, lo, hi, edges, args.outdir,
                              args.channel, args.lumi, args.no_stamp, mass_axis=args.mass_axis))
            med = {}
            for source in ("data", "pythia", "herwig"):
                v = np.clip(sliced[source][0][i], 0.0, None)
                if v.sum() > 0:
                    med[source] = float(np.interp(0.5, np.cumsum(v) / v.sum(), centers))
            if len(med) == 3:
                label = f"{lo:.0f}-{hi:.0f}" if hi < 1e4 else f">{lo:.0f}"
                print(f"{mode:10s} {label:14s} "
                      f"{10 ** ((med['data'] - med['pythia']) / 2):9.3f} "
                      f"{10 ** ((med['herwig'] - med['pythia']) / 2):9.3f}")
    print(f"\n{len(made)} figures in {args.outdir}")


if __name__ == "__main__":
    main()
