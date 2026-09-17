"""Gen-level rho spectra, HERWIG vs PYTHIA, for any channel (2026-09-17).

Companion to plot_zjet_ptbalance_gen.py.  For Z+jet the pT balance agrees
between the two generators, so a difference in rho = 2 log10(m/(pT R)) is a
difference in the JET MASS.  This script draws, per gen pT bin, the normalized
gen rho spectrum of both generators with a ratio panel, plus the implied
median mass-scale ratio versus jet pT.

    # Z+jet: merged-era production pickles
    python scripts/diagnostics/plot_zjet_gen_rho_generators.py --channel zjet

    # dijet / trijet: the per-era producer pickles, summed over eras
    python scripts/diagnostics/plot_zjet_gen_rho_generators.py --channel dijet

Bins below `--rho-floor` (default -5) are the buffer/catch-all region of the
production axes and carry almost no entries in HERWIG; they are dropped so an
empty-bin ratio cannot dominate the figure.  Ratio bins where either generator
carries more than 50% MC error are omitted for the same reason.
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
COLORS = {"pythia": "#5790fc", "herwig": "#e42536"}
LABELS = {"pythia": "Pythia8 (CP5)", "herwig": "Herwig7 (CH3)"}
MODES = {"groomed": "ptjet_rhojet_g_gen", "ungroomed": "ptjet_rhojet_u_gen"}
HADRONIC_ERAS = ("2016APV", "2016", "2017", "2018")
CHANNEL_LABEL = {"zjet": "Z+jet", "dijet": "Dijet", "trijet": "Trijet"}


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
    """Sum over the dataset axis and take the nominal systematic category."""
    names = [a.name for a in hist_obj.axes]
    index = {}
    if "dataset" in names:
        index["dataset"] = sum
    if "systematic" in names:
        index["systematic"] = ("nominal" if "nominal" in list(hist_obj.axes["systematic"])
                               else sum)
    return hist_obj[index] if index else hist_obj


def load_channel(channel, inputs_root):
    """{generator: {mode: (values, variances, gen rho edges, pT edges)}}.

    Z+jet ships merged-era pickles; dijet/trijet ship one pickle per era per
    generator, summed here.
    """
    root = Path(inputs_root) / channel / "rho" / "original"
    out = {}
    for generator in ("herwig", "pythia"):
        per_mode = {}
        if channel == "zjet":
            payloads = [pickle.load(open(root / f"{generator}_all.pkl", "rb"))]
        else:
            stem = "herwig" if generator == "herwig" else "mg_pythia8"
            folder = f"{channel}_herwig" if generator == "herwig" else f"{channel}_mc"
            payloads = [pickle.load(open(
                root / era / folder / f"minimal_rho_{channel}_{stem}_{era}.pkl", "rb"))
                for era in HADRONIC_ERAS]
        for mode, key in MODES.items():
            values = variances = None
            edges = pt_edges = None
            for payload in payloads:
                h = project(payload[key])
                v, var = np.asarray(h.values()), np.asarray(h.variances())
                values = v.copy() if values is None else values + v
                variances = var.copy() if variances is None else variances + var
                edges, pt_edges = h.axes["mpt_gen"].edges, h.axes["ptgen"].edges
            per_mode[mode] = (values, variances, edges, pt_edges)
        out[generator] = per_mode
    return out


def density(values, variances, edges):
    """Unit-area density and its MC-stat error on the same normalization."""
    widths = np.diff(edges)
    total = values.sum()
    if total <= 0:
        return None, None
    return values / widths / total, np.sqrt(np.clip(variances, 0.0, None)) / widths / total


def pt_label(lo, hi):
    return (rf"${lo:.0f} < p_{{T}}^{{jet}} < {hi:.0f}$ GeV" if hi < 1e4
            else rf"$p_{{T}}^{{jet}} > {lo:.0f}$ GeV")


def panel(data, mode, pt_index, lo, hi, edges, outdir, no_stamp, channel):
    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})
    curves, ymax = {}, 0.0
    for generator in ("pythia", "herwig"):
        all_values, all_variances = data[generator]
        dens, err = density(all_values[pt_index], all_variances[pt_index], edges)
        if dens is None:
            continue
        curves[generator] = (dens, err)
        ax.stairs(dens, edges, color=COLORS[generator], lw=2.5,
                  ls="dotted" if generator == "pythia" else "dashdot",
                  label=LABELS[generator], baseline=None)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.errorbar(centers, dens, yerr=err, fmt="none",
                    ecolor=COLORS[generator], elinewidth=1.2, capsize=2)
        ymax = max(ymax, float((dens + err).max()))

    hep.cms.label("", data=False, loc=2, ax=ax, rlabel="Run 2 (13 TeV)")
    ax.set_ylabel("Normalized", fontsize=PUB_LABEL_FS)
    ax.tick_params(labelsize=PUB_TICK_FS)
    ax.set_ylim(0, ymax * 1.55)
    cms_x = next((t.get_position()[0] for t in ax.texts if t.get_text() == "CMS"), 0.045)
    # Below the CMS block and above the rising edge of the spectrum; the rho
    # distributions peak on the right, so the left half stays free.
    ax.text(cms_x, 0.70, f"{CHANNEL_LABEL[channel]}, {mode}, GEN\n{pt_label(lo, hi)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=PUB_ANNOT_FS,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0))
    ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS, frameon=False)

    if "pythia" in curves and "herwig" in curves:
        dp, ep = curves["pythia"]
        dh, eh = curves["herwig"]
        ratio = np.divide(dh, dp, out=np.full_like(dh, np.nan), where=dp > 0)
        rel_h = np.divide(eh, dh, out=np.zeros_like(dh), where=dh > 0)
        rel_p = np.divide(ep, dp, out=np.zeros_like(dp), where=dp > 0)
        err = np.abs(ratio) * np.sqrt(rel_h ** 2 + rel_p ** 2)
        # A bin where either generator's own MC error exceeds 50% of its content
        # carries no information about the shape; its ratio would be an
        # unbounded spike that sets the panel range for everything else.
        usable = np.isfinite(ratio) & (rel_h < 0.5) & (rel_p < 0.5)
        centers = 0.5 * (edges[:-1] + edges[1:])
        rax.stairs(np.where(usable, ratio, np.nan), edges, color=COLORS["herwig"],
                   lw=2.5, baseline=None)
        rax.errorbar(centers[usable], ratio[usable], yerr=err[usable], fmt="none",
                     ecolor=COLORS["herwig"], elinewidth=1.2, capsize=2)
        if (~usable & np.isfinite(ratio)).any():
            # Top-left of the ratio panel with a white box: the drawn ratio can
            # run anywhere along the bottom, so a bottom-right note sits on it.
            rax.text(0.02, 0.94, "bins with >50% MC error omitted",
                     transform=rax.transAxes, ha="left", va="top",
                     fontsize=PUB_LEGEND_FS - 4, color="0.45",
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        if usable.any():
            hi_draw = float((ratio + err)[usable].max())
            lo_draw = float((ratio - err)[usable].min())
            pad = 0.22 * max(hi_draw - lo_draw, 0.2)
            rax.set_ylim(max(0.0, lo_draw - pad), hi_draw + pad)
    rax.axhline(1.0, color="gray", ls="--", lw=1)
    rax.set_ylabel("HW / PY", fontsize=PUB_LABEL_FS - 4)
    rax.set_xlabel(r"$\log_{10}(\rho^{2}) = 2\log_{10}(m/(p_{T}R))$, GEN",
                   fontsize=PUB_LABEL_FS)
    rax.tick_params(labelsize=PUB_TICK_FS)

    stamp(fig, f"{channel}/rho/original gen", enabled=not no_stamp)
    out = Path(outdir) / f"gen_rho_{channel}_{mode}_pt{pt_index}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def mass_scale_summary(summary, outdir, no_stamp, channel):
    """Implied median mass ratio m(HERWIG)/m(PYTHIA) versus jet pT, both modes."""
    fig, ax = plt.subplots(layout="constrained")
    styles = {"groomed": ("o", "-", "#5790fc"), "ungroomed": ("s", "--", "#964a8b")}
    for mode in summary:
        marker, style, color = styles[mode]
        x = [0.5 * (lo + min(hi, 600.0)) for lo, hi, _ in summary[mode]]
        y = [r for _, _, r in summary[mode]]
        ax.plot(x, y, marker=marker, ls=style, lw=2.5, ms=11, color=color, label=mode)
    ax.axhline(1.0, color="gray", ls=":", lw=1.5)
    hep.cms.label("", data=False, loc=2, ax=ax, rlabel="Run 2 (13 TeV)")
    ax.set_xlabel(r"$p_{T}^{jet}$ (GeV)", fontsize=PUB_LABEL_FS)
    ax.set_ylabel(r"median $m_{\mathrm{Herwig7}} / m_{\mathrm{Pythia8}}$",
                  fontsize=PUB_LABEL_FS)
    ax.tick_params(labelsize=PUB_TICK_FS)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * 1.45)
    ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS, frameon=False)
    ax.text(0.5, 0.02, CHANNEL_LABEL[channel], transform=ax.transAxes,
            ha="center", va="bottom", fontsize=PUB_ANNOT_FS)
    stamp(fig, f"{channel}/rho/original gen", enabled=not no_stamp)
    out = Path(outdir) / f"gen_mass_scale_ratio_{channel}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--channel", default="zjet", choices=("zjet", "dijet", "trijet"))
    ap.add_argument("--inputs", type=Path,
                    default=Path.home() / "Projects/unfold/inputs")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/plots/gen_rho_generators"))
    ap.add_argument("--rho-floor", type=float, default=-5.0)
    ap.add_argument("--modes", default="groomed",
                    help="comma-separated grooming modes to draw (default: groomed; "
                         "the ungroomed mass agrees between the generators)")
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw = load_channel(args.channel, args.inputs)
    made, summary = [], {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        values_p, variances_p, rho_edges, pt_edges = raw["pythia"][mode]
        keep = 0.5 * (rho_edges[:-1] + rho_edges[1:]) > args.rho_floor
        first = int(np.argmax(keep))
        edges = rho_edges[first:]
        sliced = {g: (raw[g][mode][0][:, keep], raw[g][mode][1][:, keep])
                  for g in ("herwig", "pythia")}
        rows = []
        for i, (lo, hi) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
            made.append(panel(sliced, mode, i, lo, hi, edges, args.outdir,
                              args.no_stamp, args.channel))
            med = {}
            for g in ("pythia", "herwig"):
                v = sliced[g][0][i]
                if v.sum() <= 0:
                    continue
                c = 0.5 * (edges[:-1] + edges[1:])
                med[g] = float(np.interp(0.5, np.cumsum(v) / v.sum(), c))
            if len(med) == 2:
                rows.append((lo, hi, 10 ** ((med["herwig"] - med["pythia"]) / 2)))
        summary[mode] = rows
    made.append(mass_scale_summary(summary, args.outdir, args.no_stamp, args.channel))

    print(f"\n{args.channel}: {'mode':12s} {'pT bin':22s} {'median m(HW)/m(PY)':>20s}")
    for mode, rows in summary.items():
        for lo, hi, r in rows:
            label = f"{lo:.0f}-{hi:.0f}" if hi < 1e4 else f">{lo:.0f}"
            print(f"{mode:12s} {label:22s} {r:20.3f}")
    print(f"\n{len(made)} figures in {args.outdir}")


if __name__ == "__main__":
    main()
