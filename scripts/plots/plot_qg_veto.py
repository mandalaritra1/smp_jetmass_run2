#!/usr/bin/env python
"""CMS-style plots of the hadronic q/g content and the trijet-veto effect.

Inputs: the 2026-07-31 veto'd `minimal_rho` production (dijet MC / trijet MC /
dijet data) plus the pre-veto production for the veto-effect ratio. One square
panel per figure (document grids handle layout); every figure carries the
provenance stamp (scripts/plot_stamp.py).

    python scripts/plot_qg_veto.py --outdir <dir>
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plot_stamp import stamp  # noqa: E402

hep.style.use(hep.style.CMS)

#### Petroff 6-color scheme (CMS default categorical palette)
P6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
FLAVS = ["Gluon", "UDS", "Charm", "Bottom"]
INPUT_TAG = "minrho_veto_20260731 (adb7024)"
#### publication sizes (reviewer-driven; mirrors unfold's unfolder_core PUB_*)
PUB_LABEL_FS, PUB_TICK_FS, PUB_LEGEND_FS, PUB_ANNOT_FS = 30, 26, 20, 22
#### display cap for the open-ended last pt bin (real edge 13000)
PT_CAP = 1000.0


def load_postprocessed(pkl, channel):
    out = pickle.load(open(pkl, "rb"))
    if channel == "dijet":
        from smp_jetmass_run2.dijet_processor import DijetProcessor as P
    else:
        from smp_jetmass_run2.trijet_processor import TrijetProcessor as P
    return P(do_gen=True, mode="minimal_rho", jet_systematics=["nominal"],
             systematics=[]).postprocess(out)


def flav_vs_pt(out):
    """{flav: fraction array over pt bins}, plus the pt edges (capped)."""
    h = out["ptjet_rhojet_g_reco_flav"][{"dataset": sum}]
    edges = h.axes["ptreco"].edges.copy()
    per = {fl: h[{"partonFlav": fl}].project("ptreco").values() for fl in FLAVS}
    tot = np.sum(list(per.values()), axis=0)
    with np.errstate(invalid="ignore"):
        frac = {fl: np.where(tot > 0, v / tot, np.nan) for fl, v in per.items()}
    edges[-1] = min(edges[-1], PT_CAP)
    return edges, frac


def gluon_vs_rho(out, ptlo, pthi):
    h = out["ptjet_rhojet_g_reco_flav"][{"dataset": sum}]
    sl = {"ptreco": slice(hist_loc(h.axes["ptreco"], ptlo),
                          hist_loc(h.axes["ptreco"], pthi), sum)}
    g = h[{"partonFlav": "Gluon"}][sl].values()
    tot = sum(h[{"partonFlav": fl}][sl].values() for fl in FLAVS)
    with np.errstate(invalid="ignore"):
        return h.axes["mpt_reco"].edges, np.where(tot > 0, g / tot, np.nan)


def hist_loc(axis, value):
    return int(np.searchsorted(axis.edges, value + 1e-6) - 1)


def reco_pt(out):
    h = out["ptjet_rhojet_g_reco"][{"dataset": sum, "systematic": "nominal"}]
    hh = h.project("ptreco")
    return hh.axes[0].edges, hh.values()


def new_panel(rlabel, data=False):
    fig, ax = plt.subplots(layout="constrained")
    #### CMS block INSIDE the frame (loc=2) -- house preference 2026-07-31;
    #### keep the top-left clear of curves/legend/text
    hep.cms.label("Preliminary", data=data, loc=2, rlabel=rlabel, ax=ax)
    ax.tick_params(labelsize=PUB_TICK_FS)
    return fig, ax


def overflow_break(ax, x, gap=0.006, half=0.014, top=False):
    """Broken-axis crack marking the compressed overflow last bin (820-13000
    drawn to PT_CAP); pairs with the true-edge last tick label."""
    xlim = ax.get_xlim()
    xa = (x - xlim[0]) / (xlim[1] - xlim[0])
    lw = plt.rcParams["axes.linewidth"]
    #### bottom spine only by default: a top crack slices through the rlabel
    #### ("2018 (13 TeV)") sitting just above the frame
    for y in ((0, 1) if top else (0,)):
        ax.plot([xa - gap, xa + gap], [y, y], transform=ax.transAxes,
                color="white", lw=lw * 1.8, clip_on=False, zorder=4)
        for o in (-gap, gap):
            ax.plot([xa + o - 0.006, xa + o + 0.006], [y - half, y + half],
                    transform=ax.transAxes, color="k", lw=lw, clip_on=False,
                    zorder=5)


def pt_overflow_axis(ax):
    """Shared pt-axis dressing: crack after the 820 edge, true 13000 edge tick."""
    overflow_break(ax, 860.0)
    ax.set_xticks([200, 400, 600, 800, PT_CAP],
                  ["200", "400", "600", "800", r"$\infty$"])


def cms_x(ax):
    """Left x (axes coords) of the inside CMS block, so in-panel text
    left-aligns with it exactly (mplhep pads loc=2 by ~0.045, not 0)."""
    for t in ax.texts:
        if t.get_text() == "CMS":
            return t.get_position()[0]
    return 0.03


def finish(ax, xlab, ylab):
    ax.set_xlabel(xlab, fontsize=PUB_LABEL_FS)
    ax.set_ylabel(ylab, fontsize=PUB_LABEL_FS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path.home() / "Downloads/minrho_veto_20260731"
    ap.add_argument("--dijet-mc", type=Path, default=base / "minimal_rho_dijet_mg_pythia8_2018.pkl")
    ap.add_argument("--trijet-mc", type=Path, default=base / "minimal_rho_trijet_mg_pythia8_2018.pkl")
    ap.add_argument("--dijet-data", type=Path, default=base / "minimal_rho_dijet_data_2018.pkl")
    ap.add_argument("--preveto-mc", type=Path,
                    default=Path.home() / "Projects/unfold/inputs/dijet/rho/minimal_rho_dijet_mg_pythia8_2018.pkl")
    ap.add_argument("--preveto-data", type=Path,
                    default=Path.home() / "Projects/unfold/inputs/dijet/rho/minimal_rho_dijet_data_2018.pkl")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/figs"))
    ap.add_argument("--no-stamp", action="store_true")
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    do_stamp = not args.no_stamp

    di = load_postprocessed(args.dijet_mc, "dijet")
    tri = load_postprocessed(args.trijet_mc, "trijet")

    #### 1. gluon fraction vs pt, both channels
    fig, ax = new_panel("2018 (13 TeV)")
    for out, label, color in ((di, "dijet (jets 1+2)", P6[0]),
                              (tri, "trijet (jet 3)", P6[1])):
        edges, frac = flav_vs_pt(out)
        ax.stairs(frac["Gluon"], edges, label=label, color=color, lw=2.5)
    ax.text(cms_x(ax), 0.80, "QCD MG+Pythia8", transform=ax.transAxes,
            ha="left", va="top", fontsize=PUB_ANNOT_FS)
    finish(ax, r"measured-jet $p_\mathrm{T}^\mathrm{reco}$ (GeV)", "Gluon fraction")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(185, PT_CAP)
    pt_overflow_axis(ax)
    ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS)
    stamp(fig, INPUT_TAG, do_stamp)
    fig.savefig(args.outdir / "hadronic_qg_gluon_fraction_fullstat.png", dpi=150)
    plt.close(fig)

    #### 2. flavour composition vs pt, one panel per channel
    for out, ch, meas in ((di, "dijet", "jets 1+2"), (tri, "trijet", "jet 3")):
        fig, ax = new_panel("2018 (13 TeV)")
        edges, frac = flav_vs_pt(out)
        for fl, color in zip(FLAVS, P6):
            ax.stairs(frac[fl], edges, label=fl, color=color, lw=2.5)
        ax.text(cms_x(ax), 0.80, f"QCD MG+Pythia8\n{ch} ({meas})", transform=ax.transAxes,
                ha="left", va="top", fontsize=PUB_ANNOT_FS)
        finish(ax, r"measured-jet $p_\mathrm{T}^\mathrm{reco}$ (GeV)", "Flavour fraction")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(185, PT_CAP)
        pt_overflow_axis(ax)
        ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS)
        stamp(fig, INPUT_TAG, do_stamp)
        fig.savefig(args.outdir / f"hadronic_qg_flavour_composition_{ch}.png", dpi=150)
        plt.close(fig)

    #### 3. gluon fraction vs rho_g, pt 200-290
    fig, ax = new_panel("2018 (13 TeV)")
    for out, label, color in ((di, "dijet (jets 1+2)", P6[0]),
                              (tri, "trijet (jet 3)", P6[1])):
        redges, gfrac = gluon_vs_rho(out, 200., 290.)
        ax.stairs(gfrac, redges, label=label, color=color, lw=2.5)
    ax.text(cms_x(ax), 0.80, "QCD MG+Pythia8\n" + r"$200 < p_\mathrm{T}^\mathrm{reco} < 290$ GeV",
            transform=ax.transAxes, ha="left", va="top", fontsize=PUB_ANNOT_FS)
    finish(ax, r"$\log_{10}(\rho^2)$ (groomed, detector)", "Gluon fraction")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-6, 0)
    ax.legend(loc="lower center", fontsize=PUB_LEGEND_FS)
    stamp(fig, INPUT_TAG, do_stamp)
    fig.savefig(args.outdir / "hadronic_qg_gluon_fraction_vs_rho_pt200290.png", dpi=150)
    plt.close(fig)

    #### 4. veto effect: veto'd / pre-veto reco yield vs pt (MC + data)
    fig, ax = new_panel("2018 (13 TeV)", data=True)
    for new_pkl, old_pkl, ch, label, color in (
            (args.dijet_mc, args.preveto_mc, "dijet", "QCD MG+Pythia8", P6[2]),
            (args.dijet_data, args.preveto_data, "dijet", "JetHT (prescale-weighted)", "black")):
        en, vn = reco_pt(load_postprocessed(new_pkl, ch) if "data" not in str(new_pkl)
                         else pickle.load(open(new_pkl, "rb")))
        eo, vo = reco_pt(load_postprocessed(old_pkl, ch) if "data" not in str(old_pkl)
                         else pickle.load(open(old_pkl, "rb")))
        with np.errstate(invalid="ignore"):
            r = np.where(vo > 0, vn / vo, np.nan)
        en = en.copy(); en[-1] = min(en[-1], PT_CAP)
        ax.stairs(r, en, label=label, color=color, lw=2.5)
    ax.axhline(1.0, color="0.6", lw=1, ls="--")
    finish(ax, r"jet $p_\mathrm{T}^\mathrm{reco}$ (GeV)",
           "dijet reco yield ratio (veto / no veto)")
    ax.set_ylim(0.8, 1.1)
    ax.set_xlim(185, PT_CAP)
    pt_overflow_axis(ax)
    ax.legend(loc="lower left", title="trijet-priority veto",
              fontsize=PUB_LEGEND_FS, title_fontsize=PUB_LEGEND_FS)
    stamp(fig, INPUT_TAG, do_stamp)
    fig.savefig(args.outdir / "dijet_trijet_veto_effect_vs_pt.png", dpi=150)
    plt.close(fig)

    for f in sorted(args.outdir.glob("*.png")):
        print(f)


if __name__ == "__main__":
    main()
