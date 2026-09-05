#!/usr/bin/env python
"""Figure set for the dijet/trijet groomed-rho unfolding on the combined Run 2
pair_split inputs.

Everything is read back from the Phase A/B products under
``outputs/pairsplit_unfold``; no unfolding driver is rerun here EXCEPT the
common six-bin dijet-vs-trijet comparison (figure 6), which needs its own
tau=0 pass on a coarser gen merge.  That pass lives behind ``--common-unfold``
and needs ROOT:

    source ~/Projects/unfold/scripts/setup_root.sh
    ~/Projects/unfold/.venv/bin/python scripts/plots/plot_hadronic_rho_pairsplit_run2.py --common-unfold

then the figures themselves (no ROOT needed):

    ~/Projects/GluonJetMass/.venv/bin/python scripts/plots/plot_hadronic_rho_pairsplit_run2.py

``--variant`` writes the presentation variants of the unfolded family into
``outputs/figs/hadronic_rho_pairsplit_run2/variants`` and touches nothing else;
see ``variants()``.

Conventions
-----------
* production gen merge = the ``proposal_coarse_tail`` row of the stability JSON
  (dijet 10 reported rho bins per pT slice, trijet 6).
* the [-10, -4] rho buffer bin is never shown; the reported window starts at
  the first physics bin.
* "normalized shape" here = unit area over the SHOWN bins of a pT slice,
  divided by the bin width (a density in log(rho^2)).  The relative systematic
  and statistical bands come from the Phase B JSONs, where the same shapes are
  normalized over the narrower core window instead -- the difference is a
  per-slice constant and is negligible for a band.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

hep.style.use(hep.style.CMS)

REPO = Path(__file__).resolve().parents[2]
IN = REPO / "outputs" / "pairsplit_unfold"
OUT = REPO / "outputs" / "figs" / "hadronic_rho_pairsplit_run2"
VARIANT_OUT = OUT / "variants"
COMMON_NPZ = OUT / "compare_channels_common6.npz"
#### save() writes here; --variant repoints it so the production figures are
#### never touched.
OUT_DIR = OUT

PROD_ROW = "proposal_coarse_tail"
ERA = r"Run 2, 137.6 fb$^{-1}$ (13 TeV)"
SIM_ERA = "13 TeV"
DATA_LABEL = "Data (JetHT, Run 2)"
MC_LABEL = "MG+Pythia8 CP5"
XLAB = r"Groomed $\log(\rho^{2})$"
XLAB_RECO = r"Groomed $\log(\rho^{2})$ (detector)"
YLAB = r"$(1/N)\;\mathrm{d}N/\mathrm{d}\log(\rho^{2})$"
CHANNELS = ("dijet", "trijet")

PETROFF = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
PETROFF10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
             "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
DET_COLOR = "#5790fc"
FSR_COLOR = "#f89c20"
JES_COLOR = "#a96b59"

#### the 27-source era-correlated JES band lives in the unfold repo (currently
#### ~/Projects/unfold/_from_smp_20260813/studies_unfold/rho_jes_era_correlated.py).
#### Off by default so the production figure set
#### is unchanged; ``--jes-update`` points it at the JSON and writes ONLY the
#### uncertainty family, into OUT/"jes_update".
JES_JSON: Path | None = None
JES_SPLIT_MODE = "sqrt"
JES_UPDATE_OUT = OUT / "jes_update"

#### the common gen binning both channels are re-merged onto for figure 6:
#### rho [-4, -2.25, -1.8, -1.5, -1.1, -0.75, 0] (plus the [-10,-4] buffer)
#### x pT {200-290, 290-400, >400}.  The reco rho-group indices below are into
#### each channel's own reco axis, which is why they differ.
COMMON_RHO_EDGES = [-10.0, -4.0, -2.25, -1.8, -1.5, -1.1, -0.75, 0.0]
COMMON_MERGE = {
    "dijet": dict(pt_groups=[[0], [1], [2, 3, 4]],
                  rho_groups=[[0], [1, 2, 3], [4], [5], [6, 7], [8, 9],
                              [10, 11, 12]]),
    "trijet": dict(pt_groups=[[0], [1], [2]],
                   rho_groups=[[0], [1, 2], [3], [4], [5], [6], [7]]),
}


# --------------------------------------------------------------------------
# small style helpers (the cms-plot-style skill, minus the grid)
# --------------------------------------------------------------------------
def cms_label(ax, rlabel=ERA, simulation=False, loc=2, label="Internal"):
    hep.cms.label(label, data=not simulation, loc=loc, ax=ax, rlabel=rlabel)
    ax.margins(x=0.02)
    return ax


def anchor_x(ax, fallback=0.045):
    for t in ax.texts:
        if t.get_text() == "CMS":
            return t.get_position()[0]
    return fallback


def cms_block_bottom(ax, fallback=0.86):
    """Bottom of the inside CMS block ("CMS" + the italic "Internal") in axes
    coords.  Measured, not guessed: at the CMS style's absolute font sizes the
    block ends at ~0.859, so a hand-set annotation at 0.86 lands right on the
    descenders of "Internal"."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    ys = []
    for t in ax.texts:
        bb = t.get_window_extent(rend).transformed(inv)
        if bb.y1 <= 1.0 + 1e-6:      # skip the rlabel, which sits above the frame
            ys.append(bb.y0)
    return min(ys) if ys else fallback


def in_panel_text(ax, text, fontsize=22, pad=0.035, linespacing=1.6, **kw):
    """Annotation left-aligned with the inside CMS block and pushed clear of
    its bottom edge.  Returns the bottom of the annotation in axes coords so
    the caller can anchor a legend underneath it."""
    t = ax.text(anchor_x(ax), cms_block_bottom(ax) - pad, text,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=fontsize, linespacing=linespacing, **kw)
    fig = ax.figure
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(
        ax.transAxes.inverted())
    return bb.y0


def headroom(ax, ymax, frac=1.45):
    lo, _ = ax.get_ylim()
    ax.set_ylim(lo, max(ymax * frac, 1e-12))


def panel(**kw):
    return plt.subplots(layout="constrained", **kw)


def grid(nrows=1, ncols=1, gap=1.06, base=10.0, **kw):
    return plt.subplots(nrows, ncols, layout="constrained",
                        figsize=(base * ncols * gap, base * nrows * gap), **kw)


def ratio_panel(base=10.0):
    """Main + ratio panel keeping a roughly square main axes."""
    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, layout="constrained",
        figsize=(base, base * 1.18),
        gridspec_kw=dict(height_ratios=[3.0, 1.0]))
    try:
        fig.get_layout_engine().set(hspace=0.02, h_pad=0.02)
    except Exception:
        pass
    return fig, ax, rax


_VER = None


def stamp(fig, inputs="pairsplit Run 2 unfold", bottom=0.03):
    global _VER
    if _VER is None:
        try:
            _VER = subprocess.run(
                ["git", "describe", "--tags", "--always", "--dirty"],
                capture_output=True, text=True, timeout=10,
                cwd=str(REPO)).stdout.strip()
        except Exception:
            _VER = "unknown"
    if os.environ.get("NO_STAMP"):
        return
    try:
        #### rect is (left, bottom, WIDTH, HEIGHT), NOT (x0, y0, x1, y1).  The
        #### height must therefore be 1 - bottom so the layout top lands
        #### exactly on the canvas edge; a fixed 0.97 with a raised bottom
        #### puts the top at bottom + 0.97 > 1 and clips the y-axis titles,
        #### the top spine and the topmost tick labels.
        fig.get_layout_engine().set(rect=(0, bottom, 1, 1.0 - bottom))
    except Exception:
        pass
    fig.text(0.99, 0.002,
             f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 "
             f"{_VER}  |  inputs: {inputs}",
             ha="right", va="bottom", fontsize=7, color="0.45",
             family="monospace")


MADE: list[tuple[str, str]] = []
ONLY: str | None = None      # regex; when set, only matching figures are written
CMS_TEXTS = {"CMS", "Internal", "Simulation", "Preliminary"}


def geom_check(fig, name):
    """GEOM_CHECK=1: report the three collisions that bite these panels --
    CMS block vs annotation, annotation vs legend, legend vs the data it
    covers.  All gaps are in axes fractions and must be positive."""
    rend = fig.canvas.get_renderer()
    for ax in fig.axes:
        inv = ax.transAxes.inverted()
        cms, ann = [], []
        for t in ax.texts:
            bb = t.get_window_extent(rend).transformed(inv)
            if bb.y1 > 1.0 + 1e-6:
                continue
            (cms if t.get_text().split("\n")[0] in CMS_TEXTS else ann).append(bb)
        if not cms:
            continue
        msg = [f"  [{name}] ax{fig.axes.index(ax)}"]
        cms_bot = min(b.y0 for b in cms)
        if ann:
            a_top, a_bot = max(b.y1 for b in ann), min(b.y0 for b in ann)
            msg.append(f"cms->annot {cms_bot - a_top:+.3f}")
        else:
            a_bot = cms_bot
        leg = ax.get_legend()
        if leg is not None:
            lb = leg.get_window_extent(rend).transformed(inv)
            msg.append(f"annot->legend {a_bot - lb.y1:+.3f}")
            #### highest plotted point under the legend's x-span
            pts = [ln.get_xydata() for ln in ax.lines if len(ln.get_xydata())]
            pts += [np.array([[p.get_x() + p.get_width(),
                               p.get_y() + p.get_height()]])
                    for p in ax.patches if hasattr(p, "get_height")]
            top = -np.inf
            for arr in pts:
                q = ax.transAxes.inverted().transform(
                    ax.transData.transform(np.asarray(arr, float)))
                sel = (q[:, 0] >= lb.x0 - 0.01) & (q[:, 0] <= lb.x1 + 0.01)
                if sel.any():
                    top = max(top, np.nanmax(q[sel, 1]))
            if np.isfinite(top):
                msg.append(f"legend->data {lb.y0 - top:+.3f}")
        print("  ".join(msg))


def save(fig, name, caption):
    import re
    if ONLY is not None and not re.search(ONLY, name):
        plt.close(fig)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("GEOM_CHECK"):
        fig.canvas.draw()
        geom_check(fig, name)
    stamp(fig, bottom=getattr(fig, "_stamp_bottom", 0.03))
    png, pdf = OUT_DIR / f"{name}.png", OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=140)
    fig.savefig(pdf)
    plt.close(fig)
    MADE.append((name, caption))
    print(f"  wrote {png.name} / {pdf.name}")


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def pt_label(edges, i, sym=r"$p_{T}$"):
    lo, hi = edges[i], edges[i + 1]
    if hi >= 1e4:
        return f"{sym} > {lo:.0f} GeV"
    return f"{lo:.0f} < {sym} < {hi:.0f} GeV"


def load(channel, row=PROD_ROW, syst="hadronic_rho_systematics_run2.json",
         bayes="hadronic_rho_bayes_run2.json"):
    """Everything for one channel, on the production merge.

    ``row``/``syst``/``bayes`` default to the production merge; pointing them
    at a different stability row and its matching Phase B products (the
    ``_2to1`` twins) loads an alternative gen binning with the same layout."""
    z = dict(np.load(IN / "npz" / f"{channel}_run2_nominal.npz"))
    rows = json.loads((IN / f"{channel}_run2_stability.json").read_text())
    st = [r for r in rows if r["label"] == row][0]
    sy = json.loads((IN / syst).read_text())[channel]
    by = json.loads((IN / bayes).read_text())[channel]
    assert sy["gen_rho_edges"] == st["gen_rho_edges"] == by["gen_rho_edges"]
    assert (np.array(sy["shown"], bool) == np.array(st["shown"], bool)).all()

    gedges = np.asarray(st["gen_rho_edges"], float)
    n_rho = len(gedges) - 1
    pt_groups = st["pt_groups"]
    n_pt = len(pt_groups)
    pt_edges = np.asarray(z["pt_edges"], float)
    gen_pt_edges = np.r_[[pt_edges[g[0]] for g in pt_groups], pt_edges[-1]]

    def r2(a):
        return np.asarray(a, float).reshape(n_pt, n_rho)

    d = dict(
        channel=channel, z=z, st=st, sy=sy, by=by,
        gedges=gedges, n_rho=n_rho, n_pt=n_pt,
        pt_edges=gen_pt_edges, reco_pt_edges=pt_edges,
        reco_rho_edges=np.asarray(z["rho_edges"], float),
        pt_groups=pt_groups,
        shown=np.asarray(st["shown"], bool).reshape(n_pt, n_rho),
        core=np.asarray(sy["core"], bool).reshape(n_pt, n_rho),
        x=r2(st["unfolded"]), err=r2(st["err"]), gen=r2(st["gen"]),
        amp=r2(st["amp"]), rho_i=r2(st["rho_i"]),
        stat=r2(sy["relative_data_stat"]),
        resp=r2(sy["relative_response_stat"]),
        jer=r2(sy["relative_half_difference"]["JER"]),
        jms=r2(sy["relative_half_difference"]["JMS"]),
        jmr=r2(sy["relative_half_difference"]["JMR"]),
        det=r2(sy["total_half_difference"]),
        fsr=r2(sy["parton_shower"]["relative_half_difference"]["fsr"]),
        isr=r2(sy["parton_shower"]["relative_half_difference"]["isr"]),
        bayes_shape=r2(by["reference"]["normalized_shape"]),
        tunf_shape=r2(by["tunfold_reference"]["normalized_shape"]),
        norm_window=st["norm_window"],
    )
    d["ptlab"] = [pt_label(gen_pt_edges, i) for i in range(n_pt)]
    if JES_JSON is not None:
        je = json.loads(Path(JES_JSON).read_text())[channel]
        assert je["gen_rho_edges"] == st["gen_rho_edges"]
        assert (np.array(je["shown"], bool) == np.array(st["shown"], bool)).all()
        d["jes"] = r2(je["jes_total"][JES_SPLIT_MODE])
        d["jes_clean"] = r2(je["jes_total_excl_defective"][JES_SPLIT_MODE])
        d["jes_meta"] = je
    return d


def density(values, edges, mask, norm=None):
    """Unit area over the masked bins, per unit bin width; NaN elsewhere.

    ``norm`` picks the bins that carry the unit area and defaults to ``mask``
    (the production convention).  A narrower ``norm`` still draws every
    ``mask`` bin but sets the normalization from the core window only."""
    w = np.diff(edges)
    out = np.full_like(values, np.nan, dtype=float)
    if norm is None:
        norm = mask
    tot = values[norm].sum()
    if tot > 0:
        out[mask] = values[mask] / tot / w[mask]
    return out


def centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def window_mask(edges, lo, hi, eps=1e-9):
    """Bins overlapping ``[lo, hi]``.  On a binning that carries both window
    edges this is exactly the set of bins inside the window; on a coarser
    binning it is the closest window the binning can realize."""
    return (edges[1:] > lo + eps) & (edges[:-1] < hi - eps)


def renorm_signed(shift, counts, wmask):
    """Move a relative normalized-shape shift onto another normalization
    window, exactly and without re-unfolding.

    The stored shift is ``r_i = v_i/n_i / <v/n>_W - 1`` with ``<.>_W`` the
    ``n``-weighted mean over the old window ``W``, so ``1 + r_i`` already is
    the per-bin variation ratio up to one per-slice constant.  Dividing by its
    ``counts``-weighted mean over the new window replaces that constant."""
    t = 1.0 + np.asarray(shift, float)
    num = np.nansum(counts[:, wmask] * t[:, wmask], axis=1)
    den = np.nansum(counts[:, wmask], axis=1)
    return t / (num / den)[:, None] - 1.0


def window_bands(d, wmask):
    """Detector-quadrature and FSR relative bands, each variation renormalized
    over ``wmask`` before the half-difference is taken."""
    sy, n = d["sy"], d["x"]

    def r2(a):
        return np.asarray(a, float).reshape(d["n_pt"], d["n_rho"])

    def half(signed):
        up = renorm_signed(r2(signed["Up"]), n, wmask)
        down = renorm_signed(r2(signed["Down"]), n, wmask)
        return 0.5 * np.abs(up - down)

    det = np.sqrt(sum(half(sy["signed_relative_shift"][s]) ** 2
                      for s in ("JER", "JMS", "JMR")))
    fsr = half(sy["parton_shower"]["signed_relative_shift"]["fsr"])
    return det, fsr


# --------------------------------------------------------------------------
# 1. unfolded data vs MC gen prior
# --------------------------------------------------------------------------
def fig_unfolded(d, norm_window=None, suffix="", extra_line=None,
                 head_frac=1.90):
    """Unfolded shape vs the gen prior.

    Defaults reproduce the production figure exactly.  ``norm_window`` swaps
    the unit-area normalization onto the core window ``(lo, hi)`` for data,
    MC and every systematic variation, while all shown bins are still drawn;
    ``suffix``/``extra_line`` name and annotate the variant.  A second
    annotation line pushes the legend down by about one line height, so
    ``head_frac`` has to grow with it or the legend lands on the data."""
    ge, ch = d["gedges"], d["channel"]
    xc, w = centers(ge), np.diff(ge)
    wmask = None if norm_window is None else window_mask(ge, *norm_window)
    if wmask is None:
        det_rel, fsr_rel = d["det"], d["fsr"]
    else:
        det_rel, fsr_rel = window_bands(d, wmask)
    for p in range(d["n_pt"]):
        m = d["shown"][p]
        nm = m if wmask is None else (m & wmask)
        dat = density(d["x"][p], ge, m, norm=nm)
        mc = density(d["gen"][p], ge, m, norm=nm)
        stat = d["stat"][p] * dat
        inner = det_rel[p] * dat
        outer = np.hypot(det_rel[p], fsr_rel[p]) * dat

        fig, ax, rax = ratio_panel()
        cms_label(ax)
        for band, color, lab, z in (
                (outer, FSR_COLOR, "Detector + FSR", 1),
                (inner, DET_COLOR, "Detector (JER+JMS+JMR)", 2)):
            ax.bar(xc[m], 2 * band[m], width=w[m], bottom=(dat - band)[m],
                   color=color, alpha=0.55, zorder=z, linewidth=0,
                   label=lab)
        ax.step(np.r_[ge[:-1][m], ge[1:][m][-1]],
                np.r_[mc[m], mc[m][-1]], where="post", color="#e42536",
                lw=2.6, zorder=3, label=MC_LABEL)
        ax.errorbar(xc[m], dat[m], yerr=stat[m], xerr=w[m] / 2,
                    fmt="o", ms=8, color="k", lw=1.6, capsize=0, zorder=4,
                    label=DATA_LABEL)
        ax.set_ylabel(YLAB)
        ax.set_ylim(bottom=0)
        headroom(ax, np.nanmax([np.nanmax(dat[m] + outer[m]),
                                np.nanmax(mc[m])]), frac=head_frac)
        label = d["ptlab"][p] if extra_line is None else \
            f"{d['ptlab'][p]}\n{extra_line}"
        ybot = in_panel_text(ax, label, fontsize=22)
        ax.legend(loc="upper left", framealpha=0.0, fontsize=20,
                  bbox_to_anchor=(0.0, ybot - 0.03))

        r = dat / mc
        rax.axhline(1.0, color="#9c9ca1", lw=1.4)
        for band, color in ((np.hypot(det_rel[p], fsr_rel[p]), FSR_COLOR),
                            (det_rel[p], DET_COLOR)):
            rax.bar(xc[m], 2 * (band * r)[m], width=w[m],
                    bottom=(r - band * r)[m], color=color, alpha=0.55,
                    linewidth=0)
        rax.errorbar(xc[m], r[m], yerr=(d["stat"][p] * r)[m], xerr=w[m] / 2,
                     fmt="o", ms=7, color="k", lw=1.5, capsize=0)
        rax.set_ylabel("Data / MC", fontsize=24)
        rax.set_xlabel(XLAB)
        lim = np.nanmax(np.abs(np.r_[r[m], (r + outer / mc)[m],
                                     (r - outer / mc)[m]] - 1.0))
        rax.set_ylim(1 - 1.35 * lim, 1 + 1.35 * lim)
        rax.set_xlim(ge[:-1][m].min(), ge[-1])
        rax.tick_params(labelsize=22)
        caption = (f"{ch}, {d['ptlab'][p]}: unfolded normalized groomed-rho "
                   f"shape vs the MC gen prior, with the detector and "
                   f"detector+FSR bands.")
        if wmask is not None:
            caption += (f" Unit area over {ge[:-1][nm].min():g} < "
                        f"log(rho^2) < {ge[1:][nm].max():g} only"
                        + (f"; {extra_line}." if extra_line else "."))
            print(f"    pt{p + 1} data/MC over the window: "
                  f"{np.nanmin(r[nm]):.3f} .. {np.nanmax(r[nm]):.3f}"
                  f"   (all shown bins {np.nanmin(r[m]):.3f} .. "
                  f"{np.nanmax(r[m]):.3f})")
        save(fig, f"unfolded_{ch}_pt{p + 1}{suffix}", caption)


# --------------------------------------------------------------------------
# 2. uncertainty breakdown
# --------------------------------------------------------------------------
def fig_uncertainty(d):
    ge, ch = d["gedges"], d["channel"]
    comps = [("Data stat.", "stat", "k", "-", 3.0),
             ("Response MC stat.", "resp", PETROFF[4], "-", 2.2),
             ("JER", "jer", PETROFF[0], "-", 2.2),
             ("JMS", "jms", PETROFF[1], "-", 2.2),
             ("JMR", "jmr", PETROFF[2], "-", 2.2),
             ("FSR", "fsr", PETROFF[3], "-", 2.2),
             ("ISR (not in band)", "isr", PETROFF[5], "--", 2.2)]
    #### the era-correlated JES band goes in right after the stat entries so
    #### the two largest curves are adjacent in the legend
    if "jes" in d:
        comps.insert(2, ("JES (27 sources)", "jes", JES_COLOR, "-", 2.8))
    for p in range(d["n_pt"]):
        m = d["shown"][p]
        fig, ax = panel()
        cms_label(ax)
        top = 0.0
        for lab, key, color, ls, lw in comps:
            y = 100.0 * d[key][p]
            ax.step(np.r_[ge[:-1][m], ge[1:][m][-1]], np.r_[y[m], y[m][-1]],
                    where="post", color=color, ls=ls, lw=lw, label=lab)
            top = max(top, np.nanmax(y[m]))
        ax.set_xlabel(XLAB)
        ax.set_ylabel("Uncertainty on the normalized shape [%]")
        ax.set_xlim(ge[:-1][m].min(), ge[-1])
        ax.set_ylim(bottom=0)
        headroom(ax, top, frac=2.15)
        ybot = in_panel_text(ax, d["ptlab"][p], fontsize=22)
        ax.legend(loc="upper left", ncol=2, fontsize=19, framealpha=0.0,
                  bbox_to_anchor=(0.0, ybot - 0.03))
        caption = (f"{ch}, {d['ptlab'][p]}: per-bin uncertainty components on "
                   f"the unfolded normalized shape; ISR is shown for "
                   f"observation only.")
        if "jes" in d:
            caption += (" JES is the quadrature of all 27 split sources over "
                        "the four era-correlation legs "
                        f"(f_corr = sqrt(rho), f_uncorr = sqrt(1 - rho)); "
                        "RelativeJEREC1/EC2/HF carry the JER shift in these "
                        "production histograms (processor dispatch defect) "
                        "and inflate the JES curve by ~5%.")
        save(fig, f"uncertainty_{ch}_pt{p + 1}", caption)


# --------------------------------------------------------------------------
# 3. reco-level data vs MC
# --------------------------------------------------------------------------
def fig_datamc_reco(d):
    z, ch = d["z"], d["channel"]
    re = d["reco_rho_edges"]
    n_rho = len(re) - 1
    n_pt = len(d["reco_pt_edges"]) - 1
    xc, w = centers(re), np.diff(re)
    dat_all = np.asarray(z["data_reco"], float).reshape(n_pt, n_rho)
    dw2 = np.asarray(z["data_w2"], float).reshape(n_pt, n_rho)
    mc_all = np.asarray(z["reco"], float).reshape(n_pt, n_rho)
    mw2 = np.asarray(z["reco_w2"], float).reshape(n_pt, n_rho)
    buffer_bins = int(z["n_buffer"])
    m = np.ones(n_rho, bool)
    m[:buffer_bins] = False

    for p in range(n_pt):
        dat = density(dat_all[p], re, m)
        mc = density(mc_all[p], re, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_d = np.where(dat_all[p] > 0, np.sqrt(dw2[p]) / dat_all[p], 0.0)
            rel_m = np.where(mc_all[p] > 0, np.sqrt(mw2[p]) / mc_all[p], 0.0)

        fig, ax, rax = ratio_panel()
        cms_label(ax)
        ax.step(np.r_[re[:-1][m], re[1:][m][-1]], np.r_[mc[m], mc[m][-1]],
                where="post", color="#e42536", lw=2.6, label=MC_LABEL)
        ax.bar(xc[m], 2 * (rel_m * mc)[m], width=w[m],
               bottom=(mc - rel_m * mc)[m], color="#e42536", alpha=0.3,
               linewidth=0, label="MC stat.")
        ax.errorbar(xc[m], dat[m], yerr=(rel_d * dat)[m], xerr=w[m] / 2,
                    fmt="o", ms=8, color="k", lw=1.6, capsize=0,
                    label=DATA_LABEL)
        ax.set_ylabel(YLAB)
        ax.set_ylim(bottom=0)
        headroom(ax, np.nanmax([np.nanmax(dat[m]), np.nanmax(mc[m])]),
                 frac=1.90)
        ybot = in_panel_text(ax, pt_label(d["reco_pt_edges"], p), fontsize=22)
        ax.legend(loc="upper left", framealpha=0.0, fontsize=20,
                  bbox_to_anchor=(0.0, ybot - 0.03))

        r = dat / mc
        rax.axhline(1.0, color="#9c9ca1", lw=1.4)
        rax.bar(xc[m], 2 * (rel_m * r)[m], width=w[m],
                bottom=(r - rel_m * r)[m], color="#e42536", alpha=0.3,
                linewidth=0)
        rax.errorbar(xc[m], r[m], yerr=(rel_d * r)[m], xerr=w[m] / 2,
                     fmt="o", ms=7, color="k", lw=1.5, capsize=0)
        rax.set_ylabel("Data / MC", fontsize=24)
        rax.set_xlabel(XLAB_RECO)
        lim = np.nanmax(np.abs(np.r_[r[m] + (rel_d * r)[m],
                                     r[m] - (rel_d * r)[m]] - 1.0))
        rax.set_ylim(1 - 1.35 * lim, 1 + 1.35 * lim)
        rax.set_xlim(re[:-1][m].min(), re[-1])
        rax.tick_params(labelsize=22)
        save(fig, f"datamc_reco_{ch}_pt{p + 1}",
             f"{ch}, {pt_label(d['reco_pt_edges'], p)}: detector-level "
             f"normalized groomed-rho shape, data vs MC (statistical only).")


# --------------------------------------------------------------------------
# 4. purity and stability
# --------------------------------------------------------------------------
def rho_purity_stability(d):
    """rho-only purity/stability of the production gen merge.

    purity(p, i)    = P(gen rho bin i | reco in pT slice p and reco rho in the
                      reco bins that make up gen bin i), gen pT unconstrained
    stability(p, i) = P(reco rho in those bins | gen in (p, i)), reco pT
                      unconstrained
    """
    z = d["z"]
    re = d["reco_rho_edges"]
    n_rho_r = len(re) - 1
    n_pt_r = len(d["reco_pt_edges"]) - 1
    n_pt_g, n_rho_g = d["n_pt"], d["n_rho"]
    ge = d["gedges"]

    #### nested merge: each reco rho bin sits inside exactly one gen rho bin.
    #### The gen axis of A in the NPZ is still the FINE (= reco) binning, so
    #### the production merge has to be applied here.
    rc = centers(re)
    reco2gen = np.clip(np.searchsorted(ge, rc) - 1, 0, n_rho_g - 1)
    M = np.zeros((n_pt_r * n_rho_r, n_pt_g * n_rho_g))
    for p in range(n_pt_r):
        for r in range(n_rho_r):
            M[p * n_rho_r + r, p * n_rho_g + reco2gen[r]] = 1.0
    A = np.asarray(z["A"], float) @ M

    A4 = A.reshape(n_pt_r, n_rho_r, n_pt_g, n_rho_g)
    matched = A4.sum(axis=(2, 3))                    # (reco pT, reco rho)
    purity = np.full((n_pt_r, n_rho_g), np.nan)
    stability = np.full((n_pt_g, n_rho_g), np.nan)
    for i in range(n_rho_g):
        rows = reco2gen == i
        if not rows.any():
            continue
        sub = A4[:, :, :, i]                         # (reco pT, reco rho, gen pT)
        num_p = sub[:, rows, :].sum(axis=(1, 2))     # gen pT unconstrained
        den_p = matched[:, rows].sum(axis=1)
        purity[:, i] = np.where(den_p > 0, num_p / den_p, np.nan)
        num_s = sub[:, rows, :].sum(axis=(0, 1))     # reco pT unconstrained
        den_s = sub.sum(axis=(0, 1))
        stability[:, i] = np.where(den_s > 0, num_s / den_s, np.nan)
    return purity, stability, reco2gen, A


def fig_purity_stability(d):
    ge, ch = d["gedges"], d["channel"]
    purity, stability, _, _ = rho_purity_stability(d)
    m = d["shown"][0]  # same shown pattern in every pT slice
    x = np.r_[ge[:-1][m], ge[1:][m][-1]]

    fig, axes = grid(1, 2)
    for ax, arr, name in ((axes[0], purity, "Purity"),
                          (axes[1], stability, "Stability")):
        cms_label(ax, rlabel=SIM_ERA, simulation=True)
        for p in range(d["n_pt"]):
            y = arr[p] if p < arr.shape[0] else np.full(d["n_rho"], np.nan)
            ax.step(x, np.r_[y[m], y[m][-1]], where="post", lw=2.4,
                    color=PETROFF[p % 6], label=d["ptlab"][p])
        ax.axhline(0.5, color="#9c9ca1", ls=":", lw=2.0)
        ax.set_xlabel(XLAB)
        ax.set_ylabel(f"{name} (groomed $\\rho$ only)")
        ax.set_xlim(ge[:-1][m].min(), ge[-1])
        ax.set_ylim(0.0, 1.19)
        ax.legend(loc="lower left", fontsize=19, framealpha=0.0)
    save(fig, f"purity_stability_{ch}",
         f"{ch}: rho-only purity (left) and stability (right) of the "
         f"production gen merge, one line per pT slice; guide at 0.5.")


# --------------------------------------------------------------------------
# 5. error amplification and global correlation
# --------------------------------------------------------------------------
def fig_amplification(d):
    ge, ch = d["gedges"], d["channel"]
    m0 = d["shown"][0]
    x = np.r_[ge[:-1][m0], ge[1:][m0][-1]]
    fig, axes = grid(1, 2)
    specs = ((axes[0], d["amp"], "Error amplification vs bin-by-bin", 1.0),
             (axes[1], np.abs(d["rho_i"]), r"Global correlation $|\rho_i|$",
              None))
    for ax, arr, name, guide in specs:
        cms_label(ax)
        top = 0.0
        for p in range(d["n_pt"]):
            m = d["shown"][p]
            y = arr[p]
            ax.step(x, np.r_[y[m], y[m][-1]], where="post", lw=2.4,
                    color=PETROFF[p % 6], label=d["ptlab"][p])
            top = max(top, np.nanmax(y[m]))
        if guide is not None:
            ax.axhline(guide, color="#9c9ca1", ls=":", lw=2.0)
        ax.set_xlabel(XLAB)
        ax.set_ylabel(name)
        ax.set_xlim(ge[:-1][m0].min(), ge[-1])
        ax.set_ylim(bottom=0)
        #### these curves fill the panel from left to right, so there is no
        #### in-frame corner a legend can own -- park it outside, shared
        headroom(ax, top, frac=1.35)
    #### "outside lower center" is pinned to the figure edge no matter what
    #### the layout rect says, i.e. straight onto the provenance stamp.  So
    #### reserve the strip with the rect (honoured by save()) and place a
    #### plain figure legend in it.
    fig._stamp_bottom = 0.115
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.025), bbox_transform=fig.transFigure,
               ncol=d["n_pt"], fontsize=20, frameon=False)
    save(fig, f"amplification_{ch}",
         f"{ch}: tau=0 error amplification against the bin-by-bin reference "
         f"(left) and the TUnfold global correlation (right), per reported "
         f"bin and pT slice.")


# --------------------------------------------------------------------------
# 6. dijet vs trijet on the common six-bin binning
# --------------------------------------------------------------------------
def common_unfold(out_path=COMMON_NPZ):
    """One tau=0 pass per channel on the common gen binning (needs ROOT)."""
    sys.path.insert(0, str(REPO / "scripts" / "studies" / "unfold"))
    from rho_unfold_stability import gen_map, tunfold  # noqa: E402

    res = {}
    for ch in CHANNELS:
        z = dict(np.load(IN / "npz" / f"{ch}_run2_nominal.npz"))
        pg = COMMON_MERGE[ch]["pt_groups"]
        rg = COMMON_MERGE[ch]["rho_groups"]
        n_pt = len(z["pt_edges"]) - 1
        n_rho = len(z["rho_edges"]) - 1
        M = gen_map(n_pt, n_rho, pg, rg)
        A, Aw2 = z["A"] @ M, z["A_w2"] @ M
        gen, gen_w2 = z["gen"] @ M, z["gen_w2"] @ M
        reco = z["reco"]
        fakes = reco - A.sum(axis=1)
        misses = gen - A.sum(axis=0)
        misses_w2 = np.maximum(gen_w2 - Aw2.sum(axis=0), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(reco > 0, fakes / reco, 0.0)
        surv = 1.0 - np.clip(f, 0.0, 1.0)
        y = z["data_reco"] * surv
        Vy = z["data_V"] * np.outer(surv, surv)
        da = tunfold(A, Aw2, misses, misses_w2, y, Vy, f"_common_{ch}",
                     quiet=True)
        err = np.sqrt(np.maximum(np.diag(da["Ein"]), 0.0))
        gedges = np.asarray(COMMON_RHO_EDGES, float)
        pt_edges = np.r_[[z["pt_edges"][g[0]] for g in pg], z["pt_edges"][-1]]
        res[f"{ch}_x"] = da["x"].reshape(len(pg), len(rg))
        res[f"{ch}_err"] = err.reshape(len(pg), len(rg))
        res[f"{ch}_gen"] = gen.reshape(len(pg), len(rg))
        res[f"{ch}_pt_edges"] = pt_edges
        res["gedges"] = gedges
        print(f"  {ch}: chi2A/ndf = {da['chi2A']:.1f}/{da['ndf']}, "
              f"negatives {(da['x'] < 0).sum()}, "
              f"median rel stat "
              f"{np.median(err[da['x'] > 0] / da['x'][da['x'] > 0]):.3%}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **res)
    print(f"wrote {out_path}")
    return 0


def fig_compare_channels():
    if not COMMON_NPZ.exists():
        print(f"  SKIP compare_channels: {COMMON_NPZ} missing "
              f"(run with --common-unfold under ROOT first)")
        return
    c = dict(np.load(COMMON_NPZ))
    ge = c["gedges"]
    n_rho = len(ge) - 1
    m = np.ones(n_rho, bool)
    m[0] = False  # the [-10,-4] buffer
    xc, w = centers(ge), np.diff(ge)
    pt_edges = c["dijet_pt_edges"]
    colors = {"dijet": PETROFF[0], "trijet": PETROFF[2]}
    for p in range(len(pt_edges) - 1):
        fig, ax, rax = ratio_panel()
        cms_label(ax)
        shp, rel = {}, {}
        for ch in CHANNELS:
            x = c[f"{ch}_x"][p]
            e = c[f"{ch}_err"][p]
            shp[ch] = density(x, ge, m)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel[ch] = np.where(x > 0, e / x, np.nan)
            ax.errorbar(xc[m], shp[ch][m], yerr=(rel[ch] * shp[ch])[m],
                        xerr=w[m] / 2, fmt="o" if ch == "dijet" else "s",
                        ms=9, color=colors[ch], lw=1.8, capsize=0,
                        label=f"{ch.capitalize()} (unfolded data)")
        ax.set_ylabel(YLAB)
        ax.set_ylim(bottom=0)
        headroom(ax, np.nanmax([np.nanmax(shp[ch][m]) for ch in CHANNELS]),
                 frac=1.95)
        ybot = in_panel_text(
            ax, pt_label(pt_edges, p) + "\nCommon 6-bin gen binning",
            fontsize=22)
        ax.legend(loc="upper left", framealpha=0.0, fontsize=20,
                  bbox_to_anchor=(0.0, ybot - 0.03))

        r = shp["trijet"] / shp["dijet"]
        re_ = r * np.hypot(rel["trijet"], rel["dijet"])
        rax.axhline(1.0, color="#9c9ca1", lw=1.4)
        rax.errorbar(xc[m], r[m], yerr=re_[m], xerr=w[m] / 2, fmt="o", ms=7,
                     color="k", lw=1.5, capsize=0)
        rax.set_ylabel("Trijet / dijet", fontsize=24)
        rax.set_xlabel(XLAB)
        lim = np.nanmax(np.abs(np.r_[r[m] + re_[m], r[m] - re_[m]] - 1.0))
        rax.set_ylim(1 - 1.3 * lim, 1 + 1.3 * lim)
        rax.set_xlim(ge[:-1][m].min(), ge[-1])
        rax.tick_params(labelsize=22)
        save(fig, f"compare_channels_pt{p + 1}",
             f"{pt_label(pt_edges, p)}: dijet vs trijet unfolded normalized "
             f"shapes re-merged onto the common six-bin gen binning "
             f"(statistical errors only).")


# --------------------------------------------------------------------------
# 7. iterative Bayes n=4 against the tau=0 result
# --------------------------------------------------------------------------
def fig_bayes(d):
    ge, ch = d["gedges"], d["channel"]
    m0 = d["shown"][0]
    x = np.r_[ge[:-1][m0], ge[1:][m0][-1]]
    fig, ax = panel()
    cms_label(ax)
    lo, hi = 1.0, 1.0
    for p in range(d["n_pt"]):
        m = d["shown"][p]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = d["bayes_shape"][p] / d["tunf_shape"][p]
        ax.step(x, np.r_[r[m], r[m][-1]], where="post", lw=2.4,
                color=PETROFF[p % 6], label=d["ptlab"][p])
        lo, hi = min(lo, np.nanmin(r[m])), max(hi, np.nanmax(r[m]))
    ax.axhline(1.0, color="#9c9ca1", ls=":", lw=2.0)
    ax.set_xlabel(XLAB)
    ax.set_ylabel("Bayes ($n$ = 4) / TUnfold ($\\tau$ = 0)")
    ax.set_xlim(ge[:-1][m0].min(), ge[-1])
    pad = max(hi - 1.0, 1.0 - lo)
    ax.set_ylim(1 - 1.25 * pad, 1 + 2.0 * pad)
    ax.legend(loc="upper left", fontsize=19, framealpha=0.0,
              bbox_to_anchor=(0.0, 0.80))
    save(fig, f"bayes_vs_tunfold_{ch}",
         f"{ch}: ratio of the iterative-Bayes (n = 4) normalized shape to the "
         f"production tau = 0 TUnfold result, per reported bin and pT slice.")


# --------------------------------------------------------------------------
# 8. response mosaic
# --------------------------------------------------------------------------
def fig_response(d):
    ch = d["channel"]
    n_pt_r = len(d["reco_pt_edges"]) - 1
    n_rho_r = len(d["reco_rho_edges"]) - 1
    n_pt_g, n_rho_g = d["n_pt"], d["n_rho"]
    #### the response already merged onto the production gen binning
    _, _, _, A = rho_purity_stability(d)
    col = A.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(col > 0, A / np.where(col > 0, col, 1.0), np.nan)
    P = np.where(P > 0, P, np.nan)

    fig, ax = panel(figsize=(11.5, 10.0))
    im = ax.pcolormesh(np.arange(A.shape[1] + 1), np.arange(A.shape[0] + 1),
                       P, cmap="viridis",
                       norm=LogNorm(vmin=max(np.nanmin(P), 1e-5),
                                    vmax=np.nanmax(P)))
    for p in range(1, n_pt_g):
        ax.axvline(p * n_rho_g, color="w", lw=2.0)
    for p in range(1, n_pt_r):
        ax.axhline(p * n_rho_r, color="w", lw=2.0)
    ax.set_xlabel("Generator bin (pT $\\otimes$ groomed $\\rho$)")
    ax.set_ylabel("Detector bin (pT $\\otimes$ groomed $\\rho$)")
    ax.set_xticks(np.arange(0, A.shape[1] + 1, n_rho_g))
    ax.set_yticks(np.arange(0, A.shape[0] + 1, n_rho_r))
    #### pT-block tags inside the frame: the production merge leaves the pT
    #### axis un-merged, so the same edges label the gen and the reco blocks
    for p in range(n_pt_g):
        ax.text((p + 0.5) * n_rho_g, A.shape[0] * 0.985,
                (f">{d['pt_edges'][p]:.0f}" if d["pt_edges"][p + 1] >= 1e4
                 else f"{d['pt_edges'][p]:.0f}-{d['pt_edges'][p + 1]:.0f}"),
                ha="center", va="top", fontsize=15, color="0.15",
                bbox=dict(facecolor="white", alpha=0.75, pad=1.5,
                          edgecolor="none"))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of the generator column")
    cms_label(ax, rlabel=SIM_ERA, simulation=True, loc=0)
    save(fig, f"response_{ch}",
         f"{ch}: response mosaic on the production binning, normalized within "
         f"each generator column; white lines separate the pT blocks, whose "
         f"ranges (GeV) are tagged along the top and apply to both axes.")


# --------------------------------------------------------------------------
# 9. data/MC disagreement before and after unfolding
# --------------------------------------------------------------------------
def fig_before_after(d):
    ge, ch, z = d["gedges"], d["channel"], d["z"]
    re = d["reco_rho_edges"]
    n_rho_r = len(re) - 1
    n_pt_r = len(d["reco_pt_edges"]) - 1
    dat_r = np.asarray(z["data_reco"], float).reshape(n_pt_r, n_rho_r)
    mc_r = np.asarray(z["reco"], float).reshape(n_pt_r, n_rho_r)
    mr = np.ones(n_rho_r, bool)
    mr[:int(z["n_buffer"])] = False

    fig, ax = panel()
    cms_label(ax)
    top = 0.0
    for p in range(d["n_pt"]):
        color = PETROFF[p % 6]
        #### after: gen space, production merge
        mg = d["shown"][p]
        a = np.abs(density(d["x"][p], ge, mg)
                   / density(d["gen"][p], ge, mg) - 1.0) * 100.0
        ax.plot(centers(ge)[mg], a[mg], marker="o", ms=10, lw=2.0,
                color=color, label=d["ptlab"][p])
        #### before: folded space, one reco pT slice per gen slice (the
        #### production merge keeps the pT axis un-merged in both channels)
        q = d["pt_groups"][p][0]
        b = np.abs(density(dat_r[q], re, mr) / density(mc_r[q], re, mr)
                   - 1.0) * 100.0
        ax.plot(centers(re)[mr], b[mr], marker="o", ms=10, lw=2.0, ls="--",
                mfc="none", mew=2.0, color=color)
        top = max(top, np.nanmax(a[mg]), np.nanmax(b[mr]))
    ax.set_xlabel(XLAB)
    ax.set_ylabel(r"$|\,$data/MC $-\,1|$ of the normalized shape [%]")
    ax.set_xlim(min(ge[:-1][d["shown"][0]].min(), re[:-1][mr].min()), 0.0)
    ax.set_ylim(bottom=0)
    headroom(ax, top, frac=1.8)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", ls="--", marker="o", mfc="none",
                       mew=2.0, ms=10, lw=2.0),
                Line2D([], [], color="0.3", ls="-", marker="o", ms=10,
                       lw=2.0)]
    labels += ["Before (detector level)", "After (unfolded)"]
    ax.legend(handles, labels, loc="upper left", ncol=2, fontsize=18,
              framealpha=0.0, bbox_to_anchor=(0.0, 0.80))
    save(fig, f"before_after_{ch}",
         f"{ch}: absolute normalized-shape data/MC deviation per bin before "
         f"(detector level, open markers) and after (unfolded, filled) the "
         f"unfolding, one colour per pT slice.")


# --------------------------------------------------------------------------
# presentation variants (never overwrite a production figure)
# --------------------------------------------------------------------------
#### the narrower core window the variant normalizes over, per channel
CORE_WINDOW = {"dijet": (-1.8, -0.55), "trijet": (-1.8, -0.75)}
#### the variants carry a second annotation line, and the core-window
#### normalization lifts the tail bins, so the production 1.90 leaves the
#### legend sitting on the data (dijet pT > 1000 worst); measured with
#### GEOM_CHECK=1, 2.25 keeps every legend->data gap positive.
VARIANT_HEAD_FRAC = 2.25


def variants():
    """Two presentation variants into ``OUT/variants``:

    1. ``_win18`` -- the production unfolded family with the unit area taken
       over the core window only (both channels, every pT slice).
    2. ``_2to1``  -- trijet on the coarser ``proposal_2to1`` gen merge, which
       removes the oscillation the production merge shows at 200-290 GeV.
       Its own normalization window is already the core window, so the same
       renormalization is applied and is the identity there.
    """
    global OUT_DIR
    OUT_DIR = VARIANT_OUT
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ch in CHANNELS:
        lo, hi = CORE_WINDOW[ch]
        print(f"== {ch} (core-window normalization)")
        fig_unfolded(load(ch), norm_window=(lo, hi), suffix="_win18",
                     extra_line=f"Normalized over [{lo:g}, {hi:g}]",
                     head_frac=VARIANT_HEAD_FRAC)
    print("== trijet (coarser 2:1 gen binning)")
    d = load("trijet", row="proposal_2to1",
             syst="hadronic_rho_systematics_run2_2to1.json",
             bayes="hadronic_rho_bayes_run2_2to1.json")
    fig_unfolded(d, norm_window=CORE_WINDOW["trijet"], suffix="_2to1",
                 extra_line="Coarser gen binning (2:1)",
                 head_frac=VARIANT_HEAD_FRAC)
    print(f"{len(MADE)} variant figures in {OUT_DIR}")
    return 0


def jes_update(jes_json, split_mode="sqrt"):
    """The 8 uncertainty panels with the era-correlated JES band added.

    Written into ``OUT/jes_update`` so the production uncertainty figures --
    which predate the JES propagation -- stay byte-identical until the
    coordinator promotes these.
    """
    global OUT_DIR, JES_JSON, JES_SPLIT_MODE
    JES_JSON, JES_SPLIT_MODE = jes_json, split_mode
    OUT_DIR = JES_UPDATE_OUT
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ch in CHANNELS:
        print(f"== {ch} (JES band, {split_mode} split)")
        d = load(ch)
        core = d["core"]
        for key, lab in (("jes", "27 sources"),
                         ("jes_clean", "excl. RelativeJER*")):
            print(f"   JES {lab:<18} core median "
                  f"{np.nanmedian(d[key][core]):.3%}, core max "
                  f"{np.nanmax(d[key][core]):.3%}, shown max "
                  f"{np.nanmax(d[key][d['shown']]):.3%}")
        fig_uncertainty(d)
    print(f"{len(MADE)} figures in {OUT_DIR}")
    return 0


# --------------------------------------------------------------------------
def write_index():
    lines = ["# Hadronic groomed-rho figures -- pair_split Run 2",
             "",
             "Combined Run 2 (2016APV+2016+2017+2018, 137.6 fb$^{-1}$), "
             "groomed rho, tau = 0 TUnfoldDensity on the `coarse_tail` gen "
             "merge. Every entry has a `.png` and a `.pdf` twin.",
             "",
             "| figure | caption |",
             "|---|---|"]
    for name, cap in MADE:
        lines.append(f"| `{name}` | {cap} |")
    lines += ["", "Normalization: unit area over the shown bins of each pT "
                  "slice, divided by the bin width. The buffer bin "
                  "$\\log(\\rho^2) \\in [-10, -4]$ is never shown. The "
                  "relative statistical and systematic bands are taken from "
                  "the Phase B JSONs, where the shapes are normalized over "
                  "the narrower core window (dijet [-2.85, -0.55], trijet "
                  "[-3.0, -0.7]); the difference is a per-slice constant.",
              ""]
    (OUT / "index.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'index.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--common-unfold", action="store_true",
                    help="rerun the tau=0 unfold on the common six-bin gen "
                         "binning for figure 6 (needs ROOT)")
    ap.add_argument("--only", default=None,
                    help="regex; write only the figures whose name matches "
                         "(index.md is left alone in that case)")
    ap.add_argument("--variant", action="store_true",
                    help="write only the presentation variants of the "
                         "unfolded family into outputs/.../variants; the "
                         "production figures and index.md are left alone")
    ap.add_argument("--jes-update", action="store_true",
                    help="write only the uncertainty family, with the "
                         "era-correlated JES band added, into "
                         "outputs/.../jes_update; nothing else is touched")
    ap.add_argument("--jes-json", type=Path,
                    default=IN / "hadronic_rho_jes_run2.json")
    ap.add_argument("--jes-split-mode", choices=("sqrt", "linear"),
                    default="sqrt")
    args = ap.parse_args(argv)
    if args.common_unfold:
        return common_unfold()

    global ONLY
    ONLY = args.only
    if args.variant:
        return variants()
    if args.jes_update:
        return jes_update(args.jes_json, args.jes_split_mode)
    OUT.mkdir(parents=True, exist_ok=True)
    for ch in CHANNELS:
        print(f"== {ch}")
        d = load(ch)
        fig_unfolded(d)
        fig_uncertainty(d)
        fig_datamc_reco(d)
        fig_purity_stability(d)
        fig_amplification(d)
        fig_bayes(d)
        fig_response(d)
        fig_before_after(d)
    print("== channel comparison")
    fig_compare_channels()
    if ONLY is None:
        write_index()
    else:
        print(f"index.md untouched ({len(MADE)} figures rewritten)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
