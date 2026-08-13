#!/usr/bin/env python
"""Exhaustive appendix figure set for the pair_split Run 2 groomed-rho unfolds.

One figure per unfolding test, per pT slice, per channel -- the personal
appendix companion to the curated production set written by
``plot_hadronic_rho_pairsplit_run2.py``.  Style helpers are imported from that
driver so the two sets are visually identical; only the output directory
differs:

    outputs/figs/hadronic_rho_pairsplit_run2/appendix/

Families (``appendix_<family>_<channel>[_pt<i>][_extra]``):

  1  closure      tau=0 self-closure of the nominal MC, per reported bin
  2  systshift    signed per-bin normalized shifts of all ten variations
  3  varclosure   shifted input through the NOMINAL response (shape + absolute)
  4  pulls        pooled input-toy pull distribution + per-bin pull width
  5  coverage     per-bin 68% coverage with binomial errors
  6  config       per-bin ratio of every estimator challenger to production
  7  bayesiter    D'Agostini n = 1, 2, 4, 8 against tau = 0
  8  refold       measured/folded per reco bin, prefit and profiled; + pulls
  9  erareco      per-era detector-level data/MC; + chi2A per era
 10  erapurity    per-era purity and stability
 11  ladder       all five gen-merge ladder candidates

Sources: Phase B JSONs where they carry per-bin arrays, and
``appendix_rerun_run2.json`` (written by ``appendix_rerun_pairsplit_run2.py``)
where they carry only summaries.  ``appendix_manifest.md`` records which is
which for every figure.

    ~/Projects/GluonJetMass/.venv/bin/python \\
        scripts/plots/plot_hadronic_rho_pairsplit_appendix.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "plots"))

import plot_hadronic_rho_pairsplit_run2 as drv  # noqa: E402

IN = REPO / "outputs" / "pairsplit_unfold"
OUT = REPO / "outputs" / "figs" / "hadronic_rho_pairsplit_run2" / "appendix"
RERUN = IN / "appendix_rerun_run2.json"

CHANNELS = ("dijet", "trijet")
ERAS = ("2016APV", "2016", "2017", "2018")
ERA_COLORS = dict(zip(ERAS, drv.PETROFF[:4]))
#### one colour per systematic source, Up solid / Down dashed
SOURCES = ("JER", "JMS", "JMR", "fsr", "isr")
SOURCE_COLOR = dict(zip(SOURCES, drv.PETROFF[:5]))
SOURCE_LABEL = {"JER": "JER", "JMS": "JMS", "JMR": "JMR",
                "fsr": "FSR", "isr": "ISR"}
BAYES_N = (1, 2, 4, 8)
LADDER_COLORS = drv.PETROFF10
PROD_ROW = drv.PROD_ROW

MANIFEST: list[tuple[str, str, str]] = []   # (family, name, caption)
BLANK: list[str] = []


# --------------------------------------------------------------------------
# small helpers on top of the production driver's
# --------------------------------------------------------------------------
def annotate(ax, text, loc, fontsize=21):
    """In-panel annotation; returns its bottom edge in axes coordinates.

    ``loc == 2`` defers to the driver's helper, which left-aligns with the
    inside CMS block.  ``loc == 0`` (CMS block above the frame, used where the
    curves span the full panel height) has no block to align to, so the text
    is placed at a fixed inset instead.
    """
    if loc == 2:
        return drv.in_panel_text(ax, text, fontsize=fontsize)
    t = ax.text(0.028, 0.962, text, transform=ax.transAxes, ha="left",
                va="top", fontsize=fontsize, linespacing=1.5)
    fig = ax.figure
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(
        ax.transAxes.inverted())
    return bb.y0


def bottom_legend(fig, ax, ncol, fontsize=18, rows=2, handles=None,
                  labels=None):
    """Figure-level legend parked in a reserved strip under the axes.

    A ten-entry legend has no in-frame corner to live in on a symmetric
    signed-shift panel, and the axis range must stay symmetric, so the strip
    is reserved with the same ``rect`` mechanism ``save()`` uses for the
    provenance stamp.  The font is shrunk until the legend fits inside the
    canvas -- a "lower center" figure legend is NOT clipped by matplotlib, it
    simply runs off the edge and loses the last label.
    """
    fig._stamp_bottom = 0.075 + 0.035 * rows
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    legend = None
    for size in range(int(fontsize), int(fontsize) - 7, -1):
        if legend is not None:
            legend.remove()
        legend = fig.legend(handles, labels, loc="lower center",
                            bbox_to_anchor=(0.5, 0.02),
                            bbox_transform=fig.transFigure, ncol=ncol,
                            fontsize=size, frameon=False, columnspacing=1.3,
                            handlelength=1.9, handletextpad=0.5)
        fig.canvas.draw()
        box = legend.get_window_extent(
            fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        if box.width <= 0.975:
            break
    return legend


def source_legend_keys():
    """Compact key for the ten-variation panels: colour = source, style = sign.

    Ten explicit "SOURCE Up"/"SOURCE Down" entries do not fit on one canvas
    width; five colours plus two line styles carry exactly the same
    information in seven entries.
    """
    handles = [Line2D([], [], color=SOURCE_COLOR[s], lw=3.0) for s in SOURCES]
    labels = [SOURCE_LABEL[s] for s in SOURCES]
    handles += [Line2D([], [], color="0.25", lw=2.6, ls="-"),
                Line2D([], [], color="0.25", lw=2.6, ls="--")]
    labels += ["Up", "Down"]
    return handles, labels


def symmetric(ax, values, pad=1.18, floor=1e-9):
    """Symmetric y range around zero sized on the plotted values."""
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    lim = max(float(np.max(np.abs(finite))) if finite.size else floor, floor)
    ax.set_ylim(-pad * lim, pad * lim)
    return lim


def step(ax, edges, values, mask, **kw):
    """Post-step curve over the masked bins of ``edges``."""
    x = np.r_[edges[:-1][mask], edges[1:][mask][-1]]
    y = np.r_[values[mask], values[mask][-1]]
    return ax.step(x, y, where="post", **kw)


def check(name, *arrays):
    """Record a figure whose every plotted point is NaN/absent."""
    ok = any(np.isfinite(np.asarray(a, float)).any() for a in arrays if
             np.size(a))
    if not ok:
        BLANK.append(name)
        print(f"  !! {name}: nothing finite to draw")
    return ok


def emit(fig, family, name, caption):
    drv.save(fig, name, caption)
    MANIFEST.append((family, name, caption))


def r2(values, n_pt, n_rho):
    return np.asarray(values, float).reshape(n_pt, n_rho)


def load_rerun():
    if not RERUN.exists():
        raise SystemExit(
            f"missing {RERUN}\nrun, under ROOT:\n"
            "  source ~/Projects/unfold/scripts/setup_root.sh\n"
            "  ~/Projects/unfold/.venv/bin/python "
            "scripts/plots/appendix_rerun_pairsplit_run2.py")
    return json.loads(RERUN.read_text())


def per_pt_shape(values, n_pt, n_rho, in_window):
    """Per-pT unit area over the ``in_window`` bins (the Phase B convention)."""
    v = r2(values, n_pt, n_rho)
    totals = v[:, in_window].sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(totals > 0, v / totals, np.nan)


# --------------------------------------------------------------------------
# 1. tau = 0 self-closure
# --------------------------------------------------------------------------
def fig_closure(d, rr):
    ch, ge = d["channel"], d["gedges"]
    n_pt, n_rho = d["n_pt"], d["n_rho"]
    dev = r2(rr["closure"]["ratio"], n_pt, n_rho) - 1.0
    shown = d["shown"]
    scale = 10.0 ** np.floor(np.log10(
        max(np.nanmax(np.abs(dev[shown])), 1e-18)))
    for p in range(n_pt):
        m = shown[p]
        y = dev[p] / scale
        if not check(f"appendix_closure_{ch}_pt{p + 1}", y[m]):
            continue
        fig, ax = drv.panel()
        drv.cms_label(ax, rlabel=drv.SIM_ERA, simulation=True)
        ax.axhline(0.0, color="#9c9ca1", lw=1.4)
        ax.plot(drv.centers(ge)[m], y[m], marker="o", ms=11, lw=2.0,
                color=drv.PETROFF[0])
        ax.set_xlabel(drv.XLAB)
        ax.set_ylabel(f"Unfolded MC / gen $-$ 1  [$10^{{{int(np.log10(scale))}}}$]")
        ax.set_xlim(ge[:-1][m].min(), ge[-1])
        lim = symmetric(ax, y[m], pad=1.0)
        ax.set_ylim(-2.6 * lim, 2.6 * lim)
        worst = np.nanmax(np.abs(dev[p][m]))
        annotate(ax, f"{d['ptlab'][p]}\nmax $|$dev$|$ = {worst:.1e}\n"
                     f"criterion $10^{{-10}}$", loc=2, fontsize=21)
        emit(fig, "closure", f"appendix_closure_{ch}_pt{p + 1}",
             f"{ch}, {d['ptlab'][p]}: tau = 0 self-closure -- nominal MC "
             f"unfolded through its own response, relative deviation from "
             f"gen per reported bin (axis in units of "
             f"1e{int(np.log10(scale))}; the 1e-10 criterion is "
             f"{1e-10 / scale:.0f} times the full axis range).")


# --------------------------------------------------------------------------
# 2. signed systematic shifts
# --------------------------------------------------------------------------
def signed_shifts(d):
    sy = d["sy"]
    out = {}
    for source in ("JER", "JMS", "JMR"):
        out[source] = {k: r2(v, d["n_pt"], d["n_rho"]) for k, v in
                       sy["signed_relative_shift"][source].items()}
    for source in ("fsr", "isr"):
        out[source] = {
            k: r2(v, d["n_pt"], d["n_rho"]) for k, v in
            sy["parton_shower"]["signed_relative_shift"][source].items()}
    return out


def fig_systshift(d):
    ch, ge = d["channel"], d["gedges"]
    shifts = signed_shifts(d)
    for p in range(d["n_pt"]):
        m = d["shown"][p]
        pool = []
        fig, ax = drv.panel()
        drv.cms_label(ax, loc=0)
        ax.axhline(0.0, color="#9c9ca1", lw=1.6)
        for source in SOURCES:
            for direction, ls in (("Up", "-"), ("Down", "--")):
                y = 100.0 * shifts[source][direction][p]
                pool.append(y[m])
                step(ax, ge, y, m, color=SOURCE_COLOR[source], ls=ls, lw=2.4,
                     label=f"{SOURCE_LABEL[source]} {direction}")
        if not check(f"appendix_systshift_{ch}_pt{p + 1}", *pool):
            plt.close(fig)
            continue
        ax.set_xlabel(drv.XLAB)
        ax.set_ylabel("Signed shift of the normalized shape [%]")
        ax.set_xlim(ge[:-1][m].min(), ge[-1])
        symmetric(ax, np.concatenate(pool))
        annotate(ax, d["ptlab"][p], loc=0, fontsize=21)
        keys, key_labels = source_legend_keys()
        bottom_legend(fig, ax, ncol=7, fontsize=18, rows=1, handles=keys,
                      labels=key_labels)
        emit(fig, "systshift", f"appendix_systshift_{ch}_pt{p + 1}",
             f"{ch}, {d['ptlab'][p]}: signed per-bin shift of the normalized "
             f"shape for every propagated variation (JER/JMS/JMR/FSR/ISR, Up "
             f"solid and Down dashed); ISR is shown for observation only.")


# --------------------------------------------------------------------------
# 3. shifted input through the nominal response
# --------------------------------------------------------------------------
def fig_varclosure(d, rr, mode="shape"):
    ch, ge = d["channel"], d["gedges"]
    key = "ratio_shape" if mode == "shape" else "ratio_absolute"
    suffix = "" if mode == "shape" else "_absolute"
    what = ("per-pT normalized shape" if mode == "shape"
            else "absolute yield")
    for p in range(d["n_pt"]):
        m = d["shown"][p]
        pool = []
        fig, ax = drv.panel()
        drv.cms_label(ax, rlabel=drv.SIM_ERA, simulation=True, loc=0)
        ax.axhline(0.0, color="#9c9ca1", lw=1.6)
        for source in SOURCES:
            for direction, ls in (("Up", "-"), ("Down", "--")):
                y = 100.0 * (r2(rr["varclosure"][f"{source}{direction}"][key],
                                d["n_pt"], d["n_rho"])[p] - 1.0)
                pool.append(y[m])
                step(ax, ge, y, m, color=SOURCE_COLOR[source], ls=ls, lw=2.4,
                     label=f"{SOURCE_LABEL[source]} {direction}")
        name = f"appendix_varclosure_{ch}_pt{p + 1}{suffix}"
        if not check(name, *pool):
            plt.close(fig)
            continue
        ax.set_xlabel(drv.XLAB)
        ax.set_ylabel(f"Unfolded / truth $-$ 1, {what} [%]")
        ax.set_xlim(ge[:-1][m].min(), ge[-1])
        symmetric(ax, np.concatenate(pool))
        annotate(ax, f"{d['ptlab'][p]}\nShifted input, nominal response",
                 loc=0, fontsize=21)
        keys, key_labels = source_legend_keys()
        bottom_legend(fig, ax, ncol=7, fontsize=18, rows=1, handles=keys,
                      labels=key_labels)
        shower = max(rr["varclosure"][f"{s}{x}"]["gen_max_relative_change"]
                     for s in ("fsr", "isr") for x in ("Up", "Down"))
        detector = max(rr["varclosure"][f"{s}{x}"]["gen_max_relative_change"]
                       for s in ("JER", "JMS", "JMR") for x in ("Up", "Down"))
        emit(fig, "varclosure", name,
             f"{ch}, {d['ptlab'][p]}: each variation's matched reco spectrum "
             f"unfolded through the NOMINAL response and compared with the "
             f"nominal truth, {what} per reported bin. For JER/JMS/JMR the gen "
             f"spectrum is invariant to {detector:.0e}, so the curve isolates "
             f"response mismodelling; the FSR/ISR truth itself moves by up to "
             f"{shower:.0%}, so those two are dominated by the genuine "
             f"particle-level shift and are not a closure failure.")


# --------------------------------------------------------------------------
# 4/5. input-stat pseudo-experiments
# --------------------------------------------------------------------------
def fig_pulls(d, rr):
    ch, ge = d["channel"], d["gedges"]
    toys = rr["input_toys"]
    n_pt, n_rho = d["n_pt"], d["n_rho"]
    shown = np.asarray(rr["shown"], bool)
    pulls = np.asarray(toys["pulls"], float)[:, shown]
    flat = pulls[np.isfinite(pulls)].ravel()
    width = r2(toys["pull_width"], n_pt, n_rho)
    if not check(f"appendix_pulls_{ch}", flat, width[d["shown"]]):
        return

    fig, axes = drv.grid(1, 2)
    ax = axes[0]
    drv.cms_label(ax, rlabel=drv.SIM_ERA, simulation=True)
    bins = np.linspace(-4.0, 4.0, 41)
    ax.hist(flat, bins=bins, density=True, histtype="stepfilled",
            color=drv.PETROFF[0], alpha=0.45, lw=0.0)
    ax.hist(flat, bins=bins, density=True, histtype="step",
            color=drv.PETROFF[0], lw=2.6,
            label=f"{toys['n_toys']} toys $\\times$ "
                  f"{int(shown.sum())} reported bins")
    xx = np.linspace(-4.0, 4.0, 401)
    ax.plot(xx, np.exp(-0.5 * xx ** 2) / np.sqrt(2.0 * np.pi), color="k",
            lw=2.4, ls="--", label="Unit Gaussian")
    ax.set_xlabel(r"Pull  (unfolded $-$ truth) / $\sigma$")
    ax.set_ylabel("Probability density")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(bottom=0)
    drv.headroom(ax, 0.45, frac=2.05)
    ybot = annotate(ax, f"{ch}\npooled mean {flat.mean():+.3f}\n"
                        f"pooled width {flat.std(ddof=1):.3f}", loc=2,
                    fontsize=21)
    ax.legend(loc="upper left", framealpha=0.0, fontsize=18,
              bbox_to_anchor=(0.0, ybot - 0.03))

    ax = axes[1]
    drv.cms_label(ax, rlabel=drv.SIM_ERA, simulation=True)
    for p in range(n_pt):
        m = d["shown"][p]
        step(ax, ge, width[p], m, color=drv.PETROFF[p % 6], lw=2.4,
             label=d["ptlab"][p])
    ax.axhline(1.0, color="#9c9ca1", ls=":", lw=2.2)
    ax.set_xlabel(drv.XLAB)
    ax.set_ylabel(r"Pull width  $\sigma_{\mathrm{toys}}\,/\,\sigma_{\mathrm{analytic}}$")
    ax.set_xlim(ge[:-1][d["shown"][0]].min(), ge[-1])
    ax.set_ylim(0.80, 1.34)
    ax.legend(loc="upper left", fontsize=18, framealpha=0.0,
              bbox_to_anchor=(0.0, 0.80))
    emit(fig, "pulls", f"appendix_pulls_{ch}",
         f"{ch}: input-stat pseudo-experiments -- pooled pull distribution "
         f"over all reported bins with a unit Gaussian overlay (left) and the "
         f"per-bin pull width (right); {toys['n_toys']} toys, seed "
         f"{toys['seed']}.")


def fig_coverage(d, rr):
    ch, ge = d["channel"], d["gedges"]
    toys = rr["input_toys"]
    n = toys["n_toys"]
    cov = r2(toys["coverage_1sigma"], d["n_pt"], d["n_rho"])
    err = np.sqrt(np.clip(cov * (1.0 - cov), 0.0, None) / n)
    if not check(f"appendix_coverage_{ch}", cov[d["shown"]]):
        return
    fig, ax = drv.panel()
    drv.cms_label(ax, rlabel=drv.SIM_ERA, simulation=True)
    for p in range(d["n_pt"]):
        m = d["shown"][p]
        ax.errorbar(drv.centers(ge)[m], cov[p][m], yerr=err[p][m],
                    xerr=np.diff(ge)[m] / 2, fmt="o", ms=8, lw=1.8,
                    capsize=0, color=drv.PETROFF[p % 6], label=d["ptlab"][p])
    ax.axhline(0.68, color="#9c9ca1", ls=":", lw=2.4)
    ax.set_xlabel(drv.XLAB)
    ax.set_ylabel("68% coverage of the analytic error")
    ax.set_xlim(ge[:-1][d["shown"][0]].min(), ge[-1])
    #### a three-column legend of "200 < pT < 290 GeV" entries is wider than
    #### the panel, so it is kept at two columns and the data pushed down
    ax.set_ylim(0.50, 1.00)
    ybot = annotate(ax, f"{n} input-stat toys, seed {toys['seed']}", loc=2,
                    fontsize=21)
    ax.legend(loc="upper left", fontsize=18, framealpha=0.0, ncol=2,
              bbox_to_anchor=(0.0, ybot - 0.03))
    emit(fig, "coverage", f"appendix_coverage_{ch}",
         f"{ch}: fraction of {n} input-stat toys within one analytic sigma of "
         f"the truth, per reported bin, with binomial errors; guide at 0.68.")


# --------------------------------------------------------------------------
# 6. estimator configurations
# --------------------------------------------------------------------------
CONFIG_ROWS = (
    ("tau0_noarea_full", "No area", drv.PETROFF[0], "-"),
    ("tau0_area_diag", "Diagonal cov.", drv.PETROFF[1], "-"),
    ("gls", "GLS", drv.PETROFF[3], "--"),
    ("bayes_n4", "Bayes $n$ = 4", drv.PETROFF[2], "-"),
)


def coarser_map(fine_edges, coarse_edges):
    """Index of the coarse bin each fine bin falls into (nested binnings)."""
    centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    return np.clip(np.searchsorted(coarse_edges, centers) - 1, 0,
                   len(coarse_edges) - 2)


def fig_config(d):
    ch, ge = d["channel"], d["gedges"]
    cfg = json.loads(
        (IN / "hadronic_rho_config_comparison_run2.json").read_text())[ch]
    n_pt, n_rho = d["n_pt"], d["n_rho"]

    #### the coarser 2to1 candidate, rebinned against the production result:
    #### both sides are renormalized over the 2to1 window so the comparison is
    #### a pure shape statement on the coarse axis.
    twin = json.loads(
        (IN / "hadronic_rho_config_comparison_run2_2to1.json").read_text())[ch]
    ge2 = np.asarray(twin["gen_rho_edges"], float)
    n_rho2 = len(ge2) - 1
    win2 = np.asarray(twin["in_window"], bool)
    shown2 = np.asarray(twin["shown"], bool).reshape(n_pt, n_rho2)
    fine2coarse = coarser_map(ge, ge2)
    merge = np.zeros((n_rho, n_rho2))
    merge[np.arange(n_rho), fine2coarse] = 1.0
    prod_fine = r2(cfg["configurations"]["tau0_area_full"]["per_bin"]["unfolded"],
                   n_pt, n_rho)
    prod_coarse = prod_fine @ merge
    twin_coarse = r2(twin["configurations"]["tau0_area_full"]["per_bin"]["unfolded"],
                     n_pt, n_rho2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sp = prod_coarse / prod_coarse[:, win2].sum(axis=1, keepdims=True)
        st = twin_coarse / twin_coarse[:, win2].sum(axis=1, keepdims=True)
        ratio2 = st / sp - 1.0

    for p in range(n_pt):
        m = d["shown"][p]
        pool = []
        fig, ax = drv.panel()
        drv.cms_label(ax, loc=0)
        ax.axhline(0.0, color="#9c9ca1", lw=1.6)
        for key, label, color, ls in CONFIG_ROWS:
            y = 100.0 * r2(
                cfg["configurations"][key]["per_bin"]
                ["signed_shape_shift_vs_production"], n_pt, n_rho)[p]
            pool.append(y[m])
            step(ax, ge, y, m, color=color, ls=ls, lw=2.6, label=label)
        m2 = shown2[p]
        y2 = 100.0 * ratio2[p]
        pool.append(y2[m2])
        step(ax, ge2, y2, m2, color=drv.PETROFF[5], ls="-", lw=2.6,
             label="2:1 gen merge")
        name = f"appendix_config_{ch}_pt{p + 1}"
        if not check(name, *pool):
            plt.close(fig)
            continue
        ax.set_xlabel(drv.XLAB)
        ax.set_ylabel("Challenger / production shape $-$ 1 [%]")
        ax.set_xlim(min(ge[:-1][m].min(), ge2[:-1][m2].min()), ge[-1])
        symmetric(ax, np.concatenate(pool))
        annotate(ax, d["ptlab"][p], loc=0, fontsize=21)
        bottom_legend(fig, ax, ncol=5, fontsize=18, rows=1)
        emit(fig, "config", name,
             f"{ch}, {d['ptlab'][p]}: per-bin ratio of every estimator "
             f"challenger to the production tau = 0 result (no area "
             f"constraint, diagonal data covariance, independent GLS, "
             f"iterative Bayes n = 4, and the coarser 2:1 gen merge "
             f"renormalized onto its own window).")


# --------------------------------------------------------------------------
# 7. iterative-Bayes iteration scan
# --------------------------------------------------------------------------
def fig_bayesiter(d, rr):
    ch, ge = d["channel"], d["gedges"]
    n_pt, n_rho = d["n_pt"], d["n_rho"]
    for p in range(n_pt):
        m = d["shown"][p]
        pool = []
        fig, ax = drv.panel()
        drv.cms_label(ax, loc=0)
        ax.axhline(0.0, color="#9c9ca1", lw=1.6)
        for index, n in enumerate(BAYES_N):
            y = 100.0 * (r2(
                rr["bayes_iter"]["iterations"][str(n)]["ratio_vs_tunfold"],
                n_pt, n_rho)[p] - 1.0)
            pool.append(y[m])
            step(ax, ge, y, m, color=drv.PETROFF[index], lw=2.6,
                 label=f"$n$ = {n}" + ("  (reference)" if n == 4 else ""))
        name = f"appendix_bayesiter_{ch}_pt{p + 1}"
        if not check(name, *pool):
            plt.close(fig)
            continue
        ax.set_xlabel(drv.XLAB)
        ax.set_ylabel(r"Bayes / TUnfold ($\tau$ = 0) $-$ 1 [%]")
        ax.set_xlim(ge[:-1][m].min(), ge[-1])
        symmetric(ax, np.concatenate(pool))
        annotate(ax, f"{d['ptlab'][p]}\nD'Agostini, MC prior", loc=0,
                 fontsize=21)
        bottom_legend(fig, ax, ncol=4, fontsize=18, rows=1)
        emit(fig, "bayesiter", name,
             f"{ch}, {d['ptlab'][p]}: unregularized D'Agostini normalized "
             f"shape at n = 1, 2, 4, 8 relative to the production tau = 0 "
             f"TUnfold result, per reported bin.")


# --------------------------------------------------------------------------
# 8. profiled refolding
# --------------------------------------------------------------------------
def fig_refold(d, rr):
    ch = d["channel"]
    re = np.asarray(rr["reco_rho_edges"], float)
    n_rho_r = len(re) - 1
    n_pt_r = len(d["reco_pt_edges"]) - 1
    ref = rr["refold"]
    pre = r2(ref["ratio_prefit"], n_pt_r, n_rho_r)
    post = r2(ref["ratio_postfit"], n_pt_r, n_rho_r)
    data = r2(ref["data_prefit"], n_pt_r, n_rho_r)
    sigma = r2(ref["sigma_data"], n_pt_r, n_rho_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(data > 0, sigma / data, np.nan)
    x = np.arange(n_rho_r) + 0.5
    pulls = ", ".join(f"{k} {v:+.3f}" for k, v in ref["nuisance_pulls"].items())

    for p in range(n_pt_r):
        name = f"appendix_refold_{ch}_pt{p + 1}"
        if not check(name, pre[p], post[p]):
            continue
        fig, ax = drv.panel()
        drv.cms_label(ax)
        ax.axhline(1.0, color="#9c9ca1", lw=1.6)
        ax.errorbar(x - 0.12, pre[p], yerr=(rel * pre)[p], fmt="o", ms=10,
                    mfc="none", mew=2.2, lw=1.8, capsize=0,
                    color=drv.PETROFF[0], label="Prefit (baseline GLS)")
        ax.errorbar(x + 0.12, post[p], yerr=(rel * post)[p], fmt="s", ms=9,
                    lw=1.8, capsize=0, color=drv.PETROFF[2],
                    label="Profiled JER/JMS/JMR")
        ax.set_xlabel(drv.XLAB_RECO)
        ax.set_ylabel("Measured / folded")
        ax.set_xlim(0.0, float(n_rho_r))
        ax.set_xticks(np.arange(n_rho_r + 1))
        ax.set_xticklabels([f"{v:g}" for v in re], fontsize=17, rotation=45)
        lim = np.nanmax(np.abs(np.r_[pre[p], post[p],
                                     (pre + rel * pre)[p],
                                     (post - rel * post)[p]] - 1.0))
        #### three annotation lines plus a two-entry legend need the top of
        #### the data to sit below ~0.55 of the panel height
        ax.set_ylim(1.0 - 1.30 * lim, 1.0 + 3.60 * lim)
        ybot = annotate(
            ax,
            f"{drv.pt_label(d['reco_pt_edges'], p)}\n"
            f"$\\chi^2$ {ref['chi2_prefit']:.1f} $\\rightarrow$ "
            f"{ref['chi2_postfit_data']:.1f} (all slices)\n{pulls}",
            loc=2, fontsize=20)
        ax.legend(loc="upper left", fontsize=19, framealpha=0.0,
                  bbox_to_anchor=(0.0, ybot - 0.03))
        emit(fig, "refold", name,
             f"{ch}, {drv.pt_label(d['reco_pt_edges'], p)}: fake-corrected "
             f"data over the folded prediction per detector bin, before and "
             f"after the profiled JER/JMS/JMR fit; the leftmost bin is the "
             f"[-10, -4] buffer.")


def fig_refoldpulls(d):
    ch = d["channel"]
    prof = json.loads(
        (IN / "hadronic_rho_profiled_refold_run2.json").read_text())[ch]
    fits = [
        ("profile_JER_JMS_JMR_data_stat_only", "Data stat only", None),
        ("profile_JER_JMS_JMR_with_approx_response_mc_stat",
         "+ response MC stat", None),
    ]
    for feed in ("10pct", "20pct", "50pct", "100pct"):
        fits.append((feed, f"+ feed-in {feed}", "data_stat_only"))

    records = []
    for key, label, sub in fits:
        node = (prof["profile_with_first_pt_feed_scan"][key][sub] if sub
                else prof[key])
        records.append((label, node["nuisance_pulls"],
                        node.get("nuisance_at_bounds", {})))
    names = []
    for _, pulls, _ in records:
        for k in pulls:
            short = "feed" if k.startswith("feed_firstpt") else k
            if short not in names:
                names.append(short)
    if not check(f"appendix_refoldpulls_{ch}",
                 [v for _, p, _ in records for v in p.values()]):
        return

    fig, ax = drv.panel()
    drv.cms_label(ax)
    width = 0.8 / len(records)
    for index, (label, pulls, bounds) in enumerate(records):
        xs, ys, edge = [], [], []
        for j, short in enumerate(names):
            key = next((k for k in pulls
                        if (k.startswith("feed_firstpt") and short == "feed")
                        or k == short), None)
            if key is None:
                continue
            xs.append(j - 0.4 + width * (index + 0.5))
            ys.append(pulls[key])
            edge.append("k" if bounds.get(key) else "none")
        ax.bar(xs, ys, width=width * 0.92, color=drv.PETROFF10[index],
               edgecolor=edge, linewidth=1.6, label=label)
    for level in (-1.0, 1.0):
        ax.axhline(level, color="#9c9ca1", ls="--", lw=1.8)
    ax.axhline(0.0, color="k", lw=1.2)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n.upper() if n != "feed" else "First-pT feed-in"
                        for n in names])
    ax.set_ylabel("Fitted nuisance pull [$\\sigma$]")
    values = [v for _, pulls, _ in records for v in pulls.values()]
    lo = min(-1.45, min(values) * 1.2)
    hi = max(1.45, max(values) * 1.2)
    #### keep the tallest bar under 55% of the panel height so the two-column
    #### legend below the annotation never lands on a bar
    ax.set_ylim(lo, lo + (hi - lo) / 0.55)
    ybot = annotate(ax, "Dashed: template range $\\pm$1; black outline = at "
                        "bound", loc=2, fontsize=20)
    ax.legend(loc="upper left", ncol=2, fontsize=18, framealpha=0.0,
              bbox_to_anchor=(0.0, ybot - 0.03))
    emit(fig, "refold", f"appendix_refoldpulls_{ch}",
         f"{ch}: fitted nuisance pulls of the profiled refolding fit for every "
         f"variant in the Phase B scan (data stat only, with approximate "
         f"response MC stat, and the four first-pT feed-in priors).")


# --------------------------------------------------------------------------
# 9. per-era detector level
# --------------------------------------------------------------------------
def era_inputs(channel):
    out = {}
    for era in ERAS:
        z = dict(np.load(IN / "npz_era" / f"{channel}_{era}_nominal.npz"))
        out[era] = z
    return out


def fig_erareco(d, eras):
    ch = d["channel"]
    re = d["reco_rho_edges"]
    n_rho = len(re) - 1
    n_pt = len(d["reco_pt_edges"]) - 1
    lo, hi = d["norm_window"]
    win = (re[:-1] >= lo - 1e-9) & (re[1:] <= hi + 1e-9)
    m = np.ones(n_rho, bool)
    m[:int(d["z"]["n_buffer"])] = False
    xc, w = drv.centers(re), np.diff(re)

    ratios, errors = {}, {}
    for era, z in eras.items():
        A = np.asarray(z["A"], float)
        matched = A.sum(axis=1).reshape(n_pt, n_rho)
        reco = np.asarray(z["reco"], float).reshape(n_pt, n_rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            survival = 1.0 - np.clip(
                np.where(reco > 0, (reco - matched) / reco, 0.0), 0.0, 1.0)
        data = np.asarray(z["data_reco"], float).reshape(n_pt, n_rho) * survival
        dw2 = np.asarray(z["data_w2"], float).reshape(n_pt, n_rho) * survival ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            sd = data / data[:, win].sum(axis=1, keepdims=True)
            sm = matched / matched[:, win].sum(axis=1, keepdims=True)
            ratios[era] = sd / sm
            errors[era] = np.where(data > 0, np.sqrt(dw2) / data, np.nan) \
                * ratios[era]

    for p in range(n_pt):
        name = f"appendix_erareco_{ch}_pt{p + 1}"
        if not check(name, *[ratios[e][p][m] for e in ERAS]):
            continue
        fig, ax = drv.panel()
        drv.cms_label(ax)
        ax.axhline(1.0, color="#9c9ca1", lw=1.6)
        pool = []
        for era in ERAS:
            y, e = ratios[era][p], errors[era][p]
            pool.append(np.r_[(y + e)[m], (y - e)[m]])
            ax.errorbar(xc[m], y[m], yerr=e[m], xerr=w[m] / 2, fmt="o", ms=8,
                        lw=1.8, capsize=0, color=ERA_COLORS[era], label=era)
        ax.set_xlabel(drv.XLAB_RECO)
        ax.set_ylabel("Data / MC of the normalized shape")
        ax.set_xlim(re[:-1][m].min(), re[-1])
        lim = np.nanmax(np.abs(np.concatenate(pool) - 1.0))
        ax.set_ylim(1.0 - 1.30 * lim, 1.0 + 2.35 * lim)
        ybot = annotate(
            ax, f"{drv.pt_label(d['reco_pt_edges'], p)}\n"
                f"Normalized over [{lo:g}, {hi:g}]", loc=2, fontsize=21)
        ax.legend(loc="upper left", ncol=2, fontsize=19, framealpha=0.0,
                  bbox_to_anchor=(0.0, ybot - 0.03))
        emit(fig, "erareco", name,
             f"{ch}, {drv.pt_label(d['reco_pt_edges'], p)}: detector-level "
             f"data/MC of the normalized groomed-rho shape for each era, "
             f"normalized inside the core window; statistical errors only.")


def fig_erachi2(d):
    ch = d["channel"]
    era_json = json.loads(
        (IN / "hadronic_rho_era_chi2_run2.json").read_text())[ch]
    values = [era_json[era]["tunfold_chi2A_at_run2_yield"] for era in ERAS]
    combined = era_json["run2_combined"]["tunfold_chi2A"]
    if not check(f"appendix_erachi2_{ch}", values):
        return
    fig, ax = drv.panel()
    drv.cms_label(ax)
    ax.bar(np.arange(len(ERAS)), values, width=0.62,
           color=[ERA_COLORS[e] for e in ERAS])
    ax.axhline(combined, color="k", ls="--", lw=2.2,
               label=f"Run 2 combined ({combined:.0f})")
    ax.set_xticks(np.arange(len(ERAS)))
    ax.set_xticklabels(ERAS)
    ax.set_ylabel(r"Folded $\chi^2_{A}$ rescaled to the Run 2 yield")
    ax.set_ylim(bottom=0)
    drv.headroom(ax, max(max(values), combined), frac=2.05)
    #### the annotation must be placed BEFORE the per-bar value labels are
    #### added: ``in_panel_text`` anchors under the LOWEST in-frame text, and
    #### a bar label would drag it down onto the bars.
    ybot = annotate(ax, "Each era rebuilt from its own data, MC and "
                        "covariance", loc=2, fontsize=20)
    ax.legend(loc="upper left", fontsize=19, framealpha=0.0,
              bbox_to_anchor=(0.0, ybot - 0.03))
    for index, value in enumerate(values):
        ax.text(index, value * 1.02, f"{value:.0f}", ha="center", va="bottom",
                fontsize=19)
    emit(fig, "erareco", f"appendix_erachi2_{ch}",
         f"{ch}: tau = 0 folded chi2A of each era's own unfold, rescaled to "
         f"the Run 2 data yield so the eras are comparable; dashed line is the "
         f"combined Run 2 value.")


# --------------------------------------------------------------------------
# 10. per-era purity and stability
# --------------------------------------------------------------------------
def era_purity_stability(z):
    """rho-only purity/stability, marginal and per pT slice.

    Same definitions as ``era_check/era_binning_check.json``: purity is
    reco-conditioned (per-pT slices condition on reco pT), stability is
    gen-conditioned (per-pT slices condition on gen pT).  Reproduced from the
    response matrix so all four eras are computed identically -- the JSON
    covers only three of them.
    """
    n_pt = len(z["pt_edges"]) - 1
    n_rho = len(z["rho_edges"]) - 1
    A = np.asarray(z["A"], float).reshape(n_pt, n_rho, n_pt, n_rho)
    marg = A.sum(axis=(0, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        purity_marg = np.diag(marg) / marg.sum(axis=1)
        stab_marg = np.diag(marg) / marg.sum(axis=0)
        by_reco_pt = A.sum(axis=2)                     # (pt_reco, rho_reco, rho_gen)
        purity_pt = np.array([
            np.diag(by_reco_pt[p]) / by_reco_pt[p].sum(axis=1)
            for p in range(n_pt)])
        by_gen_pt = A.sum(axis=0)                      # (rho_reco, pt_gen, rho_gen)
        stab_pt = np.array([
            np.diag(by_gen_pt[:, p, :]) / by_gen_pt[:, p, :].sum(axis=0)
            for p in range(n_pt)])
    return purity_marg, stab_marg, purity_pt, stab_pt


def fig_erapurity(d, eras):
    ch = d["channel"]
    re = d["reco_rho_edges"]
    n_rho = len(re) - 1
    n_pt = len(d["reco_pt_edges"]) - 1
    m = np.ones(n_rho, bool)
    m[:int(d["z"]["n_buffer"])] = False
    computed = {era: era_purity_stability(z) for era, z in eras.items()}

    fig, axes = drv.grid(1, 2)
    for ax, index, name in ((axes[0], 0, "Purity"), (axes[1], 1, "Stability")):
        drv.cms_label(ax, rlabel=drv.SIM_ERA, simulation=True)
        for era in ERAS:
            step(ax, re, computed[era][index], m, color=ERA_COLORS[era],
                 lw=2.6, label=era)
        ax.axhline(0.5, color="#9c9ca1", ls=":", lw=2.2)
        ax.set_xlabel(drv.XLAB_RECO)
        ax.set_ylabel(f"{name} (groomed $\\rho$ only)")
        ax.set_xlim(re[:-1][m].min(), re[-1])
        ax.set_ylim(0.0, 1.19)
        ax.legend(loc="lower left", ncol=2, fontsize=19, framealpha=0.0)
    emit(fig, "erapurity", f"appendix_erapurity_{ch}",
         f"{ch}: rho-only purity (left) and stability (right) of the detector "
         f"binning for each era, marginalized over pT; guide at 0.5.")

    for p in range(n_pt):
        name = f"appendix_erapurity_{ch}_pt{p + 1}"
        if not check(name, computed[ERAS[0]][2][p][m]):
            continue
        fig, axes = drv.grid(1, 2)
        for ax, index, what in ((axes[0], 2, "Purity"),
                                (axes[1], 3, "Stability")):
            drv.cms_label(ax, rlabel=drv.SIM_ERA, simulation=True)
            for era in ERAS:
                step(ax, re, computed[era][index][p], m,
                     color=ERA_COLORS[era], lw=2.6, label=era)
            ax.axhline(0.5, color="#9c9ca1", ls=":", lw=2.2)
            ax.set_xlabel(drv.XLAB_RECO)
            ax.set_ylabel(f"{what} (groomed $\\rho$ only)")
            ax.set_xlim(re[:-1][m].min(), re[-1])
            ax.set_ylim(0.0, 1.19)
            drv.in_panel_text(ax, drv.pt_label(d["reco_pt_edges"], p),
                              fontsize=21)
            ax.legend(loc="lower left", ncol=2, fontsize=19, framealpha=0.0)
        emit(fig, "erapurity", name,
             f"{ch}, {drv.pt_label(d['reco_pt_edges'], p)}: rho-only purity "
             f"(left, conditioned on reco pT) and stability (right, "
             f"conditioned on gen pT) per era in this pT slice.")


# --------------------------------------------------------------------------
# 11. the gen-merge stability ladder
# --------------------------------------------------------------------------
def ladder_shape_ratio(row):
    """Per-pT normalized unfolded/gen shape ratio, the stability convention."""
    edges = np.asarray(row["gen_rho_edges"], float)
    n_rho = len(edges) - 1
    n_pt = len(row["pt_groups"])
    x = r2(row["unfolded"], n_pt, n_rho)
    g = r2(row["gen"], n_pt, n_rho)
    err = r2(row["err"], n_pt, n_rho)
    shown = np.asarray(row["shown"], bool).reshape(n_pt, n_rho)
    lo, hi = row["norm_window"]
    win = (edges[:-1] >= lo - 1e-9) & (edges[1:] <= hi + 1e-9)
    ratio = np.full((n_pt, n_rho), np.nan)
    rel = np.full((n_pt, n_rho), np.nan)
    for p in range(n_pt):
        m, w = shown[p], shown[p] & win
        if m.sum() < 2 or w.sum() < 1:
            continue
        scale = (g[p][w].sum() / g[p][m].sum()) / (x[p][w].sum() / x[p][m].sum())
        r = np.full(n_rho, np.nan)
        r[m] = (x[p][m] / x[p][m].sum()) / (g[p][m] / g[p][m].sum()) * scale
        ratio[p] = r
        with np.errstate(divide="ignore", invalid="ignore"):
            rel[p] = np.where(x[p] > 0, err[p] / x[p], np.nan)
    return edges, shown, ratio, rel


def fig_ladder(d):
    ch = d["channel"]
    rows = json.loads((IN / f"{ch}_run2_stability.json").read_text())
    prepared = {}
    for row in rows:
        prepared[row["label"]] = (row, *ladder_shape_ratio(row))
    n_pt = d["n_pt"]

    for p in range(n_pt):
        pool = []
        fig, ax = drv.panel()
        drv.cms_label(ax, loc=0)
        ax.axhline(1.0, color="#9c9ca1", lw=1.6)
        for index, (label, (row, edges, shown, ratio, rel)) in enumerate(
                prepared.items()):
            #### a pT-merged candidate has no one-to-one slice: draw the group
            #### this production slice belongs to and say so in the legend.
            group = next((a for a, g in enumerate(row["pt_groups"])
                          if p in g), None)
            if group is None:
                continue
            merged = len(row["pt_groups"][group]) > 1
            m = shown[group]
            if not m.any():
                continue
            y = ratio[group]
            pool.append(y[m])
            name = label.replace("proposal_", "")
            tag = f"{name} ({int(shown.sum())} reported)"
            if merged:
                tag += ", pT-merged"
            if label == PROD_ROW:
                ax.errorbar(drv.centers(edges)[m], y[m],
                            yerr=(rel[group] * y)[m],
                            xerr=np.diff(edges)[m] / 2, fmt="o", ms=8,
                            color="k", lw=2.0, capsize=0, zorder=5,
                            label=f"{tag}  [production]")
            else:
                step(ax, edges, y, m, color=LADDER_COLORS[index], lw=2.4,
                     zorder=2, label=tag)
        name = f"appendix_ladder_{ch}_pt{p + 1}"
        if not check(name, *pool):
            plt.close(fig)
            continue
        ax.set_xlabel(drv.XLAB)
        ax.set_ylabel("Unfolded / gen, per-pT normalized shape")
        ax.set_xlim(-4.0, 0.0)
        lim = np.nanmax(np.abs(np.concatenate(pool) - 1.0))
        ax.set_ylim(1.0 - 1.25 * lim, 1.0 + 1.25 * lim)
        annotate(ax, d["ptlab"][p], loc=0, fontsize=21)
        bottom_legend(fig, ax, ncol=2, fontsize=17, rows=3)
        emit(fig, "ladder", name,
             f"{ch}, {d['ptlab'][p]}: unfolded/gen normalized-shape ratio for "
             f"all five gen-merge ladder candidates; the production "
             f"coarse_tail row is drawn with its statistical errors, the "
             f"others as thin lines on their own gen binning.")


# --------------------------------------------------------------------------
def write_manifest():
    groups: dict[str, list[tuple[str, str]]] = {}
    for family, name, caption in MANIFEST:
        groups.setdefault(family, []).append((name, caption))
    order = ["closure", "systshift", "varclosure", "pulls", "coverage",
             "config", "bayesiter", "refold", "erareco", "erapurity",
             "ladder"]
    heading = {
        "closure": ("1. tau = 0 self-closure", "rerun",
                    "`appendix_rerun_run2.json` -> `closure`; the crosschecks "
                    "driver's nominal MC unfold, reproduced per bin (the "
                    "Phase B JSON stores only the maximum)."),
        "systshift": ("2. Signed systematic shifts", "JSON",
                      "`hadronic_rho_systematics_run2.json` -> "
                      "`signed_relative_shift` and "
                      "`parton_shower.signed_relative_shift`; already per bin."),
        "varclosure": ("3. Shifted input through the nominal response",
                       "rerun",
                       "`appendix_rerun_run2.json` -> `varclosure`; the "
                       "crosschecks `_systematic_closure` logic extended to "
                       "the FSR/ISR slices, kept per bin instead of "
                       "summarized. The JER/JMS/JMR core maxima reproduce the "
                       "Phase B check register exactly (dijet JER 0.36%, JMS "
                       "4.84%, JMR 0.42%). The FSR/ISR curves are NOT a "
                       "response test: those weights move the gen spectrum "
                       "itself by 4-15%, so what is drawn is mostly the "
                       "genuine particle-level shift."),
        "pulls": ("4. Input-stat pull distribution", "rerun",
                  "`appendix_rerun_run2.json` -> `input_toys`; 300 toys on the "
                  "same rng stream as Phase B (seed 20260726 for dijet, "
                  "20260727 for trijet), so the medians reproduce the "
                  "published ones exactly."),
        "coverage": ("5. 68% coverage per bin", "rerun",
                     "`appendix_rerun_run2.json` -> `input_toys.coverage_"
                     "1sigma`; binomial errors from the same 300 toys."),
        "config": ("6. Estimator configurations", "JSON",
                   "`hadronic_rho_config_comparison_run2.json` -> "
                   "`configurations.*.per_bin.signed_shape_shift_vs_"
                   "production`, plus the `_2to1` twin's `per_bin.unfolded` "
                   "rebinned against the production result (both sides "
                   "renormalized over the 2:1 window)."),
        "bayesiter": ("7. Iterative-Bayes iteration scan", "rerun",
                      "`appendix_rerun_run2.json` -> `bayes_iter`; the "
                      "D'Agostini driver rerun to keep the per-bin shape "
                      "vectors (the Bayes JSON stores per-bin arrays only at "
                      "n = 4)."),
        "refold": ("8. Profiled refolding", "rerun + JSON",
                   "Folded ratios: `appendix_rerun_run2.json` -> `refold` "
                   "(the profiled-refold driver rerun, keeping the reco-space "
                   "folded vectors). Nuisance pulls: "
                   "`hadronic_rho_profiled_refold_run2.json`, which already "
                   "stores them for every fit variant."),
        "erareco": ("9. Per-era detector level", "NPZ + JSON",
                    "Ratios recomputed from `npz_era/<channel>_<era>_"
                    "nominal.npz` (fake-corrected data over matched MC, "
                    "normalized inside the core window). chi2A bars from "
                    "`hadronic_rho_era_chi2_run2.json`."),
        "erapurity": ("10. Per-era purity and stability", "NPZ",
                      "Recomputed from the per-era response matrices in "
                      "`npz_era/`, with the definitions of "
                      "`era_check/era_binning_check.json` (verified to "
                      "reproduce its 2016APV/2016/2017 numbers bit for bit; "
                      "that JSON has no 2018 entry, which is why all four "
                      "eras are recomputed the same way)."),
        "ladder": ("11. Gen-merge stability ladder", "JSON",
                   "`<channel>_run2_stability.json`; the per-bin `unfolded`, "
                   "`err`, `gen` and `shown` arrays of all five candidates, "
                   "with the shape ratio built exactly as "
                   "`rho_unfold_stability.analyse` does."),
    }
    lines = [
        "# Appendix figure set -- pair_split Run 2 groomed-rho unfolding tests",
        "",
        "Every unfolding test from the Phase B campaign, drawn per reported "
        "bin, per pT slice, for both channels (dijet 5 slices, trijet 3). "
        "Combined Run 2, 137.6 fb$^{-1}$, tau = 0 TUnfoldDensity on the "
        "production `proposal_coarse_tail` gen merge. Each entry has a `.png` "
        "and a `.pdf` twin.",
        "",
        "The curated production families (`unfolded`, `uncertainty`, "
        "`datamc_reco`, `purity_stability`, `amplification`, "
        "`compare_channels`, `bayes_vs_tunfold`, `response`, `before_after` "
        "and the presentation variants) are NOT duplicated here -- they live "
        "one directory up.",
        "",
        f"**{len(MANIFEST)} figures.**",
        "",
    ]
    for family in order:
        if family not in groups:
            continue
        title, source, note = heading[family]
        lines += [f"## {title}", "",
                  f"*Source: {source}.* {note}", "",
                  "| figure | caption |", "|---|---|"]
        for name, caption in sorted(groups[family]):
            lines.append(f"| `{name}` | {caption} |")
        lines.append("")
    if BLANK:
        lines += ["## Figures that could not be drawn", ""]
        lines += [f"* `{name}`" for name in BLANK]
        lines.append("")
    (OUT / "appendix_manifest.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'appendix_manifest.md'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None,
                        help="regex; write only the matching figures")
    args = parser.parse_args(argv)
    drv.ONLY = args.only
    drv.OUT_DIR = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    rerun = load_rerun()

    for channel in CHANNELS:
        print(f"== {channel}")
        d = drv.load(channel)
        rr = rerun[channel]
        eras = era_inputs(channel)
        fig_closure(d, rr)
        fig_systshift(d)
        fig_varclosure(d, rr, mode="shape")
        fig_varclosure(d, rr, mode="absolute")
        fig_pulls(d, rr)
        fig_coverage(d, rr)
        fig_config(d)
        fig_bayesiter(d, rr)
        fig_refold(d, rr)
        fig_refoldpulls(d)
        fig_erareco(d, eras)
        fig_erachi2(d)
        fig_erapurity(d, eras)
        fig_ladder(d)

    if args.only is None:
        write_manifest()
    print(f"\n{len(MANIFEST)} figures in {OUT}")
    if BLANK:
        print(f"BLANK: {BLANK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
