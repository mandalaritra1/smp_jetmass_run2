#!/usr/bin/env python3
"""Per-pT-bin SINGLE-panel figures for the ARC backup slides (gridded in Typst).

The earlier deck figures were wide 3-panel matplotlib grids (one image holding all
three pT bins) which overflow a slide. This script instead writes ONE square panel
per pT bin, named <plot>_<obs>_pt1/pt2/pt3.png, so the .typ can place them in a
`#grid(columns: (1fr,1fr,1fr) ...)` (the same pattern the rest of the deck uses for
closure_systematics_*_1/2/3.png) and Typst controls the sizing.

CMS plot-style rules (see unfold auto-memory `cms-plot-style-rules`):
  - mplhep CMS style; `hep.cms.label("Preliminary", data=False, ...)` -> "Simulation
    Preliminary", NEVER plt.title; pT range carried in `rlabel`.
  - leave legend headroom: ylim top raised so an in-frame legend clears the data;
    the Herwig panel no longer clips the magenta band at 80% (the old ylim bug).

Three plot types (all groomed rho by default; pass obs list on argv):
  modelunc_persource : per-source fractional envelope (CR/Had/Shower/Total)
  modelunc_statbars  : per-variation shape deviation +- MC stat ("unc of the unc")
  herwig_vs_internal : Pythia<->Herwig band vs the Pythia-internal Total

Run: ~/Projects/GluonJetMass/.venv/bin/python make_deck_figs.py [prefix=hi] [obs...]
Writes into ../review/figs/ with a wp_ prefix.
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

# Global CMS plot style (ROOT-like ticks/fonts) before any figure is made.
try:
    hep.style.use(hep.style.CMS)
except Exception:
    hep.style.use("CMS")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.expanduser("~/Projects/smp25010-docs/review/figs")
# --no-vincia (anywhere on argv): drop the Shower (Vincia) source so the envelope +
# Total exclude the baseline-tune-dependent shower-model swap. Outputs get a
# "novincia_" tag so they never overwrite the with-Vincia figures.
NOVINCIA = "--no-vincia" in sys.argv[1:]
_ARGS = [a for a in sys.argv[1:] if a != "--no-vincia"]
PREFIX = _ARGS[0] if _ARGS else "hi"
OBSES = _ARGS[1:] if len(_ARGS) > 1 else ["rho_g"]
YODA = lambda name: os.path.join(HERE, "out", f"{PREFIX}_{name}.yoda")
# Output filename tag: the Monash high-stats set ("hi") keeps the unprefixed names
# the deck already references; any other tune (cp5, ...) is tagged so it doesn't
# overwrite them -> wp_cp5_modelunc_persource_rho_g_pt1.png, etc.
TAG = "" if PREFIX == "hi" else f"{PREFIX}_"
if NOVINCIA:
    TAG = f"{TAG}novincia_"

NOMINAL = "pythia"
SOURCES = {"CR": ["pythia_cr1", "pythia_cr2"],
           "Hadronization": ["pythia_fragsoft", "pythia_fraghard"],
           "Shower": ["pythia_vincia"]}
SRC_COLOR = {"CR": "#1f77b4", "Hadronization": "#ff7f0e", "Shower": "#2ca02c", "Total": "black"}
VAR_STYLE = {"pythia_cr1": ("CR1 (QCD)", "#1f77b4"),
             "pythia_cr2": ("CR2 (move)", "#17becf"),
             "pythia_fragsoft": ("frag-soft", "#ff7f0e"),
             "pythia_fraghard": ("frag-hard", "#d62728"),
             "pythia_vincia": ("Vincia", "#2ca02c")}
if NOVINCIA:
    SOURCES.pop("Shower", None)
    VAR_STYLE.pop("pythia_vincia", None)
# The unfolded observable is log10(rho^2) with rho = m/(p_T R) (NOT rho itself) —
# matches the analysis axis label (unfold/scripts/run_combine_rho_full.py etc.).
XLAB = {"rho_g": r"$\log_{10}(\rho_g^2)$", "rho_u": r"$\log_{10}(\rho_u^2)$",
        "mass_g": r"$m_g$ [GeV]", "mass_u": r"$m_u$ [GeV]"}
# (yoda key, short rlabel that fits beside "CMS Simulation" on a small panel)
PT = [("pt200_290", "200$-$290 GeV"),
      ("pt290_400", "290$-$400 GeV"),
      ("pt400_Inf", "$p_T$ > 400 GeV")]
# x-axis lower cut per observable: hide the near-empty low-mass tail (whose fractional
# uncertainty is pure noise). The y-axis then auto-scales to the visible range.
XMIN = {"rho_g": -4.5, "rho_u": -2.5}


def parse(path):
    """{leaf_name: (edges, values, stat_err)} for finalized ESTIMATE1D_V3 histos."""
    out = {}
    if not os.path.exists(path):
        return out
    lines = open(path).read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = re.match(r"BEGIN YODA_ESTIMATE1D_V3 (\S+)", lines[i])
        if not m:
            i += 1; continue
        hp = m.group(1); edges = None; vals = []; errs = []; tab = False; i += 1
        while i < n and not lines[i].startswith("END YODA_ESTIMATE1D_V3"):
            ln = lines[i]
            if ln.startswith("Edges(A1):"):
                edges = [float(x) for x in re.findall(r"[-+0-9.eE]+", ln.split(":", 1)[1])]
            elif ln.startswith("# value"):
                tab = True
            elif tab:
                tok = ln.split()
                if tok:
                    if tok[0] == "nan":
                        vals.append(np.nan); errs.append(np.nan)
                    else:
                        vals.append(float(tok[0]))
                        errs.append(abs(float(tok[2])) if len(tok) > 2 and tok[2] != "---" else np.nan)
            i += 1
        if hp.startswith("/CMS_ZJET_JETMASS/") and edges is not None and len(vals) >= 2:
            v = np.array(vals[1:-1]); e = np.array(errs[1:-1]); ed = np.array(edges)
            if len(v) == len(ed) - 1:
                out[hp.split("/")[-1]] = (ed, np.nan_to_num(v), np.nan_to_num(e))
        i += 1
    return out


def norm_shape(ed, v, e):
    w = np.diff(ed); S = np.sum(v * w)
    if S <= 0:
        return v * 0.0, np.zeros_like(v)
    rel = np.where(v > 0, e / v, 0.0)
    return v / S, rel  # normalized shape, per-bin relative stat error


def new_panel(ptrange):
    # NO custom figsize: the CMS style sets absolute font sizes tuned for its own
    # (square) default figure, so a small figsize makes the labels look humongous.
    # Let the style drive the size; the panels are scaled to fit by Typst's grid.
    # layout="constrained" (NOT a later tight_layout): mplhep freezes the
    # CMS<->Simulation gap as an axes-fraction at label time, and tight_layout's
    # post-hoc resize stretches it (~8% -> ~19%). Constrained sizes the axes first.
    fig, ax = plt.subplots(layout="constrained")
    # exp label "CMS Simulation" (left) + pT range (right), both above the frame so
    # they never collide with the in-frame legend (cms-plot-style-rules: no plt.title).
    hep.cms.label("", data=False, loc=0, ax=ax, rlabel=ptrange)
    ax.grid(alpha=0.25)
    ax.margins(x=0.02)
    return fig, ax


def save(fig, ax, name):
    ax.legend(loc="upper left", framealpha=0.92)
    # no tight_layout: the figure uses constrained layout (see new_panel) so the
    # CMS<->Simulation gap is not stretched by a post-hoc axes resize.
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


def headroom_top(ax, ymax, frac=1.30):
    lo, _ = ax.get_ylim()
    ax.set_ylim(lo, max(ymax * frac, 1e-9))


CAP = 0.6  # y-axis is capped here only if data exceeds it; else it auto-scales.


def _annotate_over(ax, x, y, color, cap, sign=1):
    """Write the true value at the clamp line for a bin that runs off the axis."""
    ax.annotate(f"{y:.2f}", (x, sign * cap), ha="center",
                va="top" if sign > 0 else "bottom", fontsize=10, fontweight="bold",
                color=color, xytext=(0, -2 if sign > 0 else 2),
                textcoords="offset points", annotation_clip=False)


def cap_and_annotate(ax, ctr, series, mask=None, cap=CAP, frac=1.5):
    """0-based axis. If the peak (within `mask`) fits under `cap`, auto-scale with
    headroom; else clamp the top to `cap` and print the real value over every
    overflowing bin (no masking). `series` = [(yvals, colour), ...]; `mask` restricts
    the scale + annotations to the visible x-window."""
    ax.set_ylim(bottom=0)
    stack = np.vstack([y for y, _ in series])
    if mask is None:
        mask = np.ones(stack.shape[1], bool)
    vis = stack[:, mask]
    vmax = float(np.nanmax(vis)) if vis.size else 0.0
    if vmax <= cap:
        headroom_top(ax, vmax, frac)
        return
    ax.set_ylim(0, cap)
    top = np.nanmax(stack, axis=0)
    for j, x in enumerate(ctr):
        if mask[j] and np.isfinite(top[j]) and top[j] > cap:
            col = next((c for y, c in series if np.isfinite(y[j]) and y[j] >= top[j] - 1e-12), "black")
            _annotate_over(ax, x, top[j], col, cap, sign=1)


def cap_symmetric(ax, ctr, dev_list, mask=None, cap=CAP):
    """Deviation axis (±). Auto-scale (extra top room for the legend) if the peak |dev|
    within `mask` fits under `cap`; else clamp to ±`cap` and annotate overflow points."""
    if mask is None:
        mask = np.ones(len(ctr), bool)
    peak = max((float(np.nanmax(np.abs(d[mask]))) for d, _, _ in dev_list), default=0.0)
    if peak <= cap:
        amax = max((float(np.nanmax((np.abs(d) + s)[mask])) for d, s, _ in dev_list), default=cap)
        ax.set_ylim(-amax * 1.2, amax * 1.6)
        return
    ax.set_ylim(-cap, cap)
    for dev, _s, col in dev_list:
        for j, (x, y) in enumerate(zip(ctr, dev)):
            if mask[j] and np.isfinite(y) and abs(y) > cap:
                _annotate_over(ax, x, y, col, cap, sign=1 if y > 0 else -1)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    nom = parse(YODA(NOMINAL))
    members = {m: parse(YODA(m)) for ms in SOURCES.values() for m in ms}
    hw = parse(YODA("herwig"))

    for obs in OBSES:
        xl = XLAB.get(obs, obs)
        for pidx, (ptb, ptrange) in enumerate(PT, start=1):
            key = f"{obs}_{ptb}"
            if key not in nom:
                continue
            ed, vn, en = nom[key]
            nn, reln = norm_shape(ed, vn, en)
            base = np.where(nn > 0, nn, np.nan)
            ctr = 0.5 * (ed[:-1] + ed[1:])
            # No masking: every visible bin is plotted. We crop the x-axis to the
            # physical range (XMIN) so the near-empty low-mass tail is out of frame, and
            # cap the y-axis at CAP (printing the value over any in-window overflow).
            xmin = XMIN.get(obs)
            inwin = np.ones(len(ctr), bool) if xmin is None else (ctr >= xmin)

            # ---- per-source fractional envelope (CR/Had/Shower/Total) ----------
            fig, ax = new_panel(ptrange)
            totsq = np.zeros_like(nn)
            for src, mem in SOURCES.items():
                env = np.zeros_like(nn)
                for mm in mem:
                    if key not in members.get(mm, {}):
                        continue
                    nm, _ = norm_shape(*members[mm][key])
                    env = np.fmax(env, np.nan_to_num(np.abs(nm - nn) / base))
                ax.step(ctr, env, where="mid", color=SRC_COLOR[src], lw=1.9, label=src)
                totsq += env ** 2
            internal_total = np.sqrt(totsq)
            ax.step(ctr, internal_total, where="mid", color="black", lw=2.4, label="Total")
            ax.set_xlabel(xl); ax.set_ylabel("Modelling uncertainty")
            cap_and_annotate(ax, ctr, [(internal_total, "black")], mask=inwin, frac=1.5)
            ax.set_xlim(left=xmin)
            save(fig, ax, f"wp_{TAG}modelunc_persource_{obs}_pt{pidx}.png")

            # ---- per-variation shape deviation +- MC stat ("unc of the unc") ---
            fig, ax = new_panel(ptrange)
            dev_list = []
            for mm, (lbl, col) in VAR_STYLE.items():
                if key not in members.get(mm, {}):
                    continue
                nm, relm = norm_shape(*members[mm][key])
                dev = np.nan_to_num((nm - nn) / base)
                ratio = np.where(nn > 0, nm / nn, 0.0)
                sdev = np.nan_to_num(ratio * np.sqrt(relm ** 2 + reln ** 2))
                ax.errorbar(ctr, dev, yerr=sdev, color=col, label=lbl, lw=1.3,
                            marker="o", ms=3, capsize=1.6, elinewidth=0.8)
                dev_list.append((dev, sdev, col))
            ax.axhline(0, color="grey", lw=0.9, zorder=1)
            ax.set_xlabel(xl); ax.set_ylabel("Shape deviation")
            cap_symmetric(ax, ctr, dev_list, mask=inwin)
            ax.set_xlim(left=xmin)
            save(fig, ax, f"wp_{TAG}modelunc_statbars_{obs}_pt{pidx}.png")

            # ---- Pythia<->Herwig band vs internal Total -----------------------
            if key in hw:
                fig, ax = new_panel(ptrange)
                nh, relh = norm_shape(*hw[key])
                dev = np.nan_to_num(np.abs(nh - nn) / base)
                ratio = np.where(nn > 0, nh / nn, 0.0)
                sdev = np.nan_to_num(ratio * np.sqrt(relh ** 2 + reln ** 2))
                ax.step(ctr, dev, where="mid", color="#d62728", lw=2.2, label=r"Pythia$\leftrightarrow$Herwig")
                ax.errorbar(ctr, dev, yerr=sdev, fmt="none", ecolor="#d62728",
                            elinewidth=0.8, capsize=1.6, alpha=0.75)
                ax.step(ctr, internal_total, where="mid", color="black", lw=2.2, label="Pythia-internal Total")
                ax.set_xlabel(xl); ax.set_ylabel("Fractional difference")
                cap_and_annotate(ax, ctr, [(dev, "#d62728"), (internal_total, "black")], mask=inwin, frac=1.4)
                ax.set_xlim(left=xmin)
                save(fig, ax, f"wp_{TAG}herwig_vs_internal_{obs}_pt{pidx}.png")


if __name__ == "__main__":
    main()
