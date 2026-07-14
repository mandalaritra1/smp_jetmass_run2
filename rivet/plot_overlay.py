#!/usr/bin/env python3
"""Generator overlay: area-normalised nominal vs each modelling variation (and Herwig
if present), per pT bin, with a ratio-to-nominal sub-panel.

Line styling is grouped by SOURCE so lines are readable: solid=nominal/shower,
dashed=colour-reconnection pair, dotted=hadronisation pair, dash-dot=alt generator,
each with a distinct colour + marker (the default matplotlib palette puts the CR pair
and the frag pair on near-identical hues, which blur together).

Run: ~/Projects/GluonJetMass/.venv/bin/python plot_overlay.py [hi|cp5|mglo]
Outputs out/overlay_<PREFIX>/overlay_<obs>.png for rho_g, mass_g, rho_u, mass_u.
"""
import os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "hi"
OUTDIR = os.path.join(HERE, "out", f"overlay_{PREFIX}")
YODA = lambda name: os.path.join(HERE, "out", f"{PREFIX}_{name}.yoda")

NOMINAL = "pythia"
# name -> (label, colour, linestyle, marker)  -- grouped by source via linestyle
VARS = [("pythia_vincia",   "Vincia (shower)",  "#2ca02c", "-",  "o"),   # green  solid
        ("pythia_cr1",      "CR1 (QCD)",        "#1f77b4", "--", "s"),   # blue   dashed
        ("pythia_cr2",      "CR2 (gluon-move)", "#7b3fbf", "--", "^"),   # purple dashed
        ("pythia_fragsoft", "frag-soft",        "#ff7f0e", ":",  "v"),   # orange dotted
        ("pythia_fraghard", "frag-hard",        "#d62728", ":",  "D")]   # red    dotted
HERWIG = ("Herwig (cluster)", "#e6194B", "-.", "*")                       # crimson dash-dot
FAMILIES = {"rho_g": r"$\rho_g = 2\log_{10}(m_g/p_T R)$", "mass_g": r"$m_g$ [GeV]",
            "rho_u": r"$\rho_u$", "mass_u": r"$m_u$ [GeV]"}
PT_BINS = ["pt200_290", "pt290_400", "pt400_Inf"]
PLOT_FAMS = ["rho_g", "mass_g", "rho_u", "mass_u"]


def parse(path):
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
                        errs.append(abs(float(tok[2])) if len(tok) > 2 and tok[2] != "---" else 0.0)
            i += 1
        if hp.startswith("/CMS_ZJET_JETMASS/") and edges is not None and len(vals) >= 2:
            v = np.nan_to_num(np.array(vals[1:-1])); e = np.nan_to_num(np.array(errs[1:-1]))
            ed = np.array(edges)
            if len(v) == len(ed) - 1:
                out[hp.split("/")[-1]] = (ed, v, e)
        i += 1
    return out


def norm(ed, v, e):
    w = np.diff(ed); S = np.sum(v * w)
    if S <= 0:
        return v * 0.0, e * 0.0
    return v / S, e / S


def draw(ax, axr, x, y, ynom, color, ls, marker, lw, label, z=3, ms=4):
    ax.plot(x, y, drawstyle="steps-mid", color=color, ls=ls, lw=lw,
            marker=marker, ms=ms, markevery=1, label=label, zorder=z, alpha=0.95)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ynom > 0, y / ynom, np.nan)
    axr.plot(x, r, drawstyle="steps-mid", color=color, ls=ls, lw=lw,
             marker=marker, ms=ms, markevery=1, zorder=z, alpha=0.95)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    nom = parse(YODA(NOMINAL))
    data = {v[0]: parse(YODA(v[0])) for v in VARS}
    herwig = parse(YODA("herwig"))

    for fam in PLOT_FAMS:
        fig = plt.figure(figsize=(15, 4.8))
        gs = GridSpec(2, len(PT_BINS), height_ratios=[3, 1.3], hspace=0.06, wspace=0.2)
        drew = False
        for c, ptb in enumerate(PT_BINS):
            key = f"{fam}_{ptb}"
            if key not in nom:
                continue
            ed, vn, en = nom[key]
            nn, enn = norm(ed, vn, en)
            ctr = 0.5 * (ed[:-1] + ed[1:])
            reln = np.where(nn > 0, enn / nn, 0.0)

            ax = fig.add_subplot(gs[0, c]); axr = fig.add_subplot(gs[1, c], sharex=ax)
            # nominal: thick black, stat band
            ax.plot(ctr, nn, drawstyle="steps-mid", color="black", lw=2.6, label="Pythia (nominal)", zorder=6)
            ax.fill_between(ctr, nn * (1 - reln), nn * (1 + reln), step="mid", color="black", alpha=0.12, lw=0)
            for vname, lbl, col, ls, mk in VARS:
                if key in data[vname]:
                    nv, _ = norm(*data[vname][key])
                    draw(ax, axr, ctr, nv, nn, col, ls, mk, 1.6, lbl)
            if key in herwig:
                nh, _ = norm(*herwig[key])
                draw(ax, axr, ctr, nh, nn, HERWIG[1], HERWIG[2], HERWIG[3], 2.3, HERWIG[0], z=7, ms=6)
            # ratio reference + nominal band
            axr.fill_between(ctr, 1 - reln, 1 + reln, step="mid", color="black", alpha=0.12, lw=0)
            axr.axhline(1, color="black", lw=1.0, zorder=1)

            ax.set_title(ptb.replace("pt", "pT ").replace("_", "–") + " GeV", fontsize=10)
            ax.set_ylabel("(1/N) dN/dx" if c == 0 else ""); ax.set_ylim(bottom=0)
            ax.tick_params(labelbottom=False); ax.grid(alpha=0.25)
            axr.set_ylim(0.6, 1.4); axr.grid(alpha=0.25)
            axr.set_ylabel("ratio to nom." if c == 0 else "", fontsize=9)
            axr.set_xlabel(FAMILIES[fam], fontsize=11)
            if c == 0:
                ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
            drew = True
        if not drew:
            plt.close(fig); continue
        fig.suptitle(f"Generator overlay — {fam}  (mglo+Pythia8 {PREFIX}, area-normalised; band = nominal MC stat)",
                     fontsize=11)
        fig.savefig(os.path.join(OUTDIR, f"overlay_{fam}.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("wrote", os.path.join(OUTDIR, f"overlay_{fam}.png"))


if __name__ == "__main__":
    main()
