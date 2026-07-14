#!/usr/bin/env python3
"""Herwig cross-check on the dimensionless jet mass rho = 2*log10(m/(pt*R)), R=0.8.

Overlays three *gen/particle-level* predictions, normalized to unit area (shape):
  1. Pythia        -- our nominal, coffea pythia_all.pkl  (ptjet_rhojet_{g,u}_gen)
  2. Herwig (cof)  -- coffea herwig_all.pkl               (same hist, sum over eras)
  3. Herwig (Rivet)-- rivet/out/<prefix>_herwig.yoda      (rho_{g,u}_pt* leaves)

All three share the analysis rho binning [-10,-6,-5,-4.5,...,0], so they overlay
bin-for-bin; we drop the wide [-10,-6] catch-all bin from the view. Densities (per
unit rho) are plotted so unequal bin widths compare fairly, with a ratio-to-Pythia
sub-panel. One figure per grooming (3 pT columns x main+ratio).

  ~/Projects/GluonJetMass/.venv/bin/python tools/arc_diagnostics/herwig_compare_rho.py
"""
from __future__ import annotations
import os, re, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep

hep.style.use(hep.style.CMS)

ORIG = "/Users/aritra/Projects/unfold/inputs/zjet/rho/original"
HERE = os.path.dirname(os.path.abspath(__file__))
RIVET_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "rivet", "out"))
OUTDIR = os.path.join(HERE, "figs")
PREFIX = "hi"                       # high-stats rivet set
XMIN = -6.0                        # drop the wide [-10,-6] catch-all bin from view

# coffea ptgen axis: bin 0 = 0-200 (below measurement); we use 1,2,3.
PT = [(1, "200$-$290 GeV"), (2, "290$-$400 GeV"), (3, "$p_T$ > 400 GeV")]
PT_YODA = {1: "pt200_290", 2: "pt290_400", 3: "pt400_Inf"}
XLAB = {"g": r"$\log_{10}(\rho_g^2)$", "u": r"$\log_{10}(\rho_u^2)$"}
TITLE = {"g": "groomed (soft-drop)", "u": "ungroomed"}

C_PYTHIA = "black"; C_HWCOF = "#d62728"; C_HWRIV = "#1f77b4"


def density(edges, content, var=None):
    """content per bin -> (density, density_err). Area-normalized (integrates to 1);
    stat error propagated as sqrt(var)/area (normalization treated as fixed)."""
    edges = np.asarray(edges, float); content = np.asarray(content, float)
    w = np.diff(edges); area = np.sum(content * w)
    if area <= 0:
        return np.zeros_like(content), np.zeros_like(content)
    d = content / area
    if var is None:
        return d, np.zeros_like(d)
    return d, np.sqrt(np.asarray(var, float)) / area


def parse_yoda(path):
    """{leaf: (edges, content, err)} for the finalized rho/mass histos (stats errUp)."""
    out = {}
    if not os.path.exists(path):
        return out
    lines = open(path).read().splitlines(); i = 0; n = len(lines)
    while i < n:
        m = re.match(r"BEGIN YODA_\S+ (\S+)", lines[i])
        if not m:
            i += 1; continue
        hp = m.group(1); edges = None; vals = []; errs = []; tab = False; i += 1
        while i < n and not lines[i].startswith("END YODA"):
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
        if edges is not None and len(vals) >= 2:
            v = np.nan_to_num(np.array(vals[1:-1]))   # strip under/overflow
            e = np.nan_to_num(np.array(errs[1:-1]))
            if len(v) == len(edges) - 1:
                out[hp.split("/")[-1]] = (np.array(edges), v, e)
        i += 1
    return out


def coffea_gen(pkl, obs):
    """(edges, {ptbin: (content, variance)}) from a coffea *_gen hist, nominal,
    summed over eras. Variances add under the era sum."""
    d = pickle.load(open(os.path.join(ORIG, pkl), "rb"))
    h = d[f"ptjet_rhojet_{obs}_gen"]
    names = [getattr(ax, "name", "") for ax in h.axes]
    edges = h.axes[names.index("mpt_gen")].edges
    vals = h.values()
    var = h.variances()
    if var is None:                                # unweighted fallback: var ~ content
        var = vals
    out = {}
    for ptb, _ in PT:
        if "dataset" in names:                     # herwig_all: (dataset,ptgen,mpt,sys)
            content = vals[:, ptb, :, 0].sum(axis=0)
            variance = var[:, ptb, :, 0].sum(axis=0)
        else:                                      # pythia_all: (ptgen,mpt,sys)
            content = vals[ptb, :, 0]
            variance = var[ptb, :, 0]
        out[ptb] = (content, variance)
    return np.array(edges), out


def trim(edges, content, xmin):
    keep = edges[:-1] >= xmin - 1e-9
    return edges, content, keep


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    riv = parse_yoda(os.path.join(RIVET_OUT, f"{PREFIX}_herwig.yoda"))

    for obs in ("g", "u"):
        ed_p, py = coffea_gen("pythia_all.pkl", obs)
        ed_h, hw = coffea_gen("herwig_all.pkl", obs)
        assert np.allclose(ed_p, ed_h), "pythia/herwig gen edges differ"

        fig = plt.figure(figsize=(31, 11), layout="constrained")
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1], figure=fig)

        for col, (ptb, ptrange) in enumerate(PT):
            ax = fig.add_subplot(gs[0, col]); rax = fig.add_subplot(gs[1, col], sharex=ax)
            ctr = 0.5 * (ed_p[:-1] + ed_p[1:])
            keep = ed_p[:-1] >= XMIN - 1e-9

            dp, ep = density(ed_p, *py[ptb])
            dh, eh = density(ed_h, *hw[ptb])
            rk = riv.get(f"rho_{obs}_{PT_YODA[ptb]}")
            dr, er = density(rk[0], rk[1], rk[2] ** 2) if rk is not None else (None, None)

            def band(a, x, d, e, color, **kw):
                a.step(x[keep], d[keep], where="mid", color=color, **kw)
                a.fill_between(x[keep], (d - e)[keep], (d + e)[keep], step="mid",
                               color=color, alpha=0.22, lw=0)

            band(ax, ctr, dp, ep, C_PYTHIA, lw=2.6, label="Pythia (nominal)")
            band(ax, ctr, dh, eh, C_HWCOF, lw=2.2, label="Herwig (coffea)")
            if dr is not None:
                band(ax, ctr, dr, er, C_HWRIV, lw=2.2, ls="--", label="Herwig (Rivet)")

            hep.cms.label("", data=False, loc=0, ax=ax, rlabel=ptrange)
            ax.set_ylabel("normalized density")
            ax.grid(alpha=0.25); ax.margins(x=0.02)
            ymax = np.nanmax([(dp + ep)[keep].max(), (dh + eh)[keep].max(),
                              (dr + er)[keep].max() if dr is not None else 0])
            ax.set_ylim(0, ymax * 1.45)
            ax.legend(loc="upper left", framealpha=0.92,
                      title=f"$\\rho_{obs}$ {TITLE[obs]}  (band = MC stat)")
            plt.setp(ax.get_xticklabels(), visible=False)

            # ratio to Pythia, with stat bands (numerator+denominator added in quad)
            def rel(d, e):
                return np.where(d > 0, e / d, 0.0)
            relp = rel(dp, ep)
            with np.errstate(divide="ignore", invalid="ignore"):
                rh = np.where(dp > 0, dh / dp, np.nan)
                rr = np.where(dp > 0, dr / dp, np.nan) if dr is not None else None
            # grey band around 1 = Pythia's own stat
            rax.fill_between(ctr[keep], (1 - relp)[keep], (1 + relp)[keep], step="mid",
                             color=C_PYTHIA, alpha=0.15, lw=0)
            rax.axhline(1.0, color=C_PYTHIA, lw=1.4)
            erh = rh * np.sqrt(rel(dh, eh) ** 2 + relp ** 2)
            band(rax, ctr, rh, erh, C_HWCOF, lw=2.0)
            if rr is not None:
                err = rr * np.sqrt(rel(dr, er) ** 2 + relp ** 2)
                band(rax, ctr, rr, err, C_HWRIV, lw=2.0, ls="--")
            rax.set_ylim(0.5, 1.5)
            rax.set_ylabel("Herwig / Pythia"); rax.set_xlabel(XLAB[obs])
            rax.grid(alpha=0.25); rax.set_xlim(XMIN, 0.0)

            # quantify groomed precision: median per-bin rel stat error in-window
            if obs == "g":
                w = keep & (dr if dr is not None else dh > 0).astype(bool)
                print(f"  [rho_g pt{ptb}] median rel-stat  "
                      f"Pythia={np.median(relp[keep]):.3f}  "
                      f"Herwig-cof={np.median(rel(dh, eh)[keep]):.3f}  "
                      + (f"Herwig-Riv={np.median(rel(dr, er)[keep]):.3f}" if dr is not None else ""))

        out = os.path.join(OUTDIR, f"herwig_cmp_rho_{obs}.png")
        fig.savefig(out, dpi=110); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
