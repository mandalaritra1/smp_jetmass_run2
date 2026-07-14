#!/usr/bin/env python3
"""Herwig7 CH3 (LPC condor, pT-sliced) vs the unfolded data and the analysis Pythia,
for the measured groomed/ungroomed rho, per pT bin, with a ratio-to-data sub-panel.

Data + true_pythia come from refdata/hepdata_export_{groomed,ungroomed}.npz (already on
the published rho binning); Herwig CH3 comes from out/herwig_ch3.yoda and is rebinned
onto the coarser data edges. All curves area-normalised to unit integral (shape compare).

Run: ~/Projects/GluonJetMass/.venv/bin/python plot_ch3_vs_data.py
Outputs out/ch3_vs_data/ch3_vs_data_{rho_g,rho_u}.png
"""
import os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mplhep as hep

hep.style.use(hep.style.CMS)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "out", "ch3_vs_data")
PT_KEYS = ["pt200_290", "pt290_400", "pt400_Inf"]
PT_LABELS = ["200–290 GeV", "290–400 GeV", "400–∞ GeV"]
# (observable key in yoda, npz file tag, x-axis label)
OBS = [("rho_g", "groomed",   r"$\rho_g = 2\log_{10}(m_g/(p_T R))$"),
       ("rho_u", "ungroomed", r"$\rho_u = 2\log_{10}(m_u/(p_T R))$")]


def parse_yoda(path):
    out = {}
    lines = open(path).read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = re.match(r"BEGIN YODA_ESTIMATE1D_V3 (\S+)", lines[i])
        if not m:
            i += 1; continue
        hp = m.group(1); edges = None; vals = []; tab = False; i += 1
        while i < n and not lines[i].startswith("END YODA_ESTIMATE1D_V3"):
            ln = lines[i]
            if ln.startswith("Edges(A1):"):
                edges = [float(x) for x in re.findall(r"[-+0-9.eE]+", ln.split(":", 1)[1])]
            elif ln.startswith("# value"):
                tab = True
            elif tab:
                tok = ln.split()
                if tok:
                    vals.append(np.nan if tok[0] == "nan" else float(tok[0]))
            i += 1
        if hp.startswith("/CMS_ZJET_JETMASS/") and edges is not None and len(vals) >= 2:
            v = np.nan_to_num(np.array(vals[1:-1])); ed = np.array(edges)
            if len(v) == len(ed) - 1:
                out[hp.split("/")[-1]] = (ed, v)
        i += 1
    return out


def rebin_to(mc_edges, mc_val, target_edges):
    """Sum MC content (val*width) into the coarser target bins (target edges are a
    subset of mc edges). Returns content per target bin."""
    mc_content = mc_val * np.diff(mc_edges)
    out = np.zeros(len(target_edges) - 1)
    for k in range(len(target_edges) - 1):
        lo, hi = target_edges[k], target_edges[k + 1]
        mask = (mc_edges[:-1] >= lo - 1e-6) & (mc_edges[1:] <= hi + 1e-6)
        out[k] = mc_content[mask].sum()
    return out


def norm_density(edges, content):
    """content per bin -> unit-area density (content/sum, then /width)."""
    S = content.sum()
    if S <= 0:
        return content * 0.0
    return (content / S) / np.diff(edges)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ch3 = parse_yoda(os.path.join(HERE, "out", "herwig_ch3.yoda"))

    for obs, tag, xlabel in OBS:
        npz = np.load(os.path.join(HERE, "refdata", f"hepdata_export_{tag}.npz"))
        fig = plt.figure(figsize=(31, 13))
        gs = GridSpec(2, 3, height_ratios=[3, 1], hspace=0.05, wspace=0.16,
                      figure=fig)
        for c, (ptk, ptlab) in enumerate(zip(PT_KEYS, PT_LABELS)):
            edges = npz[f"pt{c}__edges"]
            w = np.diff(edges); ctr = 0.5 * (edges[:-1] + edges[1:])
            # data: value already ~unit-area density; renormalise for safety
            dval = npz[f"pt{c}__value"]; S = (dval * w).sum()
            d = dval / S
            d_stat = npz[f"pt{c}__stat"] / S
            d_tot = 0.5 * (npz[f"pt{c}__total_up"] + npz[f"pt{c}__total_down"]) / S
            # analysis Pythia (truth), same binning
            pyth = npz[f"pt{c}__true_pythia"]; pyth = pyth / (pyth * w).sum()
            # Herwig CH3: rebin onto data edges, then unit-area density
            mc_ed, mc_v = ch3[f"{obs}_{ptk}"]
            h = norm_density(edges, rebin_to(mc_ed, mc_v, edges))

            ax = fig.add_subplot(gs[0, c]); axr = fig.add_subplot(gs[1, c], sharex=ax)
            # data: total-unc band + stat error bars
            ax.fill_between(ctr, d - d_tot, d + d_tot, step="mid", color="0.6",
                            alpha=0.45, lw=0, label="Data total unc.", zorder=1)
            ax.errorbar(ctr, d, yerr=d_stat, fmt="o", color="black", ms=7, lw=1.6,
                        capsize=3, label="Data (unfolded)", zorder=5)
            ax.step(ctr, pyth, where="mid", color="#1f77b4", lw=2.6,
                    label="Pythia8 CP5", zorder=3)
            ax.step(ctr, h, where="mid", color="#e6194B", lw=2.8, ls="-",
                    label="Herwig7 CH3", zorder=4)
            hep.cms.label("Preliminary", data=True, loc=0, ax=ax,
                          rlabel=f"{ptlab}  (13 TeV)")
            ax.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}\rho$" if c == 0 else "")
            ax.set_ylim(bottom=0)
            ymax = np.nanmax([np.nanmax(d + d_tot), np.nanmax(pyth), np.nanmax(h)])
            ax.set_ylim(0, ymax * 1.45)
            ax.tick_params(labelbottom=False); ax.grid(alpha=0.25)
            if c == 0:
                ax.legend(fontsize=17, loc="upper left", framealpha=0.9)
            # ratio to data
            with np.errstate(divide="ignore", invalid="ignore"):
                rp = np.where(d > 0, pyth / d, np.nan)
                rh = np.where(d > 0, h / d, np.nan)
                rtot = np.where(d > 0, d_tot / d, np.nan)
            axr.fill_between(ctr, 1 - rtot, 1 + rtot, step="mid", color="0.6",
                             alpha=0.45, lw=0)
            axr.axhline(1, color="black", lw=1.2)
            axr.step(ctr, rp, where="mid", color="#1f77b4", lw=2.4)
            axr.step(ctr, rh, where="mid", color="#e6194B", lw=2.6)
            axr.set_ylim(0.5, 1.5); axr.grid(alpha=0.25)
            axr.set_ylabel("MC / data" if c == 0 else "", fontsize=18)
            axr.set_xlabel(xlabel)
        fig.savefig(os.path.join(OUTDIR, f"ch3_vs_data_{obs}.png"), dpi=90,
                    bbox_inches="tight")
        plt.close(fig)
        print("wrote", os.path.join(OUTDIR, f"ch3_vs_data_{obs}.png"))


if __name__ == "__main__":
    main()
