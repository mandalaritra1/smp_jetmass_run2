"""Plot the hadronic model-variation groomed-rho shapes: one CMS panel per
pT bin, cp5 reference plus the CR/fragmentation variations, with a ratio
sub-panel carrying the statistical uncertainties.

    python3 plot_shapes.py shapes_dijet.npz out/ --channel dijet

The point of the ratio panel is honesty about statistics: the grey band is
the cp5 statistical uncertainty, so the reader can see directly whether the
variations are resolved or are sitting inside the noise.
"""
import datetime
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

PUB_LABEL_FS, PUB_TICK_FS, PUB_LEGEND_FS, PUB_ANNOT_FS = 30, 26, 20, 22
# Petroff 6-colour (CVD-safe); cp5 reference is black like data.
COLORS = {"vincia": "#009e9e", "cr1": "#5790fc", "cr2": "#f89c20",
          "fraghard": "#e42536", "fragsoft": "#964a8b"}
#### vincia is the SHOWER leg, not hadronisation. It gets a teal hue, a
#### dashed line AND a diamond marker: in the ratio panel every series is
#### drawn as plain dots, so linestyle alone does not separate it, and a
#### violet would twin with fragsoft's purple.
STYLE = {"vincia": (0, (5, 2))}
MARKER = {"vincia": "D"}
NICE = {"cp5": "CP5 (reference)", "vincia": "Shower: Vincia", "cr1": "CR: QCD-inspired", "cr2": "CR: gluon-move",
        "fraghard": "Hadronisation (hard)", "fragsoft": "Hadronisation (soft)"}

npz = np.load(sys.argv[1])
outdir = sys.argv[2] if len(sys.argv) > 2 else "."
channel = "dijet"
for i, a in enumerate(sys.argv):
    if a == "--channel":
        channel = sys.argv[i + 1]
os.makedirs(outdir, exist_ok=True)

rho = npz["rho_edges"]
configs = [str(c) for c in npz["configs"]]
PT = [(200, 290), (290, 400), (400, None)]

# drop the unreported low-rho sink bins for display (< -4): they are catch-alls
first = int(np.searchsorted(rho, -4.0))
centers = 0.5 * (rho[:-1] + rho[1:])

ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                     capture_output=True, text=True,
                     cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()

for ipt, (lo, hi) in enumerate(PT):
    key = lambda c, s: "%s_pt%d%s" % (c, lo, s)
    ref = npz[key("cp5", "_h")].astype(float)
    refe = npz[key("cp5", "_e")].astype(float)
    tot = ref.sum()
    refn, refne = ref / tot, refe / tot

    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.06},
        layout="constrained")
    fig.set_size_inches(10, 11)

    sl = slice(first, len(centers))
    w = np.diff(rho)

    # --- main panel: normalized shapes (density, so variable bins are fair)
    ax.step(rho[first:], np.append(refn[sl] / w[sl], (refn / w)[-1]),
            where="post", color="k", lw=2.6, label=NICE["cp5"])
    ax.errorbar(centers[sl], (refn / w)[sl], yerr=(refne / w)[sl],
                fmt="none", ecolor="k", elinewidth=1.6, capsize=0)
    for c in configs:
        if c == "cp5":
            continue
        h = npz[key(c, "_h")].astype(float)
        n = h / h.sum()
        ax.step(rho[first:], np.append(n[sl] / w[sl], (n / w)[-1]),
                where="post", color=COLORS[c], lw=2.2, label=NICE[c], alpha=0.9,
                linestyle=STYLE.get(c, "-"))

    ax.set_ylabel(r"$(1/N)\,dN/d\log_{10}(\rho^2_g)$")
    ax.set_ylim(bottom=0)
    ymax = max((npz[key(c, "_h")].astype(float) / npz[key(c, "_h")].sum() / w)[sl].max()
               for c in configs)
    #### Headroom: the groomed-rho distribution RISES toward the right, i.e.
    #### straight into the upper-right legend, and the CMS block owns the
    #### upper left. 1.55x was not enough -- the legend sat on the curves.
    ax.set_ylim(0, ymax * 2.05)
    # data=False already prepends "Simulation" -- passing it again double-prints
    hep.cms.label("Preliminary", data=False, loc=2, ax=ax, rlabel=r"13 TeV")
    ax.grid(alpha=0.25)
    anchor = [t for t in ax.texts if t.get_text() == "CMS"]
    ax0 = anchor[0].get_position()[0] if anchor else 0.03
    ptxt = r"$p_{T} > %d$ GeV" % lo if hi is None else r"$%d < p_{T} < %d$ GeV" % (lo, hi)
    ax.text(ax0, 0.775, "%s jets\n%s" % (channel.capitalize(), ptxt),
            transform=ax.transAxes, ha="left", va="top", fontsize=PUB_ANNOT_FS)
    # opaque + above the data: a tall error bar in the peak bin otherwise runs
    # straight through the legend text (seen in the trijet panels)
    leg = ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS - 2, frameon=True,
                    facecolor="white", edgecolor="none", framealpha=1.0,
                    borderpad=0.4, labelspacing=0.35, handlelength=1.6,
                    borderaxespad=0.6)
    leg.set_zorder(10)

    # --- ratio panel: variation / cp5 with the cp5 stat band
    rax.axhline(1.0, color="k", lw=1.4)
    rax.fill_between(rho[first:],
                     np.append(1 - refne[sl] / refn[sl], 1),
                     np.append(1 + refne[sl] / refn[sl], 1),
                     step="post", color="0.6", alpha=0.45, lw=0,
                     label="CP5 stat. unc.")
    # only bins with enough raw entries to mean anything (same N>=50 cut the
    # numeric summary uses); the sub-50 bins have runaway errors that would
    # stretch the axis by several units and hide everything else
    nraw = npz[key("cp5", "_n")][sl]
    keep = nraw >= 50
    lo_seen, hi_seen, devs, errs = [], [], [], []
    for c in configs:
        if c == "cp5":
            continue
        h = npz[key(c, "_h")].astype(float)
        e = npz[key(c, "_e")].astype(float)
        n, ne = h / h.sum(), e / h.sum()
        r = n[sl] / refn[sl]
        re = r * np.hypot(ne[sl] / n[sl], refne[sl] / refn[sl])
        rax.errorbar(centers[sl][keep], r[keep], yerr=re[keep], ms=6,
                     fmt=MARKER.get(c, "o"),
                     color=COLORS[c], elinewidth=1.6, capsize=0)
        r, re = r[keep], re[keep]
        good = np.isfinite(r) & np.isfinite(re)
        lo_seen.append((r - re)[good].min()); hi_seen.append((r + re)[good].max())
        devs.append(np.abs(r - 1)[good]); errs.append(re[good])
    # never clip the error bars -- the statistical size IS the message here
    rlo, rhi = min(lo_seen), max(hi_seen)
    pad = 0.13 * (rhi - rlo)
    rax.set_ylim(rlo - pad, rhi + pad)
    devs = np.concatenate(devs); errs = np.concatenate(errs)
    #### In the MAIN panel, left-aligned under the pT text: inside the ratio
    #### panel this line collided with the low-rho points.
    ax.text(ax0, 0.60,
            "bins with $N\\geq50$:\nmedian |ratio$-$1| = %.1f%%\nmedian stat. unc. = %.1f%%"
            % (100 * np.median(devs), 100 * np.median(errs)),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=PUB_LEGEND_FS - 2, color="0.30", linespacing=1.4)
    rax.set_ylabel("Ratio to CP5")
    rax.set_xlabel(r"$\log_{10}(\rho^2_g)$")
    rax.grid(alpha=0.25)
    rax.legend(loc="upper left", fontsize=PUB_LEGEND_FS - 3, frameon=False, ncol=1)

    for a in (ax, rax):
        a.xaxis.label.set_size(PUB_LABEL_FS)
        a.yaxis.label.set_size(PUB_LABEL_FS)
        a.tick_params(labelsize=PUB_TICK_FS)
    rax.yaxis.label.set_size(PUB_LABEL_FS - 4)

    # stamp below everything; the constrained-layout rect keeps it off the xlabel
    fig.get_layout_engine().set(rect=(0, 0.022, 1, 0.978))
    fig.text(0.99, 0.001,
             "%s  |  smp_jetmass_run2 %s  |  inputs: hadronic_model_prod cluster 11943799"
             % (datetime.date.today().isoformat(), ver),
             ha="right", va="bottom", fontsize=7, color="0.45", family="monospace")

    out = os.path.join(outdir, "model_rho_g_%s_pt%d.png" % (channel, ipt + 1))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)
