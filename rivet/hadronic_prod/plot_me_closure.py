"""ME closure: standalone HardQCD (2->2) CP5 vs standalone madgraphMLM CP5,
same shower/tune/selection, so the difference is purely the matrix element.

    python3 plot_me_closure.py shapes_<ch>.npz shapes_hq_<ch>.npz out/ --channel dijet
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

mg = np.load(sys.argv[1]); hq = np.load(sys.argv[2])
outdir = sys.argv[3] if len(sys.argv) > 3 else "."
channel = "dijet"
for i, a in enumerate(sys.argv):
    if a == "--channel":
        channel = sys.argv[i + 1]
os.makedirs(outdir, exist_ok=True)

RHO = mg["rho_edges"]
first = int(np.searchsorted(RHO, -4.0))
centers = 0.5 * (RHO[:-1] + RHO[1:])
w = np.diff(RHO)
PT = [(200, 290), (290, 400), (400, None)]
ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                     capture_output=True, text=True,
                     cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()


def norm(z, lo):
    h = z["cp5_pt%d_h" % lo].astype(float)
    e = z["cp5_pt%d_e" % lo].astype(float)
    n = z["cp5_pt%d_n" % lo].astype(float)
    return h / h.sum(), e / h.sum(), n


for ipt, (lo, hi) in enumerate(PT):
    a, ae, an = norm(mg, lo)
    b, be, bn = norm(hq, lo)
    sl = slice(first, len(a))
    keep = (an[sl] >= 50) & (bn[sl] >= 50) & (a[sl] > 0)

    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.06},
        layout="constrained")
    fig.set_size_inches(10, 11)

    ax.step(RHO[first:], np.append((a / w)[sl], (a / w)[-1]), where="post",
            color="k", lw=2.6, label="madgraphMLM + Pythia")
    ax.errorbar(centers[sl], (a / w)[sl], yerr=(ae / w)[sl], fmt="none",
                ecolor="k", elinewidth=1.6, capsize=0)
    ax.step(RHO[first:], np.append((b / w)[sl], (b / w)[-1]), where="post",
            color="#e42536", lw=2.4, label=r"HardQCD 2$\to$2")
    ax.errorbar(centers[sl], (b / w)[sl], yerr=(be / w)[sl], fmt="none",
                ecolor="#e42536", elinewidth=1.6, capsize=0)

    ax.set_ylabel(r"$(1/N)\,dN/d\log_{10}(\rho^2_g)$")
    #### enough headroom that the legend clears the CMS block AND the curves
    ax.set_ylim(0, max((a / w)[sl].max(), (b / w)[sl].max()) * 2.0)
    hep.cms.label("Preliminary", data=False, loc=2, ax=ax, rlabel=r"13 TeV")
    ax.grid(alpha=0.25)
    anchor = [t for t in ax.texts if t.get_text() == "CMS"]
    ax0 = anchor[0].get_position()[0] if anchor else 0.03
    ptxt = r"$p_{T} > %d$ GeV" % lo if hi is None else r"$%d < p_{T} < %d$ GeV" % (lo, hi)
    ax.text(ax0, 0.775, "%s jets, CP5\n%s" % (channel.capitalize(), ptxt),
            transform=ax.transAxes, ha="left", va="top", fontsize=PUB_ANNOT_FS)
    leg = ax.legend(loc="upper right", fontsize=PUB_LEGEND_FS - 2, frameon=True,
                    facecolor="white", edgecolor="none", framealpha=1.0,
                    borderpad=0.4, labelspacing=0.35, handlelength=1.4,
                    borderaxespad=0.6)
    leg.set_zorder(10)

    r = (b / a)[sl][keep]
    re = r * np.hypot((be / b)[sl][keep], (ae / a)[sl][keep])
    rax.axhline(1.0, color="k", lw=1.4)
    rax.fill_between(RHO[first:], np.append(1 - (ae / a)[sl], 1),
                     np.append(1 + (ae / a)[sl], 1), step="post",
                     color="0.6", alpha=0.45, lw=0, label="madgraphMLM stat. unc.")
    rax.errorbar(centers[sl][keep], r, yerr=re, fmt="o", ms=6,
                 color="#e42536", elinewidth=1.6, capsize=0)
    pad = 0.13 * ((r + re).max() - (r - re).min())
    rax.set_ylim((r - re).min() - pad, (r + re).max() + pad)
    rax.set_ylabel("HardQCD / MLM")
    rax.set_xlabel(r"$\log_{10}(\rho^2_g)$")
    rax.grid(alpha=0.25)
    rax.legend(loc="upper left", fontsize=PUB_LEGEND_FS - 3, frameon=False)
    ax.text(ax0, 0.60,
            "median ratio = %.2f\nmax deviation = %.0f%%"
            % (np.median(r), 100 * np.max(np.abs(r - 1))),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=PUB_LEGEND_FS - 2, color="0.30", linespacing=1.4)

    for a_ in (ax, rax):
        a_.xaxis.label.set_size(PUB_LABEL_FS)
        a_.yaxis.label.set_size(PUB_LABEL_FS)
        a_.tick_params(labelsize=PUB_TICK_FS)
    rax.yaxis.label.set_size(PUB_LABEL_FS - 6)

    fig.get_layout_engine().set(rect=(0, 0.022, 1, 0.978))
    fig.text(0.99, 0.001,
             "%s  |  smp_jetmass_run2 %s  |  inputs: hadronic_model_prod + _hardqcd"
             % (datetime.date.today().isoformat(), ver),
             ha="right", va="bottom", fontsize=7, color="0.45", family="monospace")
    out = os.path.join(outdir, "me_closure_%s_pt%d.png" % (channel, ipt + 1))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)
