"""Closure figures: Rivet routine vs the coffea gen path on the same FullSim events.

    python3 plot_closure.py rivet_gen.npz coffea_gen_nomatch.npz coffea_gen_matched.npz <outdir>

One square CMS panel per (grooming, pT slice) with a ratio sub-panel, following the
cms-plot-style conventions: global CMS style, no forced figsize, no plt.title,
constrained layout, legend headroom, plain fractions.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use(hep.style.CMS)

# Petroff CVD-safe categorical colors
C_RIVET = "#5790fc"
C_COFFEA = "#f89c20"
C_MATCHED = "#9c9ca1"

PUB_RHO = np.array([-10., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0.])
SLICES = [(0, r"$200 < p_{T}^{jet} < 290$ GeV"),
          (1, r"$290 < p_{T}^{jet} < 400$ GeV"),
          (2, r"$p_{T}^{jet} > 400$ GeV")]


def rebin(values, edges, target):
    edges = np.round(np.asarray(edges, float), 6)
    target = np.round(np.asarray(target, float), 6)
    out = np.zeros(len(target) - 1)
    for i in range(len(target) - 1):
        mask = (edges[:-1] >= target[i] - 1e-9) & (edges[1:] <= target[i + 1] + 1e-9)
        out[i] = values[mask].sum()
    return out


def density(counts, edges):
    """Unit-area density: sum(y * dx) = 1."""
    w = np.diff(edges)
    tot = (counts).sum()
    return counts / (tot * w) if tot else counts


def panel(target, y_r, y_c, y_m, rlabel, outpath):
    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        layout="constrained")

    # The whole point is that the first two agree, so they must not hide each other:
    # Rivet as a solid step, coffea as open markers on top of it.
    ctr = 0.5 * (target[1:] + target[:-1])
    ax.stairs(y_r, target, baseline=None, lw=3.0, color=C_RIVET,
              label="Rivet routine (MiniAOD)")
    ax.plot(ctr, y_c, "o", ms=9, mfc="none", mew=2.2, color=C_COFFEA,
            ls="none", label="coffea gen path (NanoAOD)")
    if y_m is not None:
        ax.stairs(y_m, target, baseline=None, lw=1.8, ls=":", color=C_MATCHED,
                  label="coffea gen, reco-matched")

    ax.set_ylabel(r"$\frac{1}{N}\,\frac{dN}{d\log_{10}(\rho^{2})}$")
    ax.set_ylim(bottom=0, top=max(y_r.max(), y_c.max()) * 1.45)
    ax.legend(loc="upper left", fontsize="small", frameon=False)
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel=rlabel)
    ax.grid(alpha=0.25)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y_c > 0, y_r / y_c, np.nan)
    # baseline=None: a closed step would drop to y=0 at the panel edges, which in a
    # ratio panel zoomed around 1 looks like a data point slammed into the frame.
    rax.stairs(ratio, target, baseline=None, lw=3.0, color=C_RIVET)
    rax.plot(ctr, ratio, "o", ms=6, color=C_RIVET, ls="none")
    rax.axhline(1.0, color="black", lw=1.0, ls="-")
    rax.set_xlabel(r"$\log_{10}(\rho^{2})$")
    rax.set_ylabel("Rivet / coffea")
    finite = ratio[np.isfinite(ratio)]
    span = max(0.02, np.nanmax(np.abs(finite - 1.0)) * 1.45) if finite.size else 0.05
    rax.set_ylim(1 - span, 1 + span)
    rax.grid(alpha=0.25)

    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print("wrote", outpath)


def main(rivet_npz, coffea_npz, matched_npz, outdir):
    R, C = np.load(rivet_npz), np.load(coffea_npz)
    M = np.load(matched_npz) if matched_npz != "none" else None

    for groom, cname in (("ungroomed", "ptjet_rhojet_u_gen"),
                         ("groomed", "ptjet_rhojet_g_gen")):
        rv = R[f"zjets_{groom}__values"]
        rrho = R[f"zjets_{groom}__rhoedges"]
        cv = C[f"{cname}__values"]
        crho = C[f"{cname}__edges__mpt_gen"]
        mv = M[f"{cname}__values"] if M is not None else None
        # groomed coffea axis has no -0.5 edge -> merge the last two published bins
        target = PUB_RHO if groom == "ungroomed" else np.delete(PUB_RHO, -2)

        for i, rlab in SLICES:
            y_r = density(rebin(rv[i], rrho, target), target)
            y_c = density(rebin(cv[i + 1], crho, target), target)  # skip 185-200 sink
            y_m = density(rebin(mv[i + 1], crho, target), target) if mv is not None else None
            panel(target, y_r, y_c, y_m, rlab,
                  f"{outdir}/closure_{groom}_pt{i}.png")


if __name__ == "__main__":
    main(*sys.argv[1:5])
