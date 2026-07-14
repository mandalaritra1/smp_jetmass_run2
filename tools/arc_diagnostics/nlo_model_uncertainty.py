#!/usr/bin/env python
"""Model (Herwig) uncertainty on the groomed-rho unfolding via gen-reweighting non-closure.

The unfolded result depends on the physics model behind the RESPONSE (the within-gen-bin
shape sets the migration). To quantify that dependence:
  1. reweight the nominal MC (NLO skims) at GEN level so its gen-rho shape matches Herwig
     (per pT bin; smooth interpolation of the Herwig/NLO density ratio),
  2. fold the reweighted MC through the SAME nominal response -> Herwig-like pseudo-data,
  3. unfold that pseudo-data with the NOMINAL (un-reweighted) response,
  4. non-closure = unfolded / reweighted-gen-truth - 1  ==  the model systematic.

A response built on the right physics would close; the residual is the model bias the
nominal response imprints. Herwig gen shapes are pre-exported (herwig_gen_shapes.npz).

Run in the unfold venv (ROOT + awkward):
  cd ~/Projects/unfold && source scripts/setup_root.sh && source .venv/bin/activate
  python ~/Projects/smp_jetmass_run2/tools/arc_diagnostics/nlo_model_uncertainty.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nlo_tunfold_nested as T

SP = ("/private/tmp/claude-501/-Users-aritra-Projects-smp-jetmass-run2/"
      "5475621c-e50a-4ac9-9c83-0b954dcd7fe5/scratchpad/herwig_gen_shapes.npz")
OUTDIR = T.OUTDIR
PT_LAB = T.PT_LAB


def reweight_fn(rg_nom, w_nom, hw_edges, hw_counts):
    """Per-event Herwig/NLO gen DENSITY ratio, as a smooth (interpolated) function of rho.
    Both shapes are area-normalized within the pT bin, so the reweight morphs shape only."""
    nlo, _ = np.histogram(rg_nom, bins=hw_edges, weights=w_nom)
    bw = np.diff(hw_edges)
    nd = nlo / max(np.sum(nlo), 1e-30) / bw                # NLO density
    hd = hw_counts / max(np.sum(hw_counts), 1e-30) / bw    # Herwig density
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nd > 0, hd / nd, 1.0)
    cx = 0.5 * (hw_edges[:-1] + hw_edges[1:])
    good = nd > 0
    # piecewise-linear in rho (varies within the unfolding gen bins); flat extrapolation
    return lambda x: np.interp(x, cx[good], ratio[good], left=ratio[good][0], right=ratio[good][-1])


def main():
    z = np.load(SP)
    c = T.load()
    res = []
    print(f"{'pT bin':16s} {'bins':>4s} {'max|model unc|':>14s} {'mean|model unc|':>15s}")
    for ip, (plo, phi) in enumerate(zip(T.PT_EDGES[:-1], T.PT_EDGES[1:])):
        ge = T.RHO_GEN[ip]
        s = ((c["gpt"] >= plo) & (c["gpt"] < phi) & (c["rpt"] >= plo) & (c["rpt"] < phi)
             & (c["gmg"] > T.FLOOR) & (c["rmg"] > T.FLOOR))
        rg = T.rho(c["gmg"][s], c["gpt"][s]); rr = T.rho(c["rmg"][s], c["rpt"][s])
        w = c["w"][s]
        fid = np.isfinite(rg) & np.isfinite(rr) & (rg >= ge[0]) & (rg <= ge[-1])
        rg, rr, w = rg[fid], rr[fid], w[fid]

        rwf = reweight_fn(rg, w, z["edges_g"], z[f"hw_g_pt{ip+1}"])
        wr = w * rwf(rg)                                   # Herwig-like per-event weight

        # nominal response (un-reweighted), Herwig-like pseudo-data + truth
        hResp = T.th2_response(rg, rr, w, ge)
        hData = T.th1_reco(rr, wr)
        uval, _ = T.run_tunfold(hResp, hData, full_err=False)
        truth_rw, _ = np.histogram(rg, bins=ge, weights=wr)
        with np.errstate(divide="ignore", invalid="ignore"):
            munc = np.where(truth_rw > 0, uval / truth_rw - 1.0, np.nan)
        print(f"{PT_LAB[ip]:16s} {len(ge)-1:4d} {np.nanmax(np.abs(munc))*100:13.1f}% "
              f"{np.nanmean(np.abs(munc))*100:14.1f}%")
        res.append(dict(ge=ge, munc=munc, truth=truth_rw, uval=uval))

    # save for downstream combination + plot
    np.savez(os.path.join(OUTDIR, "model_unc_rho_g.npz"),
             **{f"munc_pt{i+1}": r["munc"] for i, r in enumerate(res)},
             **{f"edges_pt{i+1}": r["ge"] for i, r in enumerate(res)})

    fig, axes = plt.subplots(1, 3, figsize=(31, 10.6), layout="constrained")
    for ip, ax in enumerate(axes):
        d = res[ip]; ge = d["ge"]; cx = 0.5 * (ge[:-1] + ge[1:])
        ax.step(ge, np.append(d["munc"], d["munc"][-1]) * 100, where="post",
                color="#7a21dd", lw=2.6)
        ax.fill_between(cx, d["munc"] * 100, step="mid", color="#7a21dd", alpha=0.18)
        ax.axhline(0, color="black", lw=1.0)
        ax.set_ylim(-25, 25); ax.set_xlim(ge[0], 0)
        ax.set_ylabel("Herwig model uncertainty [%]")
        ax.set_xlabel(r"groomed $\log_{10}(\rho_g^2)$"); ax.grid(alpha=0.25)
        hep.cms.label("", data=False, loc=0, ax=ax, rlabel="NLO $\\to$ Herwig")
        ax.text(0.05, 0.93, f"groomed model unc.\n{PT_LAB[ip]} ({len(ge)-1} bins)",
                transform=ax.transAxes, va="top", fontsize=18)
    out = os.path.join(OUTDIR, "nlo_model_uncertainty_g.png")
    fig.savefig(out, dpi=100); plt.close(fig)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
