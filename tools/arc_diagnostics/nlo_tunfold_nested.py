#!/usr/bin/env python
"""TUnfold the groomed rho spectrum on the NESTED binning + m_g>2 GeV floor (NLO skims).

Well-posed TUnfold: FINE reco axis (0.1) -> nested coarse gen axis (resolution-matched,
overlapping edges across pT bins). Groomed only. Half-sample closure: response built on
even-event jets, the odd-event reco spectrum unfolded with it and compared to the odd
truth. tau=0 (gen bins already resolution-matched) -> least-squares with full stat cov,
so the unfolded errors are the propagated statistical uncertainties. NO scale factor:
the 2 GeV floor + nested binning replace it.

Run in the unfold venv (has ROOT + awkward):
  cd ~/Projects/unfold && source scripts/setup_root.sh && source .venv/bin/activate
  python ~/Projects/smp_jetmass_run2/tools/arc_diagnostics/nlo_tunfold_nested.py
"""
import os, glob, re
from array import array
import numpy as np
import awkward as ak
import ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError
hep.style.use(hep.style.CMS)

SKIM = "/Users/aritra/Projects/unfold/inputs/zjet/nlo_skims/nlo_ptz_skims"
OUTDIR = os.path.expanduser("~/Projects/smp_jetmass_run2/tools/arc_diagnostics/figs")
R8, FLOOR = 0.8, 2.0
XS = {"100To250": 97.2, "250To400": 3.701, "400To650": 0.5086, "650ToInf": 0.04728}
PT_EDGES = [200., 290., 400., 13000.]
PT_LAB = ["200$-$290 GeV", "290$-$400 GeV", "$p_T>$400 GeV"]
RECO_FINE = np.round(np.arange(-7.0, 0.0 + 0.1, 0.1), 4)
# nested gen edges (m_g>2), -6.0 = fixed floor catch-all replacing the -7 merge artifact.
# Tail coarsened (no 0.1-wide bins above -0.8) so the unregularized inversion stays
# stable near the rho->0 edge; clean nested chain pt1 c pt2 c pt3, all subsets of pt3.
RHO_GEN = [
    np.array([-6.0, -3.1, -2.1, -1.6, -1.2, -1.0, -0.8, 0.0]),
    np.array([-6.0, -3.1, -2.1, -1.8, -1.6, -1.4, -1.2, -1.0, 0.0]),
    np.array([-6.0, -3.9, -3.1, -2.5, -2.1, -1.8, -1.6, -1.4, -1.2, -1.0, 0.0]),
]


def load():
    """Pool NLO skims -> reco/gen groomed+ungroomed mass, pt, weight, event (xs-stitched).
    rmg/gmg = groomed (msoftdrop); rmu/gmu = ungroomed (mass). Floor is always on groomed."""
    K = {k: [] for k in ("rpt", "rmg", "rmu", "gpt", "gmg", "gmu", "w", "ev")}
    for f in sorted(glob.glob(SKIM + "/*/merged.parquet")):
        ds = os.path.basename(os.path.dirname(f))
        xs = XS[re.search(r"ptz_(\d+To\w+?)_UL", ds).group(1)]
        a = ak.from_parquet(f)
        pb = ak.to_numpy(a.passes_both); gw = ak.to_numpy(a.weight)
        rpt = ak.to_numpy(ak.fill_none(a.pt, np.nan)); rmg = ak.to_numpy(ak.fill_none(a.msoftdrop, np.nan))
        rmu = ak.to_numpy(ak.fill_none(a.mass, np.nan))
        gpt = ak.to_numpy(ak.fill_none(a.gen_pt, np.nan)); gmg = ak.to_numpy(ak.fill_none(a.gen_msoftdrop, np.nan))
        gmu = ak.to_numpy(ak.fill_none(a.gen_mass, np.nan))
        ev = ak.to_numpy(a.event).astype(np.uint64)
        sel = pb & np.isfinite(gpt) & np.isfinite(rpt)
        norm = xs / np.sum(np.abs(gw[sel]))
        K["rpt"].append(rpt[sel]); K["rmg"].append(rmg[sel]); K["rmu"].append(rmu[sel])
        K["gpt"].append(gpt[sel]); K["gmg"].append(gmg[sel]); K["gmu"].append(gmu[sel])
        K["w"].append(np.sign(gw[sel]) * np.abs(gw[sel]) * norm); K["ev"].append(ev[sel])
    return {k: np.concatenate(v) for k, v in K.items()}


def rho(m, pt):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(m / (pt * R8))


def th2_response(rg, rr, w, gen_edges):
    ng, nr = len(gen_edges) - 1, len(RECO_FINE) - 1
    H, _, _ = np.histogram2d(rg, rr, bins=[gen_edges, RECO_FINE], weights=w)
    H2, _, _ = np.histogram2d(rg, rr, bins=[gen_edges, RECO_FINE], weights=w * w)
    h = ROOT.TH2D("resp", ";gen;reco", ng, array('d', gen_edges), nr, array('d', RECO_FINE))
    h.Sumw2()
    for i in range(ng):
        for j in range(nr):
            h.SetBinContent(i + 1, j + 1, H[i, j]); h.SetBinError(i + 1, j + 1, np.sqrt(H2[i, j]))
    return h


def th1_reco(vals, w):
    h = ROOT.TH1D("m", ";reco", len(RECO_FINE) - 1, array('d', RECO_FINE)); h.Sumw2()
    hv, _ = np.histogram(vals, bins=RECO_FINE, weights=w)
    hv2, _ = np.histogram(vals, bins=RECO_FINE, weights=w * w)
    for i, (c, v) in enumerate(zip(hv, hv2)):
        h.SetBinContent(i + 1, c); h.SetBinError(i + 1, np.sqrt(max(v, 0)))
    return h


TAUS = np.concatenate([[0.0], np.logspace(-4.0, 2.0, 31)])


def unfold(hResp, hData, tau=0.0, reg=False):
    """TUnfoldDensity object. reg=False -> tau=0 least squares; reg=True -> curvature
    regularization (bin-width density, for the variable gen binning)."""
    mode = ROOT.TUnfold.kRegModeCurvature if reg else ROOT.TUnfold.kRegModeNone
    dens = (ROOT.TUnfoldDensity.kDensityModeBinWidth if reg
            else ROOT.TUnfoldDensity.kDensityModeNone)
    unf = ROOT.TUnfoldDensity(hResp, ROOT.TUnfold.kHistMapOutputHoriz, mode,
                              ROOT.TUnfold.kEConstraintNone, dens)
    unf.SetInput(hData)
    unf.DoUnfold(tau)
    return unf


def val_err(unf, full_err=True):
    hOut = unf.GetOutput("out"); n = hOut.GetNbinsX()
    val = np.array([hOut.GetBinContent(i + 1) for i in range(n)])
    if not full_err:
        return val, np.array([hOut.GetBinError(i + 1) for i in range(n)])
    eIn = unf.GetEmatrixInput("eIn"); eMC = unf.GetEmatrixSysUncorr("eMC")
    err = np.array([np.sqrt(max(eIn.GetBinContent(i + 1, i + 1)
                                + eMC.GetBinContent(i + 1, i + 1), 0.0)) for i in range(n)])
    return val, err


def rho_avg(unf, n):
    """Mean global correlation coefficient of the unfolded result (TUnfold's tau metric)."""
    C = unf.GetEmatrixTotal("C")
    cov = np.array([[C.GetBinContent(i + 1, j + 1) for j in range(n)] for i in range(n)])
    try:
        cinv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.nan
    d = np.diag(cov) * np.diag(cinv)
    return np.nanmean(np.sqrt(np.clip(1.0 - 1.0 / d, 0.0, 1.0)))


def scan_tau(hResp, hData, n):
    """Pick tau minimizing the mean global correlation (rho_avg)."""
    best = (np.inf, 0.0)
    for tau in TAUS:
        r = rho_avg(unfold(hResp, hData, tau, reg=(tau > 0)), n)
        if np.isfinite(r) and r < best[0]:
            best = (r, tau)
    return best[1], best[0]


def run_tunfold(hResp, hData, full_err=True):           # tau=0 convenience (diagnostics)
    return val_err(unfold(hResp, hData, 0.0, reg=False), full_err)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    c = load()
    print(f"pooled matched gen jets: {len(c['gpt']):,d}")
    def indep_chi2(rg, rr, w, even, ge, tau):
        """Two-disjoint-half closure chi2 at a given tau (data-stat errors)."""
        uo, eo = val_err(unfold(th2_response(rg[even], rr[even], w[even], ge),
                                th1_reco(rr[~even], w[~even]), tau, reg=(tau > 0)), full_err=False)
        ue, ee = val_err(unfold(th2_response(rg[~even], rr[~even], w[~even], ge),
                                th1_reco(rr[even], w[even]), tau, reg=(tau > 0)), full_err=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(np.hypot(eo, ee) > 0, (uo - ue) / np.hypot(eo, ee), np.nan)
        return np.nansum(p ** 2) / np.isfinite(p).sum()

    res = []
    print(f"\n{'pT bin':16s} {'bins':>4s} {'tau':>9s} {'selfbias':>8s} "
          f"{'indepChi2 tau0':>14s} {'indepChi2 reg':>13s}")
    for ip, (plo, phi) in enumerate(zip(PT_EDGES[:-1], PT_EDGES[1:])):
        ge = RHO_GEN[ip]
        s = ((c["gpt"] >= plo) & (c["gpt"] < phi) & (c["rpt"] >= plo) & (c["rpt"] < phi)
             & (c["gmg"] > FLOOR) & (c["rmg"] > FLOOR))
        rg = rho(c["gmg"][s], c["gpt"][s]); rr = rho(c["rmg"][s], c["rpt"][s])
        w, ev = c["w"][s], c["ev"][s]
        fid = np.isfinite(rg) & np.isfinite(rr) & (rg >= ge[0]) & (rg <= ge[-1])
        rg, rr, w, ev = rg[fid], rr[fid], w[fid], ev[fid]
        even = (ev % 2 == 0)

        # FIGURE = unregularized (tau=0): regularization was tested (below) but gives only
        # marginal indep-chi2 gain at the cost of bias, and hurts pt1 -> not adopted.
        hRespE = th2_response(rg[even], rr[even], w[even], ge)
        hOdd = th1_reco(rr[~even], w[~even])
        uval, uerr = val_err(unfold(hRespE, hOdd, 0.0, reg=False), full_err=True)
        truth_odd, _ = np.histogram(rg[~even], bins=ge, weights=w[~even])
        tau, _ = scan_tau(hRespE, hOdd, len(ge) - 1)        # reg optimum, for the table only

        # diagnostics: self-bias (reg) + indep chi2 at tau=0 vs chosen tau
        sval, _ = val_err(unfold(th2_response(rg, rr, w, ge), th1_reco(rr, w), tau,
                                 reg=(tau > 0)), full_err=False)
        strue, _ = np.histogram(rg, bins=ge, weights=w)
        selfbias = np.nanmax(np.abs(np.where(strue > 0, sval / strue - 1, np.nan)))
        ic0 = indep_chi2(rg, rr, w, even, ge, 0.0)
        icr = indep_chi2(rg, rr, w, even, ge, tau)
        print(f"{PT_LAB[ip]:16s} {len(ge)-1:4d} {tau:9.1e} {selfbias:8.4f} "
              f"{ic0:14.2f} {icr:13.2f}")
        res.append(dict(ge=ge, uval=uval, uerr=uerr, truth=truth_odd, tau=0.0))

    # ---- plot: closure ratio + unfolded vs truth ----
    fig, axes = plt.subplots(2, 3, figsize=(31, 16), height_ratios=[3, 1.3], layout="constrained")
    for ip in range(3):
        d = res[ip]; ge = d["ge"]; cx = 0.5 * (ge[:-1] + ge[1:]); wid = np.diff(ge)
        ax, rax = axes[0, ip], axes[1, ip]
        tn = d["truth"] / wid / d["truth"].sum()
        un = d["uval"] / wid / d["uval"].sum()
        une = d["uerr"] / wid / d["uval"].sum()
        ax.step(ge, np.append(tn, tn[-1]), where="post", color="#e42536", lw=2.6,
                label="MC truth (odd half)")
        ax.errorbar(cx, un, yerr=une, fmt="o", color="black", ms=10, lw=2,
                    label="Unfolded (TUnfold, even resp.)")
        ax.set_ylim(0, max(np.nanmax(tn), np.nanmax(un + une)) * 1.45)
        ax.set_xlim(ge[0], 0); ax.set_ylabel(r"$(1/N)\,dN/d\rho_g$")
        hep.cms.label("", data=False, loc=0, ax=ax, rlabel="NLO  ($m_g>2$)")
        treg = "unregularized" if d["tau"] == 0 else f"$\\tau$={d['tau']:.1e}"
        ax.text(0.05, 0.93, f"groomed — half-sample closure\n{PT_LAB[ip]} ({len(ge)-1} bins)\n{treg}",
                transform=ax.transAxes, va="top", fontsize=18)
        ax.legend(loc="upper right", fontsize=15, frameon=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d["truth"] > 0, d["uval"] / d["truth"], np.nan)
            re = np.where(d["truth"] > 0, d["uerr"] / d["truth"], np.nan)
        rax.errorbar(cx, r, yerr=re, fmt="o", color="#3f90da", ms=10, lw=2)
        rax.axhline(1.0, color="black", ls="--", lw=1.2)
        rax.set_ylim(0.7, 1.3); rax.set_xlim(ge[0], 0)
        rax.set_ylabel("unf / truth"); rax.set_xlabel(r"groomed $\log_{10}(\rho_g^2)$")
        rax.grid(alpha=0.25)
    out = os.path.join(OUTDIR, "nlo_tunfold_nested_closure_g.png")
    fig.savefig(out, dpi=100); plt.close(fig)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
