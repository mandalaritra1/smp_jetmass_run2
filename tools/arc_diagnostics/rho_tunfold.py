#!/usr/bin/env python
"""TUnfold the rho spectra on the adaptive common binning (+SF), with proper stat errors.

Standard well-posed TUnfold setup: FINE reco axis (0.1-wide) -> ADAPTIVE coarse gen
axis (the resolution-matched common binning). Per pT bin, groomed & ungroomed,
Pythia + data 2018. SF (reco-indexed median m_reco/m_gen) applied to ALL reco mass
before rho. Regularization off (tau=0) since the gen bins are already resolution-
matched; TUnfold then gives the least-squares solution with the full data-stat
covariance -> the unfolded bin errors are the propagated statistical uncertainties.

  - half-sample closure (response on even events, unfold odd events' reco)
  - data unfold (MC-fake-subtracted), bands = TUnfold stat errors

Run in the unfold venv with ROOT:
  cd ~/Projects/unfold && source scripts/setup_root.sh && source .venv/bin/activate
  python ~/Projects/smp_jetmass_run2/tools/arc_diagnostics/rho_tunfold.py
"""
import os
import pickle
from array import array

import numpy as np
import ROOT
import matplotlib.pyplot as plt
import mplhep as hep

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
hep.style.use(hep.style.CMS)

D = os.path.expanduser("~/Downloads")
OUT = os.path.expanduser("~/Projects/smp_jetmass_run2/review/figs/nlo_deck")
R8 = 0.8
PT_EDGES = [200., 290., 400., 13000.]
PT_LAB = ["200-290", "290-400", ">400"]
RECO_FINE = np.round(np.arange(-8.0, 0.0 + 0.1, 0.1), 4)
RHO_GEN = {"g": np.array([-6.0, -3.6, -2.6, -2.0, -1.6, -1.3, -1.1, -0.9, -0.7, 0.0]),
           "u": np.array([-6.0, -1.5, -1.1, -0.8, -0.6, 0.0])}
GROOM = {"g": ("Groomed", "msoftdrop", "gen_msoftdrop"),
         "u": ("Ungroomed", "mass",     "gen_mass")}


def ntuple(tag):
    o = pickle.load(open(os.path.join(D, f"mass_diagnostic_ntuple_{tag}_2018.pkl"),
                         "rb"))["reco_jet_ntuple"]
    return {k: np.asarray(o[k].value if hasattr(o[k], "value") else o[k])
            for k in ("pt", "gen_pt", "msoftdrop", "gen_msoftdrop", "mass",
                      "gen_mass", "weight", "passes_both", "event")}


def rho(m, pt):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(m / (pt * R8))


def sf_table(mc, rcol, gcol, plo, phi):
    s = (mc["passes_both"].astype(bool) & (mc[gcol] > 0) & (mc[rcol] > 0)
         & (mc["pt"] >= plo) & (mc["pt"] < phi))
    reco, r = mc[rcol][s], mc[rcol][s] / mc[gcol][s]
    grid = np.arange(0.0, 202.0, 2.0); cx = 0.5 * (grid[:-1] + grid[1:])
    idx = np.digitize(reco, grid) - 1
    sf = np.full(len(cx), np.nan)
    for b in range(len(cx)):
        m = idx == b
        if m.sum() >= 50:
            sf[b] = np.median(r[m])
    g = np.isfinite(sf)
    return cx[g], sf[g]


def corr(mreco, sx, sy):
    return mreco / np.interp(mreco, sx, sy, left=sy[0], right=sy[-1])


def th2_response(rg, rr, w, gen_edges):
    h = ROOT.TH2D("resp", ";gen;reco", len(gen_edges) - 1, array('d', gen_edges),
                  len(RECO_FINE) - 1, array('d', RECO_FINE))
    h.Sumw2()
    for a, b, ww in zip(rg, rr, w):
        h.Fill(a, b, ww)
    return h


def th1_reco(vals, w):
    h = ROOT.TH1D("m", ";reco", len(RECO_FINE) - 1, array('d', RECO_FINE))
    h.Sumw2()
    hv, _ = np.histogram(vals, bins=RECO_FINE, weights=w)
    hv2, _ = np.histogram(vals, bins=RECO_FINE, weights=w * w)
    for i, (c, v) in enumerate(zip(hv, hv2)):
        h.SetBinContent(i + 1, c); h.SetBinError(i + 1, np.sqrt(v))
    return h


def run_tunfold(hResp, hData):
    unf = ROOT.TUnfoldDensity(hResp, ROOT.TUnfold.kHistMapOutputHoriz,
                              ROOT.TUnfold.kRegModeNone,
                              ROOT.TUnfold.kEConstraintNone,
                              ROOT.TUnfoldDensity.kDensityModeNone)
    unf.SetInput(hData)
    unf.DoUnfold(0.0)                                  # tau=0: least-squares, full stat cov
    hOut = unf.GetOutput("out")
    n = hOut.GetNbinsX()
    val = np.array([hOut.GetBinContent(i + 1) for i in range(n)])
    err = np.array([hOut.GetBinError(i + 1) for i in range(n)])
    return val, err


def main():
    mc, data = ntuple("pythia"), ntuple("data")
    results = {}
    for gkey, (gname, rcol, gcol) in GROOM.items():
        gen_edges = RHO_GEN[gkey]; ng = len(gen_edges) - 1
        per_pt = []
        for ip, (plo, phi) in enumerate(zip(PT_EDGES[:-1], PT_EDGES[1:])):
            sx, sy = sf_table(mc, rcol, gcol, plo, phi)
            # ---- matched MC in this pT bin, fiducial gen rho ----
            s = (mc["passes_both"].astype(bool) & (mc[gcol] > 0) & (mc[rcol] > 0)
                 & (mc["pt"] >= plo) & (mc["pt"] < phi)
                 & (mc["gen_pt"] >= plo) & (mc["gen_pt"] < phi))
            rg = rho(mc[gcol][s], mc["gen_pt"][s])
            rr = rho(corr(mc[rcol][s], sx, sy), mc["pt"][s])
            w, ev = mc["weight"][s], mc["event"][s]
            fid = (rg >= gen_edges[0]) & (rg <= gen_edges[-1])
            rg, rr, w, ev = rg[fid], rr[fid], w[fid], ev[fid]
            truth, _ = np.histogram(rg, bins=gen_edges, weights=w)

            # ---- fake fraction per fine reco bin (all reco vs matched-fiducial) ----
            sa = (mc[rcol] > 0) & (mc["pt"] >= plo) & (mc["pt"] < phi)
            rr_all = rho(corr(mc[rcol][sa], sx, sy), mc["pt"][sa])
            wa = mc["weight"][sa]
            matched_fid = np.zeros(sa.sum(), bool)
            sb = (mc["passes_both"][sa].astype(bool) & (mc[gcol][sa] > 0))
            rga = np.full(sa.sum(), -99.0)
            rga[sb] = rho(mc[gcol][sa][sb], mc["gen_pt"][sa][sb])
            matched_fid = sb & (rga >= gen_edges[0]) & (rga <= gen_edges[-1])
            allr, _ = np.histogram(rr_all, bins=RECO_FINE, weights=wa)
            mreco, _ = np.histogram(rr_all[matched_fid], bins=RECO_FINE, weights=wa[matched_fid])
            with np.errstate(divide="ignore", invalid="ignore"):
                phi_f = np.where(allr > 0, 1.0 - mreco / allr, 0.0)

            # ---- data reco (SF applied), fake-subtracted ----
            ds = (data["pt"] >= plo) & (data["pt"] < phi)
            rrd = rho(corr(data[rcol][ds], sx, sy), data["pt"][ds])
            dv, _ = np.histogram(rrd, bins=RECO_FINE, weights=data["weight"][ds])
            dsub = dv * (1.0 - phi_f)
            dsub_err = np.sqrt(np.maximum(dv, 0)) * (1.0 - phi_f)

            # ---- TUnfold: data ----
            hResp = th2_response(rg, rr, w, gen_edges)
            hData = ROOT.TH1D("d", ";reco", len(RECO_FINE) - 1, array('d', RECO_FINE))
            for i, (cc, ee) in enumerate(zip(dsub, dsub_err)):
                # empty bins keep zero error so TUnfold ignores them (not a tight 0-constraint)
                hData.SetBinContent(i + 1, cc); hData.SetBinError(i + 1, ee)
            uval, uerr = run_tunfold(hResp, hData)

            # ---- TUnfold: half-sample closure ----
            even = ev % 2 == 0
            hRespE = th2_response(rg[even], rr[even], w[even], gen_edges)
            hOdd = th1_reco(rr[~even], w[~even])
            cval, cerr = run_tunfold(hRespE, hOdd)
            truth_odd, _ = np.histogram(rg[~even], bins=gen_edges, weights=w[~even])

            per_pt.append(dict(truth=truth, uval=uval, uerr=uerr,
                               cval=cval, cerr=cerr, truth_odd=truth_odd))
        results[gkey] = dict(gen_edges=gen_edges, per_pt=per_pt, gname=gname)
    pickle.dump(results, open(os.path.join(D, "rho_tunfold_results_2018.pkl"), "wb"))
    plot(results)


def plot(results):
    for gkey, R in results.items():
        ge = R["gen_edges"]; cx = 0.5 * (ge[:-1] + ge[1:]); wid = np.diff(ge)
        gname = R["gname"]
        # ---- closure ----
        fig, axes = plt.subplots(1, 3, figsize=(31, 10.6), layout="constrained")
        for ip, ax in enumerate(axes):
            d = R["per_pt"][ip]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(d["truth_odd"] > 0, d["cval"] / d["truth_odd"], np.nan)
                re = np.where(d["truth_odd"] > 0, d["cerr"] / d["truth_odd"], np.nan)
            ax.errorbar(cx, r, yerr=re, fmt="o", color="#3f90da", ms=11, lw=2)
            ax.axhline(1.0, color="black", ls="--", lw=1.2)
            ax.set_ylim(0.5, 1.5); ax.set_xlim(ge[0], 0)
            ax.set_ylabel("unfolded / truth"); ax.set_xlabel(r"$\rho = 2\log_{10}(m/(p_T R))$")
            hep.cms.label("", data=False, loc=0, ax=ax, rlabel="(2018, 13 TeV)")
            ax.text(0.05, 0.93, f"{gname} — TUnfold half-sample closure\n$p_T$ {PT_LAB[ip]} GeV",
                    transform=ax.transAxes, va="top", fontsize=20)
        fig.savefig(os.path.join(OUT, f"tunfold_closure_{gkey}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        # ---- data ----
        fig, axes = plt.subplots(1, 3, figsize=(31, 10.6), layout="constrained")
        for ip, ax in enumerate(axes):
            d = R["per_pt"][ip]
            tn = d["truth"] / wid / (d["truth"] * 1.0).sum()
            un = d["uval"] / wid / d["uval"].sum()
            une = d["uerr"] / wid / d["uval"].sum()
            ax.step(ge, np.append(tn, tn[-1]), where="post", color="#e42536", lw=2.6,
                    label="MC gen (Pythia)")
            ax.errorbar(cx, un, yerr=une, fmt="o", color="black", ms=11, lw=2,
                        label="Unfolded data (TUnfold)")
            ax.set_ylim(bottom=0, top=max(np.nanmax(tn), np.nanmax(un + une)) * 1.45)
            ax.set_xlim(ge[0], 0)
            ax.set_ylabel(r"$(1/N)\,dN/d\rho$"); ax.set_xlabel(r"$\rho = 2\log_{10}(m/(p_T R))$")
            hep.cms.label("Preliminary", data=True, loc=0, ax=ax, rlabel="59.8 fb$^{-1}$ (2018)")
            ax.text(0.05, 0.93, f"{gname} — unfolded data\n$p_T$ {PT_LAB[ip]} GeV",
                    transform=ax.transAxes, va="top", fontsize=20)
            ax.legend(loc="upper right", fontsize=18, frameon=False)
        fig.savefig(os.path.join(OUT, f"tunfold_data_{gkey}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote tunfold_{{closure,data}}_{gkey}.png")


if __name__ == "__main__":
    main()
