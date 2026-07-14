#!/usr/bin/env python
"""Build unfoldable inputs on the adaptive common rho binning (+SF) and run the unfolding.

Pipeline (rho = 2*log10(m/(pt*0.8)), Pythia + data 2018, groomed & ungroomed):
  1. derive reco-indexed mass SF per pT bin (median m_reco/m_gen) from Pythia
  2. apply m_reco -> m_reco/SF to ALL reco (MC + data), then rho_reco
  3. bin the 2D (pT x rho) response on the COMMON adaptive binning (flattened to 1D)
     from matched (passes_both) Pythia jets; also MC gen truth, MC reco, MC fakes,
     and the SF-corrected data reco spectrum
  4. save the unfoldable pkl (response + vectors + edges)
  5. unfold with iterative D'Agostini:
       - half-sample closure  (response on even events, unfold odd events' reco)
       - data unfold          (MC-fake-subtracted data, prior = MC gen)

Inputs: ~/Downloads/mass_diagnostic_ntuple_{pythia,data}_2018.pkl
Outputs: ~/Projects/unfold/inputs/zjet/rho/adaptive/rho_adaptive_unfold_inputs_{g,u}_2018.pkl
         review/figs/nlo_deck/unfold_adaptive_{closure,data}_{g,u}.png
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

D = os.path.expanduser("~/Downloads")
OUT = os.path.expanduser("~/Projects/smp_jetmass_run2/review/figs/nlo_deck")
PKLDIR = os.path.expanduser("~/Projects/unfold/inputs/zjet/rho/adaptive")
os.makedirs(PKLDIR, exist_ok=True)
R8 = 0.8
N_ITER = 4
PT_EDGES = np.array([200., 290., 400., 13000.])
PT_LAB = ["200-290", "290-400", ">400"]
RHO_EDGES = {
    "g": np.array([-6.0, -3.6, -2.6, -2.0, -1.6, -1.3, -1.1, -0.9, -0.7, 0.0]),
    "u": np.array([-6.0, -1.5, -1.1, -0.8, -0.6, 0.0]),
}
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


def sf_curves(mc, rcol, gcol):
    """reco-indexed median SF(m_reco) per pT bin (interp tables)."""
    grid = np.arange(0.0, 202.0, 2.0); cx = 0.5 * (grid[:-1] + grid[1:])
    out = {}
    for ip, (plo, phi) in enumerate(zip(PT_EDGES[:-1], PT_EDGES[1:])):
        s = (mc["passes_both"].astype(bool) & (mc[gcol] > 0) & (mc[rcol] > 0)
             & (mc["pt"] >= plo) & (mc["pt"] < phi))
        reco, r = mc[rcol][s], mc[rcol][s] / mc[gcol][s]
        idx = np.digitize(reco, grid) - 1
        sf = np.full(len(cx), np.nan)
        for b in range(len(cx)):
            m = idx == b
            if m.sum() >= 50:
                sf[b] = np.median(r[m])
        g = np.isfinite(sf)
        out[ip] = (cx[g], sf[g])
    return out


def apply_sf(mreco, ptreco, sf):
    """corrected reco mass using the per-pT SF table."""
    out = mreco.copy()
    for ip, (plo, phi) in enumerate(zip(PT_EDGES[:-1], PT_EDGES[1:])):
        m = (ptreco >= plo) & (ptreco < phi)
        sx, sy = sf[ip]
        out[m] = mreco[m] / np.interp(mreco[m], sx, sy, left=sy[0], right=sy[-1])
    return out


def flat_index(pt, rho_v, redges):
    nrho = len(redges) - 1
    ip = np.digitize(pt, PT_EDGES) - 1
    ir = np.digitize(rho_v, redges) - 1
    ok = (ip >= 0) & (ip < len(PT_EDGES) - 1) & (ir >= 0) & (ir < nrho)
    return ip * nrho + ir, ok


def dagostini(meas, R, prior, n_iter=N_ITER):
    """Iterative Bayesian unfolding. R[r,g] = matched response counts."""
    colsum = R.sum(0, keepdims=True)
    Prg = np.divide(R, colsum, out=np.zeros_like(R), where=colsum > 0)  # P(r|g)
    n = prior.astype(float).copy()
    n[n < 0] = 0
    for _ in range(n_iter):
        num = Prg * n[None, :]
        den = num.sum(1, keepdims=True)
        Pgr = np.divide(num, den, out=np.zeros_like(num), where=den > 0)   # P(g|r)
        n = (Pgr * meas[:, None]).sum(0)
    return n


def build(mc, data, rcol, gcol, redges, sf):
    """Return response + vectors (flattened pT x rho) and per-event arrays for closure."""
    ngen = (len(PT_EDGES) - 1) * (len(redges) - 1)
    # ---- MC matched response ----
    s = (mc["passes_both"].astype(bool) & (mc[gcol] > 0) & (mc[rcol] > 0)
         & (mc["gen_pt"] >= 200) & (mc["pt"] >= 200))
    mrec = apply_sf(mc[rcol][s], mc["pt"][s], sf)
    rg = rho(mc[gcol][s], mc["gen_pt"][s]); rr = rho(mrec, mc["pt"][s])
    gi, gok = flat_index(mc["gen_pt"][s], rg, redges)
    ri, rok = flat_index(mc["pt"][s], rr, redges)
    w = mc["weight"][s]; ev = mc["event"][s]
    ok = gok & rok
    gi, ri, w, ev = gi[ok], ri[ok], w[ok], ev[ok]
    R = np.zeros((ngen, ngen)); np.add.at(R, (ri, gi), w)               # R[reco, gen]
    t = np.zeros(ngen); np.add.at(t, gi, w)                            # MC gen truth
    rmc = np.zeros(ngen); np.add.at(rmc, ri, w)                        # MC reco
    # half-sample response/truth for closure
    even = ev % 2 == 0
    Re = np.zeros((ngen, ngen)); np.add.at(Re, (ri[even], gi[even]), w[even])
    te = np.zeros(ngen); np.add.at(te, gi[even], w[even])
    todd = np.zeros(ngen); np.add.at(todd, gi[~even], w[~even])
    rodd = np.zeros(ngen); np.add.at(rodd, ri[~even], w[~even])
    # ---- MC fakes (reco, no gen match) -> fake fraction per reco bin ----
    sf_all = (mc[rcol] > 0) & (mc["pt"] >= 200)
    mrec_all = apply_sf(mc[rcol][sf_all], mc["pt"][sf_all], sf)
    ri_all, rok_all = flat_index(mc["pt"][sf_all], rho(mrec_all, mc["pt"][sf_all]), redges)
    matched_all = mc["passes_both"][sf_all].astype(bool)
    allreco = np.zeros(ngen); np.add.at(allreco, ri_all[rok_all], mc["weight"][sf_all][rok_all])
    fake = np.zeros(ngen)
    fk = rok_all & (~matched_all)
    np.add.at(fake, ri_all[fk], mc["weight"][sf_all][fk])
    phi = np.divide(fake, allreco, out=np.zeros(ngen), where=allreco > 0)  # fake frac
    # ---- data reco (SF applied), fake-subtracted ----
    ds = data["pt"] >= 200
    drec = apply_sf(data[rcol][ds], data["pt"][ds], sf)
    di, dok = flat_index(data["pt"][ds], rho(drec, data["pt"][ds]), redges)
    d = np.zeros(ngen); np.add.at(d, di[dok], data["weight"][ds][dok])
    return dict(R=R, t=t, rmc=rmc, Re=Re, te=te, todd=todd, rodd=rodd,
                phi=phi, d=d, redges=redges)


def panels(redges, title, series, fname, ratio=False, hline=1.0):
    """1x3 pT panels; series = list of (label, color, fmt, values[ngen], err or None)."""
    nrho = len(redges) - 1
    cx = 0.5 * (redges[:-1] + redges[1:]); wid = np.diff(redges)
    fig, axes = plt.subplots(1, 3, figsize=(31, 10.6))
    for ip, ax in enumerate(axes):
        sl = slice(ip * nrho, (ip + 1) * nrho)
        for lab, col, fmt, val, err in series:
            v = val[sl]
            e = err[sl] if err is not None else None
            if fmt == "step":
                ax.step(redges, np.append(v, v[-1]), where="post", color=col, lw=2.6, label=lab)
            else:
                ax.errorbar(cx, v, yerr=e, fmt=fmt, color=col, ms=10, lw=2, label=lab)
        if ratio:
            ax.axhline(hline, color="black", ls="--", lw=1.2); ax.set_ylim(0.5, 1.5)
            ax.set_ylabel("unfolded / truth")
        else:
            ax.set_ylim(bottom=0); ax.set_ylabel(r"$(1/N)\,dN/d\rho$")
        ax.set_xlabel(r"$\rho = 2\log_{10}(m/(p_T R))$"); ax.set_xlim(redges[0], 0)
        hep.cms.label("" if ratio else "Preliminary",
                      data=not ratio, loc=0, ax=ax,
                      rlabel=("59.8 fb$^{-1}$ (2018)" if not ratio else "(2018)"))
        ax.text(0.05, 0.93, f"{title}\n$p_T$ {PT_LAB[ip]} GeV", transform=ax.transAxes,
                va="top", fontsize=20)
        ax.legend(loc="upper right", fontsize=17, frameon=False)
    fig.savefig(os.path.join(OUT, fname), dpi=100, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)


def main():
    mc, data = ntuple("pythia"), ntuple("data")
    for gkey, (gname, rcol, gcol) in GROOM.items():
        redges = RHO_EDGES[gkey]; nrho = len(redges) - 1; wid = np.diff(redges)
        sf = sf_curves(mc, rcol, gcol)
        B = build(mc, data, rcol, gcol, redges, sf)
        pickle.dump({**B, "pt_edges": PT_EDGES, "sf": sf, "grooming": gname},
                    open(os.path.join(PKLDIR, f"rho_adaptive_unfold_inputs_{gkey}_2018.pkl"), "wb"))

        # ---- half-sample closure ----
        unf_odd = dagostini(B["rodd"], B["Re"], B["te"])
        with np.errstate(divide="ignore", invalid="ignore"):
            clo = np.where(B["todd"] > 0, unf_odd / B["todd"], np.nan)
            cloerr = np.where(B["todd"] > 0, np.sqrt(np.abs(unf_odd)) / B["todd"], np.nan)
        panels(redges, f"{gname} — half-sample closure",
               [("unfolded / truth", "#3f90da", "o", clo, cloerr)],
               f"unfold_adaptive_closure_{gkey}.png", ratio=True)

        # ---- data unfold (fake-subtracted) ----
        d_sub = B["d"] * (1.0 - B["phi"])
        unf = dagostini(d_sub, B["R"], B["t"])
        # normalize per pT bin to unit area for the shape comparison
        def norm_by_pt(v):
            out = np.zeros_like(v, float)
            for ip in range(len(PT_EDGES) - 1):
                sl = slice(ip * nrho, (ip + 1) * nrho)
                s = (v[sl] * wid).sum()
                if s > 0:
                    out[sl] = v[sl] / wid / s
            return out
        und = norm_by_pt(unf); tnd = norm_by_pt(B["t"])
        with np.errstate(divide="ignore", invalid="ignore"):
            uerr = np.where(unf > 0, und * np.sqrt(np.abs(unf)) / unf, 0.0)
        panels(redges, f"{gname} — unfolded data",
               [("MC gen (Pythia)", "#e42536", "step", tnd, None),
                ("Unfolded data", "black", "o", und, uerr)],
               f"unfold_adaptive_data_{gkey}.png", ratio=False)


if __name__ == "__main__":
    main()
