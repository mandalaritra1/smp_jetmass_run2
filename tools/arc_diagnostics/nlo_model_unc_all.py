#!/usr/bin/env python
"""Three model-uncertainty estimates for the rho unfolding (groomed + ungroomed), on data.

Methods (all exploit that only a WITHIN-gen-bin-varying reweight changes P(reco|gen)):
  A) MC non-closure        : reweight NLO gen->Herwig, fold through nominal resp, unfold
                             with nominal resp, compare to Herwig truth.  (no data)
  B) data, nom vs Herwig   : unfold the SAME fake-corrected data with R_nom and R_hw.
  C) data, nom vs data-rew : unfold data with R_nom, reweight NLO to it, unfold again.

Reweight = ITERATED Herwig/NLO density ratio, anchored on well-populated bins
(validated <1% non-closure -- see nlo_reweight_validation / _beforeafter). The floor
catch-all bin (sparse, near the m_g>2 edge) is FLAGGED: B there is data-stat-dominated,
so the quoted model systematic uses the stable MC non-closure A in that bin.

Backgrounds neglected (cancel in the same-data ratio); fakes corrected (NLO fiducial).

Run in the unfold venv:
  cd ~/Projects/unfold && source scripts/setup_root.sh && source .venv/bin/activate
  python ~/Projects/smp_jetmass_run2/tools/arc_diagnostics/nlo_model_unc_all.py
"""
import os, sys, glob
from array import array
import numpy as np
import awkward as ak
import ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

ROOT.gErrorIgnoreLevel = ROOT.kError
hep.style.use(hep.style.CMS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nlo_tunfold_nested as T

DATA = "/Users/aritra/Projects/unfold/inputs/zjet/nlo_skims/zjet_data_skims"
SP = ("/private/tmp/claude-501/-Users-aritra-Projects-smp-jetmass-run2/"
      "5475621c-e50a-4ac9-9c83-0b954dcd7fe5/scratchpad/herwig_gen_shapes.npz")
OUTDIR = T.OUTDIR
RHO_GEN = {"g": T.RHO_GEN,
           "u": [np.array([-6.0, -1.3, -0.9, -0.5, 0.0]),
                 np.array([-6.0, -1.9, -1.3, -0.9, -0.5, 0.0]),
                 np.array([-6.0, -1.9, -1.6, -1.3, -1.1, -0.9, -0.7, -0.5, 0.0])]}
OBSLAB = {"g": "groomed", "u": "ungroomed"}


def load_data():
    pt, mg, mu = [], [], []
    for f in sorted(glob.glob(DATA + "/*/merged.parquet")):
        a = ak.from_parquet(f)
        pt.append(ak.to_numpy(ak.fill_none(a.pt, np.nan)))
        mg.append(ak.to_numpy(ak.fill_none(a.msoftdrop, np.nan)))
        mu.append(ak.to_numpy(ak.fill_none(a.mass, np.nan)))
    return np.concatenate(pt), np.concatenate(mg), np.concatenate(mu)


def iter_reweight(rg, w, edges, target, niter=6):
    """Continuous NLO->target reweight, iterated to closure, anchored on populated bins."""
    bw = np.diff(edges); cx = 0.5 * (edges[:-1] + edges[1:])
    nlo0, _ = np.histogram(rg, bins=edges, weights=w); pop = nlo0 > 0.02 * nlo0.max()
    hd = np.where(pop, target, 0.0); hd = hd / max(hd.sum(), 1e-30) / bw
    rw = np.ones_like(cx)
    mk = lambda n: (lambda x: np.interp(x, cx[pop], n[pop], left=n[pop][0], right=n[pop][-1]))
    for _ in range(niter):
        cur, _ = np.histogram(rg, bins=edges, weights=w * mk(rw)(rg))
        cd = np.where(pop, cur, 0.0); cd = cd / max(cd.sum(), 1e-30) / bw
        with np.errstate(divide="ignore", invalid="ignore"):
            rw = np.clip(np.where((cd > 0) & pop, rw * hd / cd, rw), 0.2, 5.0)
    return mk(rw)


def th1_from(vals, errs):
    h = ROOT.TH1D("d", ";reco", len(T.RECO_FINE) - 1, array('d', T.RECO_FINE)); h.Sumw2()
    for i, (v, e) in enumerate(zip(vals, errs)):
        h.SetBinContent(i + 1, v); h.SetBinError(i + 1, e)
    return h


def uf(hResp, hData):
    return T.val_err(T.unfold(hResp, hData, 0.0, reg=False), full_err=False)[0]


def main():
    z = np.load(SP); c = T.load(); d_pt, d_mg, d_mu = load_data()
    print(f"NLO matched {len(c['gpt']):,d} | data {len(d_pt):,d}\n")
    out_npz = {}
    for obs in ("g", "u"):
        gm = c["gmg"] if obs == "g" else c["gmu"]; rm = c["rmg"] if obs == "g" else c["rmu"]
        dm = d_mg if obs == "g" else d_mu
        edges_t = z[f"edges_{obs}"]
        res = []
        print(f"=== {OBSLAB[obs]} ===")
        for ip, (plo, phi) in enumerate(zip(T.PT_EDGES[:-1], T.PT_EDGES[1:])):
            ge = RHO_GEN[obs][ip]
            sa = (c["rpt"] >= plo) & (c["rpt"] < phi) & (c["rmg"] > T.FLOOR)   # floor on GROOMED
            rr_a = T.rho(rm[sa], c["rpt"][sa]); w_a = c["w"][sa]
            rg_a = T.rho(gm[sa], c["gpt"][sa])
            mfid = ((c["gpt"][sa] >= plo) & (c["gpt"][sa] < phi) & (c["gmg"][sa] > T.FLOOR)
                    & np.isfinite(rg_a) & np.isfinite(rr_a) & (rg_a >= ge[0]) & (rg_a <= ge[-1]))
            rg_m, rr_m, w_m = rg_a[mfid], rr_a[mfid], w_a[mfid]

            allh, _ = np.histogram(rr_a, bins=T.RECO_FINE, weights=w_a)
            mh, _ = np.histogram(rr_m, bins=T.RECO_FINE, weights=w_m)
            with np.errstate(divide="ignore", invalid="ignore"):
                fake = np.where(allh > 0, 1.0 - mh / allh, 0.0)

            rw_hw = iter_reweight(rg_m, w_m, edges_t, z[f"hw_{obs}_pt{ip+1}"])
            w_hw = w_m * rw_hw(rg_m)
            R_nom = T.th2_response(rg_m, rr_m, w_m, ge)
            R_hw = T.th2_response(rg_m, rr_m, w_hw, ge)

            # A: MC non-closure
            uA = uf(R_nom, T.th1_reco(rr_m, w_hw))
            truth_hw, _ = np.histogram(rg_m, bins=ge, weights=w_hw)
            uncA = np.where(truth_hw > 0, uA / truth_hw - 1, np.nan)

            # data, fake-corrected
            ds = (d_pt >= plo) & (d_pt < phi) & (d_mg > T.FLOOR)   # floor on GROOMED reco mass
            d_rr = T.rho(dm[ds], d_pt[ds])
            ddata, _ = np.histogram(d_rr[np.isfinite(d_rr)], bins=T.RECO_FINE)
            hData = th1_from(ddata * (1 - fake), np.sqrt(np.maximum(ddata, 0)) * (1 - fake))

            # B: data nom vs Herwig
            uB_nom = uf(R_nom, hData); uB_hw = uf(R_hw, hData)
            uncB = np.where(uB_nom > 0, uB_hw / uB_nom - 1, np.nan)

            # C: data nom vs data-reweighted
            truth_nom, _ = np.histogram(rg_m, bins=ge, weights=w_m)
            cx = 0.5 * (ge[:-1] + ge[1:])
            ratio_c = np.where(truth_nom > 0, uB_nom / truth_nom, 1.0)
            ratio_c *= truth_nom.sum() / max(uB_nom.sum(), 1e-30)
            rw_c = iter_reweight(rg_m, w_m, ge, np.clip(ratio_c, 0.2, 5.0) * truth_nom)
            R_data = T.th2_response(rg_m, rr_m, w_m * rw_c(rg_m), ge)
            uC = uf(R_data, hData)
            uncC = np.where(uB_nom > 0, uC / uB_nom - 1, np.nan)

            # quoted: A (stable) in the floor catch-all bin, else max(|A|,|B|); C separate
            quoted = np.where(np.arange(len(ge) - 1) == 0, np.abs(uncA),
                              np.maximum(np.abs(uncA), np.abs(uncB)))
            res.append(dict(ge=ge, A=uncA, B=uncB, C=uncC, quoted=quoted))
            print(f"  pt{ip+1} {len(ge)-1:2d}b  |A|max {np.nanmax(np.abs(uncA))*100:4.1f}"
                  f"  |B|max {np.nanmax(np.abs(uncB))*100:5.1f}  |C|max {np.nanmax(np.abs(uncC))*100:4.1f}"
                  f"  quoted(no floor) {np.nanmax(quoted[1:])*100:4.1f}%")
            for k in ("A", "B", "C", "quoted"):
                out_npz[f"{obs}_pt{ip+1}_{k}"] = res[-1][k]
            out_npz[f"{obs}_pt{ip+1}_edges"] = ge

        fig, axes = plt.subplots(1, 3, figsize=(31, 10.6), layout="constrained")
        for ip, ax in enumerate(axes):
            d = res[ip]; ge = d["ge"]; cx = 0.5 * (ge[:-1] + ge[1:])
            for key, col, lab in [("A", "#7a21dd", "A: MC non-closure"),
                                  ("B", "#f89c20", "B: data nom/Herwig"),
                                  ("C", "#3f90da", "C: data prior")]:
                ax.step(cx, np.clip(d[key], -0.25, 0.25) * 100, where="mid", color=col, lw=2.6, label=lab)
            ax.axhline(0, color="black", lw=1.0)
            ax.set_ylim(-25, 25); ax.set_xlim(ge[0], 0)
            ax.set_ylabel("model uncertainty [%]"); ax.set_xlabel(rf"$\log_{{10}}(\rho_{obs}^2)$")
            ax.grid(alpha=0.25)
            hep.cms.label("", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
            ax.text(0.05, 0.95, f"{OBSLAB[obs]} {T.PT_LAB[ip]} ({len(ge)-1} bins)",
                    transform=ax.transAxes, va="top", fontsize=17)
            if ip == 0:
                ax.legend(loc="lower left", fontsize=13, frameon=False)
        out = os.path.join(OUTDIR, f"nlo_model_unc_all_{obs}.png")
        fig.savefig(out, dpi=90); plt.close(fig)
        print(f"wrote {out}\n")
    np.savez(os.path.join(OUTDIR, "model_unc_summary.npz"), **out_npz)


if __name__ == "__main__":
    main()
