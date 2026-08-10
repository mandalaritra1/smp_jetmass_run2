#!/usr/bin/env python
"""Pair-level (theory-friendly) dijet/trijet splits: no third-jet variable.

Aritra, 2026-08-06: the explicit third-jet veto is dropped -- a pT3 veto is a
jet veto (clustering logs / NGLs) and hard for theorists; the split must be a
function of the LEADING PAIR only. Candidates, dijet side (trijet = exact
complement on the same quantity from its own jets):

  B1(X):        dphi12 > pi - X/pT1
  B1c(X, cap):  dphi12 > min(pi - X/pT1, cap)   [caps the angular tightening
                at high pT, where B1's fixed recoil scale over-tightens]
  I(X):         |vec pT1 + vec pT2| < X          [the recoil / qT of the pair;
                equally resummation-friendly, sees unbalanced pairs too]
  Ic(X, c):     |pT,jj| < max(X, c*pT1)

All replace the current dphi>2 AND A<0.3 topology cuts (baseline = every
other dijet cut). Metrics as in channel_asym_split.py; weights are the stored
skim weights (already xs-scaled -- never re-apply xs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.channel_overlap import (  # noqa: E402
    DIJET_CUTS, TRIJET_CUTS, PT_EDGES, bit, dedupe)
from scripts.channel_asym_split import (  # noqa: E402
    load_channel, allbits, imbalance, frac,
    DIJET_RECO_NOBAL, DIJET_GEN_NOBAL)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dijet-mc", default="/Users/aritra/cernbox (2)/dijet_skims")
    ap.add_argument("--trijet-mc", default="/Users/aritra/cernbox (2)/trijet_skims/mc")
    ap.add_argument("--dijet-data", default="/Users/aritra/Downloads/dijet_data_skims")
    ap.add_argument("--trijet-data", default="/Users/aritra/cernbox (2)/trijet_skims/data")
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--plot-fate", default=None,
                    help="directory for the overlap-fate figure")
    args = ap.parse_args(argv)

    if args.data:
        dpaths = sorted(Path(args.dijet_data).glob("*.pkl"))
        tpaths = sorted(Path(args.trijet_data).glob("*.pkl"))
    else:
        dpaths = sorted(Path(args.dijet_mc).glob("*.pkl"))
        tpaths = sorted(Path(args.trijet_mc).glob("*.pkl"))
    print("dijet skims:")
    D, _ = load_channel(dpaths)
    print("trijet skims:")
    T, _ = load_channel(tpaths)

    dm, tm = dedupe(D["key"]), dedupe(T["key"])
    D = {k: v[dm] for k, v in D.items()}
    T = {k: v[tm] for k, v in T.items()}

    def view(a):
        return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()
    do, to = np.argsort(view(D["key"])), np.argsort(view(T["key"]))
    _, di, ti = np.intersect1d(view(D["key"])[do], view(T["key"])[to],
                               return_indices=True)
    di, ti = do[di], to[ti]
    w = np.ones(len(di)) if args.data else D["weight"][di].astype(float)

    pt1, dphi = D["pt1"][di], D["dphi12"][di]
    pt1_t, dphi_t = T["pt1"][ti], T["dphi12"][ti]
    I = imbalance(D["pt1"][di], D["pt2"][di], dphi)
    I_t = imbalance(T["pt1"][ti], T["pt2"][ti], dphi_t)
    gpt1, gdphi = D["gpt1"][di], D["gdphi12"][di]
    Igen = imbalance(D["gpt1"][di], D["gpt2"][di], gdphi)

    d_tot = (D["cutbits"][di] & bit(DIJET_CUTS, "recoTot_seq")) != 0
    t_tot = (T["cutbits"][ti] & bit(TRIJET_CUTS, "recoTot_seq")) != 0
    d_nobal = allbits(D["cutbits"][di], DIJET_CUTS, DIJET_RECO_NOBAL) & np.isfinite(dphi)
    g_nobal = (allbits(D["cutbits"][di], DIJET_CUTS, DIJET_GEN_NOBAL)
               & np.isfinite(gdphi))
    Wd, Wt = w[d_tot].sum(), w[t_tot].sum()

    def cutfn_b1(X, cap=None):
        def f(p, d):
            thr = np.pi - X / p
            if cap is not None:
                thr = np.minimum(thr, cap)
            return d > thr
        return f

    def cutfn_i(X, c=None):
        def f(p, i):
            thr = X if c is None else np.maximum(X, c * p)
            return i < thr
        return f

    #### Sal, 2026-08-06: pT1 is soft-sensitive (leading-jet label swaps under
    #### soft emission near pT1 ~ pT2; pT1 itself smears with full single-jet
    #### JER). Variant: same cut with the SWAP-SYMMETRIC average
    #### ptavg = (pT1+pT2)/2, which is continuous under relabeling and has
    #### ~1/sqrt(2) the threshold smearing. ptavg <= pT1, so matched working
    #### points sit at slightly smaller X.
    ptavg = 0.5 * (D["pt1"][di] + D["pt2"][di])
    ptavg_t = 0.5 * (T["pt1"][ti] + T["pt2"][ti])
    gptavg = 0.5 * (D["gpt1"][di] + D["gpt2"][di])

    cands = [
        ("B1(150)",        "ang", cutfn_b1(150.)),
        ("B1(200) ~ G1",   "ang", cutfn_b1(200.)),
        ("B1(250)",        "ang", cutfn_b1(250.)),
        ("B1(300)",        "ang", cutfn_b1(300.)),
        ("B1c(200, 2.90)", "ang", cutfn_b1(200., 2.90)),
        ("B1c(250, 2.90)", "ang", cutfn_b1(250., 2.90)),
        ("B1c(250, 2.95)", "ang", cutfn_b1(250., 2.95)),
        ("B1c(300, 2.90)", "ang", cutfn_b1(300., 2.90)),
        ("B1c(300, 2.95)", "ang", cutfn_b1(300., 2.95)),
        ("A1avg(150)",     "avg", cutfn_b1(150.)),
        ("A1avg(175)",     "avg", cutfn_b1(175.)),
        ("A1avg(185)",     "avg", cutfn_b1(185.)),
        ("A1avg(200)",     "avg", cutfn_b1(200.)),
        ("A1avg(230)",     "avg", cutfn_b1(230.)),
        ("A1avg(250)",     "avg", cutfn_b1(250.)),
        ("I<200",          "imb", cutfn_i(200.)),
        ("I<250",          "imb", cutfn_i(250.)),
        ("Ic(200, 0.30)",  "imb", cutfn_i(200., 0.30)),
        ("Ic(200, 0.35)",  "imb", cutfn_i(200., 0.35)),
        ("Ic(250, 0.30)",  "imb", cutfn_i(250., 0.30)),
    ]

    label = "2018 JetHT" if args.data else "QCD MG+Pythia8 2018 (xs-weighted)"
    print(f"\n=== pair-level splits (no third-jet variable) -- {label} ===")
    print(f"{'rule':>16s} {'di keep':>8s} {'tri keep':>9s} {'orphans':>8s} "
          f"{'fakes':>6s} {'misses':>7s}  per-pT dijet keep (8 bins)")
    for name, kind, f in cands:
        if kind == "imb":
            var, var_t, var_g = I, I_t, Igen
        else:
            var, var_t, var_g = dphi, dphi_t, gdphi
        if kind == "avg":
            p_r, p_rt, p_g = ptavg, ptavg_t, gptavg
        else:
            p_r, p_rt, p_g = pt1, pt1_t, gpt1
        dsel = d_nobal & f(p_r, var)
        tsel = t_tot & ~f(p_rt, var_t)
        orph = frac(w[d_tot & ~dsel & ~t_tot].sum(), Wd)
        if args.data:
            fk = ms = float("nan")
        else:
            G = g_nobal & f(p_g, var_g)
            R = dsel
            fk = frac(w[R & g_nobal & ~f(p_g, var_g)].sum(), w[R].sum())
            ms = frac(w[G & d_nobal & ~f(p_r, var)].sum(), w[G].sum())
        pp = []
        for i in range(len(PT_EDGES) - 1):
            m = (pt1 >= PT_EDGES[i]) & (pt1 < PT_EDGES[i + 1])
            pp.append(frac(w[m & dsel].sum(), w[m & d_tot].sum()))
        print(f"{name:>16s} {frac(w[dsel].sum(), Wd):>8.3f} "
              f"{frac(w[tsel].sum(), Wt):>9.3f} {orph:>8.3f} "
              f"{fk:>6.3f} {ms:>7.3f}  " + "/".join(f"{x:.2f}" for x in pp))
    #### fate of the baseline overlap and of the lost trijet events
    Wb = w[d_tot & t_tot].sum()
    print(f"\n=== where the baseline overlap (di AND tri, {frac(Wb, Wd):.3f} of "
          f"dijet / {frac(Wb, Wt):.3f} of trijet) goes, and trijet losses ===")
    print(f"{'rule':>16s} {'ovl->dijet':>11s} {'ovl->trijet':>12s} "
          f"{'ovl->neither':>13s} {'tri->dijet':>11s} {'tri orphaned':>13s}")
    for name, kind, f in cands:
        var, var_t = (I, I_t) if kind == "imb" else (dphi, dphi_t)
        p_r, p_rt = (ptavg, ptavg_t) if kind == "avg" else (pt1, pt1_t)
        dsel = d_nobal & f(p_r, var)
        tsel = t_tot & ~f(p_rt, var_t)
        both = d_tot & t_tot
        lost_t = t_tot & ~tsel
        print(f"{name:>16s} {frac(w[both & dsel].sum(), Wb):>11.3f} "
              f"{frac(w[both & tsel].sum(), Wb):>12.3f} "
              f"{frac(w[both & ~dsel & ~tsel].sum(), Wb):>13.3f} "
              f"{frac(w[lost_t & dsel].sum(), Wt):>11.3f} "
              f"{frac(w[lost_t & ~dsel].sum(), Wt):>13.3f}")
    print("  ovl->X = fraction of the baseline overlap assigned to X by the split")
    print("  tri->dijet / tri orphaned = lost trijet yield that lands in dijet / "
          "leaves the measurement")

    #### per-pT fate of the overlap for the two adopted X values
    print(f"\n=== overlap fate per pT1 bin (fraction of that bin's overlap) ===")
    fate = {}
    for X in (200., 250.):
        f = cutfn_b1(X)
        dsel = d_nobal & f(pt1, dphi)
        tsel = t_tot & ~f(pt1_t, dphi_t)
        rows = []
        print(f"-- dphi12 > pi - {X:.0f}/pT1 --")
        print(f"{'pT1 bin':>12s} {'N_raw':>6s} {'->dijet':>8s} {'->trijet':>9s} "
              f"{'->neither':>10s}")
        for i in range(len(PT_EDGES) - 1):
            m = (pt1 >= PT_EDGES[i]) & (pt1 < PT_EDGES[i + 1]) & d_tot & t_tot
            wb = w[m].sum()
            if wb <= 0:
                rows.append((np.nan, np.nan, 0))
                continue
            fd = w[m & dsel].sum() / wb
            ft = w[m & tsel].sum() / wb
            neff = wb ** 2 / max((w[m] ** 2).sum(), 1e-30)
            rows.append((fd, ft, neff))
            print(f"{PT_EDGES[i]:>5.0f}-{PT_EDGES[i+1]:<6.0f} {int(m.sum()):>6d} "
                  f"{fd:>8.3f} {ft:>9.3f} {1 - fd - ft:>10.3f}")
        fate[X] = rows

    if args.plot_fate:
        import matplotlib.pyplot as plt
        import mplhep as hep
        import datetime as _dt
        import subprocess as _sp
        hep.style.use(hep.style.CMS)
        fig, ax = plt.subplots(layout="constrained")
        ctr = 0.5 * (PT_EDGES[:-1] + np.minimum(PT_EDGES[1:], 1000.))
        xerr = (np.minimum(PT_EDGES[1:], 1000.) - PT_EDGES[:-1]) / 2
        for X, col, mk in ((200., "#5790fc", "o"), (250., "#e42536", "s")):
            fd = np.array([r[0] for r in fate[X]])
            ne = np.array([max(r[2], 1.) for r in fate[X]])
            err = np.sqrt(np.clip(fd * (1 - fd), 0, None) / ne)
            ax.errorbar(ctr, fd, xerr=xerr, yerr=err, fmt=mk, ms=8, lw=2,
                        color=col, label=rf"$X = {X:.0f}$ GeV: overlap $\to$ dijet")
        ax.axhline(0.5, color="0.6", lw=1, ls=":")
        ax.set_xlabel(r"$p_{T,1}$ [GeV]")
        ax.set_ylabel(r"fraction of overlap assigned to dijet")
        ax.set_xlim(200, 1050)
        ax.set_ylim(0, 1.35)
        ax.legend(loc="upper left", fontsize=17)
        hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
        ver = _sp.run(["git", "describe", "--tags", "--always", "--dirty"],
                      capture_output=True, text=True,
                      cwd=str(Path(__file__).resolve().parent.parent)).stdout.strip()
        fig.text(0.99, 0.002,
                 f"{_dt.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  "
                 f"inputs: dijet+trijet rho_skim event tables, 2018 MC",
                 ha="right", va="bottom", fontsize=7, color="0.45",
                 family="monospace")
        out = Path(args.plot_fate) / "overlap_fate_vs_pt.png"
        fig.savefig(out, dpi=110)
        print("wrote", out)

    #### trijet per-pT for the two leading candidates
    for name, kind, f in cands:
        if name not in ("B1c(250, 2.90)", "Ic(200, 0.30)", "B1(200) ~ G1", "I<200",
                        "A1avg(185)", "A1avg(230)"):
            continue
        var_t = I_t if kind == "imb" else dphi_t
        p_rt = ptavg_t if kind == "avg" else pt1_t
        pp = []
        for i in range(len(PT_EDGES) - 1):
            m = (pt1_t >= PT_EDGES[i]) & (pt1_t < PT_EDGES[i + 1])
            pp.append(frac(w[m & t_tot & ~f(p_rt, var_t)].sum(),
                           w[m & t_tot].sum()))
        print(f"per-pT trijet keep {name:>16s}: " + "/".join(f"{x:.2f}" for x in pp))


if __name__ == "__main__":
    main()
