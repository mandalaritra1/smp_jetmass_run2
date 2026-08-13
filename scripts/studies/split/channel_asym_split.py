#!/usr/bin/env python
"""Can the leading-pair pT asymmetry alone make dijet and trijet orthogonal?

Proposal under test (Aritra, 2026-08-05): drop the trijet-priority veto entirely
and split the two channels on ONE variable,

    A = |pT1 - pT2| / (pT1 + pT2)   (leading + subleading AK8, same collection)

    dijet  : A < thr   (replacing the current A < 0.3)
    trijet : A > thr   (a NEW cut; the trijet selection has no A cut today)

with thr = 0.1 as the proposed working point. Because both channels compute A
from the same two jets, the split is orthogonal BY CONSTRUCTION -- the residual
overlap is identically zero and is only reported as a closure check. The real
questions this script answers are the costs:

  1. how much of each channel's event yield the split throws away,
  2. how the loss depends on leading-jet pT (at low pT1 the AK8 storage
     threshold pT > ~170 GeV already caps A, so `A < 0.1` bites differently
     across the spectrum),
  3. how much fake/miss migration the boundary creates, i.e. how often an event
     sits on opposite sides of A = thr at gen and reco level. A cut placed in a
     steeply populated region migrates far more than one out in the tail, and
     that migration is what the unfolding pays for.

Everything is computed on the joined 1/1000 event-display sample used by
channel_overlap.py / channel_orthogonality.py, so both channels' true cut bits
are known per event. NO trijet-priority veto is applied anywhere here (the
skims predate those bits, which are identically 0 -- see AGENTS.md).

    python scripts/channel_asym_split.py             # QCD MG+Pythia8 2018
    python scripts/channel_asym_split.py --data      # 2018 JetHT
    python scripts/channel_asym_split.py --plots DIR # + the two figures
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.channel_overlap import (  # noqa: E402
    DIJET_CUTS, TRIJET_CUTS, PT_EDGES, bit, dedupe)
from scripts.studies.split.channel_orthogonality import _topology  # noqa: E402

#### dijet reco cuts EXCLUDING the asymmetry bit (the N-1 sample the A cut acts
#### on). vetoTrijetReco is deliberately absent: this study is "no trijet
#### priority", and the bit is 0 in these skims anyway.
DIJET_RECO_NM1 = ("npv", "METfilters", "twoRecoJet", "recoRap2p5", "recodphi2",
                  "muonIso0p4", "jetId", "hemveto")
DIJET_GEN_NM1 = ("twoGenJet", "genRap2p5", "dphiGen2")
#### the trijet selection has no asymmetry cut, so its own recoTot_seq/genTot_seq
#### bits ARE the N-1 sample for a proposed A > thr.
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def allbits(cutbits, cuts, names):
    m = np.ones(len(cutbits), bool)
    for n in names:
        m &= (cutbits & bit(cuts, n)) != 0
    return m


def asym(pt1, pt2):
    with np.errstate(invalid="ignore"):
        return np.abs(pt1 - pt2) / (pt1 + pt2)


def load_channel(paths):
    """Events table + leading-3-jet reco/gen topology, per channel."""
    out = {k: [] for k in ("key", "cutbits", "weight", "dsid",
                           "pt1", "pt2", "pt3", "njet", "dphimin", "dphi12",
                           "gpt1", "gpt2", "gpt3", "gnjet", "gdphimin",
                           "gdphi12")}
    datasets = set()
    for p in paths:
        sk = pickle.load(open(p, "rb"))["skim"]
        ev = sk["events"]
        n = len(ev["event"].value)
        if n == 0:
            print(f"  !! empty events table: {Path(p).name}", file=sys.stderr)
            continue
        datasets |= set(sk.get("datasets", []))
        ek = np.stack([np.asarray(ev[c].value, dtype=np.int64) for c in
                       ("dataset_id", "run", "lumi", "event")], axis=1)
        topo = _topology(ek, sk["alljets"], n)
        gtopo = _topology(ek, sk["allgenjets"], n)
        out["key"].append(ek)
        out["cutbits"].append(np.asarray(ev["cutbits"].value))
        out["weight"].append(np.asarray(ev["weight"].value))
        out["dsid"].append(np.asarray(ev["dataset_id"].value))
        for k in ("pt1", "pt2", "pt3", "njet", "dphimin", "dphi12"):
            out[k].append(topo[k])
        out["gpt1"].append(gtopo["pt1"])
        out["gpt2"].append(gtopo["pt2"])
        out["gpt3"].append(gtopo["pt3"])
        out["gnjet"].append(gtopo["njet"])
        out["gdphimin"].append(gtopo["dphimin"])
        out["gdphi12"].append(gtopo["dphi12"])
        print(f"  {Path(p).name:<55s} {n:>9d} events")
    return {k: np.concatenate(v) for k, v in out.items()}, datasets


def frac(num, den):
    return num / den if den else float("nan")


def imbalance(pt1, pt2, dphi12):
    """|vec pT1 + vec pT2|, the transverse recoil of the leading pair.

    By momentum conservation this is minus the vector sum of everything else in
    the event, so for a 3-jet event it is essentially pT3 -- which is exactly
    the variable that separates the two channels. It also transfers verbatim to
    Z+jet as |vec pT(Z) + vec pT(jet)|.
    """
    with np.errstate(invalid="ignore"):
        return np.sqrt(pt1 ** 2 + pt2 ** 2 + 2 * pt1 * pt2 * np.cos(dphi12))


#### cuts left when the dphi AND asymmetry cuts are both replaced by |vec pT1 +
#### vec pT2| < X
DIJET_RECO_NOBAL = ("npv", "METfilters", "twoRecoJet", "recoRap2p5",
                    "muonIso0p4", "jetId", "hemveto")
DIJET_GEN_NOBAL = ("twoGenJet", "genRap2p5")
IMB_THRESHOLDS = (100., 150., 185., 200., 250., 300.)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dijet-mc", default="/Users/aritra/cernbox (2)/dijet_skims")
    ap.add_argument("--trijet-mc", default="/Users/aritra/cernbox (2)/trijet_skims/mc")
    ap.add_argument("--dijet-data", default="/Users/aritra/Downloads/dijet_data_skims")
    ap.add_argument("--trijet-data", default="/Users/aritra/cernbox (2)/trijet_skims/data")
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--plots", default=None, help="directory for the two figures")
    args = ap.parse_args(argv)

    if args.data:
        dpaths = sorted(Path(args.dijet_data).glob("*.pkl"))
        tpaths = sorted(Path(args.trijet_data).glob("*.pkl"))
    else:
        dpaths = sorted(Path(args.dijet_mc).glob("*.pkl"))
        tpaths = sorted(Path(args.trijet_mc).glob("*.pkl"))

    print("dijet skims:")
    D, ddsets = load_channel(dpaths)
    print("trijet skims:")
    T, tdsets = load_channel(tpaths)

    dm, tm = dedupe(D["key"]), dedupe(T["key"])
    D = {k: v[dm] for k, v in D.items()}
    T = {k: v[tm] for k, v in T.items()}

    def view(a):
        return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()
    do, to = np.argsort(view(D["key"])), np.argsort(view(T["key"]))
    _, di, ti = np.intersect1d(view(D["key"])[do], view(T["key"])[to],
                               return_indices=True)
    di, ti = do[di], to[ti]
    print(f"\njoin: {len(di)} common events")

    #### The skim `weight` column ALREADY carries xs*lumi*1000/sumw:
    #### HadronicProcessorBase.postprocess calls _scale_skim_weights on it.
    #### channel_overlap.py / channel_orthogonality.py multiply by xs/sumw a
    #### SECOND time, which reweights the HT bins against each other by up to a
    #### factor ~10 (getXSweight ranges 0.51 for HT200to300 to 4.98 for
    #### HT2000toInf) and puts a visible shoulder in the combined pT spectrum.
    #### Use the stored weight as-is. Data is unweighted counts.
    w = np.ones(len(di)) if args.data else D["weight"][di].astype(float)

    pt1 = D["pt1"][di]
    A = asym(D["pt1"][di], D["pt2"][di])
    A_t = asym(T["pt1"][ti], T["pt2"][ti])
    Agen = asym(D["gpt1"][di], D["gpt2"][di])

    #### closure 1: the two channels store the same jets, so A should agree.
    #### It does for 87% of events EXACTLY and disagrees for ~1.6% by more than
    #### 0.01 -- coffea's JER stochastic smearing (CorrectedJetsFactory.
    #### rand_gauss) seeds its PCG64 from the first and last raw pt IN THE
    #### CHUNK, so gen-unmatched jets get different random draws in two
    #### independent productions. Gen jets (never smeared) are bit-identical
    #### here, which confirms it. Consequence: the residual overlap below is a
    #### production artifact -- within ONE processor run the split is exactly
    #### orthogonal.
    both_have = np.isfinite(A) & np.isfinite(A_t)
    if both_have.any():
        dA = np.abs(A[both_have] - A_t[both_have])
        print(f"closure: A(dijet skim) vs A(trijet skim) over "
              f"{int(both_have.sum())} events -- identical {np.mean(dA == 0):.3f}, "
              f"|dA|>0.01 {np.mean(dA > 0.01):.4f}, max {dA.max():.2e} "
              f"(JER stochastic-smearing seeds differ between productions)")

    d_nm1 = allbits(D["cutbits"][di], DIJET_CUTS, DIJET_RECO_NM1) & np.isfinite(A)
    d_tot = (D["cutbits"][di] & bit(DIJET_CUTS, "recoTot_seq")) != 0
    t_tot = (T["cutbits"][ti] & bit(TRIJET_CUTS, "recoTot_seq")) != 0
    #### closure 2: N-1 & A<0.3 must reproduce the stored dijet recoTot_seq
    rebuilt = d_nm1 & (A < 0.3)
    print(f"closure: rebuilt dijet(A<0.3) vs stored recoTot_seq -- "
          f"{int((rebuilt & ~d_tot).sum())} extra, {int((~rebuilt & d_tot).sum())} "
          f"missing, of {int(d_tot.sum())} stored")

    label = "2018 JetHT" if args.data else "QCD MG+Pythia8 2018 (xs-weighted)"
    Wd, Wt = w[d_tot].sum(), w[t_tot].sum()
    Wb = w[d_tot & t_tot].sum()
    print(f"\n=== baseline, NO trijet-priority veto -- {label} ===")
    print(f"dijet (A<0.3): {Wd:.5g}    trijet: {Wt:.5g}    both: {Wb:.5g}")
    print(f"P(tri|di) = {frac(Wb, Wd):.3f}    P(di|tri) = {frac(Wb, Wt):.3f}    "
          f"Jaccard = {frac(Wb, w[d_tot | t_tot].sum()):.3f}")

    #### ---------------- threshold scan ----------------
    #### reference scheme: the trijet-priority veto in production today (dijet
    #### drops any event in the trijet phase space, pt3 > 185 floor).
    pt3 = np.nan_to_num(D["pt3"][di], nan=0.0)
    prio_veto = t_tot & (pt3 > 185.0)
    print(f"\n=== split: dijet A < thr, trijet A > thr (no trijet priority) ===")
    print(f"{'scheme':>22s} {'dijet keep':>11s} {'trijet keep':>12s} "
          f"{'resid. overlap':>15s} {'union yield':>12s}")
    Wu = w[d_tot | t_tot].sum()
    print(f"{'trijet priority (now)':>22s} "
          f"{frac(w[d_tot & ~prio_veto].sum(), Wd):>11.3f} {1.0:>12.3f} "
          f"{frac(w[d_tot & ~prio_veto & t_tot].sum(), Wb):>15.4f} "
          f"{frac(w[(d_tot & ~prio_veto) | t_tot].sum(), Wu):>12.3f}")
    for thr in THRESHOLDS:
        dsel = d_nm1 & (A < thr)
        tsel = t_tot & (A_t > thr)
        print(f"{'A split @ %.2f' % thr:>22s} {frac(w[dsel].sum(), Wd):>11.3f} "
              f"{frac(w[tsel].sum(), Wt):>12.3f} "
              f"{frac(w[dsel & tsel].sum(), Wb):>15.4f} "
              f"{frac(w[dsel | tsel].sum(), Wu):>12.3f}")
    print("  dijet keep     = yield / baseline dijet (A<0.3, no veto)")
    print("  trijet keep    = yield / baseline trijet (no A cut)")
    print("  resid. overlap = events still in BOTH / baseline overlap "
          "(A-split value is the JER-seed artifact, exactly 0 in one run)")

    #### ---------------- per-pT detail at the working point ----------------
    thr = args.thr
    dsel, tsel = d_nm1 & (A < thr), t_tot & (A_t > thr)
    print(f"\n--- per leading-jet pT bin at thr = {thr:g} ---")
    print(f"{'pT bin':>12s} {'dijet keep':>11s} {'trijet keep':>12s} "
          f"{'baseline overlap':>17s} {'A<thr | trijet':>15s}")
    for i in range(len(PT_EDGES) - 1):
        m = (pt1 >= PT_EDGES[i]) & (pt1 < PT_EDGES[i + 1])
        wd, wt = w[m & d_tot].sum(), w[m & t_tot].sum()
        if wd + wt == 0:
            continue
        print(f"{PT_EDGES[i]:>5.0f}-{PT_EDGES[i+1]:<6.0f} "
              f"{frac(w[m & dsel].sum(), wd):>11.3f} "
              f"{frac(w[m & tsel].sum(), wt):>12.3f} "
              f"{frac(w[m & d_tot & t_tot].sum(), wd):>17.3f} "
              f"{frac(w[m & t_tot & (A_t < thr)].sum(), wt):>15.3f}")

    #### ---------------- orphans ----------------
    #### The split is only "free" if every event it removes from one channel is
    #### picked up by the other. It is not: a 2-jet event with 0.1 < A < 0.3
    #### fails the new dijet cut AND fails the trijet selection (no 3rd jet), so
    #### it leaves the measurement entirely.
    print(f"\n--- where the removed events go ---")
    print(f"{'thr':>6s} {'dijet loses':>12s} {'of that, orphaned':>18s} "
          f"{'orphans / dijet':>16s}")
    for t in THRESHOLDS:
        lost = d_tot & ~(A < t)
        orphan = lost & ~t_tot          # not picked up by the trijet channel
        if not lost.any():
            continue
        print(f"{t:>6.2f} {frac(w[lost].sum(), Wd):>12.3f} "
              f"{frac(w[orphan].sum(), w[lost].sum()):>18.3f} "
              f"{frac(w[orphan].sum(), Wd):>16.3f}")
    print("  orphaned = removed from dijet AND not trijet-selected, i.e. gone "
          "from the measurement entirely")

    #### ---------------- vector-sum imbalance ----------------
    #### Alternative proposal: drop BOTH the dphi and the asymmetry cut and
    #### require |vec pT1 + vec pT2| < X instead (dijet), > X (trijet).
    I = imbalance(D["pt1"][di], D["pt2"][di], D["dphi12"][di])
    I_t = imbalance(T["pt1"][ti], T["pt2"][ti], T["dphi12"][ti])
    Igen = imbalance(D["gpt1"][di], D["gpt2"][di], D["gdphi12"][di])
    d_nobal = allbits(D["cutbits"][di], DIJET_CUTS, DIJET_RECO_NOBAL) & np.isfinite(I)
    g_nobal = (allbits(D["cutbits"][di], DIJET_CUTS, DIJET_GEN_NOBAL)
               & np.isfinite(Igen))

    print(f"\n=== alternative: |vec pT1 + vec pT2| < X (dijet) / > X (trijet), "
          f"dphi AND asymmetry cuts REMOVED ===")
    #### how close is the imbalance to being a pT3 cut in disguise?
    m3 = (D["njet"][di] >= 3) & np.isfinite(I) & (pt3 > 0)
    if m3.sum() > 100:
        r = np.corrcoef(I[m3], pt3[m3])[0, 1]
        print(f"  |vec pT1+pT2| vs pT3 on >=3-jet events: corr = {r:.3f}, "
              f"median |I - pT3| = {np.median(np.abs(I[m3] - pt3[m3])):.1f} GeV")
    print(f"{'X [GeV]':>9s} {'dijet keep':>11s} {'trijet keep':>12s} "
          f"{'resid. overlap':>15s} {'orphans/dijet':>14s}")
    for X in IMB_THRESHOLDS:
        dsel = d_nobal & (I < X)
        tsel = t_tot & (I_t > X)
        lost = d_tot & ~dsel
        print(f"{X:>9.0f} {frac(w[dsel].sum(), Wd):>11.3f} "
              f"{frac(w[tsel].sum(), Wt):>12.3f} "
              f"{frac(w[dsel & tsel].sum(), Wb):>15.4f} "
              f"{frac(w[lost & ~t_tot].sum(), Wd):>14.3f}")
    print("  dijet keep is relative to the CURRENT dijet selection "
          "(dphi>2 & A<0.3); >1 means the looser topology cut gains events")

    X = 200.
    print(f"\n--- |vec pT1 + vec pT2| < {X:.0f} GeV: per leading-jet pT bin ---")
    print(f"{'pT bin':>12s} {'dijet keep':>11s} {'vs N-1':>8s} {'trijet keep':>12s} "
          f"{'orphans/dijet':>14s} {'A<0.3 equiv X':>14s}")
    for i in range(len(PT_EDGES) - 1):
        m = (pt1 >= PT_EDGES[i]) & (pt1 < PT_EDGES[i + 1])
        wd, wt = w[m & d_tot].sum(), w[m & t_tot].sum()
        if wd + wt == 0:
            continue
        wn = w[m & d_nobal].sum()
        lost = m & d_tot & ~(d_nobal & (I < X))
        #### for a back-to-back pair, A = |pT1-pT2|/(pT1+pT2) is the same cut as
        #### I < 2*A*pT1 -- i.e. the current asymmetry cut is a pT-PROPORTIONAL
        #### imbalance cut. This column is where A<0.3 sits in GeV, so the two
        #### schemes can be compared bin by bin.
        ptmid = 0.5 * (PT_EDGES[i] + min(PT_EDGES[i + 1], 1200.))
        print(f"{PT_EDGES[i]:>5.0f}-{PT_EDGES[i+1]:<6.0f} "
              f"{frac(w[m & d_nobal & (I < X)].sum(), wd):>11.3f} "
              f"{frac(w[m & d_nobal & (I < X)].sum(), wn):>8.3f} "
              f"{frac(w[m & t_tot & (I_t > X)].sum(), wt):>12.3f} "
              f"{frac(w[lost & ~t_tot].sum(), wd):>14.3f} "
              f"{0.6 * ptmid:>14.0f}")
    print("  'dijet keep' = vs the current selection; 'vs N-1' = vs the same "
          "events with NO topology cut at all (the true efficiency of the cut)")
    print("  'orphans/dijet' = removed from dijet and NOT picked up by trijet")
    print("  'A<0.3 equiv X' = the imbalance the CURRENT asymmetry cut "
          "corresponds to at that pT (2*A*pT1) -- the fixed cut is looser below "
          "~330 GeV and tighter above")

    if not args.data:
        print(f"\n--- |vec pT1 + vec pT2| boundary migration (dijet) ---")
        print(f"{'X [GeV]':>9s} {'fakes':>8s} {'misses':>8s} "
              f"{'X-only fakes':>13s} {'X-only misses':>14s}")
        for X in IMB_THRESHOLDS:
            R, G = d_nobal & (I < X), g_nobal & (Igen < X)
            wR, wG = w[R].sum(), w[G].sum()
            af = R & g_nobal & (Igen >= X)
            am = G & d_nobal & (I >= X)
            print(f"{X:>9.0f} {frac(w[R & ~G].sum(), wR):>8.3f} "
                  f"{frac(w[G & ~R].sum(), wG):>8.3f} "
                  f"{frac(w[af].sum(), wR):>13.3f} {frac(w[am].sum(), wG):>14.3f}")

    #### ---------------- SCALAR sum pT1 + pT2 ----------------
    #### Aritra, 2026-08-05: "scalar sum, just add the pTs". Reported alongside
    #### the vector sum because the two behave completely differently: the
    #### scalar sum is ~2*pT1, i.e. it IS the measurement axis, so cutting on it
    #### slices the pT spectrum rather than the topology.
    S = D["pt1"][di] + D["pt2"][di]
    S_t = T["pt1"][ti] + T["pt2"][ti]
    print(f"\n=== scalar sum pT1 + pT2 ===")
    for name, m, v in (("dijet selected", d_tot, S), ("trijet selected", t_tot, S_t)):
        ok = m & np.isfinite(v)
        q = np.percentile(v[ok], [1, 50, 99])
        print(f"  {name:>16s}: min {v[ok].min():.0f}, 1% {q[0]:.0f}, "
              f"median {q[1]:.0f}, 99% {q[2]:.0f} GeV;  "
              f"fraction < 200 GeV = {frac(w[ok & (v < 200)].sum(), w[ok].sum()):.4f}")
    #### separating power at FIXED pT1: S/pT1 = 2/(1+A), so this is just the
    #### asymmetry again, rescaled -- nothing new.
    ok = np.isfinite(S) & (pt1 > 0)
    r = S / np.where(pt1 > 0, pt1, np.nan)
    for lo, hi in ((300., 400.), (600., 800.)):
        m = ok & (pt1 >= lo) & (pt1 < hi)
        md = frac((w[m & d_tot] * r[m & d_tot]).sum(), w[m & d_tot].sum())
        mt = frac((w[m & t_tot] * r[m & t_tot]).sum(), w[m & t_tot].sum())
        print(f"  <S/pT1> for {lo:.0f} < pT1 < {hi:.0f}: "
              f"dijet {md:.3f}, trijet {mt:.3f}")
    print(f"{'X [GeV]':>9s} {'dijet keep':>11s} {'trijet keep':>12s} "
          f"{'resid. overlap':>15s} {'orphans/dijet':>14s}")
    for X in (400., 500., 600., 800., 1000.):
        dsel = d_nobal & (S < X)
        tsel = t_tot & (S_t > X)
        lost = d_tot & ~dsel
        print(f"{X:>9.0f} {frac(w[dsel].sum(), Wd):>11.3f} "
              f"{frac(w[tsel].sum(), Wt):>12.3f} "
              f"{frac(w[dsel & tsel].sum(), Wb):>15.4f} "
              f"{frac(w[lost & ~t_tot].sum(), Wd):>14.3f}")

    #### ---------------- gen/reco boundary migration ----------------
    if args.data:
        return
    g_nm1 = allbits(D["cutbits"][di], DIJET_CUTS, DIJET_GEN_NM1) & np.isfinite(Agen)
    tg_tot = (T["cutbits"][ti] & bit(TRIJET_CUTS, "genTot_seq")) != 0
    print(f"\n=== fake/miss migration induced by the A boundary ===")
    print("dijet: reco = N-1 & A_reco<thr, gen = N-1 & A_gen<thr "
          "(both N-1 = every other dijet cut)")
    print(f"{'thr':>6s} {'fakes':>8s} {'misses':>8s} {'A-only fakes':>13s} "
          f"{'A-only misses':>14s}")
    for t in THRESHOLDS:
        R, G = d_nm1 & (A < t), g_nm1 & (Agen < t)
        wR, wG = w[R].sum(), w[G].sum()
        #### "A-only": the event would have been matched if not for the boundary
        af = R & g_nm1 & (Agen >= t)
        am = G & d_nm1 & (A >= t)
        print(f"{t:>6.2f} {frac(w[R & ~G].sum(), wR):>8.3f} "
              f"{frac(w[G & ~R].sum(), wG):>8.3f} "
              f"{frac(w[af].sum(), wR):>13.3f} {frac(w[am].sum(), wG):>14.3f}")
    print("  fakes  = reco-selected, not gen-selected (fraction of reco yield)")
    print("  misses = gen-selected, not reco-selected (fraction of gen yield)")
    print("  A-only = passes every other cut at the other level; the boundary "
          "alone caused it")

    #### like-for-like reference: the trijet-priority veto in production, whose
    #### gen mirror is the same kinematic veto with the same pt3 > 185 floor.
    g_pt3 = np.nan_to_num(D["gpt3"][di], nan=0.0)
    gen_veto = (D["gnjet"][di] >= 3) & (D["gdphimin"][di] > 1.0) & (g_pt3 > 185.)
    R = d_nm1 & (A < 0.3) & ~prio_veto
    G = g_nm1 & (Agen < 0.3) & ~gen_veto
    vf = R & g_nm1 & (Agen < 0.3) & gen_veto      # reco-kept, gen-vetoed
    vm = G & d_nm1 & (A < 0.3) & prio_veto        # gen-kept, reco-vetoed
    print(f"\n  reference -- trijet-priority veto (dijet, A stays at 0.3): "
          f"veto-only fakes {frac(w[vf].sum(), w[R].sum()):.3f}, "
          f"veto-only misses {frac(w[vm].sum(), w[G].sum()):.3f}")

    print(f"\ntrijet: reco = trijet recoTot_seq & A_reco>thr, "
          f"gen = trijet genTot_seq & A_gen>thr")
    print(f"{'thr':>6s} {'fakes':>8s} {'misses':>8s} {'A-only fakes':>13s} "
          f"{'A-only misses':>14s}")
    gA = np.isfinite(Agen)
    for t in THRESHOLDS:
        R, G = t_tot & (A_t > t), tg_tot & gA & (Agen > t)
        wR, wG = w[R].sum(), w[G].sum()
        af = R & tg_tot & gA & (Agen <= t)
        am = G & t_tot & (A_t <= t)
        print(f"{t:>6.2f} {frac(w[R & ~G].sum(), wR):>8.3f} "
              f"{frac(w[G & ~R].sum(), wG):>8.3f} "
              f"{frac(w[af].sum(), wR):>13.3f} {frac(w[am].sum(), wG):>14.3f}")

    #### reference: how much migration does the CURRENT scheme cost? The
    #### trijet-priority veto is a kinematic veto, not a boundary in a smooth
    #### variable -- its one-sided migration was measured at ~1.5% (see
    #### dijet_processor._in_trijet_phase_space docstring).
    if args.plots:
        make_plots(Path(args.plots), w, A, A_t, Agen, pt1,
                   d_nm1, d_tot, t_tot, g_nm1, thr)
        make_imbalance_plots(Path(args.plots), w, I, I_t, Igen, pt1, pt3,
                             D["njet"][di], d_nobal, d_tot, t_tot, g_nobal,
                             A, d_nm1)


def make_plots(outdir, w, A, A_t, Agen, pt1, d_nm1, d_tot, t_tot, g_nm1, thr):
    import datetime
    import subprocess
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.CMS)
    outdir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0, 0.7, 71)

    fig, ax = plt.subplots(layout="constrained")
    for m, arr, c, lab in (
            (d_nm1, A, "#5790fc", "dijet N-1 (no A cut)"),
            (t_tot, A_t, "#e42536", "trijet selected"),
            (d_tot & t_tot, A, "#964a8b", "overlap (dijet AND trijet)")):
        h, _ = np.histogram(arr[m], bins=bins, weights=w[m])
        hep.histplot(h / h.sum(), bins, ax=ax, histtype="step", lw=2,
                     color=c, label=lab)
    for x, c, ls in ((thr, "#f89c20", "-"), (0.3, "0.35", ":")):
        ax.axvline(x, color=c, lw=2, ls=ls)
    ax.text(thr + 0.01, 0.55, f"proposed split $A = {thr:g}$",
            transform=ax.get_xaxis_transform(), color="#f89c20", fontsize=18,
            rotation=90, va="center")
    ax.text(0.31, 0.55, "current dijet $A < 0.3$",
            transform=ax.get_xaxis_transform(), color="0.35", fontsize=18,
            rotation=90, va="center")
    ax.set_yscale("log")
    ax.set_xlabel(r"$A = |p_{T,1}-p_{T,2}| / (p_{T,1}+p_{T,2})$")
    ax.set_ylabel("Fraction of events / 0.01")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(top=ax.get_ylim()[1] * 20)
    ax.legend(loc="upper right", fontsize=17)
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
    stamp(fig)
    f1 = outdir / "channel_asym_split_dists.png"
    fig.savefig(f1, dpi=110)
    print("wrote", f1)

    #### keep/loss and migration vs threshold
    thrs = np.linspace(0.02, 0.35, 34)
    dk, tk, fk, ms = [], [], [], []
    Wd, Wt = w[d_tot].sum(), w[t_tot].sum()
    for t in thrs:
        dk.append(w[d_nm1 & (A < t)].sum() / Wd)
        tk.append(w[t_tot & (A_t > t)].sum() / Wt)
        R, G = d_nm1 & (A < t), g_nm1 & (Agen < t)
        fk.append(w[R & g_nm1 & (Agen >= t)].sum() / max(w[R].sum(), 1e-30))
        ms.append(w[G & d_nm1 & (A >= t)].sum() / max(w[G].sum(), 1e-30))
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(thrs, dk, lw=2.5, color="#5790fc", label="dijet yield kept (rel. $A<0.3$)")
    ax.plot(thrs, tk, lw=2.5, color="#e42536", label="trijet yield kept")
    ax.plot(thrs, fk, lw=2.5, ls="--", color="#964a8b",
            label="dijet fakes from the $A$ boundary")
    ax.plot(thrs, ms, lw=2.5, ls=":", color="#f89c20",
            label="dijet misses from the $A$ boundary")
    ax.axvline(thr, color="0.35", lw=1.5, ls="-.")
    ax.set_xlabel(r"split threshold on $A$")
    ax.set_ylabel("Fraction")
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 1.45)
    ax.legend(loc="center right", fontsize=17)
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
    stamp(fig)
    f2 = outdir / "channel_asym_split_scan.png"
    fig.savefig(f2, dpi=110)
    print("wrote", f2)


def make_imbalance_plots(outdir, w, I, I_t, Igen, pt1, pt3, njet,
                         d_nobal, d_tot, t_tot, g_nobal, A, d_nm1):
    """|vec pT1 + vec pT2|: the distribution, and the pT dependence of the cost."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.CMS)
    outdir.mkdir(parents=True, exist_ok=True)
    X = 200.

    bins = np.linspace(0, 700, 71)
    fig, ax = plt.subplots(layout="constrained")
    for m, arr, c, lab in (
            (d_nobal, I, "#5790fc", "dijet, no topology cut"),
            (t_tot, I_t, "#e42536", "trijet selected"),
            (d_tot & t_tot, I, "#964a8b", "overlap (dijet AND trijet)")):
        h, _ = np.histogram(arr[m], bins=bins, weights=w[m])
        hep.histplot(h / h.sum(), bins, ax=ax, histtype="step", lw=2.5,
                     color=c, label=lab)
    ax.axvline(X, color="#f89c20", lw=2.5)
    ax.text(X + 12, 0.55, f"proposed split at {X:.0f} GeV",
            transform=ax.get_xaxis_transform(), color="#f89c20", fontsize=18,
            rotation=90, va="center")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\vec{p}_{T,1} + \vec{p}_{T,2}|$ [GeV]")
    ax.set_ylabel("Fraction of events / 10 GeV")
    ax.set_xlim(0, 700)
    ax.set_ylim(1e-5, ax.get_ylim()[1] * 8)
    ax.legend(loc="upper right", fontsize=17)
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
    stamp(fig)
    f1 = outdir / "channel_imbalance_dists.png"
    fig.savefig(f1, dpi=110)
    print("wrote", f1)

    #### efficiency vs pT1: the high-pT cost, against the A-split alternative
    edges = np.array([200., 250., 300., 400., 500., 600., 800., 1000., 1400.])
    ctr = 0.5 * (edges[1:] + edges[:-1])
    def eff(base, sel):
        out = []
        for i in range(len(edges) - 1):
            m = (pt1 >= edges[i]) & (pt1 < edges[i + 1]) & base
            out.append(frac(w[m & sel].sum(), w[m].sum()))
        return np.array(out)
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(ctr, eff(d_nobal, I < X), "o-", lw=2.5, ms=8, color="#5790fc",
            label=rf"dijet: $|\vec{{p}}_{{T,1}}+\vec{{p}}_{{T,2}}| < {X:.0f}$ GeV")
    ax.plot(ctr, eff(d_nm1, A < 0.10), "s--", lw=2.5, ms=8, color="#964a8b",
            label=r"dijet: $A < 0.1$")
    ax.plot(ctr, eff(d_nm1, A < 0.30), "^:", lw=2.5, ms=8, color="#9c9ca1",
            label=r"dijet: $A < 0.3$ (current)")
    ax.plot(ctr, eff(t_tot, I_t > X), "o-", lw=2.5, ms=8, color="#e42536",
            label=rf"trijet: $|\vec{{p}}_{{T,1}}+\vec{{p}}_{{T,2}}| > {X:.0f}$ GeV")
    ax.set_xlabel(r"$p_{T,1}$ [GeV]")
    ax.set_ylabel("Selection efficiency")
    ax.set_xlim(200, 1400)
    ax.set_ylim(0, 1.45)
    ax.legend(loc="lower left", fontsize=16)
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
    stamp(fig)
    f2 = outdir / "channel_imbalance_eff_vs_pt.png"
    fig.savefig(f2, dpi=110)
    print("wrote", f2)


def stamp(fig):
    import datetime
    import subprocess
    ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                         capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[3])).stdout.strip()
    fig.text(0.99, 0.002,
             f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  "
             f"inputs: dijet rho_skim10 + trijet rho_skim event tables, 2018",
             ha="right", va="bottom", fontsize=7, color="0.45", family="monospace")


if __name__ == "__main__":
    main()
