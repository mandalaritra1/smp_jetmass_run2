#!/usr/bin/env python
"""How to make the dijet and trijet channels orthogonal, quantified from skims.

The overlap study (channel_overlap.py) showed trijet-selected events are ~90%
a subset of dijet-selected ones. To orthogonalize, either

  (A) trijet takes priority: dijet vetoes any event passing the trijet reco
      selection (exact, uses the trijet cutbits from the join), or
  (B) a kinematic cut on the dijet side removes the 3-jet phase space:
        - require min pairwise dphi of the 3 leading jets < 1.0 whenever a 3rd
          AK8 jet exists (the exact complement of the trijet dphimin cut),
        - or a 3rd-jet pt veto (pt3 < X),
        - or tighten the back-to-back dphi12 cut.

For each candidate this script reports, per leading-jet pt bin:

    residual   = overlap events surviving the candidate veto / original overlap
    cost       = dijet-only events (NOT trijet-selected) killed by the veto
    dijet loss = total dijet events removed (overlap removal + collateral)

computed on the SAME joined 1/1000 event sample as channel_overlap.py, so both
channels' true pass bits are known per event. Topology variables come from the
alljets table (nominal-JEC pt, full AK8 collection).

    python scripts/channel_orthogonality.py            # QCD MC, xs-weighted
    python scripts/channel_orthogonality.py --data     # 2018 JetHT
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.channel_overlap import (  # noqa: E402
    DIJET_CUTS, TRIJET_CUTS, PT_EDGES, bit, dedupe, xs_lut)


def wrap_dphi(a, b):
    d = np.abs(a - b)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def _topology(ek, table, n):
    """Per-event 3-leading-jet (pt1..3, dphi12, dphimin, njet) from a
    row-per-jet table keyed like the events table."""
    jk = np.stack([np.asarray(table[c].value, dtype=np.int64) for c in
                   ("dataset_id", "run", "lumi", "event")], axis=1)
    jidx = np.asarray(table["jetidx"].value)
    jpt = np.asarray(table["pt"].value)
    jphi = np.asarray(table["phi"].value)

    def view(a):
        return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()

    #### map each jet row to its event row (the table only stores events that
    #### are in the events table, but guard anyway)
    order = np.argsort(view(ek), kind="stable")
    if len(jk):
        pos = np.searchsorted(view(ek)[order], view(jk))
        pos = np.clip(pos, 0, n - 1)
        hit = view(ek)[order][pos] == view(jk)
        evrow = order[pos]
    else:
        hit = np.zeros(0, bool)
        evrow = np.zeros(0, np.int64)

    pts = np.full((n, 3), np.nan, np.float32)
    phis = np.full((n, 3), np.nan, np.float32)
    njet = np.zeros(n, np.int32)
    np.add.at(njet, evrow[hit], 1)
    lead = hit & (jidx < 3)
    pts[evrow[lead], jidx[lead]] = jpt[lead]
    phis[evrow[lead], jidx[lead]] = jphi[lead]

    d12 = wrap_dphi(phis[:, 0], phis[:, 1])
    d13 = wrap_dphi(phis[:, 0], phis[:, 2])
    d23 = wrap_dphi(phis[:, 1], phis[:, 2])
    with np.errstate(invalid="ignore"):
        dphimin = np.nanmin(np.stack([d12, d13, d23]), axis=0)
    return {"pt1": pts[:, 0], "pt2": pts[:, 1], "pt3": pts[:, 2],
            "dphi12": d12, "dphimin": dphimin, "njet": njet}


def load_channel(paths):
    """Events table + per-event topology from alljets (and allgenjets, MC)."""
    out = {k: [] for k in ("key", "cutbits", "weight", "dsid",
                           "pt1", "pt2", "pt3", "dphi12", "dphimin", "njet",
                           "gen_pt3", "gen_dphimin", "gen_njet")}
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
        for k in ("pt1", "pt2", "pt3", "dphi12", "dphimin", "njet"):
            out[k].append(topo[k])
        out["gen_pt3"].append(gtopo["pt3"])
        out["gen_dphimin"].append(gtopo["dphimin"])
        out["gen_njet"].append(gtopo["njet"])
        print(f"  {Path(p).name:<55s} {n:>9d} events")
    return {k: np.concatenate(v) for k, v in out.items()}, datasets


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dijet-mc", default="/Users/aritra/cernbox (2)/dijet_skims")
    ap.add_argument("--trijet-mc", default="/Users/aritra/cernbox (2)/trijet_skims/mc")
    ap.add_argument("--dijet-data", default="/Users/aritra/Downloads/dijet_data_skims")
    ap.add_argument("--trijet-data", default="/Users/aritra/cernbox (2)/trijet_skims/data")
    ap.add_argument("--data", action="store_true")
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

    passd = (D["cutbits"][di] & bit(DIJET_CUTS, "recoTot_seq")) != 0
    passt = (T["cutbits"][ti] & bit(TRIJET_CUTS, "recoTot_seq")) != 0

    if args.data:
        w = np.ones(len(di))
    else:
        lut = xs_lut(ddsets | tdsets)
        w = D["weight"][di] * np.array(
            [lut.get(d) or 0.0 for d in D["dsid"][di]])

    pt1 = D["pt1"][di]
    pt3 = np.nan_to_num(D["pt3"][di], nan=0.0)
    dphimin = D["dphimin"][di]
    dphi12 = D["dphi12"][di]
    njet = D["njet"][di]
    has3 = njet >= 3

    #### candidate dijet-side vetoes: True = event REMOVED from dijet
    cands = {
        "A: veto trijet pass (exact)": passt,
        "A185: trijet pass & pt3>185": passt & (pt3 > 185.0),
        "A200: trijet pass & pt3>200": passt & (pt3 > 200.0),
        "B1: veto njet>=3 & dphimin>1": has3 & (dphimin > 1.0),
        "B2: veto pt3 > 200": pt3 > 200.0,
        "B3: veto pt3 > 0.3*pt1": pt3 > 0.3 * pt1,
        "B4: dphi12 > 2.6 instead of 2.0": ~(dphi12 > 2.6),
        "B5: dphi12 > 2.8 instead of 2.0": ~(dphi12 > 2.8),
        "B6: dphi12 > 3.0 instead of 2.0": ~(dphi12 > 3.0),
    }

    label = "2018 JetHT" if args.data else "QCD MG+Pythia8 2018 (xs-weighted)"
    both = passd & passt
    only = passd & ~passt
    Wb, Wo = w[both].sum(), w[only].sum()
    print(f"\n=== dijet-side orthogonality candidates -- {label} ===")
    print(f"overlap (dijet AND trijet): {Wb:.4g}   "
          f"dijet-only: {Wo:.4g}   overlap frac of dijet: {Wb/(Wb+Wo):.3f}\n")
    print(f"{'candidate':>34s} {'residual':>9s} {'cost':>7s} {'dijet loss':>11s}")
    for name, veto in cands.items():
        resid = w[both & ~veto].sum() / Wb if Wb else float("nan")
        cost = w[only & veto].sum() / Wo if Wo else float("nan")
        loss = w[passd & veto].sum() / (Wb + Wo)
        print(f"{name:>34s} {resid:>9.3f} {cost:>7.3f} {loss:>11.3f}")

    #### per-pt detail for the exact veto and the best kinematic proxy
    for name in ("A: veto trijet pass (exact)", "B1: veto njet>=3 & dphimin>1"):
        veto = cands[name]
        print(f"\n--- {name}: per leading-jet pt bin ---")
        print(f"{'pT bin':>12s} {'overlap/dijet':>14s} {'residual':>9s} "
              f"{'cost':>7s} {'dijet loss':>11s}")
        for i in range(len(PT_EDGES) - 1):
            m = (pt1 >= PT_EDGES[i]) & (pt1 < PT_EDGES[i + 1])
            wb = w[m & both].sum()
            wo = w[m & only].sum()
            if wb + wo == 0:
                continue
            resid = w[m & both & ~veto].sum() / wb if wb else float("nan")
            cost = w[m & only & veto].sum() / wo if wo else float("nan")
            loss = w[m & passd & veto].sum() / (wb + wo)
            print(f"{PT_EDGES[i]:>5.0f}-{PT_EDGES[i+1]:<6.0f} {wb/(wb+wo):>14.3f} "
                  f"{resid:>9.3f} {cost:>7.3f} {loss:>11.3f}")

    #### gen-level (MC only): the veto has to exist in the fiducial definition
    #### too, or the unfolding treats vetoed-at-reco events as fakes. Uses the
    #### exact genTot_seq bits from both channels' cutbits.
    if not args.data:
        gd = (D["cutbits"][di] & bit(DIJET_CUTS, "genTot_seq")) != 0
        gt = (T["cutbits"][ti] & bit(TRIJET_CUTS, "genTot_seq")) != 0
        wgb, wgo = w[gd & gt].sum(), w[gd & ~gt].sum()
        print(f"\n--- gen level ---")
        print(f"gen overlap frac of dijet-gen: {wgb/(wgb+wgo):.3f}   "
              f"P(di-gen|tri-gen): {wgb/w[gt].sum():.3f}")
        #### migration under a trijet-priority veto at BOTH levels: of dijet
        #### events in the response (reco AND gen selected), how many change
        #### category (vetoed at only one level)?
        rg = passd & gd
        wm = w[rg].sum()
        for rname, rv in (("reco-vetoed only", passt & ~gt),
                          ("gen-vetoed only", ~passt & gt),
                          ("vetoed both", passt & gt)):
            print(f"  dijet reco&gen events {rname:>17s}: "
                  f"{w[rg & rv].sum()/wm:.4f}")

        #### The raw trijet gen selection is much looser than reco (GenJetAK8
        #### is stored to lower pt than FatJet), so mirror the veto at gen with
        #### a pt3 floor and scan for the threshold that minimizes one-sided
        #### migration against the reco veto (= exact trijet pass bit).
        g_pt3 = np.nan_to_num(D["gen_pt3"][di], nan=0.0)
        g_dpm = D["gen_dphimin"][di]
        g_has3 = D["gen_njet"][di] >= 3
        for rlabel, rveto in (("trijet pass bit", passt),
                              ("trijet pass & pt3>200", passt & (pt3 > 200.0))):
            print(f"\n--- gen-veto pt3(gen) floor scan (dijet reco&gen "
                  f"events, veto_reco = {rlabel}) ---")
            print(f"{'pt3gen floor':>13s} {'gen-vetoed':>11s} "
                  f"{'reco only':>10s} {'gen only':>9s} {'mismatch':>9s}")
            for thr in (0., 120., 150., 170., 185., 200., 240., 280.):
                gv = g_has3 & (g_dpm > 1.0) & (g_pt3 > thr)
                m_r = w[rg & rveto & ~gv].sum() / wm
                m_g = w[rg & ~rveto & gv].sum() / wm
                print(f"{thr:>13.0f} {w[rg & gv].sum()/wm:>11.4f} "
                      f"{m_r:>10.4f} {m_g:>9.4f} {m_r + m_g:>9.4f}")

    #### where does the exact veto differ from the kinematic proxy?
    proxy = cands["B1: veto njet>=3 & dphimin>1"]
    exact = cands["A: veto trijet pass (exact)"]
    d_pe = w[passd & exact & ~proxy].sum()
    d_pp = w[passd & proxy & ~exact].sum()
    print(f"\nexact-veto events missed by B1 (rap/id/muIso on jet3): {d_pe:.4g}"
          f"\nB1 vetoes NOT in the exact trijet selection:            {d_pp:.4g}"
          f"  (of {w[passd].sum():.4g} dijet)")


if __name__ == "__main__":
    main()
