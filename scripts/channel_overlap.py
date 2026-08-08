#!/usr/bin/env python
"""Event overlap between the dijet and trijet selections, per leading-jet pT bin.

Uses the rho_skim EVENT-DISPLAY tables, which both channels fill for every
processed event with `event % 1000 == 0` (dijet: rho_skim10 x factor 100,
trijet: rho_skim x factor 1000 -- the SAME deterministic 1/1000 sample), with a
per-event `cutbits` mask recording every selection step. Joining the two
channels' tables on (dataset_id, run, lumi, event) therefore gives, for each
event processed by BOTH channels, whether it passed the dijet and/or the trijet
reco selection.

Definitions:

    dijet pass  = cutbits bit "recoTot_seq" of the dijet skim
    trijet pass = cutbits bit "recoTot_seq" of the trijet skim
                  (optionally also "trigsel"; the dijet selection has no trigger
                  bit -- dijet data uses prescale weights, so for data the
                  honest apples-to-apples overlap is topology-only)

    P(tri|di), P(di|tri), and Jaccard = both / (di OR tri), per pT bin of the
    leading AK8 jet (from the alljets table; nominal-JEC pT, either channel's
    copy -- dijet preferred, trijet as fallback for events dijet did not store
    jets for).

MC is combined across HT bins with xs/sumw * genWeight; data is unweighted
counts. Events with a duplicated join key (repeated MC event numbers) are
dropped entirely (<1%%).

    python scripts/channel_overlap.py            # MC
    python scripts/channel_overlap.py --data     # 2018 JetHT
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import smp_jetmass_run2.corrections as corr

#### Bit order = _SKIM_CUTS in each processor. Copied (not imported) so this
#### script does not drag in coffea; the parity test below guards the copy.
DIJET_CUTS = (
    "npv", "METfilters",
    "twoRecoJet", "recoRap2p5", "recodphi2", "recoAsym0p3",
    "muonIso0p4", "jetId", "hemveto", "recoTot_seq",
    "twoGenJet", "genRap2p5", "dphiGen2", "genAsym0p3", "genTot_seq",
    "matched_gen", "matched_reco", "final_seq",
    #### appended 2026-07-31 with the trijet-priority veto; 0 in older skims
    "vetoTrijetReco", "vetoTrijetGen",
)
TRIJET_CUTS = (
    "trigsel", "npv", "METfilters",
    "triRecoJet", "triRecoJet_seq", "recoRap2p5", "recoRap_seq",
    "recodphimin", "recodphi_seq", "muonIso0p4", "jetId", "hemveto",
    "recoTot_seq",
    "triGenJet", "rapGen", "dphiGen", "genTot_seq",
    "matched_gen", "matched_reco", "final_seq",
)

PT_EDGES = np.array([200., 290., 400., 480., 570., 680., 760., 820., 13000.])


def bit(cuts, name):
    return np.uint32(1) << np.uint32(cuts.index(name))


def load_events(paths):
    """Concatenate the events (+leading-jet pt) tables of one channel."""
    keys, cutbits, weight, dsid, leadpt = [], [], [], [], []
    datasets = set()
    for p in paths:
        sk = pickle.load(open(p, "rb"))["skim"]
        ev = sk["events"]
        n = len(ev["event"].value)
        if n == 0:
            print(f"  !! empty events table: {Path(p).name}", file=sys.stderr)
            continue
        datasets |= set(sk.get("datasets", []))
        cols = {c: np.asarray(ev[c].value) for c in
                ("run", "lumi", "event", "dataset_id", "weight", "cutbits")}
        #### leading-jet pt: max alljets pt per (run,lumi,event,dataset_id)
        aj = sk["alljets"]
        jk = np.stack([np.asarray(aj[c].value, dtype=np.int64) for c in
                       ("dataset_id", "run", "lumi", "event")], axis=1)
        ek = np.stack([cols[c].astype(np.int64) for c in
                       ("dataset_id", "run", "lumi", "event")], axis=1)
        jpt = np.asarray(aj["pt"].value)
        #### sort jets by key then take the max pt within each key block
        order = np.lexsort(jk.T[::-1])
        jk, jpt = jk[order], jpt[order]
        newblk = np.r_[True, np.any(jk[1:] != jk[:-1], axis=1)]
        blk = np.cumsum(newblk) - 1
        #### seed with -inf, not nan: np.maximum propagates nan and would wipe
        #### out every block
        blockmax = np.full(blk[-1] + 1 if len(blk) else 0, -np.inf, np.float32)
        np.maximum.at(blockmax, blk, jpt)
        #### map event rows onto jet blocks (events with no stored jet -> nan)
        uk = jk[newblk]
        lead = np.full(n, np.nan, np.float32)
        if len(uk):
            #### row index of each event key in uk via searchsorted on a view
            def view(a):
                return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()
            pos = np.searchsorted(view(uk), view(ek))
            pos = np.clip(pos, 0, len(uk) - 1)
            hit = view(uk)[pos] == view(ek)
            lead[hit] = blockmax[pos[hit]]
        keys.append(ek)
        cutbits.append(cols["cutbits"])
        weight.append(cols["weight"])
        dsid.append(cols["dataset_id"])
        leadpt.append(lead)
        print(f"  {Path(p).name:<55s} {n:>9d} events")
    return (np.concatenate(keys), np.concatenate(cutbits),
            np.concatenate(weight), np.concatenate(dsid),
            np.concatenate(leadpt), datasets)


def dedupe(keys):
    """Boolean mask keeping only keys that occur exactly once."""
    def view(a):
        return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()
    v = view(keys)
    order = np.argsort(v)
    sv = v[order]
    dup = np.zeros(len(sv), bool)
    same = sv[1:] == sv[:-1]
    dup[1:] |= same
    dup[:-1] |= same
    out = np.ones(len(v), bool)
    out[order] = ~dup
    return out


def xs_lut(datasets, year="2018"):
    lut = {}
    import zlib
    for ds in datasets:
        key = ds.split("/")[1] if ds.startswith("/") else ds
        w = None
        if key in corr.sumw_qcd_mg.get(year, {}) and key in corr.xsdb:
            w = corr.xsdb[key] / corr.sumw_qcd_mg[year][key]
        lut[zlib.crc32(ds.encode()) & 0xFFFFFFFF] = w
    return lut


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dijet-mc", default="/Users/aritra/cernbox (2)/dijet_skims")
    ap.add_argument("--trijet-mc", default="/Users/aritra/cernbox (2)/trijet_skims/mc")
    ap.add_argument("--dijet-data", default="/Users/aritra/Downloads/dijet_data_skims")
    ap.add_argument("--trijet-data", default="/Users/aritra/cernbox (2)/trijet_skims/data")
    ap.add_argument("--data", action="store_true", help="run on 2018 JetHT instead of QCD MC")
    ap.add_argument("--require-trigsel", action="store_true",
                    help="also require the trijet trigsel bit for the trijet pass")
    args = ap.parse_args(argv)

    if args.data:
        dpaths = sorted(Path(args.dijet_data).glob("*.pkl"))
        tpaths = sorted(Path(args.trijet_data).glob("*.pkl"))
    else:
        dpaths = sorted(Path(args.dijet_mc).glob("*.pkl"))
        tpaths = sorted(Path(args.trijet_mc).glob("*.pkl"))

    print("dijet skims:")
    dk, dbits, dw, dds, dpt, ddsets = load_events(dpaths)
    print("trijet skims:")
    tk, tbits, tw, tds, tpt, tdsets = load_events(tpaths)

    #### drop duplicated keys, then inner-join on the composite key
    dm, tm = dedupe(dk), dedupe(tk)
    dk, dbits, dw, dds, dpt = dk[dm], dbits[dm], dw[dm], dds[dm], dpt[dm]
    tk, tbits, tw, tds, tpt = tk[tm], tbits[tm], tw[tm], tds[tm], tpt[tm]

    def view(a):
        return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()
    dv, tv = view(dk), view(tk)
    do, to = np.argsort(dv), np.argsort(tv)
    common, di, ti = np.intersect1d(dv[do], tv[to], return_indices=True)
    di, ti = do[di], to[ti]
    print(f"\njoin: {len(dk)} dijet x {len(tk)} trijet unique events "
          f"-> {len(common)} common ({len(common)/max(len(dk),1):.1%} of dijet)")

    passd = (dbits[di] & bit(DIJET_CUTS, "recoTot_seq")) != 0
    tsel = bit(TRIJET_CUTS, "recoTot_seq")
    if args.require_trigsel:
        tsel |= bit(TRIJET_CUTS, "trigsel")
    passt = (tbits[ti] & tsel) == tsel

    #### weights: MC -> xs/sumw * stored weight (stride factor is common to all
    #### rows and cancels in every ratio); data -> unweighted
    if args.data:
        w = np.ones(len(common))
    else:
        lut = xs_lut(ddsets | tdsets)
        w = dw[di] * np.array([lut.get(d) or 0.0 for d in dds[di]])

    #### leading-jet pt: dijet's copy, fall back to trijet's
    pt = dpt[di].copy()
    use_t = ~np.isfinite(pt)
    pt[use_t] = tpt[ti][use_t]

    label = "2018 JetHT" if args.data else "QCD MG+Pythia8 2018 (xs-weighted)"
    trig = " [trijet trigsel required]" if args.require_trigsel else ""
    print(f"\n=== dijet/trijet reco-selection overlap -- {label}{trig} ===")
    print(f"{'lead-jet pT bin':>18s} {'N(di)':>12s} {'N(tri)':>12s} {'N(both)':>12s} "
          f"{'P(tri|di)':>10s} {'P(di|tri)':>10s} {'Jaccard':>9s}")
    rows = [("all pt", np.ones(len(pt), bool))]
    for i in range(len(PT_EDGES) - 1):
        rows.append((f"{PT_EDGES[i]:.0f}-{PT_EDGES[i+1]:.0f}",
                     (pt >= PT_EDGES[i]) & (pt < PT_EDGES[i + 1])))
    for name, m in rows:
        nd = w[m & passd].sum()
        nt = w[m & passt].sum()
        nb = w[m & passd & passt].sum()
        nu = w[m & (passd | passt)].sum()
        if nu == 0:
            continue
        print(f"{name:>18s} {nd:>12.4g} {nt:>12.4g} {nb:>12.4g} "
              f"{nb/nd if nd else float('nan'):>10.3f} "
              f"{nb/nt if nt else float('nan'):>10.3f} "
              f"{nb/nu:>9.3f}")
    nolead = ~np.isfinite(pt) & (passd | passt)
    if nolead.any():
        print(f"(selected events with no stored jet row, excluded from pt bins: "
              f"{int(nolead.sum())})")


if __name__ == "__main__":
    main()
