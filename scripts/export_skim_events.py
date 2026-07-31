#!/usr/bin/env python
"""Turn the dijet skim's event-display tables into a standalone 3D event viewer.

The skim stores EVERY AK8 jet (not just the leading two) plus a per-event
`cutbits` mask, precisely so an efficiency loss can be looked at instead of
guessed at. This script picks a sample of events stratified by *where they
died* in the selection ladder, recomputes the quantities the cuts actually act
on (|y|, dphi_12, pT asymmetry, gen->reco dR), and inlines the whole thing into
a single self-contained HTML file.

    python scripts/export_skim_events.py \
        --input "~/cernbox (2)/dijet_skims/rho_skim10_dijet_mg_pythia8_2018_HT2000toInf.pkl" \
        --out review/dijet_event_viewer.html

Several --input files may be given; they are concatenated (each keeps its own
dataset name via dataset_id).

The "fraction of skimmed events" numbers printed below are NOT efficiencies:
the processor returns early from a chunk with no surviving event, so chunks that
were wiped out entirely contribute no rows (see the CAVEAT in
dijet_processor._fill_skim_event's call site). Quote the cutflow for efficiency.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "scripts" / "event_viewer_template.html"

#### The ladder in the order a cut is actually reached. `cutbits` bit i is
#### _SKIM_CUTS[i]; this is the subset that is a real sequential requirement,
#### so "first cut set to 0" == "where the event died".
LADDER = [
    ("npv",         "N_PV > 0 (+ trigger, data only)"),
    ("METfilters",  "MET filters"),
    ("twoRecoJet",  "n(FatJet) > 1"),
    ("recoRap2p5",  "|y| < 2.5 for both leading reco jets"),
    ("recodphi2",   "dphi(j1,j2) > 2.0 reco"),
    ("recoAsym0p3", "|pt1-pt2|/(pt1+pt2) < 0.3 reco"),
    ("muonIso0p4",  "dR(jet, nearest muon) > 0.4"),
    ("jetId",       "jetId > 2 for both leading reco jets"),
    ("hemveto",     "HEM (2018 MC: weight, always passes)"),
    ("recoTot_seq", "all reco cuts + mass/msoftdrop not None"),
    ("twoGenJet",   "n(GenJetAK8) > 1"),
    ("genRap2p5",   "|y| < 2.5 for both leading gen jets"),
    ("dphiGen2",    "dphi(j1,j2) > 2.0 gen"),
    ("genAsym0p3",  "|pt1-pt2|/(pt1+pt2) < 0.3 gen"),
    ("genTot_seq",  "all gen cuts + gen mass/SD mass not None"),
    ("matched_gen", "dR(gen, nearest reco) < 0.4 for both  -> else MISS"),
    ("matched_reco", "both reco jets have a matched_gen      -> else FAKE"),
    ("final_seq",   "gen AND reco AND matched both ways"),
]


def load_tables(paths):
    """Concatenate the events/alljets/allgenjets tables across input files."""
    acc, names = {}, {}
    for p in paths:
        out = pickle.load(open(p, "rb"))
        skim = out["skim"]
        for ds in skim.get("datasets", []):
            names[zlib.crc32(ds.encode()) & 0xFFFFFFFF] = ds
        for t in ("events", "alljets", "allgenjets"):
            if t not in skim:
                raise SystemExit(
                    f"{p}: no '{t}' table -- this pkl predates the event-display "
                    "tables, re-run the skim")
            cols = {c: np.asarray(a.value) for c, a in skim[t].items()}
            for c, v in cols.items():
                acc.setdefault(t, {}).setdefault(c, []).append(v)
    return ({t: {c: np.concatenate(v) for c, v in cols.items()}
             for t, cols in acc.items()}, names)


def rapidity(pt, eta, mass):
    """y = 0.5 ln((E+pz)/(E-pz)); needs the mass, so eta is not enough."""
    pz = pt * np.sinh(eta)
    e = np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * np.log((e + pz) / (e - pz))


def dphi(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def group_by_event(tbl, keys):
    """Map (run, lumi, event, dataset_id) -> row indices, jetidx-ordered."""
    order = np.lexsort((tbl["jetidx"], tbl["event"], tbl["lumi"], tbl["run"]))
    idx = {}
    for r in order:
        idx.setdefault((int(tbl["run"][r]), int(tbl["lumi"][r]),
                        int(tbl["event"][r]), int(tbl["dataset_id"][r])), []).append(int(r))
    return {k: idx.get(k, []) for k in keys}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True, type=lambda s: Path(s).expanduser())
    p.add_argument("--out", type=lambda s: Path(s).expanduser(),
                   default=REPO_ROOT / "outputs" / "dijet_event_viewer.html")
    p.add_argument("--per-category", type=int, default=25,
                   help="events to keep per death category (default 25)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    T, dsnames = load_tables(args.input)
    E = T["events"]
    n = len(E["event"])
    print(f"loaded {n} events, {len(T['alljets']['pt'])} reco jets, "
          f"{len(T['allgenjets']['pt'])} gen jets")

    bit = {c: i for i, (c, _) in enumerate(LADDER)}
    cb = E["cutbits"]
    passed = {c: ((cb >> i) & 1).astype(bool) for c, i in bit.items()}

    #### where did it die: the first ladder entry the event fails
    died = np.full(n, "final_seq", dtype=object)
    remaining = np.ones(n, dtype=bool)
    for c, _ in LADDER:
        fail = remaining & ~passed[c]
        died[fail] = c
        remaining &= passed[c]
    died[remaining] = "PASS"

    cats, counts = np.unique(died.astype(str), return_counts=True)
    print("\ndeath category (fraction of skimmed events):")
    for c, k in sorted(zip(cats, counts), key=lambda x: -x[1]):
        print(f"  {c:14s} {k:6d}  {k / n:7.2%}")

    #### stratified sample: every failure mode is represented even if it is rare,
    #### which is the whole point -- the rare ones are the interesting ones
    rng = np.random.default_rng(args.seed)
    keep = []
    for c in cats:
        w = np.flatnonzero(died.astype(str) == c)
        keep.extend(rng.choice(w, size=min(args.per_category, len(w)), replace=False))
    keep = np.sort(np.array(keep, dtype=int))
    print(f"\nexporting {len(keep)} events over {len(cats)} categories")

    keys = [(int(E["run"][i]), int(E["lumi"][i]), int(E["event"][i]),
             int(E["dataset_id"][i])) for i in keep]
    reco_rows = group_by_event(T["alljets"], keys)
    gen_rows = group_by_event(T["allgenjets"], keys)

    def r3(x):
        return None if x is None or not np.isfinite(x) else round(float(x), 3)

    events = []
    for i, k in zip(keep, keys):
        rj, gj = T["alljets"], T["allgenjets"]
        ri, gi = reco_rows[k], gen_rows[k]

        def jets(tbl, rows, extra=()):
            out = []
            for r in rows:
                j = {"i": int(tbl["jetidx"][r]), "pt": r3(tbl["pt"][r]),
                     "eta": r3(tbl["eta"][r]), "phi": r3(tbl["phi"][r]),
                     "m": r3(tbl["mass"][r]),
                     "y": r3(rapidity(tbl["pt"][r], tbl["eta"][r], tbl["mass"][r]))}
                for c in extra:
                    j[c] = r3(tbl[c][r])
                out.append(j)
            return out

        R = jets(rj, ri, ("msoftdrop", "jetId", "nconst"))
        G = jets(gj, gi)

        #### the quantities the cuts are cut ON -- so a red row in the ladder can
        #### be read next to the number that caused it
        d = {}
        if len(R) > 1:
            d["dphi_reco"] = r3(dphi(np.float64(R[0]["phi"]), np.float64(R[1]["phi"])))
            d["asym_reco"] = r3(abs(R[0]["pt"] - R[1]["pt"]) / (R[0]["pt"] + R[1]["pt"]))
            d["ymax_reco"] = r3(max(abs(R[0]["y"]), abs(R[1]["y"])))
        if len(G) > 1:
            d["dphi_gen"] = r3(dphi(np.float64(G[0]["phi"]), np.float64(G[1]["phi"])))
            d["asym_gen"] = r3(abs(G[0]["pt"] - G[1]["pt"]) / (G[0]["pt"] + G[1]["pt"]))
            d["ymax_gen"] = r3(max(abs(G[0]["y"]), abs(G[1]["y"])))
        #### gen->reco dR to the nearest of the two leading reco jets: this is
        #### literally the matched_gen cut, drawn as a link in the viewer
        links = []
        if len(R) > 1 and len(G) > 1:
            for a in range(min(2, len(G))):
                best, bestd = -1, 1e9
                for b in range(min(2, len(R))):
                    dr = float(np.hypot(G[a]["eta"] - R[b]["eta"],
                                        dphi(np.float64(G[a]["phi"]), np.float64(R[b]["phi"]))))
                    if dr < bestd:
                        best, bestd = b, dr
                links.append({"g": a, "r": best, "dr": r3(bestd)})
            d["dr_match_max"] = r3(max(l["dr"] for l in links))

        events.append({
            "run": k[0], "lumi": k[1], "evt": k[2], "ds": k[3],
            "w": r3(E["weight"][i]), "npv": int(E["npv"][i]),
            "rho": r3(E["pu_rho"][i]), "met": r3(E["met"][i]),
            "htr": r3(E["ht_reco"][i]), "htg": r3(E["ht_gen"][i]),
            "nr": int(E["njet_reco"][i]), "ng": int(E["njet_gen"][i]),
            "bits": int(cb[i]), "died": str(died[i]),
            "reco": R, "gen": G, "links": links, "d": d,
        })

    payload = {
        "ladder": [{"name": c, "desc": t} for c, t in LADDER],
        "datasets": {str(k): v for k, v in dsnames.items()},
        "nTotal": int(n),
        "categoryCounts": {str(c): int(k) for c, k in zip(cats, counts)},
        "events": events,
        "source": [pp.name for pp in args.input],
    }

    html = TEMPLATE.read_text().replace(
        "/*__EVENT_DATA__*/null", json.dumps(payload, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
