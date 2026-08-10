#!/usr/bin/env python
"""ANGULAR candidate cuts that split the dijet and trijet channels orthogonally.

The two channels overlap badly (trijet-selected events are ~89%% a subset of the
dijet selection).  The production fix is a *trijet-priority veto*: the dijet
selection drops any event that also passes the trijet selection and has
pT3 > 185 GeV.  That works, but it is a two-channel bookkeeping rule rather than
a cut on a single event-level observable, and its pT dependence has never been
measured.

This script scans the ANGULAR family of replacements.  Every candidate is a rule
on ONE event-level quantity, applied to the dijet side and applied with the
opposite sign to the trijet side, so the two channels are orthogonal BY
CONSTRUCTION (the residual overlap that survives is the JER-seed artifact
described below, ~1e-3 of the baseline overlap).

Families
--------
B1  pT-dependent back-to-backness.  For a two-body recoil, the transverse
    momentum kicked out of the leading pair is ~ pT1 * (pi - dphi12), so
    "recoil < X GeV" is  dphi12 > pi - X/pT1.  dijet keeps the back-to-back
    side, trijet keeps the rest.  Both dphi and asymmetry cuts are REMOVED from
    the dijet baseline (this rule replaces them).
B2  a fixed, tighter dphi12 reference (2.6 / 2.8 / 3.0) on top of the current
    A < 0.3 -- the "just tighten what we already cut on" straw man.
B3  the generalised phase-space split, and the real point of the study: the
    dijet channel VETOES the trijet phase space
        njet >= 3  AND  dphimin > dcut  AND  pT3 > floor(pT1)
    while the trijet channel REQUIRES it.  dcut and the floor are scanned; the
    floor may be a constant or proportional to pT1 (the current scheme is the
    single point dcut = 1.0, floor = 185 GeV, with the veto evaluated from the
    other channel's selection rather than from the dijet event's own jets).

References always reported: the current trijet-priority veto, and the leading
pair asymmetry split A < 0.1 / A > 0.1 (see scripts/channel_asym_split.py).

    python scripts/split_scan_angular.py                 # MC scan + JSON + plots
    python scripts/split_scan_angular.py --no-plots      # numbers only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.channel_asym_split import (  # noqa: E402
    load_channel, allbits, asym, frac, stamp,
    DIJET_RECO_NM1, DIJET_RECO_NOBAL, DIJET_GEN_NM1, DIJET_GEN_NOBAL)
from scripts.channel_overlap import (  # noqa: E402
    DIJET_CUTS, TRIJET_CUTS, PT_EDGES, bit, dedupe)

PI = float(np.pi)
SCRATCH = Path("/private/tmp/claude-501/-Users-aritra-Projects-smp-jetmass-run2/"
               "a2621af3-b9a7-4e23-a9ea-dd4970a0445f/scratchpad")

#### the dijet gen sequence = the gen mirror of recoTot_seq
DIJET_GEN_SEQ = DIJET_GEN_NM1 + ("genAsym0p3",)

COL = ("#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd")


# --------------------------------------------------------------------------- #
#  join
# --------------------------------------------------------------------------- #
D_KEYS = ("cutbits", "weight", "pt1", "pt2", "pt3", "njet", "dphimin", "dphi12",
          "gpt1", "gpt2", "gpt3", "gnjet", "gdphimin", "gdphi12",
          #### |y| of the 3 leading reco jets, for the |y| < 2.4 acceptance test
          "ay1", "ay2", "ay3")
T_KEYS = D_KEYS


def load_rapidity(paths):
    """|y| of the 3 leading reco jets, row-aligned with load_channel's output.

    load_channel keeps only pt/phi, and the acceptance question needs rapidity
    (not pseudorapidity), so the raw alljets table is re-read here for eta and
    mass.  The empty-table skip MUST match load_channel's or the rows stop
    lining up.
    """
    import pickle
    out = []
    for p in paths:
        sk = pickle.load(open(p, "rb"))["skim"]
        ev = sk["events"]
        n = len(ev["event"].value)
        if n == 0:
            continue
        ek = np.stack([np.asarray(ev[c].value, dtype=np.int64) for c in
                       ("dataset_id", "run", "lumi", "event")], axis=1)
        aj = sk["alljets"]
        jk = np.stack([np.asarray(aj[c].value, dtype=np.int64) for c in
                       ("dataset_id", "run", "lumi", "event")], axis=1)
        jidx = np.asarray(aj["jetidx"].value)
        pt = np.asarray(aj["pt"].value, float)
        eta = np.asarray(aj["eta"].value, float)
        mass = np.asarray(aj["mass"].value, float)
        pz = pt * np.sinh(eta)
        E = np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            y = 0.5 * np.log((E + pz) / (E - pz))

        def view(a):
            return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()
        order = np.argsort(view(ek), kind="stable")
        if len(jk):
            pos = np.clip(np.searchsorted(view(ek)[order], view(jk)), 0, n - 1)
            hit = view(ek)[order][pos] == view(jk)
            evrow = order[pos]
        else:
            hit = np.zeros(0, bool)
            evrow = np.zeros(0, np.int64)
        ay = np.full((n, 3), np.nan)
        lead = hit & (jidx < 3)
        ay[evrow[lead], jidx[lead]] = np.abs(y[lead])
        out.append(ay)
    return np.concatenate(out) if out else np.zeros((0, 3))


def build_join(dpaths, tpaths, cache: Path | None):
    """Inner-join the two channels' 1/1000 event tables on (dsid,run,lumi,evt)."""
    if cache is not None and cache.exists():
        z = np.load(cache)
        D = {k: z["D_" + k] for k in D_KEYS}
        T = {k: z["T_" + k] for k in T_KEYS}
        print(f"join: {len(D['cutbits'])} common events (cached {cache.name})")
        return D, T

    print("dijet skims:")
    Draw, _ = load_channel(dpaths)
    Draw["ay1"], Draw["ay2"], Draw["ay3"] = load_rapidity(dpaths).T
    print("trijet skims:")
    Traw, _ = load_channel(tpaths)
    Traw["ay1"], Traw["ay2"], Traw["ay3"] = load_rapidity(tpaths).T

    dm, tm = dedupe(Draw["key"]), dedupe(Traw["key"])
    Draw = {k: v[dm] for k, v in Draw.items()}
    Traw = {k: v[tm] for k, v in Traw.items()}

    def view(a):
        return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()
    do, to = np.argsort(view(Draw["key"])), np.argsort(view(Traw["key"]))
    _, di, ti = np.intersect1d(view(Draw["key"])[do], view(Traw["key"])[to],
                               return_indices=True)
    di, ti = do[di], to[ti]
    print(f"\njoin: {len(di)} common events")

    D = {k: np.asarray(Draw[k])[di] for k in D_KEYS}
    T = {k: np.asarray(Traw[k])[ti] for k in T_KEYS}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{"D_" + k: v for k, v in D.items()},
                            **{"T_" + k: v for k, v in T.items()})
    return D, T


# --------------------------------------------------------------------------- #
#  floors for the B3 phase-space rule
# --------------------------------------------------------------------------- #
def make_floor(label):
    """(reco floor(pt1), gen floor(gpt1)) for a floor label.

    Gen jets are stored down to ~100 GeV while FatJets stop at ~170, so ANY gen
    third-jet condition needs its own floor: a constant floor is raised to at
    least 185 GeV, a pT-proportional floor keeps the same functional form (and
    is therefore exposed to the storage asymmetry -- that is exactly what the
    migration columns measure).
    """
    if label.startswith("max(185,"):
        f = float(label.split(",")[1].replace("pt1)", ""))
        g = r = (lambda p, f=f: np.maximum(185.0, f * p))
    elif label.endswith("pt1"):
        f = float(label[:-3])
        g = r = (lambda p, f=f: f * p)
    else:
        v = float(label)
        r = (lambda p, v=v: np.full(np.shape(p), v, float))
        g = (lambda p, v=max(v, 185.0): np.full(np.shape(p), v, float))
    return r, g


FLOORS = ("185", "250", "0.20pt1", "0.25pt1", "0.30pt1",
          "max(185,0.25pt1)", "max(185,0.30pt1)")
DCUTS = (0.5, 1.0, 1.5)


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #
class Ctx:
    """Everything a candidate rule needs, precomputed once."""

    def __init__(self, D, T, is_data):
        self.is_data = is_data
        self.w = D["weight"].astype(float)
        w = self.w
        self.pt1 = np.asarray(D["pt1"], float)
        self.pt2 = np.asarray(D["pt2"], float)
        self.pt3 = np.nan_to_num(np.asarray(D["pt3"], float), nan=0.0)
        self.njet = np.asarray(D["njet"])
        self.dpm = np.asarray(D["dphimin"], float)
        self.d12 = np.asarray(D["dphi12"], float)
        self.A = asym(self.pt1, self.pt2)

        self.ay = np.stack([np.asarray(D[k], float)
                            for k in ("ay1", "ay2", "ay3")])
        self.tay = np.stack([np.asarray(T[k], float)
                             for k in ("ay1", "ay2", "ay3")])

        self.tpt1 = np.asarray(T["pt1"], float)
        self.tpt2 = np.asarray(T["pt2"], float)
        self.tpt3 = np.nan_to_num(np.asarray(T["pt3"], float), nan=0.0)
        self.tnjet = np.asarray(T["njet"])
        self.tdpm = np.asarray(T["dphimin"], float)
        self.td12 = np.asarray(T["dphi12"], float)
        self.A_t = asym(self.tpt1, self.tpt2)

        self.gpt1 = np.asarray(D["gpt1"], float)
        self.gpt3 = np.nan_to_num(np.asarray(D["gpt3"], float), nan=0.0)
        self.gnjet = np.asarray(D["gnjet"])
        self.gdpm = np.asarray(D["gdphimin"], float)
        self.gd12 = np.asarray(D["gdphi12"], float)
        self.Agen = asym(self.gpt1, np.asarray(D["gpt2"], float))
        self.tgpt1 = np.asarray(T["gpt1"], float)
        self.tgpt3 = np.nan_to_num(np.asarray(T["gpt3"], float), nan=0.0)
        self.tgnjet = np.asarray(T["gnjet"])
        self.tgdpm = np.asarray(T["gdphimin"], float)

        cb, tb = D["cutbits"], T["cutbits"]
        self.d_tot = (cb & bit(DIJET_CUTS, "recoTot_seq")) != 0
        self.t_tot = (tb & bit(TRIJET_CUTS, "recoTot_seq")) != 0
        self.tg_tot = (tb & bit(TRIJET_CUTS, "genTot_seq")) != 0
        self.d_nm1 = allbits(cb, DIJET_CUTS, DIJET_RECO_NM1) & np.isfinite(self.A)
        self.d_nobal = (allbits(cb, DIJET_CUTS, DIJET_RECO_NOBAL)
                        & np.isfinite(self.d12) & (self.pt1 > 0))
        self.g_nm1 = (allbits(cb, DIJET_CUTS, DIJET_GEN_NM1)
                      & np.isfinite(self.Agen))
        self.g_seq = (allbits(cb, DIJET_CUTS, DIJET_GEN_SEQ)
                      & np.isfinite(self.Agen))
        self.g_nobal = (allbits(cb, DIJET_CUTS, DIJET_GEN_NOBAL)
                        & np.isfinite(self.gd12) & (self.gpt1 > 0))
        #### trijet selection with the rapidity cut removed, for the |y| < 2.4
        #### acceptance question (the only place jet 3's rapidity is uncensored)
        self.tri_rap_nm1 = allbits(tb, TRIJET_CUTS,
                                   ("npv", "METfilters", "triRecoJet",
                                    "recodphimin", "muonIso0p4", "jetId",
                                    "hemveto"))

        self.Wd = w[self.d_tot].sum()
        self.Wt = w[self.t_tot].sum()
        self.Wb = w[self.d_tot & self.t_tot].sum()
        self.Wu = w[self.d_tot | self.t_tot].sum()
        self.ptbin = [(self.pt1 >= PT_EDGES[i]) & (self.pt1 < PT_EDGES[i + 1])
                      for i in range(len(PT_EDGES) - 1)]

    #### -- reco / gen phase-space membership for the B3 family --------------
    def in3(self, dcut, floor_label):
        r, _ = make_floor(floor_label)
        with np.errstate(invalid="ignore"):
            return (self.njet >= 3) & (self.dpm > dcut) & (self.pt3 > r(self.pt1))

    def in3_t(self, dcut, floor_label):
        r, _ = make_floor(floor_label)
        with np.errstate(invalid="ignore"):
            return ((self.tnjet >= 3) & (self.tdpm > dcut)
                    & (self.tpt3 > r(self.tpt1)))

    def in3_gen(self, dcut, floor_label):
        _, g = make_floor(floor_label)
        with np.errstate(invalid="ignore"):
            return ((self.gnjet >= 3) & (self.gdpm > dcut)
                    & (self.gpt3 > g(self.gpt1)))

    def in3_gen_t(self, dcut, floor_label):
        _, g = make_floor(floor_label)
        with np.errstate(invalid="ignore"):
            return ((self.tgnjet >= 3) & (self.tgdpm > dcut)
                    & (self.tgpt3 > g(self.tgpt1)))


def metrics(C, name, params, dsel, tsel, gbase=None, gcond=None, rbase=None,
            tgbase=None, tgcond=None, sample="mc"):
    w, d_tot, t_tot = C.w, C.d_tot, C.t_tot
    row = {
        "family": "angular", "name": name, "params": params, "sample": sample,
        "dijet_keep": frac(w[dsel].sum(), C.Wd),
        "trijet_keep": frac(w[tsel].sum(), C.Wt),
        "resid_overlap": frac(w[dsel & tsel].sum(), C.Wb),
        "union": frac(w[dsel | tsel].sum(), C.Wu),
        "orphans": frac(w[d_tot & ~dsel & ~t_tot].sum(), C.Wd),
        "fakes_boundary": float("nan"), "misses_boundary": float("nan"),
        "fakes_boundary_tri": float("nan"), "misses_boundary_tri": float("nan"),
    }
    per = {"edges": [float(x) for x in PT_EDGES],
           "dijet_keep": [], "trijet_keep": [], "orphans": []}
    for m in C.ptbin:
        wd, wt = w[m & d_tot].sum(), w[m & t_tot].sum()
        per["dijet_keep"].append(frac(w[m & dsel].sum(), wd))
        per["trijet_keep"].append(frac(w[m & tsel].sum(), wt))
        per["orphans"].append(frac(w[m & d_tot & ~dsel & ~t_tot].sum(), wd))
    row["per_pt"] = per

    #### boundary-only migration: the event passes every OTHER cut at the other
    #### level, so the split boundary alone put it on the wrong side.
    if not C.is_data and gbase is not None:
        R, G = dsel, gbase & gcond
        wR, wG = w[R].sum(), w[G].sum()
        row["fakes_boundary"] = frac(w[R & gbase & ~gcond].sum(), wR)
        #### dsel = rbase & rcond, so (rbase & ~dsel) == (rbase & ~rcond)
        row["misses_boundary"] = frac(w[G & rbase & ~dsel].sum(), wG) \
            if rbase is not None else float("nan")
    if not C.is_data and tgbase is not None:
        R, G = tsel, tgbase & tgcond
        row["fakes_boundary_tri"] = frac(w[R & tgbase & ~tgcond].sum(), w[R].sum())
        row["misses_boundary_tri"] = frac(w[G & C.t_tot & ~tsel].sum(), w[G].sum())
    return row


# --------------------------------------------------------------------------- #
#  candidate rules
# --------------------------------------------------------------------------- #
B1_X = (150., 200., 250., 300.)
B2_D = (2.6, 2.8, 3.0)
#### Garvita's proposals (2026-08-06).  G1 is the EXACT equal-pT form of the
#### momentum family's |vec pT1 + vec pT2| < 2*P0: for pT1 = pT2 = p the recoil
#### is I = 2p cos(dphi12/2), so I < 2*P0  <=>  dphi12 > 2 arccos(P0/p).  Her
#### message wrote "arccos(pT/100)", whose argument exceeds 1 for every jet in
#### the sample (pT > 170 GeV) -- arccos(100/pT1) is the only dimensionally
#### sensible reading, and it reproduces her stated intent (the I < 200 GeV cut).
G1_P0 = 100.0
G2_D = 2.9


def build_configs(C):
    """[(row-builder)] -> list of metric rows."""
    rows = []

    #### ---- reference: the trijet-priority veto in production today --------
    prio = C.t_tot & (C.pt3 > 185.0)
    gprio = (C.gnjet >= 3) & (C.gdpm > 1.0) & (C.gpt3 > 185.0)
    rows.append(metrics(
        C, "current", {"rule": "veto trijet-selected & pt3>185"},
        C.d_tot & ~prio, C.t_tot,
        gbase=C.g_seq, gcond=~gprio, rbase=C.d_tot))

    #### ---- reference: the momentum-balance split on A ---------------------
    rows.append(metrics(
        C, "Asplit0.1", {"thr": 0.10},
        C.d_nm1 & (C.A < 0.10), C.t_tot & (C.A_t > 0.10),
        gbase=C.g_nm1, gcond=(C.Agen < 0.10), rbase=C.d_nm1))

    #### ---- B1: pT-dependent back-to-backness ------------------------------
    for X in B1_X:
        with np.errstate(invalid="ignore", divide="ignore"):
            rcond = C.d12 > (PI - X / C.pt1)
            tcond = C.td12 < (PI - X / C.tpt1)
            gcond = C.gd12 > (PI - X / np.where(C.gpt1 > 0, C.gpt1, np.nan))
        rows.append(metrics(
            C, "B1", {"X": X, "rule": "dphi12 > pi - X/pt1"},
            C.d_nobal & rcond, C.t_tot & tcond,
            gbase=C.g_nobal, gcond=gcond, rbase=C.d_nobal))

    #### ---- B2: fixed tighter dphi12 on top of A < 0.3 ---------------------
    for d in B2_D:
        with np.errstate(invalid="ignore"):
            rcond = (C.A < 0.3) & (C.d12 > d)
            gcond = C.gd12 > d
        rows.append(metrics(
            C, "B2", {"dphi12": d, "rule": "dphi12 > d, A<0.3"},
            C.d_nm1 & rcond, C.t_tot & (C.td12 <= d),
            gbase=C.g_seq & np.isfinite(C.gd12), gcond=gcond, rbase=C.d_nm1))

    #### ---- G1: dphi12 > 2 arccos(P0/pT1), the exact equal-pT recoil cut ----
    def g1(d12, pt1):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(pt1 > 0, G1_P0 / np.where(pt1 > 0, pt1, 1.0), 1.0)
        return d12 > 2.0 * np.arccos(np.clip(r, 0.0, 1.0))
    rows.append(metrics(
        C, "G1", {"P0": G1_P0, "rule": "dphi12 > 2*arccos(100/pt1)",
                  "equivalent_to": "|vec pT1+pT2| < 200 GeV for pT1 = pT2"},
        C.d_nobal & g1(C.d12, C.pt1),
        C.t_tot & ~g1(C.td12, C.tpt1),
        gbase=C.g_nobal, gcond=g1(C.gd12, C.gpt1), rbase=C.d_nobal))

    #### ---- G2: the 2.9 point missing from the B2 ladder -------------------
    with np.errstate(invalid="ignore"):
        rows.append(metrics(
            C, "G2", {"dphi12": G2_D, "rule": "dphi12 > 2.9, A<0.3"},
            C.d_nm1 & (C.A < 0.3) & (C.d12 > G2_D),
            C.t_tot & (C.td12 <= G2_D),
            gbase=C.g_seq & np.isfinite(C.gd12), gcond=(C.gd12 > G2_D),
            rbase=C.d_nm1))

    #### ---- B3: the generalised phase-space split --------------------------
    for dcut in DCUTS:
        for fl in FLOORS:
            rows.append(metrics(
                C, "B3", {"dcut": dcut, "floor": fl},
                C.d_tot & ~C.in3(dcut, fl), C.t_tot & C.in3_t(dcut, fl),
                gbase=C.g_seq, gcond=~C.in3_gen(dcut, fl), rbase=C.d_tot,
                tgbase=C.tg_tot, tgcond=C.in3_gen_t(dcut, fl)))
    return rows


def label_of(r):
    p = r["params"]
    if r["name"] == "B1":
        return f"B1 dphi12>pi-{p['X']:.0f}/pT1"
    if r["name"] == "B2":
        return f"B2 dphi12>{p['dphi12']:.1f}"
    if r["name"] == "B3":
        return f"B3 dphimin>{p['dcut']:.1f}, pT3>{p['floor']}"
    if r["name"] == "G1":
        return f"G1 dphi12>2acos({p['P0']:.0f}/pT1)"
    if r["name"] == "G2":
        return f"G2 dphi12>{p['dphi12']:.1f}"
    return r["name"]


# --------------------------------------------------------------------------- #
#  selection criterion for "best"
# --------------------------------------------------------------------------- #
def passes(r):
    pk = r["per_pt"]["dijet_keep"]
    tail = [x for x in pk[-2:] if np.isfinite(x)]
    return (r["dijet_keep"] > 0.95 and all(x > 0.90 for x in tail)
            and r["orphans"] < 0.03)


def rank(rows, name=None):
    c = [r for r in rows if (name is None or r["name"] == name) and passes(r)]
    return sorted(c, key=lambda r: -r["trijet_keep"])


#### The yield criterion alone happily picks a rule whose gen mirror is broken:
#### a pT3 floor proportional to pT1 falls below 185 GeV for pT1 < ~620 GeV, and
#### GenJetAK8 is stored down to 100 GeV while FatJet stops at ~160, so the gen
#### veto fires on third jets that reco could never have seen (11-17% one-sided
#### fakes).  Anything recommended has to survive the migration gate too.
#### The gate uses the DIJET-side numbers only, because that is the denominator
#### the current scheme's 1.5%% one-sided migration is quoted against.  The
#### trijet-side columns are normalised to the trijet yield, which is ~2.3%% of
#### the dijet yield, so a 15%% trijet-side number is FEWER events than a 0.7%%
#### dijet-side one; it is reported, not gated on.
def migration_ok(r, thr=0.02):
    vals = [r["fakes_boundary"], r["misses_boundary"]]
    return all((not np.isfinite(v)) or v < thr for v in vals)


def recommend(rows, name=None):
    return [r for r in rank(rows, name) if migration_ok(r)]


# --------------------------------------------------------------------------- #
#  printing
# --------------------------------------------------------------------------- #
HDR = (f"{'configuration':<34s} {'di keep':>8s} {'tri keep':>9s} {'resid':>8s} "
       f"{'union':>7s} {'orph':>7s} {'fake_b':>7s} {'miss_b':>7s} "
       f"{'fake_t':>7s} {'miss_t':>7s}  ok")


def print_table(rows, title):
    print(f"\n=== {title} ===")
    print(HDR)
    print("-" * len(HDR))
    for r in rows:
        print(f"{label_of(r):<34s} {r['dijet_keep']:>8.4f} "
              f"{r['trijet_keep']:>9.4f} {r['resid_overlap']:>8.1e} "
              f"{r['union']:>7.4f} {r['orphans']:>7.4f} "
              f"{r['fakes_boundary']:>7.4f} {r['misses_boundary']:>7.4f} "
              f"{r['fakes_boundary_tri']:>7.4f} {r['misses_boundary_tri']:>7.4f}"
              f"  {'Y' if passes(r) else '.'}")


def print_perpt(r):
    print(f"\n--- per leading-jet pT bin: {label_of(r)} ---")
    print(f"{'pT bin [GeV]':>14s} {'dijet keep':>11s} {'trijet keep':>12s} "
          f"{'orphans':>9s}")
    p = r["per_pt"]
    for i in range(len(PT_EDGES) - 1):
        hi = "inf" if PT_EDGES[i + 1] > 5000 else f"{PT_EDGES[i+1]:.0f}"
        print(f"{PT_EDGES[i]:>7.0f}-{hi:<6s} {p['dijet_keep'][i]:>11.4f} "
              f"{p['trijet_keep'][i]:>12.4f} {p['orphans'][i]:>9.4f}")


# --------------------------------------------------------------------------- #
#  G1 vs the momentum family's |vec pT1 + vec pT2| < 200 GeV
# --------------------------------------------------------------------------- #
def g1_vs_imbalance(C, P0=G1_P0):
    """How far does the equal-pT approximation behind G1 actually sit from the
    exact recoil cut it is meant to reproduce?

    I = |vec pT1 + vec pT2| = sqrt(p1^2 + p2^2 + 2 p1 p2 cos dphi12) collapses
    to 2 p cos(dphi12/2) only for p1 = p2.  For an unbalanced pair the exact I
    stays above |p1 - p2| even at dphi12 = pi, so G1 -- which only ever looks at
    the angle -- keeps events the exact cut throws out.  The size of that leak
    is the asymmetry distribution of the sample, quantified here.
    """
    from scripts.channel_asym_split import imbalance
    w, X = C.w, 2.0 * P0
    I = imbalance(C.pt1, C.pt2, C.d12)
    I_t = imbalance(C.tpt1, C.tpt2, C.td12)
    with np.errstate(invalid="ignore", divide="ignore"):
        gcond = C.d12 > 2.0 * np.arccos(np.clip(P0 / C.pt1, 0.0, 1.0))
        gcond_t = C.td12 > 2.0 * np.arccos(np.clip(P0 / C.tpt1, 0.0, 1.0))
    icond, icond_t = I < X, I_t < X

    base = C.d_nobal & np.isfinite(I)
    wb = w[base].sum()
    only_g = base & gcond & ~icond
    only_i = base & ~gcond & icond
    print(f"\n=== G1 (angle only) vs the exact recoil cut I < {X:.0f} GeV ===")
    print(f"  agreement on the dijet no-topology baseline: "
          f"{frac(w[base & (gcond == icond)].sum(), wb):.4f} of the yield")
    for nm, m in (("G1 keeps, I<200 rejects", only_g),
                  ("G1 rejects, I<200 keeps", only_i)):
        if not m.any():
            print(f"  {nm:>24s}: none")
            continue
        mA = frac((w[m] * C.A[m]).sum(), w[m].sum())
        print(f"  {nm:>24s}: {frac(w[m].sum(), wb):.4f} of the baseline, "
              f"<A> = {mA:.3f}")
    print(f"  <A> over the whole baseline: "
          f"{frac((w[base] * C.A[base]).sum(), wb):.3f}")
    #### like-for-like yield metrics for the exact cut (cross-check against the
    #### momentum family's A1 row -- NOT a new scan row)
    dsel_i, tsel_i = C.d_nobal & icond, C.t_tot & ~icond_t
    print(f"  cross-check, exact I < {X:.0f} GeV with the SAME baselines: "
          f"dijet keep {frac(w[dsel_i].sum(), C.Wd):.4f}, "
          f"trijet keep {frac(w[tsel_i].sum(), C.Wt):.4f}, "
          f"orphans {frac(w[C.d_tot & ~dsel_i & ~C.t_tot].sum(), C.Wd):.4f}")


# --------------------------------------------------------------------------- #
#  |y| < 2.4 acceptance (NOT a split cut -- a pure yield question)
# --------------------------------------------------------------------------- #
def acceptance_2p4(C, cur, best, ycut=2.4):
    """Yield surviving a tightened rapidity acceptance on the leading jets.

    IMPORTANT RESULT: these skims already implement |y| < 2.4 on the two
    leading jets, even though the processors default to ycut = 2.5 and the cut
    bit is called "recoRap2p5".  The evidence is printed below -- the |y|
    density is smooth right up to 2.40 and then identically zero, while the
    THIRD jet (which no dijet cut touches) runs on to |y| ~ 2.62.  So the
    2.5 -> 2.4 yield ratio cannot be measured from this sample: it is 1.0000
    by construction of the input, not by physics.  What CAN be done is to
    extrapolate the cost from the density just below the edge, and to measure
    the jet-3 part of it from the trijet rapidity N-1 sample.
    """
    w = C.w
    acc_d = (C.ay[0] < ycut) & (C.ay[1] < ycut)
    acc_t = (C.tay[0] < ycut) & (C.tay[1] < ycut) & (C.tay[2] < ycut)
    dsel_cur = C.d_tot & ~(C.t_tot & (C.pt3 > 185.0))
    fl = best["params"]["floor"]
    dsel_b3 = C.d_tot & ~C.in3(best["params"]["dcut"], fl)
    tsel_b3 = C.t_tot & C.in3_t(best["params"]["dcut"], fl)

    print(f"\n=== acceptance |y| < {ycut} ===")
    ymax = np.fmax(C.ay[0], C.ay[1])
    tymax = np.fmax(np.fmax(C.tay[0], C.tay[1]), C.tay[2])
    fin = np.isfinite(ymax)
    print(f"  input check -- max |y| over the 2 leading dijet jets: "
          f"{np.nanmax(ymax):.4f};  over the 3 leading trijet jets: "
          f"{np.nanmax(tymax):.4f};  max |y| of the 3rd jet in the dijet "
          f"table (uncut): {np.nanmax(C.ay[2]):.4f}")
    print(f"{'sample':>44s} {'kept':>8s}")
    for nm, m, a in (
            ("dijet baseline (2 leading jets)", C.d_tot, acc_d),
            ("trijet baseline (3 leading jets)", C.t_tot, acc_t),
            ("dijet, current trijet-priority veto", dsel_cur, acc_d),
            (f"dijet, {label_of(best)}", dsel_b3, acc_d),
            (f"trijet, {label_of(best)}", tsel_b3, acc_t)):
        print(f"{nm:>44s} {frac(w[m & a].sum(), w[m].sum()):>8.4f}")
    print("  -> all 1.0000: the sample contains nothing between 2.4 and 2.5.")

    #### the density just below the edge, and what it implies for 2.40-2.50
    base = C.d_tot & fin
    W = w[base].sum()
    edges = np.arange(2.00, 2.55, 0.05)
    dens, ctr = [], []
    print(f"\n  |y|max density (fraction of the dijet baseline per 0.05):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = base & (ymax >= lo) & (ymax < hi)
        d = frac(w[m].sum(), W)
        print(f"    {lo:.2f}-{hi:.2f}: {d:.5f}  (n = {int(m.sum())})")
        if hi <= ycut + 1e-9:
            dens.append(d)
            ctr.append(0.5 * (lo + hi))
    dens, ctr = np.array(dens), np.array(ctr)
    flat = 2.0 * dens[-1]
    lin = np.polyval(np.polyfit(ctr, dens, 1), [ycut + 0.025, ycut + 0.075])
    print(f"  extrapolated 2.40-2.50 content: {max(lin.sum(), 0.0):.4f} "
          f"(linear in |y|) to {flat:.4f} (flat at the last bin) of the dijet "
          f"yield -- i.e. IF production really cut at 2.5, moving to 2.4 would "
          f"cost ~{100*min(flat, max(lin.sum(), 0.0)):.1f}-"
          f"{100*max(flat, max(lin.sum(), 0.0)):.1f}%")

    #### The jet-3 piece IS measurable: the dijet channel never cuts on the 3rd
    #### jet, so its rapidity survives uncensored in the file.  On the trijet
    #### rapidity N-1 sample this is the part of a 2.5 -> 2.4 tightening that
    #### the input can still resolve.
    #### (the trijet channel's own N-1 is useless here: its "recodphimin" etc.
    #### bits are defined sequentially and already carry the rapidity cut, so
    #### dropping the rapidity bit does not widen the sample.)
    print("  measurable piece -- 3rd jet |y| from the DIJET table, where no cut "
          "touches it:")
    for nm, b in (("dijet baseline, >=3 jets",
                   C.d_tot & (C.njet >= 3)),
                  ("dijet copy of the trijet phase space (dphimin>1, pT3>185)",
                   C.d_tot & C.in3(1.0, "185"))):
        b = b & np.isfinite(C.ay[2])
        Wn = w[b].sum()
        if Wn <= 0:
            continue
        print(f"    {nm}: {int(b.sum())} events;  "
              f"|y3| in [{ycut}, 2.5) = "
              f"{frac(w[b & (C.ay[2] >= ycut) & (C.ay[2] < 2.5)].sum(), Wn):.4f}"
              f",  |y3| >= 2.5 = "
              f"{frac(w[b & (C.ay[2] >= 2.5)].sum(), Wn):.4f}")


# --------------------------------------------------------------------------- #
#  plots
# --------------------------------------------------------------------------- #
def setup_mpl():
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.CMS)
    return plt, hep


def finish(fig, path):
    """Reserve strips at the top and bottom of the canvas, then save.

    Bottom: channel_asym_split.stamp puts the provenance line at figure
    y = 0.002; without the reserved strip the constrained-layout engine parks
    the x-axis label on top of it.
    Top: hep.cms.label(loc=0) draws "CMS Simulation Preliminary" and the
    "(13 TeV)" rlabel ABOVE the axes with an axes-relative transform, which the
    constrained-layout engine does not measure -- with the default rect the
    glyph tops are sliced off by the canvas edge.
    """
    import matplotlib.pyplot as plt
    try:
        fig.get_layout_engine().set(rect=(0.0, 0.032, 1.0, 0.955))
    except (AttributeError, TypeError):  # pragma: no cover - old matplotlib
        pass
    stamp(fig)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def plot_dists(C, outdir, best_b3, best_b1):
    plt, hep = setup_mpl()
    w = C.w

    #### ---- dphi12 ---------------------------------------------------------
    bins = np.linspace(0.0, PI, 64)
    fig, ax = plt.subplots(layout="constrained")
    for m, arr, c, lab in (
            (C.d_nobal, C.d12, COL[0], "dijet, no topology cut"),
            (C.t_tot, C.td12, COL[2], "trijet selected"),
            (C.d_tot & C.t_tot, C.d12, COL[3], "overlap (dijet AND trijet)")):
        mm = m & np.isfinite(arr)
        h, _ = np.histogram(arr[mm], bins=bins, weights=w[mm])
        hep.histplot(h / h.sum(), bins, ax=ax, histtype="step", lw=2.5,
                     color=c, label=lab)
    X = best_b1["params"]["X"]
    for x, c, ls, lab in (
            (2.0, COL[4], ":", r"current dijet $\Delta\phi_{12} > 2$"),
            (PI - X / 300., COL[1], "-",
             rf"B1 at $p_{{T,1}}=300$: $\pi - {X:.0f}/p_{{T,1}}$"),
            (PI - X / 800., COL[5], "--",
             rf"B1 at $p_{{T,1}}=800$: $\pi - {X:.0f}/p_{{T,1}}$")):
        ax.axvline(x, color=c, lw=2, ls=ls, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta\phi_{12}$")
    ax.set_ylabel("Fraction of events / bin")
    ax.set_xlim(0.0, PI)
    ax.set_ylim(1e-5, ax.get_ylim()[1] * 30)
    ax.legend(loc="upper left", fontsize=14, frameon=False)
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
    finish(fig, outdir / "split_ang_dist_dphi12.png")

    #### ---- dphimin (>=3 jets only) ---------------------------------------
    bins = np.linspace(0.0, PI, 64)
    fig, ax = plt.subplots(layout="constrained")
    for m, arr, nj, c, lab in (
            (C.d_nobal, C.dpm, C.njet, COL[0], "dijet, no topology cut"),
            (C.t_tot, C.tdpm, C.tnjet, COL[2], "trijet selected"),
            (C.d_tot & C.t_tot, C.dpm, C.njet, COL[3],
             "overlap (dijet AND trijet)")):
        mm = m & (nj >= 3) & np.isfinite(arr)
        h, _ = np.histogram(arr[mm], bins=bins, weights=w[mm])
        hep.histplot(h / h.sum(), bins, ax=ax, histtype="step", lw=2.5,
                     color=c, label=lab)
    #### ymax keeps the split markers clear of the upper-left legend box
    for x, c, ls in ((0.5, COL[5], ":"), (1.0, COL[1], "-"), (1.5, COL[4], "--")):
        ax.axvline(x, color=c, lw=2, ls=ls, ymax=0.68,
                   label=rf"B3 split $\Delta\phi_{{min}} = {x:.1f}$"
                         + (" (current)" if x == 1.0 else ""))
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta\phi_{min}$ (3 leading jets, $N_{jet}\geq 3$)")
    ax.set_ylabel("Fraction of events / bin")
    ax.set_xlim(0.0, PI)
    ax.set_ylim(1e-4, ax.get_ylim()[1] * 40)
    ax.legend(loc="upper left", fontsize=14, frameon=False)
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
    finish(fig, outdir / "split_ang_dist_dphimin.png")


def plot_eff(rows_for_eff, outdir):
    from matplotlib.lines import Line2D
    plt, hep = setup_mpl()
    ctr = 0.5 * (PT_EDGES[:-1] + np.minimum(PT_EDGES[1:], 1200.))
    fig, ax = plt.subplots(layout="constrained")
    cols = (COL[0], COL[2], COL[4])
    for r, c in zip(rows_for_eff, cols):
        p = r["per_pt"]
        ax.plot(ctr, p["dijet_keep"], "o-", lw=2.5, ms=8, color=c)
        ax.plot(ctr, p["trijet_keep"], "s--", lw=2.5, ms=8, color=c, alpha=0.85)
    ax.axhline(1.0, color="0.7", lw=1, ls=":")
    ax.set_xlabel(r"$p_{T,1}$ [GeV]")
    ax.set_ylabel("Fraction of channel baseline kept")
    ax.set_xlim(200, 1200)
    ax.set_ylim(0.0, 2.05)
    #### colour = scheme, line style = channel: two small legends beat one
    #### six-row block sitting on top of the trijet curves
    h1 = [Line2D([], [], color=c, lw=3, label=label_of(r))
          for r, c in zip(rows_for_eff, cols)]
    h2 = [Line2D([], [], color="0.25", lw=2.5, ls="-", marker="o", ms=8,
                 label="dijet channel"),
          Line2D([], [], color="0.25", lw=2.5, ls="--", marker="s", ms=8,
                 label="trijet channel")]
    leg = ax.legend(handles=h1, loc="upper left", fontsize=14, frameon=False)
    ax.add_artist(leg)
    #### borderaxespad keeps the square marker of the "trijet channel" entry
    #### clear of the right spine
    ax.legend(handles=h2, loc="upper right", fontsize=14, frameon=False,
              borderaxespad=1.2, handletextpad=0.6)
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
    finish(fig, outdir / "split_ang_eff_vs_pt.png")


def plot_summary(rows, outdir, annotate):
    from matplotlib.lines import Line2D
    plt, hep = setup_mpl()
    fig, ax = plt.subplots(layout="constrained")
    fam_col = {"B1": COL[0], "B2": COL[1], "B3": COL[3],
               "Asplit0.1": COL[5], "current": COL[2],
               #### G1 is a B1-like angular rule, G2 a B2 ladder point: same
               #### family colours where that is true, marker shape carries the
               #### "proposed by Garvita" distinction
               "G1": COL[0], "G2": COL[1]}
    fam_mrk = {"G1": "D", "G2": "s"}
    seen = []
    for r in rows:
        if r["name"] == "current":
            continue
        s = 60 + 5200 * max(r["orphans"], 0.0)
        if r["name"] not in seen:
            seen.append(r["name"])
        ax.scatter(r["dijet_keep"], r["trijet_keep"], s=s,
                   marker=fam_mrk.get(r["name"], "o"),
                   facecolor=fam_col[r["name"]], edgecolor="k", lw=0.8,
                   alpha=0.7, zorder=4 if r["name"] in fam_mrk else 3)
    cur = [r for r in rows if r["name"] == "current"][0]
    #### hollow star: the current scheme sits on top of the B3 (dcut=1.0) points
    #### it is being compared against, and a filled marker hides them
    ax.scatter(cur["dijet_keep"], cur["trijet_keep"], s=700, marker="*",
               facecolor="none", edgecolor=COL[2], lw=2.2, zorder=6)
    for r, dx, dy in annotate:
        ax.annotate(label_of(r), (r["dijet_keep"], r["trijet_keep"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=13,
                    ha="left", va="center", color="0.15", zorder=6,
                    arrowprops=dict(arrowstyle="-", lw=0.9, color="0.45",
                                    shrinkB=6))
    ax.annotate("current scheme", (cur["dijet_keep"], cur["trijet_keep"]),
                textcoords="offset points", xytext=(14, 26), fontsize=13,
                ha="left", va="center", color=COL[2], zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.9, color=COL[2],
                                shrinkB=10))
    ax.axvline(0.95, color="0.7", lw=1.2, ls="--")
    ax.text(0.952, 0.015, "dijet keep = 0.95", rotation=90, fontsize=12,
            color="0.45", transform=ax.get_xaxis_transform(), va="bottom")
    ax.set_xlabel("Dijet yield kept")
    ax.set_ylabel("Trijet yield kept")
    ax.set_xlim(0.48, 1.16)
    ax.set_ylim(0.0, 1.62)
    #### uniform proxy markers: the real ones encode the orphan fraction
    h = [Line2D([], [], ls="", marker=fam_mrk.get(n, "o"), ms=11, mec="k",
                mew=0.8, mfc=fam_col[n], label=n, alpha=0.7) for n in seen]
    h.append(Line2D([], [], ls="", marker="*", ms=20, mec=COL[2], mew=2.0,
                    mfc="none", label="current scheme"))
    ax.legend(handles=h, loc="upper left", fontsize=14, frameon=False, ncol=2,
              handletextpad=0.4, columnspacing=1.2)
    ax.text(0.985, 0.965, "marker area $\\propto$ orphan fraction",
            transform=ax.transAxes, fontsize=13, va="top", ha="right",
            color="0.35")
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
    finish(fig, outdir / "split_ang_summary.png")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dijet-mc", default="/Users/aritra/cernbox (2)/dijet_skims")
    ap.add_argument("--trijet-mc", default="/Users/aritra/cernbox (2)/trijet_skims/mc")
    ap.add_argument("--dijet-data", default="/Users/aritra/Downloads/dijet_data_skims")
    ap.add_argument("--trijet-data", default="/Users/aritra/cernbox (2)/trijet_skims/data")
    ap.add_argument("--out", default=str(SCRATCH))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    D, T = build_join(sorted(Path(args.dijet_mc).glob("*.pkl")),
                      sorted(Path(args.trijet_mc).glob("*.pkl")),
                      outdir / "split_ang_join_mc_v2.npz")
    C = Ctx(D, T, is_data=False)

    #### ---- closure -------------------------------------------------------
    rebuilt = C.d_nm1 & (C.A < 0.3)
    print(f"\nclosure: rebuilt dijet(N-1 & A<0.3) vs stored recoTot_seq -- "
          f"{int((rebuilt & ~C.d_tot).sum())} extra, "
          f"{int((~rebuilt & C.d_tot).sum())} missing, "
          f"of {int(C.d_tot.sum())} stored")
    print(f"baselines: P(tri|di) = {frac(C.Wb, C.Wd):.4f}   "
          f"P(di|tri) = {frac(C.Wb, C.Wt):.4f}   "
          f"Jaccard = {frac(C.Wb, C.Wu):.4f}")
    #### storage floors, for the gen-mirror caveat on fractional pT3 floors
    m3 = (C.njet >= 3) & (C.pt3 > 0)
    g3 = (C.gnjet >= 3) & (C.gpt3 > 0)
    print(f"storage floors: min reco pT3 = {C.pt3[m3].min():.1f} GeV, "
          f"min gen pT3 = {C.gpt3[g3].min():.1f} GeV")
    tr = C.t_tot & (C.tpt3 > 0)
    tg = C.tg_tot & (C.tgnjet >= 3) & (C.tgpt3 > 0)
    if tr.any() and tg.any():
        print(f"trijet channel's own pT3 reach: reco min "
              f"{C.tpt3[tr].min():.1f} / 1%% {np.percentile(C.tpt3[tr], 1):.1f}"
              f" GeV;  gen min {C.tgpt3[tg].min():.1f} / 1%% "
              f"{np.percentile(C.tgpt3[tg], 1):.1f} GeV  "
              f"(a 185 GeV split floor sits ABOVE the gen reach -> the "
              f"trijet-side boundary migration below)")

    rows = build_configs(C)
    refs = [r for r in rows if r["name"] in ("current", "Asplit0.1")]
    print_table(refs, "reference schemes")
    print_table([r for r in rows if r["name"] == "B1"],
                "B1: dphi12 > pi - X/pT1  (dphi AND asymmetry cuts removed)")
    print_table([r for r in rows if r["name"] in ("B2", "G2")],
                "B2/G2: dphi12 > d on top of A < 0.3")
    print_table([r for r in rows if r["name"] == "G1"],
                "G1: dphi12 > 2*arccos(100/pT1)  (equal-pT form of I < 200 GeV)")
    print_table([r for r in rows if r["name"] == "B3"],
                "B3: dijet vetoes / trijet requires (Njet>=3, dphimin>dcut, pT3>floor)")
    print("  resid  = residual overlap / baseline overlap (JER-seed artifact)")
    print("  orph   = dijet events lost to BOTH channels / dijet baseline")
    print("  fake_b/miss_b = boundary-only migration, dijet side")
    print("  fake_t/miss_t = boundary-only migration, trijet side (B3 only)")
    print("  ok     = di keep>0.95, >0.90 in the two highest pT bins, orph<0.03")

    ranked = rank(rows)
    print_table(ranked[:12],
                "passing the yield criterion, best trijet keep first")
    rec = recommend(rows)
    print_table(rec[:12], "same, additionally passing the dijet-side "
                          "migration gate (<2%) -- the recommendation list")

    best_b3 = (recommend(rows, "B3") or rank(rows, "B3") or
               sorted([r for r in rows if r["name"] == "B3"],
                      key=lambda r: -r["trijet_keep"]))[0]
    best_b1 = (recommend(rows, "B1") or rank(rows, "B1") or
               sorted([r for r in rows if r["name"] == "B1"],
                      key=lambda r: -r["dijet_keep"]))[0]
    cur = refs[0]
    g1row = [r for r in rows if r["name"] == "G1"][0]
    for r in ([cur] + rec[:3] + [best_b1, g1row]):
        print_perpt(r)

    g1_vs_imbalance(C)
    acceptance_2p4(C, cur, best_b3)

    #### ---- data cross-check on the best few ------------------------------
    Dd, Td = build_join(sorted(Path(args.dijet_data).glob("*.pkl")),
                        sorted(Path(args.trijet_data).glob("*.pkl")),
                        outdir / "split_ang_join_data_v2.npz")
    Cd = Ctx(Dd, Td, is_data=True)
    print(f"\ndata baselines: P(tri|di) = {frac(Cd.Wb, Cd.Wd):.4f}   "
          f"P(di|tri) = {frac(Cd.Wb, Cd.Wt):.4f}")
    want = [(r["name"], json.dumps(r["params"], sort_keys=True))
            for r in ([cur] + rec[:3] + [best_b1])]
    drows = [r for r in build_configs(Cd)
             if (r["name"], json.dumps(r["params"], sort_keys=True)) in want]
    for r in drows:
        r["sample"] = "data2018"
    print_table(drows, "data 2018 cross-check (prescale-weighted, no migration)")

    #### ---- json ----------------------------------------------------------
    jf = outdir / "split_scan_angular.json"
    #### v2 (2026-08-06) adds the G1/G2 rows and wraps the row list in an object
    #### so the file can carry a version; v1 was a bare list.
    json.dump({"version": 2, "family": "angular",
               "rows": rows + drows}, open(jf, "w"), indent=1, default=float)
    print("\nwrote", jf)

    if not args.no_plots:
        annotate = [(best_b3, -265, -55), (best_b1, -175, -45),
                    (refs[1], -150, -55), (g1row, -200, -42)]
        plot_dists(C, outdir, best_b3, best_b1)
        plot_eff([best_b3, best_b1, cur], outdir)
        plot_summary(rows, outdir, annotate)


if __name__ == "__main__":
    main()
