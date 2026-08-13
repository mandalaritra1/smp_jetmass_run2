#!/usr/bin/env python
"""Momentum-balance split rules that make dijet and trijet orthogonal.

The two hadronic channels overlap badly (trijet-selected events are ~89% a
subset of the dijet selection). This script scans split rules of the form

    dijet  : V < thr        trijet : V > thr

where BOTH channels cut on the SAME event-level momentum-balance variable V, so
the split is orthogonal by construction and no priority veto is needed. The
question is not whether it works -- it does, trivially -- but what it costs:

    dijet_keep    yield kept relative to the current dijet selection
    trijet_keep   yield kept relative to the current trijet selection
    orphans       events removed from dijet AND not picked up by trijet, i.e.
                  gone from the measurement entirely -- the real price
    union         combined yield relative to the current (overlapping) union
    fakes/misses  gen<->reco migration created by the boundary itself
    per_pt        the same, per leading-jet pT bin: a rule whose cost swings
                  across the spectrum distorts the measured pT dependence

Candidate variables (all computed from the leading AK8 pair / the rest of the
event, all with an obvious Z+jet analogue):

    A1  I = |vec pT1 + vec pT2| < X          (recoil of the leading pair)
    A2  I < c * pT1                          (pT-proportional recoil)
    A3  I < max(200 GeV, c * pT1)            (floored, flat at low pT)
    A4  pT3 < X                              (third-jet veto)  [keep-topo]
    A5  pT3 < c * pT1                                          [keep-topo]
    A6  z3 = pT3 / (pT1 + pT2) < z                             [keep-topo]
    A7  HTx = sum pT of all jets beyond the leading two < X

"[keep-topo]" rules are ADDED on top of the current dijet selection; the others
REPLACE the dphi12 > 2 and asymmetry A < 0.3 cuts (they are themselves
back-to-back / balance requirements, so keeping the old ones double-counts).

Everything runs on the joined 1/1000 event-display tables (dijet rho_skim10 x
trijet rho_skim), where both channels' cut bits are known per event.

    python scripts/split_scan_momentum.py --outdir DIR
    python scripts/split_scan_momentum.py --outdir DIR --no-plots
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.channel_overlap import (  # noqa: E402
    DIJET_CUTS, TRIJET_CUTS, PT_EDGES, bit, dedupe)
from scripts.studies.split.channel_asym_split import (  # noqa: E402
    load_channel, allbits, asym, imbalance, frac, stamp,
    DIJET_RECO_NM1, DIJET_RECO_NOBAL, DIJET_GEN_NM1, DIJET_GEN_NOBAL)

#### GenJetAK8 is stored down to ~100 GeV, FatJet only from ~170, so any
#### gen-side condition on the THIRD jet needs a pT floor or it fires on jets
#### reco could never have seen (same issue that forced pT3 > 185 on the gen
#### trijet-priority veto -- see dijet_processor._in_trijet_phase_space).
GEN_PT3_FLOOR = 185.0
GEN_HT_FLOOR = 170.0

A1_X = (150., 200., 250., 300.)
A2_C = (0.2, 0.3, 0.4, 0.5)
A3_C = (0.25, 0.3, 0.4)
A4_X = (185., 220., 260., 300.)
A5_C = (0.25, 0.35, 0.45)
A6_Z = (0.10, 0.15, 0.20, 0.25)
A7_X = (100., 200., 300.)

COLORS = ("#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _view(a):
    return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()


def sum_pt_beyond_lead(ek, table, n, floor=None, nlead=2):
    """Per-event scalar sum of jet pT over jets with jetidx >= nlead.

    Same event<->jet-row join as plot_njet_hadronic.counts_above, summing pT
    instead of counting. jetidx is the index in the pT-ordered NanoAOD
    collection, so `jetidx >= 2` is exactly (HT of all stored jets) - pT1 - pT2
    with pT1/pT2 defined the way _topology defines them.
    """
    jk = np.stack([np.asarray(table[c].value, dtype=np.int64) for c in
                   ("dataset_id", "run", "lumi", "event")], axis=1)
    if len(jk) == 0:
        return np.zeros(n)
    jidx = np.asarray(table["jetidx"].value)
    jpt = np.asarray(table["pt"].value).astype(np.float64)
    order = np.argsort(_view(ek), kind="stable")
    pos = np.clip(np.searchsorted(_view(ek)[order], _view(jk)), 0, n - 1)
    hit = _view(ek)[order][pos] == _view(jk)
    evrow = order[pos]
    m = hit & (jidx >= nlead)
    if floor is not None:
        m &= jpt > floor
    s = np.zeros(n)
    np.add.at(s, evrow[m], jpt[m])
    return s


def load_htx(paths):
    """HTx = sum pT of everything past the leading pair, reco and gen.

    load_channel() does not provide it (it only keeps the leading three jets),
    so the pickles are read a second time. The events-table `ht_reco`/`ht_gen`
    columns are NOT usable for the gen side: they sum the full GenJetAK8
    collection with no pT floor.
    """
    out = {"key": [], "htx": [], "ghtx": []}
    for p in paths:
        sk = pickle.load(open(p, "rb"))["skim"]
        ev = sk["events"]
        n = len(ev["event"].value)
        if n == 0:
            continue
        ek = np.stack([np.asarray(ev[c].value, dtype=np.int64) for c in
                       ("dataset_id", "run", "lumi", "event")], axis=1)
        out["key"].append(ek)
        out["htx"].append(sum_pt_beyond_lead(ek, sk["alljets"], n))
        out["ghtx"].append(sum_pt_beyond_lead(ek, sk["allgenjets"], n,
                                              floor=GEN_HT_FLOOR))
    return {k: np.concatenate(v) for k, v in out.items()}


def load_all(paths, tag):
    print(f"{tag} skims:")
    C, dsets = load_channel(paths)
    H = load_htx(paths)
    assert (H["key"] == C["key"]).all(), "HTx pass misaligned with load_channel"
    C["htx"], C["ghtx"] = H["htx"], H["ghtx"]
    return C, dsets


def build_vars(dpaths, tpaths, is_data):
    """Join the two channels and compute every scanned variable, both levels."""
    D, _ = load_all(dpaths, "dijet")
    T, _ = load_all(tpaths, "trijet")
    dm, tm = dedupe(D["key"]), dedupe(T["key"])
    D = {k: v[dm] for k, v in D.items()}
    T = {k: v[tm] for k, v in T.items()}
    do, to = np.argsort(_view(D["key"])), np.argsort(_view(T["key"]))
    _, di, ti = np.intersect1d(_view(D["key"])[do], _view(T["key"])[to],
                               return_indices=True)
    di, ti = do[di], to[ti]
    print(f"\njoin: {len(di)} common events")

    #### The stored weight ALREADY carries xs*lumi*1000/sumw for MC (applied in
    #### HadronicProcessorBase.postprocess) and the trigger PRESCALE weight for
    #### data. Do NOT multiply by any xs lookup -- channel_overlap.xs_lut
    #### double-counts it (known bug).
    V = {"w": D["weight"][di].astype(float)}

    z = lambda x: np.nan_to_num(x, nan=0.0)  # noqa: E731
    #### ---- dijet-side reco topology
    V["pt1"], V["pt2"] = D["pt1"][di], D["pt2"][di]
    V["pt3"] = z(D["pt3"][di])
    V["dphi12"], V["njet"] = D["dphi12"][di], D["njet"][di]
    V["A"] = asym(V["pt1"], V["pt2"])
    V["I"] = imbalance(V["pt1"], V["pt2"], V["dphi12"])
    V["htx"] = D["htx"][di]
    #### ---- trijet-side reco topology, from T's OWN jets (different JER seed)
    V["tpt1"], V["tpt2"] = T["pt1"][ti], T["pt2"][ti]
    V["tpt3"] = z(T["pt3"][ti])
    V["tA"] = asym(V["tpt1"], V["tpt2"])
    V["tI"] = imbalance(V["tpt1"], V["tpt2"], T["dphi12"][ti])
    V["thtx"] = T["htx"][ti]
    #### ---- gen topology (dijet copy; gen jets are never smeared, so the two
    #### productions agree bit-for-bit here)
    V["gpt1"], V["gpt2"] = D["gpt1"][di], D["gpt2"][di]
    V["gpt3"] = z(D["gpt3"][di])
    V["gnjet"], V["gdphimin"] = D["gnjet"][di], D["gdphimin"][di]
    V["gA"] = asym(V["gpt1"], V["gpt2"])
    V["gI"] = imbalance(V["gpt1"], V["gpt2"], D["gdphi12"][di])
    V["ghtx"] = D["ghtx"][di]

    #### ---- selections
    db, tb = D["cutbits"][di], T["cutbits"][ti]
    V["d_tot"] = (db & bit(DIJET_CUTS, "recoTot_seq")) != 0
    V["t_tot"] = (tb & bit(TRIJET_CUTS, "recoTot_seq")) != 0
    V["d_nm1"] = allbits(db, DIJET_CUTS, DIJET_RECO_NM1) & np.isfinite(V["A"])
    V["d_nobal"] = allbits(db, DIJET_CUTS, DIJET_RECO_NOBAL) & np.isfinite(V["I"])
    gfin = np.isfinite(V["gI"])
    V["g_nm1"] = allbits(db, DIJET_CUTS, DIJET_GEN_NM1) & gfin
    V["g_nobal"] = allbits(db, DIJET_CUTS, DIJET_GEN_NOBAL) & gfin
    V["g_full"] = allbits(db, DIJET_CUTS,
                          DIJET_GEN_NM1 + ("genAsym0p3",)) & gfin
    V["is_data"] = np.array(is_data)
    return V


# --------------------------------------------------------------------------
# rule definitions
# --------------------------------------------------------------------------
def build_configs(V):
    """One entry per scanned working point: the reco/gen masks it implies."""
    pt1, pt2, pt3, I, htx = V["pt1"], V["pt2"], V["pt3"], V["I"], V["htx"]
    tp1, tp2, tp3, tI, thtx = V["tpt1"], V["tpt2"], V["tpt3"], V["tI"], V["thtx"]
    gp1, gp2, gp3, gI, ghtx = V["gpt1"], V["gpt2"], V["gpt3"], V["gI"], V["ghtx"]
    with np.errstate(invalid="ignore"):
        z3 = np.nan_to_num(pt3 / (pt1 + pt2), nan=0.0)
        tz3 = np.nan_to_num(tp3 / (tp1 + tp2), nan=0.0)
        gsum = np.nan_to_num(gp1 + gp2, nan=0.0)
    cfgs = []

    def add(name, params, variant, rcond, tcond, gcond, label):
        rbase = V["d_nobal"] if variant == "replace-topo" else V["d_tot"]
        gbase = V["g_nobal"] if variant == "replace-topo" else V["g_full"]
        cfgs.append(dict(family="momentum", name=name, params=params,
                         variant=variant, label=label,
                         dsel=rbase & rcond, tsel=V["t_tot"] & tcond,
                         rbase=rbase, rcond=rcond, gbase=gbase, gcond=gcond))

    for X in A1_X:
        add("A1", {"X": X}, "replace-topo", I < X, tI > X, gI < X,
            f"A1:I<{X:.0f}")
    for c in A2_C:
        add("A2", {"c": c}, "replace-topo", I < c * pt1, tI > c * tp1,
            gI < c * gp1, f"A2:I<{c:g}pT1")
    for c in A3_C:
        add("A3", {"c": c}, "replace-topo",
            I < np.maximum(200., c * pt1), tI > np.maximum(200., c * tp1),
            gI < np.maximum(200., c * gp1), f"A3:I<max(200,{c:g}pT1)")
    #### gen mirrors of the pT3 rules: the gen third jet only counts if it is
    #### above the reco-equivalent floor, hence the max(..., 185) below.
    for X in A4_X:
        add("A4", {"X": X}, "keep-topo", pt3 < X, tp3 > X, gp3 < X,
            f"A4:pT3<{X:.0f}")
    for c in A5_C:
        add("A5", {"c": c}, "keep-topo", pt3 < c * pt1, tp3 > c * tp1,
            gp3 < np.maximum(GEN_PT3_FLOOR, c * gp1), f"A5:pT3<{c:g}pT1")
    for zz in A6_Z:
        add("A6", {"z": zz}, "keep-topo", z3 < zz, tz3 > zz,
            gp3 < np.maximum(GEN_PT3_FLOOR, zz * gsum), f"A6:z3<{zz:g}")
    for X in A7_X:
        add("A7", {"X": X}, "replace-topo", htx < X, thtx > X, ghtx < X,
            f"A7:HTx<{X:.0f}")

    #### ---- references -------------------------------------------------
    #### (1) production today: trijet takes priority, dijet vetoes the trijet
    #### phase space. NOT orthogonal by construction -- it is exact only
    #### because the veto reads the other channel's decision.
    prio = V["t_tot"] & (pt3 > GEN_PT3_FLOOR)
    gveto = (V["gnjet"] >= 3) & (V["gdphimin"] > 1.0) & (gp3 > GEN_PT3_FLOOR)
    cfgs.append(dict(family="reference", name="current", params={},
                     variant="keep-topo", label="current",
                     dsel=V["d_tot"] & ~prio, tsel=V["t_tot"],
                     rbase=V["d_tot"], rcond=~prio,
                     gbase=V["g_full"], gcond=~gveto))
    #### (2) the pT-asymmetry split (scripts/channel_asym_split.py), the other
    #### single-variable orthogonal proposal, for a like-for-like cost.
    cfgs.append(dict(family="reference", name="Asplit0.1", params={"thr": 0.1},
                     variant="replace-asym", label="A<0.1",
                     dsel=V["d_nm1"] & (V["A"] < 0.1),
                     tsel=V["t_tot"] & (V["tA"] > 0.1),
                     rbase=V["d_nm1"], rcond=V["A"] < 0.1,
                     gbase=V["g_nm1"], gcond=V["gA"] < 0.1))
    return cfgs


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def metrics(V, cfg, is_data):
    w, d_tot, t_tot = V["w"], V["d_tot"], V["t_tot"]
    Wd, Wt = w[d_tot].sum(), w[t_tot].sum()
    Wb, Wu = w[d_tot & t_tot].sum(), w[d_tot | t_tot].sum()
    dsel, tsel = cfg["dsel"], cfg["tsel"]
    row = {"family": cfg["family"], "name": cfg["name"], "params": cfg["params"],
           "variant": cfg["variant"], "label": cfg["label"],
           "sample": "data2018" if is_data else "qcd_mg_pythia8_2018",
           "dijet_keep": frac(w[dsel].sum(), Wd),
           "trijet_keep": frac(w[tsel].sum(), Wt),
           "resid_overlap": frac(w[dsel & tsel].sum(), Wb),
           "orphans": frac(w[d_tot & ~dsel & ~t_tot].sum(), Wd),
           "union": frac(w[dsel | tsel].sum(), Wu)}

    #### boundary-only migration: the event passes everything except the
    #### scanned condition at the other level, so the boundary ALONE moved it.
    if is_data:
        row["fakes_boundary"] = row["misses_boundary"] = float("nan")
    else:
        R, G = dsel, cfg["gbase"] & cfg["gcond"]
        row["fakes_boundary"] = frac(w[R & cfg["gbase"] & ~cfg["gcond"]].sum(),
                                     w[R].sum())
        row["misses_boundary"] = frac(w[G & cfg["rbase"] & ~cfg["rcond"]].sum(),
                                      w[G].sum())

    pt1 = V["pt1"]
    dk, tk, orph = [], [], []
    for i in range(len(PT_EDGES) - 1):
        m = (pt1 >= PT_EDGES[i]) & (pt1 < PT_EDGES[i + 1])
        wd, wt = w[m & d_tot].sum(), w[m & t_tot].sum()
        dk.append(frac(w[m & dsel].sum(), wd))
        tk.append(frac(w[m & tsel].sum(), wt))
        orph.append(frac(w[m & d_tot & ~dsel & ~t_tot].sum(), wd))
    row["per_pt"] = {"edges": PT_EDGES.tolist(), "dijet_keep": dk,
                     "trijet_keep": tk, "orphans": orph}
    return row


def print_table(rows, title):
    print(f"\n=== {title} ===")
    hdr = (f"{'rule':>22s} {'variant':>13s} {'di keep':>8s} {'tri keep':>9s} "
           f"{'orphans':>8s} {'union':>7s} {'resid':>8s} {'fakes':>7s} "
           f"{'misses':>7s}")
    print(hdr)
    print("-" * len(hdr))
    key = lambda r: (r["family"] != "reference", r["name"],  # noqa: E731
                     list(r["params"].values()))
    for r in sorted(rows, key=key):
        print(f"{r['label']:>22s} {r['variant']:>13s} {r['dijet_keep']:>8.3f} "
              f"{r['trijet_keep']:>9.3f} {r['orphans']:>8.3f} "
              f"{r['union']:>7.3f} {r['resid_overlap']:>8.4f} "
              f"{r['fakes_boundary']:>7.3f} {r['misses_boundary']:>7.3f}")


def pick_best(rows, max_orphans=0.05, min_dijet=0.93, n=2):
    """Best n working points per sub-family: most trijet yield kept, subject to
    keeping the dijet channel and not orphaning events."""
    out = []
    for fam in sorted({r["name"] for r in rows if r["family"] == "momentum"}):
        cand = [r for r in rows if r["name"] == fam
                and r["orphans"] < max_orphans and r["dijet_keep"] > min_dijet]
        cand.sort(key=lambda r: -r["trijet_keep"])
        out += cand[:n]
    return out


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def _setup():
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.CMS)
    return plt, hep


def _footer(fig):
    """Reserve the bottom strip so the provenance stamp clears the x label."""
    try:
        fig.get_layout_engine().set(rect=(0.0, 0.024, 1.0, 0.976))
    except Exception:
        pass


def plot_dists(V, outdir):
    plt, hep = _setup()
    pt1, pt2, pt3 = V["pt1"], V["pt2"], V["pt3"]
    tp1, tp2, tp3 = V["tpt1"], V["tpt2"], V["tpt3"]
    with np.errstate(invalid="ignore"):
        z3 = np.nan_to_num(pt3 / (pt1 + pt2), nan=0.0)
        tz3 = np.nan_to_num(tp3 / (tp1 + tp2), nan=0.0)
        r1 = np.nan_to_num(V["I"] / pt1, nan=0.0)
        tr1 = np.nan_to_num(V["tI"] / tp1, nan=0.0)
    specs = [
        ("I", V["I"], V["tI"], np.linspace(0, 700, 71), 200.,
         r"$|\vec{p}_{T,1} + \vec{p}_{T,2}|$ [GeV]", "10 GeV", "A1", 8),
        ("Iopt1", r1, tr1, np.linspace(0, 2.0, 81), 0.3,
         r"$|\vec{p}_{T,1} + \vec{p}_{T,2}| / p_{T,1}$", "0.025", "A2", 8),
        #### these three carry a delta-function spike at 0 (no third jet) that
        #### tops out near 1: give them a labelled decade of headroom so the
        #### spike is unmistakably inside the frame.
        ("pt3", pt3, tp3, np.linspace(0, 900, 91), 185.,
         r"$p_{T,3}$ [GeV]", "10 GeV", "A4", 30),
        ("z3", z3, tz3, np.linspace(0, 0.8, 81), 0.15,
         r"$z_3 = p_{T,3} / (p_{T,1} + p_{T,2})$", "0.01", "A6", 30),
        ("htx", V["htx"], V["thtx"], np.linspace(0, 1200, 61), 200.,
         r"$H_T^{\mathrm{x}} = \sum_{i\geq 3} p_{T,i}$ [GeV]", "20 GeV", "A7", 30),
    ]
    written = []
    for tag, dv, tv, bins, thr, xlab, binw, rule, head in specs:
        fig, ax = plt.subplots(layout="constrained")
        hmax = 0.0
        for m, arr, c, lab in (
                (V["d_nobal"], dv, COLORS[0], "dijet, no topology cut"),
                (V["t_tot"], tv, COLORS[2], "trijet selected"),
                (V["d_tot"] & V["t_tot"], dv, COLORS[3], "overlap (both)")):
            h, _ = np.histogram(arr[m], bins=bins, weights=V["w"][m])
            if h.sum() <= 0:
                continue
            h = h / h.sum()
            hmax = max(hmax, h.max())
            hep.histplot(h, bins, ax=ax, histtype="step", lw=2.5,
                         color=c, label=lab)
        ax.axvline(thr, color=COLORS[1], lw=2.5)
        #### park the threshold label in the headroom band, not on the curves
        ax.text(thr + 0.014 * (bins[-1] - bins[0]), 0.965,
                f"{rule} split", transform=ax.get_xaxis_transform(),
                color=COLORS[1], fontsize=17, rotation=90, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
        ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(f"Fraction of events / {binw}")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(1e-5, hmax * head)
        ax.legend(loc="upper right", fontsize=16)
        hep.cms.label("Preliminary", data=False, loc=0, ax=ax,
                      rlabel="(13 TeV)")
        _footer(fig)
        stamp(fig)
        f = outdir / f"split_mom_dist_{tag}.png"
        fig.savefig(f, dpi=110)
        plt.close(fig)
        written.append(f)
        print("wrote", f)
    return written


def plot_eff_vs_pt(rows, outdir):
    plt, hep = _setup()
    edges = PT_EDGES.copy()
    edges[-1] = 1200.
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(layout="constrained")
    cur = [r for r in rows if r["family"] == "reference" and r["name"] == "current"]
    show = [(r, "k") for r in cur]
    show += [(r, COLORS[i % len(COLORS)])
             for i, r in enumerate(pick_best(rows, n=1))]
    top, bot = 1.0, 1.0
    for r, c in show:
        p = r["per_pt"]
        ax.plot(ctr, p["dijet_keep"], "-o", lw=2.5, ms=7, color=c,
                label=r["label"])
        ax.plot(ctr, p["trijet_keep"], "--s", lw=2.5, ms=7, color=c)
        top = max(top, np.nanmax(p["dijet_keep"]), np.nanmax(p["trijet_keep"]))
        bot = min(bot, np.nanmin(p["dijet_keep"]), np.nanmin(p["trijet_keep"]))
    ax.axhline(1.0, color="0.7", lw=1.2, ls=":")
    ax.set_xlabel(r"$p_{T,1}$ [GeV]")
    ax.set_ylabel("Fraction of channel yield kept")
    ax.set_xlim(200, 1120)
    #### zoom on the band the curves actually live in, then ~1.1x that span of
    #### empty headroom above for the legend
    ax.set_ylim(bot - 0.05 * (top - bot), top + 1.15 * (top - bot))
    ax.text(0.03, 0.965, "solid: dijet     dashed: trijet",
            transform=ax.transAxes, fontsize=15, va="top", color="0.3")
    ax.legend(loc="upper left", fontsize=13, ncol=3, framealpha=0.9,
              bbox_to_anchor=(0.02, 0.925))
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
    _footer(fig)
    stamp(fig)
    f = outdir / "split_mom_eff_vs_pt.png"
    fig.savefig(f, dpi=110)
    plt.close(fig)
    print("wrote", f)
    return f


def _ink(c):
    """Darkened companion of a marker colour, for text that must stay legible.

    The pale end of the palette (e.g. #92dadd) is fine as a filled marker with
    a black edge but unreadable as 11 pt text on white.
    """
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(c)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum <= 0.55:
        return c
    f = 0.52 / lum
    return mc.to_hex((r * f, g * f, b * f))


def _overlap(a, b):
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _declutter(ax, fig, items, fontsize=11, avoid=(), markers=()):
    """Place point labels on the first offset that collides with nothing.

    The interesting working points pile up in the same corner of the
    (dijet_keep, trijet_keep) plane, so fixed offsets produce unreadable
    stacked text. Candidates spiral outwards from the marker; anything pushed
    past the first ring gets a leader line so it stays attached to its point.
    A label is NEVER dropped -- if nothing is clean the least-bad slot wins.
    """
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    blocked = [a.get_window_extent(rend) for a in avoid]
    for (x, y), size in markers:
        px, py = ax.transData.transform((x, y))
        h = 0.5 * np.sqrt(size) * fig.dpi / 72.0 + 2.0
        blocked.append(Bbox.from_extents(px - h, py - h, px + h, py + h))
    rings = (1.0, 1.5, 2.1, 2.9, 3.8, 4.9, 6.2, 7.8, 9.6)
    dirs = ((13, 4), (13, -12), (-13, 4), (-13, -12), (0, 12), (0, -16),
            (15, -4), (-15, -4), (10, 11), (10, -19), (-10, 11), (-10, -19))
    placed = []
    for (x, y), text, color in items:
        best, best_pen, done = None, None, False
        for ri, r in enumerate(rings):
            for dx, dy in dirs:
                #### always leader-line: in the dense corner a bare offset
                #### label cannot be traced back to its marker by eye
                arrow = dict(arrowstyle="-", lw=0.7, color="0.6",
                             shrinkA=1, shrinkB=3.5)
                t = ax.annotate(text, (x, y), textcoords="offset points",
                                xytext=(r * dx, r * dy), fontsize=fontsize,
                                color=color, ha="left" if dx >= 0 else "right",
                                va="bottom" if dy >= 0 else "top",
                                arrowprops=arrow, zorder=6)
                bb = t.get_window_extent(rend).expanded(1.22, 1.32)
                inside = (ax.bbox.contains(bb.x0, bb.y0)
                          and ax.bbox.contains(bb.x1, bb.y1))
                pen = sum(_overlap(bb, p) for p in placed + blocked)
                pen += 0.0 if inside else 1e6
                if pen == 0:
                    placed.append(bb)
                    done = True
                    break
                if best_pen is None or pen < best_pen:
                    if best is not None:
                        best.remove()
                    best_pen, best = pen, t
                else:
                    t.remove()
            if done:
                break
        if done:
            if best is not None:
                best.remove()
        elif best is not None:      # never drop a label; take the least-bad slot
            placed.append(best.get_window_extent(rend).expanded(1.22, 1.32))


def plot_summary(rows, outdir):
    plt, hep = _setup()
    fig, ax = plt.subplots(layout="constrained")
    from matplotlib.lines import Line2D
    fams = sorted({r["name"] for r in rows if r["family"] == "momentum"})
    #### A1/A4/A5/A6/A7 all land in the same top-right knot, so they need
    #### clearly separated hues (Petroff 10); A7 additionally gets a square.
    style = {f: (c, mk) for f, c, mk in zip(
        fams, ("#5790fc", "#f89c20", "#e42536", "#832db6", "#92dadd",
               "#e76300", "#9c9ca1"), "oooooos")}
    #### the interesting working points pile up here; shown enlarged in an
    #### inset so their labels can breathe.
    ZX, ZY = (0.958, 0.988), (0.900, 1.014)
    labels, marks, inzoom = [], [], []
    for r in rows:
        c, mk = style.get(r["name"], ("k", "*" if r["name"] == "current" else "P"))
        ref = r["family"] == "reference"
        s = 700 if ref else 50 + 2400 * max(r["orphans"], 0.0)
        xy = (r["dijet_keep"], r["trijet_keep"])
        ax.scatter(*xy, s=s, marker=mk, color=c, zorder=4 if ref else 3,
                   **({} if ref else dict(alpha=0.65, edgecolor="k", lw=0.8)))
        marks.append((xy, s))
        txt = None if ref else f"{list(r['params'].values())[0]:g}"
        if ZX[0] <= xy[0] <= ZX[1] and ZY[0] <= xy[1] <= ZY[1]:
            inzoom.append((xy, s, c, mk, ref, txt))     # labelled in the inset
        elif txt is not None:
            labels.append((xy, txt, _ink(c)))
    ax.set_xlabel("Dijet yield kept")
    ax.set_ylabel("Trijet yield kept")
    ax.set_xlim(0.40, 1.26)
    ax.set_ylim(-0.08, 1.75)
    #### proxy handles: the real markers are scaled by the orphan fraction and
    #### would blow the legend apart.
    hs = [Line2D([], [], ls="", marker=style[f][1], ms=10, mfc=style[f][0],
                 mec="k", mew=0.8, alpha=0.75, label=f) for f in fams]
    hs += [Line2D([], [], ls="", marker="*", ms=17, color="k", label="current"),
           Line2D([], [], ls="", marker="P", ms=12, color="k", label="A<0.1")]
    leg = ax.legend(handles=hs, loc="lower left", fontsize=13, ncol=5,
                    framealpha=0.95, columnspacing=1.0, handletextpad=0.3,
                    title="marker area $\\propto$ orphans;  "
                          "point label = threshold", title_fontsize=12)
    #### ---- zoom inset on the top-right knot ------------------------------
    axi = ax.inset_axes([0.44, 0.615, 0.545, 0.365], xlim=ZX, ylim=ZY)
    imarks, ilabels = [], []
    for xy, s, c, mk, ref, txt in inzoom:
        axi.scatter(*xy, s=s * (0.5 if ref else 0.85), marker=mk, color=c,
                    zorder=4 if ref else 3,
                    **({} if ref else dict(alpha=0.65, edgecolor="k", lw=0.8)))
        imarks.append((xy, s * (0.5 if ref else 0.85)))
        if txt is not None:
            ilabels.append((xy, txt, _ink(c)))
    axi.tick_params(labelsize=11, length=4)
    #### only two x ticks: a 0.97 label would sit under the main-panel
    #### star that the zoom rectangle points at.
    axi.set_xticks([0.96, 0.98])
    axi.set_yticks([0.92, 0.96, 1.00])
    for sp in axi.spines.values():
        sp.set(edgecolor="0.45", lw=1.2)
    ax.indicate_inset_zoom(axi, edgecolor="0.45", lw=1.2, alpha=0.9)

    #### cms.label and the footer strip both change the axes box under
    #### constrained layout, so they must land BEFORE the labels are placed.
    hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
    _footer(fig)
    _declutter(ax, fig, labels, avoid=[leg, axi], markers=marks)
    _declutter(axi, fig, ilabels, fontsize=12, markers=imarks)
    stamp(fig)
    f = outdir / "split_mom_summary.png"
    fig.savefig(f, dpi=110)
    plt.close(fig)
    print("wrote", f)
    return f


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dijet-mc", default="/Users/aritra/cernbox (2)/dijet_skims")
    ap.add_argument("--trijet-mc", default="/Users/aritra/cernbox (2)/trijet_skims/mc")
    ap.add_argument("--dijet-data", default="/Users/aritra/Downloads/dijet_data_skims")
    ap.add_argument("--trijet-data", default="/Users/aritra/cernbox (2)/trijet_skims/data")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cache", default=None,
                    help="npz cache of the joined variables (written if absent)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-data", action="store_true")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def get(is_data):
        tag = "data" if is_data else "mc"
        cf = Path(args.cache).with_suffix(f".{tag}.npz") if args.cache else None
        if cf is not None and cf.exists():
            print(f"[cache] {cf}")
            return {k: v for k, v in np.load(cf).items()}
        dp = sorted(Path(args.dijet_data if is_data else args.dijet_mc).glob("*.pkl"))
        tp = sorted(Path(args.trijet_data if is_data else args.trijet_mc).glob("*.pkl"))
        V = build_vars(dp, tp, is_data)
        if cf is not None:
            np.savez_compressed(cf, **V)
        return V

    V = get(False)
    w = V["w"]

    #### ---- closure: the stored dijet decision must be reproducible
    rebuilt = V["d_nm1"] & (V["A"] < 0.3)
    extra = int((rebuilt & ~V["d_tot"]).sum())
    miss = int((~rebuilt & V["d_tot"]).sum())
    print(f"\nclosure: N-1 & A<0.3 vs stored recoTot_seq -- {extra} extra, "
          f"{miss} missing, of {int(V['d_tot'].sum())} stored "
          f"{'[OK]' if extra == miss == 0 else '[FAIL]'}")

    Wd, Wt = w[V["d_tot"]].sum(), w[V["t_tot"]].sum()
    Wb = w[V["d_tot"] & V["t_tot"]].sum()
    print(f"baseline: P(tri|di) = {frac(Wb, Wd):.3f}   "
          f"P(di|tri) = {frac(Wb, Wt):.3f}   "
          f"Jaccard = {frac(Wb, w[V['d_tot'] | V['t_tot']].sum()):.3f}")
    print("NOTE: the two productions used different JER stochastic-smearing "
          "seeds, so\n      reco jets differ slightly between the dijet and "
          "trijet copies of the same\n      event. A residual overlap up to "
          "~1e-3 of the baseline overlap is that\n      artifact, not a real "
          "overlap: within ONE production a same-variable\n      split is "
          "exactly orthogonal.")

    cfgs = build_configs(V)
    rows = [metrics(V, c, False) for c in cfgs]
    print_table(rows, "momentum-balance split scan -- QCD MG+Pythia8 2018")

    best = pick_best(rows)
    print("\nbest 2 per sub-family (orphans < 0.05, dijet_keep > 0.93, "
          "max trijet_keep):")
    for r in best:
        print(f"  {r['label']:>22s}  di {r['dijet_keep']:.3f}  "
              f"tri {r['trijet_keep']:.3f}  orph {r['orphans']:.3f}")

    #### ---- data cross-check on the shortlist
    if not args.no_data:
        want = {r["label"] for r in best} | {"current", "A<0.1"}
        Vd = get(True)
        dcfgs = [c for c in build_configs(Vd) if c["label"] in want]
        drows = [metrics(Vd, c, True) for c in dcfgs]
        print_table(drows, "shortlist on 2018 JetHT (prescale-weighted)")
        print("\nMC vs data, dijet_keep / trijet_keep:")
        by = {r["label"]: r for r in rows}
        for r in sorted(drows, key=lambda r: r["label"]):
            m = by[r["label"]]
            print(f"  {r['label']:>22s}  di {m['dijet_keep']:.3f} / "
                  f"{r['dijet_keep']:.3f}   tri {m['trijet_keep']:.3f} / "
                  f"{r['trijet_keep']:.3f}   orph {m['orphans']:.3f} / "
                  f"{r['orphans']:.3f}")
        rows = rows + drows

    fout = outdir / "split_scan_momentum.json"
    fout.write_text(json.dumps(rows, indent=1))
    print("\nwrote", fout)

    if not args.no_plots:
        plot_dists(V, outdir)
        mcrows = [r for r in rows if r["sample"] != "data2018"]
        plot_eff_vs_pt(mcrows, outdir)
        plot_summary(mcrows, outdir)


if __name__ == "__main__":
    main()
