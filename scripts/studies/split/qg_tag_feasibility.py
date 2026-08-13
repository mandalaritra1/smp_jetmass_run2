"""Gen-level MC-truth feasibility study: how much gluon enrichment do kinematic
tags buy in the dijet/trijet phase space of this analysis?

f_g = N_g / (N_g + N_q) of GenJetAK8 jets, from GenJetAK8_partonFlavour, per
measured-jet pT bin and per tag category.  Purely gen level -- no reco, no
detector, no trigger.

Inputs (in order of preference)
  1. local NanoAOD test file (QCD_Pt_170to300 pythia8, UL18) -- always used
  2. madgraphMLM HT-binned QCD streamed over xrootd (needs a grid proxy);
     skipped with --local-only or when the proxy probe fails.

Outputs land in the scratchpad dir (--outdir): qg_tag_feasibility.json and
qg_tag_*.png.

Usage
  python scripts/qg_tag_feasibility.py --local-only
  python scripts/qg_tag_feasibility.py                 # try HT combo too
  python scripts/qg_tag_feasibility.py --no-cache      # force re-read
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, REPO)
DEFAULT_OUT = (
    "/private/tmp/claude-501/-Users-aritra-Projects-smp-jetmass-run2/"
    "a2621af3-b9a7-4e23-a9ea-dd4970a0445f/scratchpad"
)

LOCAL_FILE = (
    "/Users/aritra/Projects/GluonJetMass/test_files/store/mc/"
    "RunIISummer20UL18NanoAODv9/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/NANOAODSIM/"
    "106X_upgrade2018_realistic_v16_L1v1-v1/120000/"
    "5C540F1F-6B0C-1047-B020-539529AB3BB6.root"
)
LOCAL_TAG = "flat170to300"  # naming kept from the task spec (pt-binned 170-300)

FILESET = os.path.join(REPO, "samples", "hadronic", "fileset_MG_pythia8_wRedirs.json")
HT_WANTED = [  # (substring of the DAS key, xsdb key, n files to stream)
    ("QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18", 2),
    ("QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18", 2),
    ("QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18", 2),
]

BRANCHES = [
    "GenJetAK8_pt", "GenJetAK8_eta", "GenJetAK8_phi", "GenJetAK8_mass",
    "GenJetAK8_partonFlavour", "genWeight",
]

PT_EDGES = np.array([170, 200, 290, 400, 480, 570, 680, 760, 820, 13000], dtype=float)
DY_EDGES = [(0.0, 1.0), (1.0, 2.0), (2.0, 99.0)]
Z3_EDGES = [(0.1, 0.2), (0.2, 0.3), (0.3, 99.0)]
Z3_PT_POOL = (185.0, 400.0)

PETROFF = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]


# ----------------------------------------------------------------------------- I/O
def read_file(url: str, max_chunks: int | None = None) -> dict:
    """Read GenJetAK8_* + genWeight from one NanoAOD file into flat numpy arrays."""
    tree = uproot.open(url)["Events"]
    arr = tree.arrays(BRANCHES, library="ak")
    pt = arr["GenJetAK8_pt"]
    return dict(
        counts=np.asarray(ak.num(pt), dtype=np.int64),
        pt=np.asarray(ak.flatten(pt), dtype=np.float32),
        eta=np.asarray(ak.flatten(arr["GenJetAK8_eta"]), dtype=np.float32),
        phi=np.asarray(ak.flatten(arr["GenJetAK8_phi"]), dtype=np.float32),
        mass=np.asarray(ak.flatten(arr["GenJetAK8_mass"]), dtype=np.float32),
        flav=np.asarray(ak.flatten(arr["GenJetAK8_partonFlavour"]), dtype=np.int32),
        genWeight=np.asarray(arr["genWeight"], dtype=np.float64),
    )


def have_proxy() -> bool:
    import subprocess
    try:
        r = subprocess.run(["voms-proxy-info", "-exists", "-valid", "1:0"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and "Couldn't find" not in (r.stdout + r.stderr)
    except Exception:
        return False


def ht_urls() -> list[tuple[str, str]]:
    """(xsdb-key, url) pairs for the HT slices we want to stream."""
    with open(FILESET) as fh:
        fs = json.load(fh)["QCD_binned"]
    out = []
    for sub, nfile in HT_WANTED:
        for das, files in fs.items():
            if sub in das.lstrip("/"):
                xskey = das.lstrip("/").split("/")[0]
                out += [(xskey, u) for u in files[:nfile]]
                break
    return out


def load_events(local_only: bool, cache: str, use_cache: bool) -> dict:
    """Return {sample_tag: dict-of-arrays}, cached to npz."""
    if use_cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        samples = json.loads(str(z["samples"]))
        out = {}
        for s in samples:
            out[s] = {k: z[f"{s}__{k}"] for k in
                      ["counts", "pt", "eta", "phi", "mass", "flav", "genWeight", "evtw"]}
        print(f"[cache] loaded {cache}: samples={samples}")
        return out

    out: dict[str, dict] = {}
    print(f"[read] local: {LOCAL_FILE}")
    d = read_file(LOCAL_FILE)
    d["evtw"] = d["genWeight"].copy()  # single sample -> genWeight is the weight
    out[LOCAL_TAG] = d
    print(f"[read]   {len(d['genWeight'])} events, {len(d['pt'])} GenJetAK8")

    if not local_only:
        if not have_proxy():
            print("[read] no valid grid proxy -> skipping HT-binned streaming")
        else:
            from smp_jetmass_run2 import corrections as corr
            parts = []
            for xskey, url in ht_urls():
                try:
                    print(f"[read] stream {xskey}: {url}")
                    p = read_file(url)
                except Exception as exc:  # noqa: BLE001
                    print(f"[read]   FAILED {type(exc).__name__}: {exc}")
                    continue
                sumw = p["genWeight"].sum()
                xs = corr.xsdb[xskey]
                p["evtw"] = p["genWeight"] * xs / sumw
                parts.append(p)
            if parts:
                out["HTcombo"] = {
                    k: np.concatenate([p[k] for p in parts])
                    for k in ["counts", "pt", "eta", "phi", "mass", "flav",
                              "genWeight", "evtw"]
                }

    payload = {"samples": json.dumps(list(out))}
    for s, d in out.items():
        for k, v in d.items():
            payload[f"{s}__{k}"] = v
    np.savez_compressed(cache, **payload)
    print(f"[cache] wrote {cache}")
    return out


# ------------------------------------------------------------------- kinematics
def rapidity(pt, eta, mass):
    pz = pt * np.sinh(eta)
    e = np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2)
    return 0.5 * np.log((e + pz) / (e - pz))


def dphi(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def build(d: dict):
    """Unflatten to jagged, return jagged arrays + per-event weight."""
    counts = d["counts"]
    j = ak.zip({
        "pt": ak.unflatten(d["pt"], counts),
        "eta": ak.unflatten(d["eta"], counts),
        "phi": ak.unflatten(d["phi"], counts),
        "mass": ak.unflatten(d["mass"], counts),
        "flav": ak.unflatten(d["flav"], counts),
        "y": ak.unflatten(rapidity(d["pt"].astype(np.float64),
                                   d["eta"].astype(np.float64),
                                   d["mass"].astype(np.float64)), counts),
    })
    return j, d["evtw"]


# ------------------------------------------------------------------- statistics
def fg_stats(flav, pt, w):
    """Weighted f_g / f_undef / n_eff / sum-w in each PT_EDGES bin."""
    flav = np.abs(np.asarray(flav))
    pt = np.asarray(pt)
    w = np.asarray(w, dtype=np.float64)
    ib = np.searchsorted(PT_EDGES, pt, side="right") - 1
    rows = []
    for k in range(len(PT_EDGES) - 1):
        m = ib == k
        if not m.any():
            rows.append(None)
            continue
        f, ww = flav[m], w[m]
        g = (f == 21)
        q = (f >= 1) & (f <= 5)
        u = f == 0
        sg, sq, su, st = ww[g].sum(), ww[q].sum(), ww[u].sum(), ww.sum()
        den = sg + sq
        if den <= 0:
            rows.append(None)
            continue
        wdef = ww[g | q]
        neff = wdef.sum() ** 2 / (wdef ** 2).sum() if (wdef ** 2).sum() > 0 else 0.0
        fg = sg / den
        rows.append(dict(f_g=float(fg), f_undef=float(su / st) if st > 0 else 0.0,
                         n_eff=float(neff), sumw=float(st), n_raw=int(m.sum()),
                         err=float(np.sqrt(max(fg * (1 - fg), 0) / neff)) if neff > 0 else np.nan))
    return rows


def fg_single(flav, w):
    """Same but with no pT binning (returns one dict or None)."""
    flav = np.abs(np.asarray(flav))
    w = np.asarray(w, dtype=np.float64)
    g, q, u = flav == 21, (flav >= 1) & (flav <= 5), flav == 0
    sg, sq, su, st = w[g].sum(), w[q].sum(), w[u].sum(), w.sum()
    if sg + sq <= 0:
        return None
    wdef = w[g | q]
    neff = wdef.sum() ** 2 / (wdef ** 2).sum()
    fg = sg / (sg + sq)
    return dict(f_g=float(fg), f_undef=float(su / st), n_eff=float(neff),
                sumw=float(st), n_raw=int(len(w)),
                err=float(np.sqrt(max(fg * (1 - fg), 0) / neff)))


# ------------------------------------------------------------------- categories
def categories(j, evtw, tag: str, verbose=True):
    """Return {category_name: (flav, pt, w)} plus the pooled trijet z3 sets."""
    out: dict[str, tuple] = {}

    # --- sanity: pt ordering ------------------------------------------------
    pt = j.pt
    two = pt[ak.num(pt) >= 2]
    ordered = ak.all(two[:, :-1] >= two[:, 1:], axis=1)
    frac_ordered = float(ak.mean(ordered))
    if verbose:
        print(f"[{tag}] pt-ordered fraction of >=2-jet events: {frac_ordered:.6f}")

    # --- global undefined-flavour fraction (all jets, pt>170) ---------------
    fl = np.abs(np.asarray(ak.flatten(j.flav)))
    ptf = np.asarray(ak.flatten(j.pt))
    m170 = ptf > 170
    if verbose:
        u = (fl[m170] == 0).mean()
        print(f"[{tag}] undefined-flavour fraction, all GenJetAK8 pt>170: {u:.4f} "
              f"({m170.sum()} jets)")

    # --- dijet selection ----------------------------------------------------
    n2 = ak.num(j.pt) >= 2
    jj = j[n2]
    w2 = evtw[np.asarray(n2)]
    y1, y2 = np.asarray(jj.y[:, 0]), np.asarray(jj.y[:, 1])
    p1, p2 = np.asarray(jj.pt[:, 0]), np.asarray(jj.pt[:, 1])
    f1, f2 = np.asarray(jj.flav[:, 0]), np.asarray(jj.flav[:, 1])
    ph1, ph2 = np.asarray(jj.phi[:, 0]), np.asarray(jj.phi[:, 1])
    asym = np.abs(p1 - p2) / (p1 + p2)
    dsel = (np.abs(y1) < 2.5) & (np.abs(y2) < 2.5) & (dphi(ph1, ph2) > 2.0) & (asym < 0.3)

    y1, y2, p1, p2, f1, f2, wd = (y1[dsel], y2[dsel], p1[dsel], p2[dsel],
                                  f1[dsel], f2[dsel], w2[dsel])
    if verbose:
        print(f"[{tag}] dijet events selected: {dsel.sum()} / {len(dsel)}")

    # trijet veto flag on the *same* dijet events
    tri_ok_all = trijet_mask(jj)
    tri_veto = ~tri_ok_all[dsel]

    ay1, ay2 = np.abs(y1), np.abs(y2)
    cen_is_1 = ay1 <= ay2
    fc = np.where(cen_is_1, f1, f2)
    pc = np.where(cen_is_1, p1, p2)
    ff = np.where(cen_is_1, f2, f1)
    pf = np.where(cen_is_1, p2, p1)
    dy = np.abs(y1 - y2)

    # 1. inclusive: both jets pooled
    out["1_dijet_inclusive"] = (np.concatenate([f1, f2]), np.concatenate([p1, p2]),
                                np.concatenate([wd, wd]))
    # 2/3. more central / more forward
    out["2_more_central"] = (fc, pc, wd)
    out["3_more_forward"] = (ff, pf, wd)
    # 4. central jet in dy bins
    for lo, hi in DY_EDGES:
        m = (dy >= lo) & (dy < hi)
        lab = f"4_central_dy_{lo:g}to{hi:g}" if hi < 90 else f"4_central_dy_ge{lo:g}"
        out[lab] = (fc[m], pc[m], wd[m])
    # 5. hard forward-partner tag: |y_meas|<0.5 and |y_partner|>1.2
    m1 = (ay1 < 0.5) & (ay2 > 1.2)
    m2 = (ay2 < 0.5) & (ay1 > 1.2)
    out["5_fwd_partner_tag"] = (np.concatenate([f1[m1], f2[m2]]),
                                np.concatenate([p1[m1], p2[m2]]),
                                np.concatenate([wd[m1], wd[m2]]))
    # 6. central-central, both jets pooled
    cc = (ay1 < 0.8) & (ay2 < 0.8)
    out["6_central_central"] = (np.concatenate([f1[cc], f2[cc]]),
                                np.concatenate([p1[cc], p2[cc]]),
                                np.concatenate([wd[cc], wd[cc]]))
    # trijet-vetoed variants of 2 and 5
    out["2v_more_central_trijetveto"] = (fc[tri_veto], pc[tri_veto], wd[tri_veto])
    m1v, m2v = m1 & tri_veto, m2 & tri_veto
    out["5v_fwd_partner_trijetveto"] = (np.concatenate([f1[m1v], f2[m2v]]),
                                        np.concatenate([p1[m1v], p2[m2v]]),
                                        np.concatenate([wd[m1v], wd[m2v]]))

    # --- trijet selection ---------------------------------------------------
    n3 = ak.num(j.pt) >= 3
    j3 = j[n3]
    w3 = evtw[np.asarray(n3)]
    ok = trijet_mask(j3, need3=True)
    t = j3[ok]
    wt = w3[ok]
    if verbose:
        print(f"[{tag}] trijet events selected: {int(ok.sum())} / {len(ok)}")
    p1t, p2t, p3t = (np.asarray(t.pt[:, 0]), np.asarray(t.pt[:, 1]),
                     np.asarray(t.pt[:, 2]))
    f3t = np.asarray(t.flav[:, 2])
    out["7_trijet_j3"] = (f3t, p3t, wt)
    z3 = p3t / (p1t + p2t)
    zsets = {}
    poolm = (p3t >= Z3_PT_POOL[0]) & (p3t < Z3_PT_POOL[1])
    for lo, hi in Z3_EDGES:
        m = poolm & (z3 >= lo) & (z3 < hi)
        lab = f"8_trijet_j3_z3_{lo:g}to{hi:g}" if hi < 90 else f"8_trijet_j3_z3_ge{lo:g}"
        zsets[lab] = (f3t[m], p3t[m], wt[m])
    out.update(zsets)

    meta = dict(frac_pt_ordered=frac_ordered,
                undef_all_jets=float((fl[m170] == 0).mean()),
                n_dijet=int(dsel.sum()), n_trijet=int(ok.sum()))
    return out, meta, list(zsets)


def trijet_mask(j, need3=False):
    """>=3 jets, |y|<2.5 for the 3 leading, min pairwise dphi > 1.0, pT3 > 185."""
    has3 = np.asarray(ak.num(j.pt) >= 3)
    out = np.zeros(len(has3), dtype=bool)
    jj = j[has3]
    if len(jj) == 0:
        return out
    y = [np.asarray(jj.y[:, i]) for i in range(3)]
    ph = [np.asarray(jj.phi[:, i]) for i in range(3)]
    p3 = np.asarray(jj.pt[:, 2])
    mind = np.minimum(np.minimum(dphi(ph[0], ph[1]), dphi(ph[0], ph[2])),
                      dphi(ph[1], ph[2]))
    sel = (np.abs(y[0]) < 2.5) & (np.abs(y[1]) < 2.5) & (np.abs(y[2]) < 2.5) \
        & (mind > 1.0) & (p3 > 185.0)
    out[has3] = sel
    return out


# ----------------------------------------------------------------------- output
def bin_label(k):
    lo, hi = PT_EDGES[k], PT_EDGES[k + 1]
    return f"{lo:.0f}-{hi:.0f}" if hi < 13000 else f"{lo:.0f}+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--min-neff", type=float, default=50.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cache = os.path.join(args.outdir, "qg_tag_arrays.npz")

    samples = load_events(args.local_only, cache, use_cache=not args.no_cache)

    rows = []
    per_sample = {}
    for tag, d in samples.items():
        j, w = build(d)
        cats, meta, zlabels = categories(j, w, tag)
        base = fg_stats(*cats["1_dijet_inclusive"])
        base_sumw = [r["sumw"] if r else 0.0 for r in base]
        bfl, bpt, bw = cats["1_dijet_inclusive"]
        bmask = (bpt >= Z3_PT_POOL[0]) & (bpt < Z3_PT_POOL[1])
        base_sumw_pool = float(np.asarray(bw)[bmask].sum())
        table = {}
        for name, (fl, pt, ww) in cats.items():
            if name.startswith("8_"):
                st = fg_single(fl, ww)
                if st is None:
                    continue
                st["yield_frac"] = (st["sumw"] / base_sumw_pool
                                    if base_sumw_pool > 0 else float("nan"))
                table[name] = {"pooled": st}
                rows.append(dict(category=name, sample=tag,
                                 pt_bin=[Z3_PT_POOL[0], Z3_PT_POOL[1]], **st))
                continue
            res = fg_stats(fl, pt, ww)
            table[name] = {}
            for k, r in enumerate(res):
                if r is None:
                    continue
                r = dict(r)
                r["yield_frac"] = (r["sumw"] / base_sumw[k]) if base_sumw[k] > 0 else float("nan")
                table[name][k] = r
                rows.append(dict(category=name, sample=tag,
                                 pt_bin=[float(PT_EDGES[k]), float(PT_EDGES[k + 1])], **r))
        base_pool = fg_single(np.asarray(bfl)[bmask], np.asarray(bw)[bmask])
        if base_pool is not None:
            base_pool["yield_frac"] = 1.0
            rows.append(dict(category="1_dijet_inclusive_pooled", sample=tag,
                             pt_bin=[Z3_PT_POOL[0], Z3_PT_POOL[1]], **base_pool))
        per_sample[tag] = dict(table=table, meta=meta, base=base, base_pool=base_pool)

    # ---------------- printed table ----------------
    for tag, blob in per_sample.items():
        table, meta, base = blob["table"], blob["meta"], blob["base"]
        good = [k for k in range(len(PT_EDGES) - 1)
                if base[k] and base[k]["n_eff"] > args.min_neff]
        print("\n" + "=" * 100)
        print(f"SAMPLE {tag}   (bins shown: n_eff(baseline) > {args.min_neff:g})")
        print("=" * 100)
        print("SANITY: pt-ordered fraction = %.6f | undefined-flavour (all jets pt>170) = %.3f%%"
              % (meta["frac_pt_ordered"], 100 * meta["undef_all_jets"]))
        for k in good:
            if PT_EDGES[k] == 200:
                b = base[k]
                print("SANITY: dijet-inclusive f_g in 200-290 = %.3f  (full-sim ref 0.531, "
                      "diff %+.1f pts)" % (b["f_g"], 100 * (b["f_g"] - 0.531)))
        hdr = f"{'category':34s}" + "".join(f"{bin_label(k):>18s}" for k in good)
        print("\nf_g +- stat   (enrichment vs dijet-inclusive, in points)")
        print(hdr)
        print("-" * len(hdr))
        for name, res in table.items():
            if name.startswith("8_"):
                continue
            line = f"{name:34s}"
            for k in good:
                r = res.get(k)
                if r is None or r["n_eff"] < args.min_neff:
                    line += f"{'--':>18s}"
                else:
                    line += f"{r['f_g']:>10.3f}+-{r['err']:.3f}"
            print(line)
        print("\nenrichment vs baseline [points]")
        print(hdr)
        print("-" * len(hdr))
        for name, res in table.items():
            if name.startswith("8_") or name == "1_dijet_inclusive":
                continue
            line = f"{name:34s}"
            for k in good:
                r = res.get(k)
                if r is None or r["n_eff"] < args.min_neff or not base[k]:
                    line += f"{'--':>18s}"
                else:
                    line += f"{100 * (r['f_g'] - base[k]['f_g']):>+18.1f}"
            print(line)
        print("\nyield fraction kept vs dijet-inclusive (same pT bin)")
        print(hdr)
        print("-" * len(hdr))
        for name, res in table.items():
            if name.startswith("8_"):
                continue
            line = f"{name:34s}"
            for k in good:
                r = res.get(k)
                line += f"{'--':>18s}" if r is None else f"{r['yield_frac']:>18.3f}"
            print(line)
        print("\nundefined-flavour fraction per category")
        print(hdr)
        print("-" * len(hdr))
        for name, res in table.items():
            if name.startswith("8_"):
                continue
            line = f"{name:34s}"
            for k in good:
                r = res.get(k)
                line += f"{'--':>18s}" if r is None else f"{r['f_undef']:>18.4f}"
            print(line)
        zl = [n for n in table if n.startswith("8_")]
        if zl:
            print(f"\ntrijet j3 vs z3  (pT3 pooled {Z3_PT_POOL[0]:g}-{Z3_PT_POOL[1]:g} GeV)")
            for n in zl:
                r = table[n]["pooled"]
                print(f"  {n:34s} f_g = {r['f_g']:.3f} +- {r['err']:.3f}   "
                      f"n_eff = {r['n_eff']:.0f}   yield_frac = {r['yield_frac']:.3f}")

    outjson = os.path.join(args.outdir, "qg_tag_feasibility.json")
    with open(outjson, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n[out] {outjson}  ({len(rows)} rows)")

    make_plots(per_sample, args.outdir, args.min_neff)


# ------------------------------------------------------------------------ plots
def make_plots(per_sample, outdir, min_neff):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from plot_stamp import stamp

    plt.style.use(hep.style.CMS)
    INPUTS = "NanoAOD gen truth"

    def finish(fig, ax, path, ylab=r"$f_{g}$ (gen-level)"):
        hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel="(13 TeV)")
        ax.set_ylabel(ylab)
        stamp(fig, inputs=INPUTS)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[out] {path}")

    tag = list(per_sample)[0] if "HTcombo" not in per_sample else "HTcombo"
    blob = per_sample[tag]
    table, base = blob["table"], blob["base"]
    good = [k for k in range(len(PT_EDGES) - 1)
            if base[k] and base[k]["n_eff"] > min_neff]

    # --- f_g vs pT ---------------------------------------------------------
    want = [("1_dijet_inclusive", "dijet, both jets"),
            ("2_more_central", "more central jet"),
            ("3_more_forward", "more forward jet"),
            ("5_fwd_partner_tag", r"$|y_{\rm meas}|<0.5,\ |y_{\rm part}|>1.2$"),
            ("6_central_central", r"both $|y|<0.8$"),
            ("7_trijet_j3", "trijet, 3rd jet")]
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    for i, (name, lab) in enumerate(want):
        res = table.get(name, {})
        xs, ys, es, xe = [], [], [], []
        for k in good:
            r = res.get(k)
            if r is None or r["n_eff"] < min_neff:
                continue
            lo, hi = PT_EDGES[k], min(PT_EDGES[k + 1], 1000)
            xs.append(0.5 * (lo + hi)); xe.append(0.5 * (hi - lo))
            ys.append(r["f_g"]); es.append(r["err"])
        if not xs:
            continue
        off = (i - 2.5) * 3.0
        ax.errorbar(np.array(xs) + off, ys, yerr=es, xerr=xe, fmt="o", ms=6,
                    color=PETROFF[i % 6], label=lab, capsize=0, lw=1.6)
    ax.set_xlabel(r"measured jet $p_{T}$ [GeV]")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    ax.legend(loc="lower left", fontsize=14, frameon=False, ncols=2)
    finish(fig, ax, os.path.join(outdir, "qg_tag_fg_vs_pt.png"))

    # --- f_g vs dy ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    xlab = [r"$0\!\leq\!\Delta y\!<\!1$", r"$1\!\leq\!\Delta y\!<\!2$",
            r"$\Delta y\!\geq\!2$"]
    names = []
    for lo, hi in DY_EDGES:
        names.append(f"4_central_dy_{lo:g}to{hi:g}" if hi < 90
                     else f"4_central_dy_ge{lo:g}")
    npl = 0
    for i, k in enumerate(good):
        ys, es, xs = [], [], []
        for x, n in enumerate(names):
            r = table.get(n, {}).get(k)
            if r is None or r["n_eff"] < min_neff:
                continue
            xs.append(x); ys.append(r["f_g"]); es.append(r["err"])
        if not xs:
            continue
        ax.errorbar(np.array(xs) + 0.02 * i, ys, yerr=es, fmt="o-", ms=7, lw=1.6,
                    color=PETROFF[i % 6],
                    label=rf"$p_{{T}}$ {bin_label(k)} GeV")
        npl += 1
    #### one grey band for the dijet-inclusive baseline across the shown pT bins,
    #### instead of one near-coincident dashed line per bin (unreadable)
    bl = table.get("1_dijet_inclusive", {})
    bvals = [bl[k]["f_g"] for k in good if bl.get(k)]
    if bvals:
        ax.axhspan(min(bvals), max(bvals), color="0.6", alpha=0.25, lw=0,
                   label="dijet inclusive (range over $p_{T}$ bins)")
    ax.set_xticks(range(len(names)), xlab)
    ax.set_xlim(-0.4, len(names) - 0.6)
    ax.set_xlabel(r"$\Delta y = |y_{1}-y_{2}|$ (dijet)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower left", fontsize=14, frameon=False,
              title="more-central jet of the dijet pair", title_fontsize=13)
    finish(fig, ax, os.path.join(outdir, "qg_tag_fg_vs_dy.png"))

    # --- f_g vs z3 ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    zn, zx = [], [r"$0.1\!\leq\!z_{3}\!<\!0.2$", r"$0.2\!\leq\!z_{3}\!<\!0.3$",
                  r"$z_{3}\!\geq\!0.3$"]
    for lo, hi in Z3_EDGES:
        zn.append(f"8_trijet_j3_z3_{lo:g}to{hi:g}" if hi < 90
                  else f"8_trijet_j3_z3_ge{lo:g}")
    #### low n_eff is kept here (with its error bar) -- the z3 scan is stats-starved
    #### by construction in a pT-hat 170-300 sample and dropping points would hide
    #### that; empty bins get an explicit "no events" annotation instead
    xs, ys, es = [], [], []
    for x, n in enumerate(zn):
        r = table.get(n, {}).get("pooled")
        if r is None or r["n_eff"] < 5:
            ax.annotate("no events\n(kinematically\ninaccessible)", (x, 0.30),
                        ha="center", va="center", fontsize=12, color="0.45")
            continue
        xs.append(x); ys.append(r["f_g"]); es.append(r["err"])
    if xs:
        ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=8, lw=1.8, color=PETROFF[0],
                    label=r"trijet 3rd jet, $185<p_{T,3}<400$ GeV")
    bp = blob.get("base_pool")
    if bp:
        ax.axhline(bp["f_g"], color=PETROFF[2], lw=1.2, ls="--",
                   label=r"dijet inclusive, $185<p_{T}<400$ GeV")
    ax.set_xticks(range(len(zn)), zx)
    ax.set_xlim(-0.4, len(zn) - 0.6)
    ax.set_xlabel(r"$z_{3} = p_{T,3}/(p_{T,1}+p_{T,2})$")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower left", fontsize=14, frameon=False)
    finish(fig, ax, os.path.join(outdir, "qg_tag_fg_vs_z3.png"))


if __name__ == "__main__":
    main()
