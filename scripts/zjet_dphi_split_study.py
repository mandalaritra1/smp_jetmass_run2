#!/usr/bin/env python
"""Gen-level study: translating the hadronic-channel pT-dependent decorrelation
cut to the Z+jet channel.

Current Z+jet topology selection:   dphi(Z, jet) > 1.57  AND  A < 0.3
    with A = |pT(Z) - pT(jet)| / (pT(Z) + pT(jet))
Proposed (hadronic-style) rule:     dphi(Z, jet) > pi - X/pT(ref),  X in {200, 250} GeV
    with pT(ref) = pT(Z) (default) or pT(jet) (variant).

Streams UL18 NanoAODv9 amcatnlo DY (PtZ-binned) over xrootd, gen truth only.

Usage:
    python scripts/zjet_dphi_split_study.py [--nfiles 2] [--refresh]
"""

import os

# CRITICAL: every remote open needs this or xrootd auth fails
os.environ.setdefault("X509_CERT_DIR", "/Users/aritra/.globus/certificates")

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-aritra-Projects-smp-jetmass-run2/"
    "a2621af3-b9a7-4e23-a9ea-dd4970a0445f/scratchpad"
)
SCRATCH.mkdir(parents=True, exist_ok=True)
CACHE = SCRATCH / "zjet_dphi_split_cache.npz"
FILECACHE = SCRATCH / "zjet_dphi_files_v2"  # v2 adds GenJetAK8_partonFlavour
FILECACHE.mkdir(exist_ok=True)
JSON_OUT = SCRATCH / "zjet_dphi_split.json"

REDIRECTORS = ["root://cmsxrootd.fnal.gov/", "root://cms-xrd-global.cern.ch/"]

# XSDB cross sections [pb] for DYJetsToLL_LHEFilterPtZ-*_MatchEWPDG20 amcatnloFXFX
XS_FALLBACK = {
    "50To100": 407.9,
    "100To250": 96.8,
    "250To400": 3.774,
    "400To650": 0.5164,
    "650ToInf": 0.04796,
}
PTZ_BINS = ["50To100", "100To250", "250To400", "400To650", "650ToInf"]
# more files where the weight-per-event is large relative to the sampled
# phase space: the high jet-pT bins are fed by the unbalanced tail of the
# low-PtZ samples, which otherwise gives a tiny effective sample size.
NFILES = {"50To100": 2, "100To250": 6, "250To400": 12,
          "400To650": 12, "650ToInf": 6}

BRANCHES = [
    "GenJetAK8_pt", "GenJetAK8_eta", "GenJetAK8_phi", "GenJetAK8_mass",
    "GenJetAK8_partonFlavour",
    "GenDressedLepton_pt", "GenDressedLepton_eta", "GenDressedLepton_phi",
    "GenDressedLepton_mass", "GenDressedLepton_pdgId",
    "genWeight",
]

JETPT_BINS = np.array([170, 200, 290, 400, 480, 570, 680, 760, 820, 13000], dtype=float)

PETROFF = ["#5790fc", "#f89c20", "#e42536", "#964a8b"]


# ----------------------------------------------------------------------------
# cross sections
# ----------------------------------------------------------------------------
def get_xs():
    xs = {}
    try:
        sys.path.insert(0, str(REPO))
        from smp_jetmass_run2 import corrections as corr

        keys = [k for k in corr.xsdb if "DY" in k or "PtZ" in k.replace("_", "")]
        print(f"[xs] DY-like keys found in corrections.xsdb: {keys if keys else 'NONE'}")
    except Exception as exc:  # pragma: no cover
        print(f"[xs] could not import corrections ({exc}); using fallback")
    for b in PTZ_BINS:
        xs[b] = XS_FALLBACK[b]
    print("[xs] using XSDB fallback values [pb]:", xs)
    return xs


# ----------------------------------------------------------------------------
# kinematics helpers (plain numpy/awkward, no coffea)
# ----------------------------------------------------------------------------
def wrap_dphi(dphi):
    return np.abs(np.mod(dphi + np.pi, 2 * np.pi) - np.pi)


def p4_from_ptetaphim(pt, eta, phi, m):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    return px, py, pz, e


def rapidity(pt, eta, phi, m):
    _, _, pz, e = p4_from_ptetaphim(pt, eta, phi, m)
    return 0.5 * np.log((e + pz) / (e - pz))


# ----------------------------------------------------------------------------
# per-file selection
# ----------------------------------------------------------------------------
def process_file(url):
    """Return dict of flat per-event arrays for events passing the kinematic baseline."""
    with uproot.open(url, timeout=120) as f:
        tree = f["Events"]
        arr = tree.arrays(BRANCHES, library="ak")

    n_raw = len(arr)
    gw = ak.to_numpy(arr["genWeight"])
    sign_all = np.sign(gw)
    sign_all[sign_all == 0] = 1.0

    # ---- Z candidate: two leading dressed muons ----
    lep_pt = arr["GenDressedLepton_pt"]
    lep_eta = arr["GenDressedLepton_eta"]
    lep_phi = arr["GenDressedLepton_phi"]
    lep_m = arr["GenDressedLepton_mass"]
    lep_id = arr["GenDressedLepton_pdgId"]

    mu = (np.abs(lep_id) == 13) & (lep_pt > 20.0) & (np.abs(lep_eta) < 2.4)
    mpt, meta, mphi, mm, mid = (
        lep_pt[mu], lep_eta[mu], lep_phi[mu], lep_m[mu], lep_id[mu],
    )
    order = ak.argsort(mpt, ascending=False)
    mpt, meta, mphi, mm, mid = (
        mpt[order], meta[order], mphi[order], mm[order], mid[order],
    )
    has2 = ak.num(mpt) >= 2
    idx = np.where(ak.to_numpy(has2))[0]
    if len(idx) == 0:
        return None, n_raw, sign_all

    def lead(a, i):
        return ak.to_numpy(ak.fill_none(ak.firsts(a[idx][:, i: i + 1]), 0.0))

    mu1 = [lead(x, 0) for x in (mpt, meta, mphi, mm)]
    mu2 = [lead(x, 1) for x in (mpt, meta, mphi, mm)]
    id1 = ak.to_numpy(ak.fill_none(ak.firsts(mid[idx][:, 0:1]), 0))
    id2 = ak.to_numpy(ak.fill_none(ak.firsts(mid[idx][:, 1:2]), 0))
    opp = (id1 * id2) < 0

    p1 = p4_from_ptetaphim(*mu1)
    p2 = p4_from_ptetaphim(*mu2)
    zx, zy, zz, ze = (a + b for a, b in zip(p1, p2))
    z_pt = np.hypot(zx, zy)
    z_phi = np.arctan2(zy, zx)
    m2 = ze**2 - zx**2 - zy**2 - zz**2
    z_mass = np.sqrt(np.clip(m2, 0, None))

    zsel = opp & (z_mass > 71.2) & (z_mass < 111.2)

    # ---- jets, cleaned against the two muons ----
    jpt = arr["GenJetAK8_pt"][idx]
    jeta = arr["GenJetAK8_eta"][idx]
    jphi = arr["GenJetAK8_phi"][idx]
    jm = arr["GenJetAK8_mass"][idx]
    jfl = arr["GenJetAK8_partonFlavour"][idx]

    def dr_to(mu):
        de = jeta - mu[1]
        dp = wrap_dphi(jphi - mu[2])
        return np.sqrt(de**2 + dp**2)

    clean = (dr_to(mu1) > 0.8) & (dr_to(mu2) > 0.8)
    jpt, jeta, jphi, jm, jfl = (
        jpt[clean], jeta[clean], jphi[clean], jm[clean], jfl[clean])
    jorder = ak.argsort(jpt, ascending=False)
    jpt, jeta, jphi, jm, jfl = (
        jpt[jorder], jeta[jorder], jphi[jorder], jm[jorder], jfl[jorder])

    has_jet = ak.to_numpy(ak.num(jpt) >= 1)
    j1_pt = ak.to_numpy(ak.fill_none(ak.firsts(jpt), 0.0))
    j1_eta = ak.to_numpy(ak.fill_none(ak.firsts(jeta), 0.0))
    j1_phi = ak.to_numpy(ak.fill_none(ak.firsts(jphi), 0.0))
    j1_m = ak.to_numpy(ak.fill_none(ak.firsts(jm), 0.0))
    j2_pt = ak.to_numpy(ak.fill_none(ak.firsts(jpt[:, 1:2]), 0.0))
    j1_flav = ak.to_numpy(ak.fill_none(ak.firsts(jfl), 0)).astype(np.int32)

    with np.errstate(divide="ignore", invalid="ignore"):
        j1_y = np.where(has_jet, rapidity(j1_pt, j1_eta, j1_phi, j1_m), 99.0)

    jetsel = has_jet & (np.abs(j1_y) < 2.4) & (j1_pt > 170.0)

    keep = zsel & jetsel
    if keep.sum() == 0:
        return None, n_raw, sign_all

    dphi = wrap_dphi(j1_phi - z_phi)
    A = np.abs(z_pt - j1_pt) / (z_pt + j1_pt)
    sub_frac = np.where(z_pt > 0, j2_pt / z_pt, 0.0)

    out = dict(
        jet_pt=j1_pt[keep],
        z_pt=z_pt[keep],
        dphi=dphi[keep],
        A=A[keep],
        sub_frac=sub_frac[keep],
        flav=j1_flav[keep].astype(np.float64),
        sign=sign_all[idx][keep],
    )
    return out, n_raw, sign_all


def cached_file(lfn):
    """process_file with an on-disk per-file cache so re-runs skip all I/O."""
    key = FILECACHE / (lfn.strip("/").replace("/", "_")[-90:] + ".npz")
    if key.exists():
        z = np.load(key)
        meta = z["_meta"]
        data = None
        if "jet_pt" in z.files:
            data = {k: z[k] for k in
                    ("jet_pt", "z_pt", "dphi", "A", "sub_frac", "flav", "sign")}
        return data, int(meta[0]), float(meta[1]), int(meta[2]), True
    last = None
    for attempt, red in enumerate(REDIRECTORS):
        try:
            data, n_raw, sign_all = process_file(red + lfn)
            n_eff = float(np.sum(sign_all))
            n_neg = int(np.sum(sign_all < 0))
            payload = dict(_meta=np.array([n_raw, n_eff, n_neg], dtype=float))
            if data is not None:
                payload.update(data)
            np.savez_compressed(key, **payload)
            return data, n_raw, n_eff, n_neg, False
        except Exception as exc:
            last = f"{type(exc).__name__}: {str(exc)[:160]}"
            print(f"   !! attempt {attempt+1} failed: {last}")
    return None, None, None, None, False


def stream(nfiles_per_bin):
    xs = get_xs()
    cols = {k: [] for k in
            ("jet_pt", "z_pt", "dphi", "A", "sub_frac", "flav", "weight")}
    ptz_tag = []
    report = []
    for b in PTZ_BINS:
        nfiles = nfiles_per_bin[b]
        lst = REPO / "samples" / "zjet" / "mc" / f"ptz_{b}_UL18NanoAODv9.txt"
        lfns = [l.strip() for l in lst.read_text().splitlines() if l.strip()]
        done = 0
        li = 0
        n_read_bin = 0
        n_neg_bin = 0
        n_tot_bin = 0
        while done < nfiles and li < len(lfns):
            lfn = lfns[li]
            li += 1
            print(f"[read] {b} file#{done} {lfn.split('/')[-1]}")
            data, n_raw, n_eff, n_neg, hit = cached_file(lfn)
            if n_raw is None:
                continue  # both redirectors failed -> next file
            done += 1
            n_tot_bin += n_raw
            n_neg_bin += n_neg
            if data is None:
                print(f"   .. {n_raw} events, 0 pass baseline")
                continue
            n_read_bin += len(data["jet_pt"])
            w = xs[b] * data["sign"] / n_eff
            for k in ("jet_pt", "z_pt", "dphi", "A", "sub_frac", "flav"):
                cols[k].append(data[k])
            cols["weight"].append(w)
            ptz_tag.append(np.full(len(w), PTZ_BINS.index(b), dtype=np.int8))
            print(f"   .. {'cache ' if hit else ''}{n_raw} events, N_eff={n_eff:.0f}, "
                  f"{len(data['jet_pt'])} pass baseline")
        report.append(
            dict(bin=b, files=done, events=n_tot_bin, baseline=n_read_bin,
                 negfrac=(n_neg_bin / n_tot_bin if n_tot_bin else 0.0))
        )
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["ptz_tag"] = np.concatenate(ptz_tag)
    out["_report"] = np.array(json.dumps(report))
    return out


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def build_masks(d):
    dphi, A, z_pt, jet_pt = d["dphi"], d["A"], d["z_pt"], d["jet_pt"]
    cur = (dphi > 1.57) & (A < 0.3)
    rules = {"current": cur}
    for X in (200.0, 250.0):
        rules[f"X{int(X)}_ptZ"] = dphi > (np.pi - X / z_pt)
        rules[f"X{int(X)}_ptJet"] = dphi > (np.pi - X / jet_pt)
    return rules


def wsum(w, m=None):
    return float(np.sum(w[m])) if m is not None else float(np.sum(w))


def table(d, rules):
    w = d["weight"]
    ib = np.digitize(d["jet_pt"], JETPT_BINS) - 1
    rows = {}
    for i in range(len(JETPT_BINS) - 1):
        sel = ib == i
        wb = w[sel]
        base = wsum(wb)
        cur = rules["current"][sel]
        wcur = wsum(wb, cur)
        entry = {
            "n_raw": int(sel.sum()),
            "n_eff": float(np.sum(wb) ** 2 / np.sum(wb**2)) if len(wb) else 0.0,
            "w_baseline": base,
            "eff_current": wcur / base if base else np.nan,
        }
        for name, m in rules.items():
            if name == "current":
                continue
            mb = m[sel]
            entry[name] = {
                "eff_new": wsum(wb, mb) / base if base else np.nan,
                "keep": wsum(wb, mb & cur) / wcur if wcur else np.nan,
                "gained": wsum(wb, mb & ~cur) / wcur if wcur else np.nan,
                "net": wsum(wb, mb) / wcur if wcur else np.nan,
            }
        lo, hi = JETPT_BINS[i], JETPT_BINS[i + 1]
        rows[f"{lo:.0f}-{hi:.0f}"] = entry
    return rows


def print_tables(rows):
    for rule in ("X200_ptZ", "X250_ptZ", "X200_ptJet", "X250_ptJet"):
        print(f"\n=== rule {rule} ===")
        print(f"{'jet pT bin':>14} {'n_raw':>8} {'n_eff':>10} {'eff_cur':>8} "
              f"{'eff_new':>8} {'keep':>8} {'gained':>8} {'net':>8}")
        for k, e in rows.items():
            r = e[rule]
            print(f"{k:>14} {e['n_raw']:>8d} {e['n_eff']:>10.1f} "
                  f"{e['eff_current']:>8.4f} {r['eff_new']:>8.4f} "
                  f"{r['keep']:>8.4f} {r['gained']:>8.4f} {r['net']:>8.4f}")


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def stamp(ax, tag):
    ax.text(
        1.0, -0.185,
        f"{datetime.date.today().isoformat()} | smp_jetmass_run2 {tag} | "
        "inputs: NanoAOD gen truth, DY PtZ-binned NLO",
        transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.35",
    )


def make_plots(d, rules, rows, tag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    w, dphi, A, jet_pt, z_pt = (
        d["weight"], d["dphi"], d["A"], d["jet_pt"], d["z_pt"],
    )

    # ---- 1. dphi distributions ----
    fig, ax = plt.subplots(layout="constrained")
    edges = np.linspace(1.2, np.pi, 80)
    panels = [(200, 290), (400, 480), (680, 13000)]
    for c, (lo, hi) in zip(PETROFF, panels):
        m = (jet_pt >= lo) & (jet_pt < hi)
        h, _ = np.histogram(dphi[m], bins=edges, weights=w[m])
        h = h / max(h.sum(), 1e-300) / np.diff(edges)
        lbl = f"{lo}-{hi} GeV" if hi < 13000 else f">{lo} GeV"
        hep.histplot(h, edges, ax=ax, color=c, label=r"$p_{T}^{jet}$ " + lbl, lw=1.8)
        # threshold band: pi - X/ptZ evaluated at the Z pT range of this bin
        # threshold band spans pi - X/pT(Z) over the 5-95% pT(Z) range of the slice
        for X, alpha in ((200, 0.20), (250, 0.10)):
            zp = z_pt[m]
            if len(zp) < 10:
                continue
            t = np.pi - X / np.percentile(zp, [5, 95])
            ax.axvspan(min(t), max(t), color=c, alpha=alpha, lw=0)
    ax.axvline(1.57, color="k", lw=2.0, label=r"current cut $\Delta\phi>1.57$")
    ax.set_xlabel(r"$\Delta\phi(Z,\,\mathrm{jet})$")
    ax.set_ylabel("normalized / rad")
    ax.set_yscale("log")
    ax.set_xlim(1.2, np.pi)
    ax.set_ylim(3e-3, ax.get_ylim()[1] * 12)
    ax.legend(loc="upper left", fontsize=13,
              title=r"bands: $\pi - X/p_T^{Z}$, X = 200 (dark) / 250 (light)",
              title_fontsize=11)
    hep.cms.label("Preliminary", data=False, loc=0, rlabel="(13 TeV)", ax=ax)
    stamp(ax, tag)
    fig.savefig(SCRATCH / "zjet_dphi_dist.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. efficiency vs jet pT ----
    fig, ax = plt.subplots(layout="constrained")
    ctr = 0.5 * (JETPT_BINS[:-1] + np.minimum(JETPT_BINS[1:], 1000.0))
    keys = list(rows.keys())
    eff_cur = [rows[k]["eff_current"] for k in keys]
    ax.plot(ctr, eff_cur, "o-", color="k", label=r"current ($\Delta\phi>1.57$, $A<0.3$)")
    style = {
        "X200_ptZ": (PETROFF[0], "-", r"$\Delta\phi>\pi-200/p_T^{Z}$"),
        "X250_ptZ": (PETROFF[1], "-", r"$\Delta\phi>\pi-250/p_T^{Z}$"),
        "X200_ptJet": (PETROFF[0], "--", r"$\Delta\phi>\pi-200/p_T^{jet}$"),
        "X250_ptJet": (PETROFF[1], "--", r"$\Delta\phi>\pi-250/p_T^{jet}$"),
    }
    for name, (c, ls, lbl) in style.items():
        ax.plot(ctr, [rows[k][name]["eff_new"] for k in keys], ls,
                marker="s" if ls == "-" else "^", ms=5, color=c, label=lbl)
    ax.set_xlabel(r"$p_{T}^{jet}$ [GeV]")
    ax.set_ylabel("efficiency w.r.t. kinematic baseline")
    ax.set_xscale("log")
    ax.set_xlim(165, 1050)
    ax.set_xticks([200, 300, 400, 600, 800, 1000])
    ax.set_xticks([], minor=True)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.0, 1.45)
    ax.legend(loc="lower left", fontsize=13, ncol=1)
    hep.cms.label("Preliminary", data=False, loc=0, rlabel="(13 TeV)", ax=ax)
    stamp(ax, tag)
    fig.savefig(SCRATCH / "zjet_dphi_eff_vs_pt.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. exchanged populations, A distribution ----
    cur = rules["current"]
    new = rules["X200_ptZ"]
    # pick the pT bin with the most raw baseline entries above 200 GeV
    cand = [k for k in rows if rows[k]["n_raw"] > 0]
    best = max(cand, key=lambda k: rows[k]["n_raw"] * (float(k.split("-")[0]) >= 200))
    lo, hi = (float(x) for x in best.split("-"))
    inb = (jet_pt >= lo) & (jet_pt < hi)
    gained = inb & new & ~cur
    lost = inb & cur & ~new
    fig, ax = plt.subplots(layout="constrained")
    ed = np.linspace(0, 1, 50)
    for m, c, lbl in (
        (inb & cur, "0.5", f"current selection (N={int(m_n(inb & cur))})"),
        (gained, PETROFF[0], f"gained by X=200 (N={int(m_n(gained))})"),
        (lost, PETROFF[2], f"lost by X=200 (N={int(m_n(lost))})"),
    ):
        if m.sum() < 5:
            continue
        h, _ = np.histogram(A[m], bins=ed, weights=w[m])
        h = h / max(h.sum(), 1e-300) / np.diff(ed)
        hep.histplot(h, ed, ax=ax, color=c, label=lbl, lw=1.8)
    ax.axvline(0.3, color="k", ls="--", lw=1.5, label=r"current $A<0.3$")
    hi_lbl = f"{hi:.0f}" if hi < 13000 else r"$\infty$"
    ax.set_xlabel(r"$A = |p_T^{Z}-p_T^{jet}| / (p_T^{Z}+p_T^{jet})$")
    ax.set_ylabel("normalized")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    ax.legend(loc="upper right", fontsize=13, labelspacing=0.4, borderpad=0.6,
              title=rf"$p_T^{{jet}}$ {lo:.0f}-{hi_lbl} GeV", title_fontsize=13)
    hep.cms.label("Preliminary", data=False, loc=0, rlabel="(13 TeV)", ax=ax)
    stamp(ax, tag)
    fig.savefig(SCRATCH / "zjet_dphi_exchanged_A.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return best, gained, lost, inb


def m_n(m):
    return m.sum()



# ----------------------------------------------------------------------------
# flavour composition of the measured jet
# ----------------------------------------------------------------------------
FLAV_SELS = ["baseline", "current", "X200_ptZ", "X200_ptJet", "X250_ptZ",
             "gained_X200_ptZ"]


def flavour_masks(d, rules):
    n = len(d["weight"])
    cur = rules["current"]
    return {
        "baseline": np.ones(n, bool),
        "current": cur,
        "X200_ptZ": rules["X200_ptZ"],
        "X200_ptJet": rules["X200_ptJet"],
        "X250_ptZ": rules["X250_ptZ"],
        "gained_X200_ptZ": rules["X200_ptZ"] & ~cur,
    }


def flavour_table(d, rules):
    w, fl = d["weight"], np.abs(d["flav"])
    is_q = (fl >= 1) & (fl <= 5)
    is_g = fl == 21
    is_u = ~(is_q | is_g)   # partonFlavour == 0 (and any other unmatched code)
    ib = np.digitize(d["jet_pt"], JETPT_BINS) - 1
    masks = flavour_masks(d, rules)
    out = {}
    for i in range(len(JETPT_BINS) - 1):
        lo, hi = JETPT_BINS[i], JETPT_BINS[i + 1]
        binkey = f"{lo:.0f}-{hi:.0f}"
        out[binkey] = {}
        inb = ib == i
        for name, m in masks.items():
            sel = inb & m
            ww = w[sel]
            wq = float(np.sum(w[sel & is_q]))
            wg = float(np.sum(w[sel & is_g]))
            wu = float(np.sum(w[sel & is_u]))
            qg = sel & (is_q | is_g)
            wqg = w[qg]
            neff = float(np.sum(wqg) ** 2 / np.sum(wqg ** 2)) if len(wqg) else 0.0
            f_q = wq / (wq + wg) if (wq + wg) else np.nan
            tot = wq + wg + wu
            out[binkey][name] = dict(
                n_raw=int(sel.sum()),
                n_eff=neff,
                f_q=f_q,
                f_g=(wg / (wq + wg)) if (wq + wg) else np.nan,
                f_undef=(wu / tot) if tot else np.nan,
                err_f_q=(float(np.sqrt(max(f_q * (1 - f_q), 0.0) / neff))
                         if neff > 0 and np.isfinite(f_q) else np.nan),
            )
    return out


def print_flavour(ftab):
    hdr = f"{'jet pT bin':>14}"
    for s in FLAV_SELS:
        hdr += f" | {s:>21}"
    print("\n=== f_q = quark / (quark + gluon) of the measured jet ===")
    print(hdr)
    print(f"{'':>14}" + " | " + " | ".join(f"{'f_q +- err   n_eff':>21}"
                                           for _ in FLAV_SELS))
    for b, row in ftab.items():
        line = f"{b:>14}"
        for s in FLAV_SELS:
            e = row[s]
            if not np.isfinite(e["f_q"]):
                line += f" | {'--':>21}"
            else:
                line += (f" | {e['f_q']:.3f}+-{e['err_f_q']:.3f} "
                         f"{e['n_eff']:>7.0f}")
        print(line)
    print("\n--- delta f_q (current - new) and undefined fraction ---")
    print(f"{'jet pT bin':>14} {'d(X200_ptZ)':>12} {'d(X200_ptJet)':>14} "
          f"{'d(X250_ptZ)':>12} {'f_undef(base)':>14} {'f_q(gained)':>12} "
          f"{'f_g(gained)':>12}")
    for b, row in ftab.items():
        c = row["current"]["f_q"]
        print(f"{b:>14} {c - row['X200_ptZ']['f_q']:>12.4f} "
              f"{c - row['X200_ptJet']['f_q']:>14.4f} "
              f"{c - row['X250_ptZ']['f_q']:>12.4f} "
              f"{row['baseline']['f_undef']:>14.5f} "
              f"{row['gained_X200_ptZ']['f_q']:>12.3f} "
              f"{row['gained_X200_ptZ']['f_g']:>12.3f}")


def plot_flavour(ftab, tag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    keys = list(ftab.keys())
    ctr = 0.5 * (JETPT_BINS[:-1] + np.minimum(JETPT_BINS[1:], 1000.0))
    style = [
        ("current", PETROFF[3], "o", "-",
         r"current ($\Delta\phi>1.57$, $A<0.3$)"),
        ("X200_ptZ", PETROFF[0], "s", "-", r"$\Delta\phi>\pi-200/p_T^{Z}$"),
        ("X200_ptJet", PETROFF[1], "^", "--", r"$\Delta\phi>\pi-200/p_T^{jet}$"),
        ("gained_X200_ptZ", PETROFF[2], "v", ":",
         r"gained by $\pi-200/p_T^{Z}$"),
    ]
    fig, ax = plt.subplots(layout="constrained")
    for name, c, mk, ls, lbl in style:
        y = np.array([ftab[k][name]["f_q"] for k in keys])
        e = np.array([ftab[k][name]["err_f_q"] for k in keys])
        ax.errorbar(ctr, y, yerr=e, color=c, marker=mk, ls=ls, ms=7, lw=1.8,
                    capsize=3, label=lbl)
    ax.set_xscale("log")
    ax.set_xlim(165, 1050)
    ax.set_xticks([200, 300, 400, 600, 800, 1000])
    ax.set_xticks([], minor=True)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"$p_{T}^{jet}$ [GeV]")
    ax.set_ylabel(r"$f_{q} = $ quark / (quark + gluon)")
    ax.set_ylim(0.30, 1.06)
    ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
    ax.legend(loc="lower left", fontsize=13)
    hep.cms.label("Preliminary", data=False, loc=0, rlabel="(13 TeV)", ax=ax)
    stamp(ax, tag)
    out = SCRATCH / "zjet_dphi_fq_vs_pt.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfiles", type=int, default=0,
                    help="override the per-PtZ-bin file counts in NFILES")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--flavour", action="store_true",
                    help="also derive parton-flavour fractions of the measured jet")
    args = ap.parse_args()
    nfiles = {b: args.nfiles for b in PTZ_BINS} if args.nfiles else NFILES

    if CACHE.exists() and not args.refresh:
        z = np.load(CACHE, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        print(f"[cache] loaded {CACHE} ({len(d['jet_pt'])} baseline events)")
    else:
        d = stream(nfiles)
        np.savez_compressed(CACHE, **d)
        print(f"[cache] wrote {CACHE}")

    report = json.loads(str(d["_report"]))
    print("\n=== streamed files ===")
    print(f"{'PtZ bin':>10} {'files':>6} {'events':>12} {'baseline':>10} {'negfrac':>9}")
    for r in report:
        print(f"{r['bin']:>10} {r['files']:>6d} {r['events']:>12d} "
              f"{r['baseline']:>10d} {r['negfrac']:>9.3f}")

    d = {k: v for k, v in d.items() if k != "_report"}
    rules = build_masks(d)
    w = d["weight"]

    print("\n=== sanity ===")
    print(f"baseline events (raw)          : {len(w)}")
    print(f"fraction of baseline with A>0.3: {wsum(w, d['A'] > 0.3)/wsum(w):.4f}")
    print(f"fraction with dphi<1.57        : {wsum(w, d['dphi'] < 1.57)/wsum(w):.4f}")
    print(f"overall eff_current            : {wsum(w, rules['current'])/wsum(w):.4f}")

    rows = table(d, rules)
    print_tables(rows)

    tag = subprocess.run(
        ["git", "describe", "--always", "--dirty"], cwd=REPO,
        capture_output=True, text=True,
    ).stdout.strip()
    best, gained, lost, inb = make_plots(d, rules, rows, tag)

    # exchanged-population characterization (all pT, and the plotted bin)
    exch = {}
    cur = rules["current"]
    for label, m in (("all_pt", np.ones(len(w), bool)), (best, inb)):
        g = m & rules["X200_ptZ"] & ~cur
        l = m & cur & ~rules["X200_ptZ"]
        b = m & cur
        def stats(mask):
            if mask.sum() < 3:
                return None
            ww = w[mask]
            return dict(
                n=int(mask.sum()),
                sumw_frac=float(np.sum(ww) / np.sum(w[b])) if b.sum() else np.nan,
                A_mean=float(np.average(d["A"][mask], weights=ww)),
                A_median=float(np.median(d["A"][mask])),
                frac_A_gt_03=float(np.sum(ww[d["A"][mask] > 0.3]) / np.sum(ww)),
                subfrac_mean=float(np.average(d["sub_frac"][mask], weights=ww)),
                subfrac_median=float(np.median(d["sub_frac"][mask])),
                frac_subfrac_gt_03=float(
                    np.sum(ww[d["sub_frac"][mask] > 0.3]) / np.sum(ww)
                ),
                dphi_mean=float(np.average(d["dphi"][mask], weights=ww)),
            )
        exch[label] = {"current": stats(b), "gained": stats(g), "lost": stats(l)}

    print("\n=== exchanged populations (X=200, pT(Z) denominator) ===")
    for label, s in exch.items():
        print(f"-- {label}")
        for pop in ("current", "gained", "lost"):
            v = s[pop]
            if v is None:
                print(f"   {pop:>8}: too few events")
                continue
            print(f"   {pop:>8}: N={v['n']:>7d} wfrac={v['sumw_frac']:.4f} "
                  f"<A>={v['A_mean']:.3f} med(A)={v['A_median']:.3f} "
                  f"f(A>0.3)={v['frac_A_gt_03']:.3f} "
                  f"<pT2/pTZ>={v['subfrac_mean']:.3f} "
                  f"med={v['subfrac_median']:.3f} "
                  f"f(>0.3)={v['frac_subfrac_gt_03']:.3f} "
                  f"<dphi>={v['dphi_mean']:.3f}")

    payload = {"version": 2, "files": report, "bins": rows,
               "exchanged": exch, "plot_bin": best, "git": tag}

    if args.flavour:
        ftab = flavour_table(d, rules)
        print_flavour(ftab)
        fq_png = plot_flavour(ftab, tag)
        payload["flavour"] = ftab
        payload["flavour_defs"] = {
            "quark": "1 <= |GenJetAK8_partonFlavour| <= 5",
            "gluon": "|GenJetAK8_partonFlavour| == 21",
            "undefined": "partonFlavour == 0 (or any other code)",
            "f_q": "w_quark / (w_quark + w_gluon)",
            "err_f_q": "sqrt(f_q (1-f_q) / n_eff), n_eff over quark+gluon jets",
        }

    JSON_OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n[out] {JSON_OUT}")
    pngs = ["zjet_dphi_dist.png", "zjet_dphi_eff_vs_pt.png",
            "zjet_dphi_exchanged_A.png"]
    if args.flavour:
        pngs.append("zjet_dphi_fq_vs_pt.png")
    for p in pngs:
        print(f"[out] {SCRATCH/p}")


if __name__ == "__main__":
    main()
