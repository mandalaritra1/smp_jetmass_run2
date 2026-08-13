#!/usr/bin/env python
"""Data/MC validation campaign for the adopted `pair_split` hadronic selection.

The pair_split fiducial (dijet_processor / trijet_processor, `selection_mode=
"pair_split"`, 2026-08-06) replaces the old asymmetry + trijet-priority-veto
scheme by ONE azimuthal boundary on the leading AK8 pair,

    ptavg = (pT1 + pT2) / 2
    dijet  : dphi12 >  pi - (200 GeV)/ptavg     (|vec pT1 + vec pT2| < 200 GeV)
    trijet : dphi12 <  pi - (200 GeV)/ptavg     (the EXACT complement)

with pT floors 200/200 (dijet) and 200/200/185 (trijet), |y| < 2.4 on the two
(three) leading jets, and the trijet dphimin > 1 kept. The two channels are
orthogonal by construction: they cut the same variable on the same two jets.

This script produces the post-production validation figures from the pair_split
skims and, where the comparison is meaningful, the same quantity from the legacy
production. Subcommands (see `--help`):

    yields        new/old reco yield ratio vs measured-jet pT, MC and data
    boundary      D = dphi12 - (pi - 200/ptavg) at full stat, data vs MC
    dphi2d        dphi12 vs ptavg 2D, MC and data, boundary curve overlaid
    orthogonality dijet AND trijet event count on the joined event tables
    kinematics    per-jet pt/eta/phi/mass/msoftdrop data-MC (the new hists)
    msdrho        m_SD and rho data/MC per measured-jet pT bin, with ratio
    etaedge       |y| of the dijet jets, old vs new (the 2.5 -> 2.4 edge move)
    sculpting     trijet jet-3 rho shape, new vs old selection, per pT3 bin
    fakesmisses   fake / miss fractions vs pT, new vs old
    all           every subcommand above

WEIGHTS (verified 2026-08-07 for this production, do not re-derive by guessing):
the pair_split pickles were written AFTER `postprocess`, so BOTH the flat skim
`weight` columns AND the hist.Hist objects already carry xs*lumi*1000/sumw for
MC. Checks: (a) the HT700to1000 dijet reco-table mean weight is 78.96 in the new
files vs 78.94 in the legacy ones, (b) sum(pt_jet1) equals the jetidx==0 skim
weight sum to 0.07%, (c) re-applying corrections.getXSweight puts a spurious
shoulder at 480-720 GeV in the combined spectrum (log-slope rms 1.06 vs 0.53).
So NOTHING here multiplies by getXSweight. Data weights are trigger prescale
weights and are always applied; their error is sqrt(sum w^2).

    python scripts/validate_pairsplit.py all --outdir DIR
"""
from __future__ import annotations

import argparse
import pickle
import sys
from math import ceil
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.channel_overlap import bit, dedupe  # noqa: E402
from scripts.plot_stamp import stamp as _stamp  # noqa: E402

#############################################################################
#### inputs
#############################################################################
NEW_ROOT = "/Users/aritra/cernbox (2)/hadronic_skims_pairsplit"
#### all_syst production: same pair_split selection, minimal_rho mode, 77
#### populated systematic categories on the per-jet kin hists and on
#### ptjet_rhojet_g_reco. No skim tables in these pickles.
MIN_ROOT = "/Users/aritra/cernbox (2)/hadronic_minimal_rho_pairsplit"
OLD_DIJET_MC = "/Users/aritra/cernbox (2)/dijet_skims"
OLD_DIJET_DATA = "/Users/aritra/cernbox (2)/hadronic_rho_skims_2018"
OLD_TRIJET_MC = "/Users/aritra/cernbox (2)/trijet_skims/mc"
OLD_TRIJET_DATA = "/Users/aritra/cernbox (2)/trijet_skims/data"

#### (production, channel, source) -> (directory, glob). The old dijet-data
#### directory also holds the old trijet data files, hence the explicit globs.
INPUTS = {
    #### both pair_split areas were re-laid-out per YEAR on 2026-08-07; only
    #### 2018 is populated (the other year directories exist but are empty)
    ("new", "dijet", "mc"):    (f"{NEW_ROOT}/2018/dijet_mc", "*.pkl"),
    ("new", "dijet", "data"):  (f"{NEW_ROOT}/2018/dijet_data", "*.pkl"),
    ("new", "trijet", "mc"):   (f"{NEW_ROOT}/2018/trijet_mc", "*.pkl"),
    ("new", "trijet", "data"): (f"{NEW_ROOT}/2018/trijet_data", "*.pkl"),
    ("old", "dijet", "mc"):    (OLD_DIJET_MC, "rho_skim10_dijet_mg_pythia8*.pkl"),
    ("old", "dijet", "data"):  (OLD_DIJET_DATA, "rho_skim10_dijet_data*.pkl"),
    ("old", "trijet", "mc"):   (OLD_TRIJET_MC, "rho_skim_trijet_mg_pythia8*.pkl"),
    ("old", "trijet", "data"): (OLD_TRIJET_DATA, "rho_skim_trijet_data*.pkl"),
    #### NB the all_syst area is laid out per YEAR; only 2018 is populated
    #### (the 2016/2016APV/2017 directories exist but are empty).
    ("min", "dijet", "mc"):    (f"{MIN_ROOT}/2018/dijet_mc", "minimal_rho_*.pkl"),
    ("min", "dijet", "data"):  (f"{MIN_ROOT}/2018/dijet_data", "minimal_rho_*.pkl"),
    ("min", "trijet", "mc"):   (f"{MIN_ROOT}/2018/trijet_mc", "minimal_rho_*.pkl"),
    ("min", "trijet", "data"): (f"{MIN_ROOT}/2018/trijet_data", "minimal_rho_*.pkl"),
}

#### Bit order = _SKIM_CUTS in each processor at THIS commit. Appended names are
#### 0 in older skims; existing indices never move (APPEND-only invariant).
DIJET_CUTS = (
    "npv", "METfilters",
    "twoRecoJet", "recoRap2p5", "recodphi2", "recoAsym0p3",
    "muonIso0p4", "jetId", "hemveto", "recoTot_seq",
    "twoGenJet", "genRap2p5", "dphiGen2", "genAsym0p3", "genTot_seq",
    "matched_gen", "matched_reco", "final_seq",
    "vetoTrijetReco", "vetoTrijetGen",
    "recoPtFloor", "recodphiPair", "genPtFloor", "dphiGenPair",
)
TRIJET_CUTS = (
    "trigsel", "npv", "METfilters",
    "triRecoJet", "triRecoJet_seq", "recoRap2p5", "recoRap_seq",
    "recodphimin", "recodphi_seq", "muonIso0p4", "jetId", "hemveto",
    "recoTot_seq",
    "triGenJet", "rapGen", "dphiGen", "genTot_seq",
    "matched_gen", "matched_reco", "final_seq",
    "recoPtFloor3", "rapLead3", "recodphiPairInv",
    "genPtFloor3", "rapGenLead3", "dphiGenPairInv",
)

#### measured-jet pT binning (the unfolding pt axes)
PT_DIJET = np.array([200., 290., 400., 480., 570., 680., 760., 820., 1500.])
PT_TRIJET = np.array([185., 200., 290., 400., 1500.])
PT_EDGES = {"dijet": PT_DIJET, "trijet": PT_TRIJET}
#### the pT bin used for the per-bin panel plots (boundary / msdrho / sculpting)
BOUNDARY_PT_BINS = ((200., 290.), (400., 570.), (760., 1500.))
SCULPT_PT_BINS = ((200., 290.), (290., 400.), (400., 1500.))

JET_R = 0.8            # AK8; rho = 2*log10(m/(pt*R)) -- HadronicProcessorBase._rho
PT_SPLIT = 200.        # GeV, the numerator of the dphi boundary
LUMI_BY_ERA = {
    "2016APV": "19.5 fb$^{-1}$ (13 TeV)",
    "2016":    "16.8 fb$^{-1}$ (13 TeV)",
    "2017":    "41.5 fb$^{-1}$ (13 TeV)",
    "2018":    "59.8 fb$^{-1}$ (13 TeV)",
}
ERA = "2018"
LUMI_LABEL = LUMI_BY_ERA[ERA]
DATA_LABEL = f"Data {ERA} (JetHT)"
RL_MC = "(13 TeV)"
STAMP_INPUTS = "dphi_12 < pi - 200/pT 2018"


def set_era(era):
    """Point the `new`/`min` inputs, the lumi label, the data legend label and
    the provenance stamp at one era. The `old` (legacy-production) inputs are
    2018-only and untouched — subcommands that compare against them are
    restricted to --era 2018 in main()."""
    global ERA, LUMI_LABEL, DATA_LABEL, STAMP_INPUTS
    ERA = era
    LUMI_LABEL = LUMI_BY_ERA[era]
    DATA_LABEL = f"Data {era} (JetHT)"
    STAMP_INPUTS = f"dphi_12 < pi - 200/pT {era}"
    for chan in ("dijet", "trijet"):
        for src in ("mc", "data"):
            INPUTS[("new", chan, src)] = (f"{NEW_ROOT}/{era}/{chan}_{src}",
                                          "*.pkl")
            INPUTS[("min", chan, src)] = (f"{MIN_ROOT}/{era}/{chan}_{src}",
                                          "minimal_rho_*.pkl")

#### Petroff 6 (CVD-safe); data is always black.
C_BLUE, C_ORANGE, C_RED, C_PURPLE, C_GREY = (
    "#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1")

#### a ratio point is drawn only if BOTH numerator and denominator have at least
#### this many raw (unweighted) entries -- prescale-weighted data is dominated by
#### a handful of very-high-weight rows otherwise
MIN_RAW = 8
#### A relative systematic above this is not a systematic, it is one high-weight
#### MC jet migrating in or out of a nearly empty bin under a JES/JER variation
#### (seen at 3.6x nominal in the sparse trijet m_SD tail). Such bins are
#### dropped from the band, the points, and the quoted numbers rather than
#### drawn -- an un-capped band there fills the whole ratio panel.
MAX_REL_SYST = 1.0


#############################################################################
#### small numerics
#############################################################################
def rho(mass, pt):
    """rho = 2*log10(m/(pt*R)) -- HadronicProcessorBase._rho, R = 0.8."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(mass / (pt * JET_R))


def rapidity(pt, eta, mass):
    """y from the skim's (pt, eta, m) columns (the selection cuts on |y|, not eta)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.sqrt(mass ** 2 + (pt * np.cosh(eta)) ** 2)
        pz = pt * np.sinh(eta)
        return 0.5 * np.log((e + pz) / (e - pz))


def dphi_wrap(phi1, phi2):
    """|dphi| folded into [0, pi]."""
    d = np.abs(phi1 - phi2)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def boundary_dphi(ptavg):
    return np.pi - PT_SPLIT / ptavg


def ratio_err(n, en, d, ed):
    """Relative-error propagation for n/d, treating the two as independent."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = n / d
        return r, np.abs(r) * np.sqrt((en / n) ** 2 + (ed / d) ** 2)


#############################################################################
#### loading
#############################################################################
def paths_for(prod, chan, src):
    d, pat = INPUTS[(prod, chan, src)]
    p = sorted(Path(d).glob(pat))
    if not p:
        raise SystemExit(f"no pickles under {d}/{pat}")
    return p


def stream_table(prod, chan, src, table, cols, quiet=False):
    """Yield (filename, {col: ndarray}) one pickle at a time.

    Streaming rather than concatenating: the dijet MC reco table alone is 19M
    rows across the HT bins, and every subcommand here is a histogram fill.
    """
    for p in paths_for(prod, chan, src):
        sk = pickle.load(open(p, "rb"))["skim"]
        acc = sk.get(table, {})
        have = list(acc.keys())
        if not have:
            if not quiet:
                print(f"  !! {p.name}: empty '{table}' table", file=sys.stderr)
            continue
        out = {}
        for c in cols:
            if c not in acc:
                raise SystemExit(f"{p.name}: table '{table}' has no column '{c}' "
                                 f"(has {have})")
            out[c] = np.asarray(acc[c].value)
        if not quiet:
            print(f"  {p.name:<58s} {table:>8s} {len(out[cols[0]]):>10d} rows")
        yield p.name, out


_NO_XS_WARNED = set()


def mc_dataset_normalized(ds, year="2018"):
    """True when corrections has an xs AND a sumw for this dataset."""
    import smp_jetmass_run2.corrections as corr
    key = ds.split("/")[1] if ds.startswith("/") else ds
    return key in corr.xsdb and key in corr.sumw_qcd_mg.get(year, {})


def unnormalized_datasets(d, h, year=None):
    """Datasets whose stored hists were NOT xs-scaled by the production.

    `postprocess` multiplies each MC dataset by getXSweight = xs*lumi*1000/sumw,
    so a dataset's mean stored weight must be <genWeight> * xsw, with
    <genWeight> = sumw / nEvents -- sumw from corrections, nEvents from the
    pickle's own cutflow. Testing that against the raw hypothesis (mean =
    <genWeight>) identifies a dataset the production could not normalize.

    This CANNOT be decided from corrections alone any more: QCD_HT100to200's
    2018 sumw entry is now committed (getXSweight = 0.374), but the pair_split
    productions predate it and stored that dataset RAW. Measured on the 2018
    dijet all_syst file: stored/<genW> = 1.010 for HT100to200 while every other
    HT bin sits at stored/(<genW>*xsw) = 1.000 +- 0.005. Left in, it injects
    2.7% of the MC yield from 1139 effective entries (mean weight 4.8e4) and
    spikes individual pT bins at the 7% level. The flat skim tables are
    unaffected -- that dataset has zero skim rows in every production here.
    """
    import smp_jetmass_run2.corrections as corr
    #### resolve at call time so set_era() is respected: the sumw tables,
    #### getXSweight and the membership check are all era-keyed
    year = ERA if year is None else year
    bad = set()
    cf = d.get("cutflow") or {}
    nev = {}
    for sub in cf.values():
        if not hasattr(sub, "items"):
            continue
        for k, v in sub.items():
            if isinstance(k, str) and k.startswith("nEvents initial /"):
                nev[k[len("nEvents initial "):]] = v
    systs = list(h.axes["systematic"])
    pick = "nominal" if "nominal" in systs else sum
    for ds in h.axes["dataset"]:
        key = ds.split("/")[1] if ds.startswith("/") else ds
        sumw = corr.sumw_qcd_mg.get(year, {}).get(key)
        n = nev.get(ds)
        if not sumw or not n:
            #### cannot test -- fall back to the corrections-membership check
            if not mc_dataset_normalized(ds, year):
                bad.add(ds)
            continue
        xsw = corr.getXSweight(ds, year)
        hh = h[{"dataset": ds, "systematic": pick}]
        v, va = float(hh.values().sum()), float(hh.variances().sum())
        if v <= 0 or va <= 0 or not xsw or xsw == 1.0:
            continue
        mean = v / (v ** 2 / va)                      # sum(w^2)/sum(w)
        mg = sumw / n
        if abs(mean / mg - 1.0) < abs(mean / (mg * xsw) - 1.0) > 0.1:
            bad.add(ds)
    for ds in bad:
        key = ds.split("/")[1] if ds.startswith("/") else ds
        if key not in _NO_XS_WARNED:
            _NO_XS_WARNED.add(key)
            print(f"  !! dropping MC dataset stored WITHOUT the xs scale: {key}",
                  file=sys.stderr)
    return bad


def hist_variations(prod, chan, src, name, quiet=False):
    """Load a hist per SYSTEMATIC category, summed over datasets and files.

    Returns (axes, {systematic: (values, variances)}) where `axes` are the
    surviving (non-dataset, non-systematic) axes, so this serves both the 1D
    per-jet kinematic hists and the 2D ptjet_rhojet_g_reco. The stored hists
    already carry the xs (MC) / prescale (data) weights -- see the module
    docstring. MC datasets without an xs/sumw entry are dropped
    (see mc_dataset_normalized); in the all_syst minimal_rho production
    QCD_HT100to200 IS present and IS unnormalized, so this matters there.
    """
    axes, per = None, {}
    for p in paths_for(prod, chan, src):
        d = pickle.load(open(p, "rb"))
        if name not in d:
            raise SystemExit(f"{p.name}: no hist '{name}' "
                             f"(this production predates it?)")
        h = d[name]
        bad = set() if src == "data" else unnormalized_datasets(d, h)
        keep = [ds for ds in h.axes["dataset"] if ds not in bad]
        if not keep:
            continue
        for s in h.axes["systematic"]:
            hh = sum(h[{"dataset": ds, "systematic": s}] for ds in keep)
            if axes is None:
                axes = list(hh.axes)
            v, va = hh.values(), hh.variances()
            if s not in per:
                per[s] = [np.zeros_like(v), np.zeros_like(v)]
            per[s][0] += v
            per[s][1] += va
        if not quiet:
            print(f"  {p.name:<58s} {name:>20s} "
                  f"nsyst={len(per)} sum={per['nominal'][0].sum():.5g}")
    return axes, {s: (v, va) for s, (v, va) in per.items()}


def hist1d_sum(prod, chan, src, name, quiet=False):
    """Nominal 1D hist (values, variances, edges, xlabel), datasets combined."""
    axes, per = hist_variations(prod, chan, src, name, quiet=quiet)
    v, va = per["nominal"] if "nominal" in per else (
        sum(x[0] for x in per.values()), sum(x[1] for x in per.values()))
    return v, va, axes[0].edges, axes[0].label


#############################################################################
#### systematics band
#############################################################################
def syst_sources(names):
    """{source: {'up': name, 'down': name}} from a systematic axis listing.

    Up/Down suffixes are stripped, so each JES_<source> pair collapses to ONE
    source (54 entries -> 27 sources) and likewise for the 16 weight entries
    (-> 8 sources); JER/JMS/JMR add 3. 38 sources in the 2018 all_syst run.
    """
    out = {}
    for n in names:
        if n == "nominal":
            continue
        if n.endswith("Up"):
            out.setdefault(n[:-2], {})["up"] = n
        elif n.endswith("Down"):
            out.setdefault(n[:-4], {})["down"] = n
        else:
            out.setdefault(n, {})["one"] = n
    return out


def syst_total(per, transform=None):
    """(nominal, total abs syst per bin, {source: abs dev per bin}).

    Per source the deviation is the SYMMETRIZED (|up-nom| + |down-nom|)/2; a
    source with only one variation contributes that single |dev|. Sources are
    added in quadrature. `transform` is applied to every variation BEFORE
    differencing -- pass an area-normalizer to get a SHAPE-only band.
    """
    f = transform or (lambda v: v)
    nom = f(per["nominal"][0])
    contrib = {}
    for s, d in syst_sources(per).items():
        devs = [np.abs(f(per[d[k]][0]) - nom)
                for k in ("up", "down", "one") if k in d]
        if devs:
            contrib[s] = np.mean(devs, axis=0)
    tot = (np.sqrt(sum(c ** 2 for c in contrib.values())) if contrib
           else np.zeros_like(nom))
    return nom, tot, contrib


def area_norm(target=1.0):
    def f(v):
        s = v.sum()
        return v * (target / s) if s else v
    return f


def report_sources(tag, nom, contrib, mask=None):
    """Print the sources ranked by their yield-weighted mean relative size."""
    m = np.ones_like(nom, bool) if mask is None else mask
    m = m & (nom > 0)
    if not m.any():
        return []
    w = nom[m]
    rank = sorted(((float((c[m] / nom[m] * w).sum() / w.sum()), s)
                   for s, c in contrib.items()), reverse=True)
    print(f"  {tag}: dominant sources (yield-weighted mean rel. dev.)")
    for v, s in rank[:5]:
        print(f"      {s:<28s} {100*v:6.2f}%")
    return rank


def load_events(prod, chan, src):
    """Event-display table (1/1000 sample) with the cut bitmask, deduped."""
    keys, bits, wt = [], [], []
    for _, t in stream_table(prod, chan, src, "events",
                             ("dataset_id", "run", "lumi", "event",
                              "cutbits", "weight")):
        keys.append(np.stack([t[c].astype(np.int64) for c in
                              ("dataset_id", "run", "lumi", "event")], axis=1))
        bits.append(t["cutbits"])
        wt.append(t["weight"])
    k, b, w = np.concatenate(keys), np.concatenate(bits), np.concatenate(wt)
    m = dedupe(k)
    return k[m], b[m], w[m]


def dijet_pairs(t):
    """Rebuild the leading pair from a dijet per-jet table.

    Rows alternate jetidx 0, 1 (see HadronicProcessorBase._fill_skim: the two
    measured jets are written per event in that order), so the even rows are the
    leading jets and the odd rows their partner. Asserted, not assumed.
    """
    ji = t["jetidx"]
    if len(ji) % 2 or not (np.all(ji[0::2] == 0) and np.all(ji[1::2] == 1)):
        raise SystemExit("dijet table is not the expected alternating 0,1 layout")
    return {k: (v[0::2], v[1::2]) for k, v in t.items()}


#############################################################################
#### plotting scaffolding (CMS house style -- see the cms-plot-style skill)
#############################################################################
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.CMS)
    return plt, hep


def cms_label(ax, is_data, rlabel=None):
    _, hep = _mpl()
    hep.cms.label("Preliminary", data=is_data, loc=0, ax=ax,
                  rlabel=LUMI_LABEL if (rlabel is None and is_data)
                  else (RL_MC if rlabel is None else rlabel))


def panel_text(ax, text, y=0.955, **kw):
    """In-frame descriptor, top-left. With loc=0 the CMS block is ABOVE the
    frame, so the top-left inside the axes is free for the pT-bin label."""
    kw.setdefault("fontsize", 20)
    return ax.text(0.035, y, text, transform=ax.transAxes, ha="left", va="top",
                   **kw)


def grid(nrows, ncols, base=10.0, gap=1.06, **kw):
    plt, _ = _mpl()
    fig, axes = plt.subplots(nrows, ncols, layout="constrained",
                             figsize=(base * ncols * gap, base * nrows * gap),
                             **kw)
    return fig, np.atleast_1d(axes).ravel()


def ratio_grid(n, ncols=None, base=10.0, gap=1.06):
    """n cells, each a (main, ratio) axes pair sharing x."""
    plt, _ = _mpl()
    ncols = ncols or min(n, 4)
    nrows = ceil(n / ncols)
    fig = plt.figure(figsize=(base * ncols * gap, base * nrows * gap),
                     layout="constrained")
    gs = fig.add_gridspec(nrows, ncols)
    cells = []
    for i in range(n):
        r, c = divmod(i, ncols)
        sub = gs[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.04)
        a0 = fig.add_subplot(sub[0])
        a1 = fig.add_subplot(sub[1], sharex=a0)
        a0.tick_params(labelbottom=False)
        cells.append((a0, a1))
    return fig, cells


#### The ratio-panel band. This production ran the `no_syst` profile and the
#### per-jet kinematic hists are filled for the NOMINAL jet systematic only, so
#### there is no experimental systematic in these files at all: the band is MC
#### STATISTICS ONLY and must never be read as a systematic band.
BAND_LABEL = "MC stat. unc."


def mc_stat_band(ax, edges, rel, label=BAND_LABEL):
    """Grey 1 +- rel band around unity in a ratio panel.

    `rel` = per-bin sqrt(sum w^2)/sum w of the MC. Returns a legend proxy so the
    band can be named in the MAIN panel's legend (the ratio panel is too short
    to carry one).
    """
    from matplotlib.patches import Patch
    rel = np.where(np.isfinite(rel), np.abs(rel), 0.0)
    ax.stairs(1.0 + rel, edges, baseline=1.0 - rel, fill=True,
              color=C_GREY, alpha=0.45, lw=0)
    return Patch(facecolor=C_GREY, alpha=0.45, label=label)


BAND2_LABEL = "stat $\\oplus$ syst"


def mc_bands(ax, edges, rel_stat, rel_syst, label=BAND_LABEL,
             label2=BAND2_LABEL):
    """Nested ratio-panel bands: outer stat (+) syst in quadrature, inner stat.

    Returns the two legend proxies (outer first is drawn first so the inner one
    stays visible on top).
    """
    from matplotlib.patches import Patch
    rs = np.where(np.isfinite(rel_stat), np.abs(rel_stat), 0.0)
    ry = np.where(np.isfinite(rel_syst), np.abs(rel_syst), 0.0)
    tot = np.sqrt(rs ** 2 + ry ** 2)
    ax.stairs(1.0 + tot, edges, baseline=1.0 - tot, fill=True,
              color=C_GREY, alpha=0.25, lw=0)
    ax.stairs(1.0 + rs, edges, baseline=1.0 - rs, fill=True,
              color=C_GREY, alpha=0.65, lw=0)
    return (Patch(facecolor=C_GREY, alpha=0.65, label=label),
            Patch(facecolor=C_GREY, alpha=0.25, label=label2))


def data_mc_ratio(sw_d, sw2_d, sw_m, sw2_m):
    """(ratio, data error, MC relative error) with 0/0 -> nan."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = sw_d / sw_m
        er = np.sqrt(sw2_d) / sw_m           # data stat only, MC in the band
        rel = np.sqrt(sw2_m) / sw_m
    return r, er, rel


#### Ratio panels are ALWAYS centred on unity (Aritra, 2026-08-07): 1.0 sits at
#### the vertical centre and the half-width is driven by the BULK of the
#### measurement, never by a single high-error outlier. Defaults to [0.5, 1.5].
RATIO_HALF_DEFAULT = 0.5
RATIO_HALF_MAX = 1.0


def _weighted_quantile(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, x))


def ratio_half_width(vals, errs, band=None, default=RATIO_HALF_DEFAULT,
                     hard=RATIO_HALF_MAX):
    """Half-width of a unity-centred ratio panel.

    Driven by (a) the widest drawn uncertainty band and (b) the central 90% of
    the points WEIGHTED BY THEIR SIGNIFICANCE (1/error), so a statistically
    meaningless point with a huge error bar cannot stretch the axis -- that was
    pushing unity off-centre and squashing the bands flat.
    """
    need = 0.0
    if band is not None:
        b = np.asarray(band)
        b = b[np.isfinite(b)]
        if b.size:
            #### the 90th percentile, NOT the max: one sparse bin whose band
            #### touches the MAX_REL_SYST ceiling would otherwise force the
            #### panel wide open and squash the band everywhere else
            need = max(need, float(np.percentile(b, 90)))
    m = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
    if m.any():
        dev, w = np.abs(vals[m] - 1.0), 1.0 / errs[m]
        need = max(need, _weighted_quantile(dev, w, 0.90))
    return float(np.clip(need * 1.25, default, hard))


def draw_ratio(ar, edges, series, rel_stat, rel_syst=None, ylabel="Data / MC",
               xlim=None):
    """One ratio panel: band(s), unity-centred limits, points with the
    out-of-range edge-marker convention. Returns (legend proxies, labels)
    for the MAIN panel's legend.

    `series` is a list of (ratio, error, good-mask, marker) -- the dijet
    kinematics panels carry one series per measured jet.

    Points outside the panel are NOT dropped and do NOT set the range: they are
    drawn as open triangles pinned just inside the boundary, pointing the way
    they went.

    `xlim` is applied here and, crucially, also gates which bins are allowed to
    SET the y-range: a bin outside the drawn x-window is invisible and must not
    open the panel up (the trijet m_SD tail at 265-475 GeV, four orders of
    magnitude down and off the 0-300 GeV panel, was forcing every trijet m_SD
    ratio to 0-2 and flattening the bands).
    """
    ctr = 0.5 * (edges[1:] + edges[:-1])
    if xlim is not None:
        ar.set_xlim(*xlim)
    xlo, xhi = ar.get_xlim() if xlim is None else xlim
    vis = (ctr >= min(xlo, xhi)) & (ctr <= max(xlo, xhi))
    gany = np.zeros(len(edges) - 1, bool)
    for _, _, g, _ in series:
        gany |= np.asarray(g, bool)
    rs = np.where(gany, np.nan_to_num(rel_stat), 0.0)
    if rel_syst is None:
        proxies, labels, env = [mc_stat_band(ar, edges, rs)], [BAND_LABEL], rs
    else:
        ry = np.where(gany, np.nan_to_num(rel_syst), 0.0)
        proxies = list(mc_bands(ar, edges, rs, ry))
        labels = [BAND_LABEL, BAND2_LABEL]
        env = np.sqrt(rs ** 2 + ry ** 2)
    allv = np.concatenate([np.where(np.asarray(g, bool) & vis, r, np.nan)
                           for r, _, g, _ in series])
    alle = np.concatenate([np.where(np.asarray(g, bool) & vis, e, np.nan)
                           for _, e, g, _ in series])
    gv = gany & vis
    half = ratio_half_width(allv, alle, band=env[gv] if gv.any() else None)
    lo, hi = 1.0 - half, 1.0 + half
    pad = 0.035 * (hi - lo)
    for r, er, g, mark in series:
        v = np.where(np.asarray(g, bool), r, np.nan)
        inside = np.isfinite(v) & (v >= lo) & (v <= hi)
        ar.errorbar(ctr[inside], v[inside], yerr=np.asarray(er)[inside],
                    fmt=mark, ms=5, mfc="k" if mark == "o" else "none",
                    color="k", lw=1.3)
        for sel, ypos, tri in ((np.isfinite(v) & (v > hi), hi - pad, "^"),
                               (np.isfinite(v) & (v < lo), lo + pad, "v")):
            if sel.any():
                ar.plot(ctr[sel], np.full(int(sel.sum()), ypos), tri, ms=10,
                        mfc="none", mec="k", mew=1.8, ls="none")
    ar.axhline(1.0, color="k", lw=1.0, ls=":")
    ar.set_ylim(lo, hi)
    ar.set_ylabel(ylabel)
    ar.grid(alpha=0.25)
    return proxies, labels


def headroom(ax, ymax, frac=2.5, log=True):
    lo, _ = ax.get_ylim()
    if log:
        ax.set_ylim(max(lo, ymax * 1e-4), ymax * frac)
    else:
        ax.set_ylim(lo, ymax * frac)


def save(fig, outdir, name, stamp_inputs=None):
    """PNG for browsing/slides plus a vector PDF twin for the AN.

    `stamp_inputs` defaults at CALL time (not def time) so set_era's rewrite
    of STAMP_INPUTS is respected."""
    plt, _ = _mpl()
    _stamp(fig, inputs=STAMP_INPUTS if stamp_inputs is None else stamp_inputs)
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / name
    fig.savefig(f, dpi=110)
    fig.savefig(f.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", f, "(+ .pdf)")
    return f


def ratio_single(base=10.0):
    """One standalone (main, ratio) figure. Per-panel figures, not grids: grid
    sheets render tiny on slides and cannot be dropped into the AN one by one
    (Aritra, 2026-08-07)."""
    fig, cells = ratio_grid(1, ncols=1, base=base)
    return fig, cells[0]


def ptbin_tag(lo, hi):
    hi_s = "Inf" if hi >= 1500 else f"{hi:.0f}"
    return f"pt{lo:.0f}to{hi_s}"


def step(ax, edges, y, **kw):
    """Filled-outline step, drawn so the first and last bin edges are closed."""
    ax.stairs(y, edges, baseline=None, **kw)


def points(ax, edges, y, yerr=None, **kw):
    ctr = 0.5 * (edges[1:] + edges[:-1])
    kw.setdefault("fmt", "o")
    kw.setdefault("ms", 5)
    kw.setdefault("color", "k")
    kw.setdefault("lw", 1.4)
    return ax.errorbar(ctr, y, yerr=yerr, **kw)


#############################################################################
#### 1. yield ratio, new / old, vs measured-jet pT
#############################################################################
def cmd_yields(args):
    outdir = Path(args.outdir)
    table = {}
    for chan in ("dijet", "trijet"):
        edges = PT_EDGES[chan]
        res = {}
        for src in ("mc", "data"):
            for prod in ("new", "old"):
                sw = np.zeros(len(edges) - 1)
                sw2 = np.zeros(len(edges) - 1)
                print(f"[yields] {chan} {src} {prod}")
                for _, t in stream_table(prod, chan, src, "reco",
                                         ("ptreco", "weight")):
                    h, _ = np.histogram(t["ptreco"], bins=edges,
                                        weights=t["weight"])
                    h2, _ = np.histogram(t["ptreco"], bins=edges,
                                         weights=t["weight"] ** 2)
                    sw += h
                    sw2 += h2
                res[(src, prod)] = (sw, np.sqrt(sw2))
        table[chan] = res

        #### ---- figure
        plt, _ = _mpl()
        fig, ax = plt.subplots(layout="constrained")
        ctr = 0.5 * (edges[1:] + edges[:-1])
        xerr = 0.5 * np.diff(edges)
        txt = []
        for src, col, mark, lab in (("mc", C_BLUE, "s", "QCD MG+Pythia8"),
                                    ("data", "k", "o", DATA_LABEL)):
            n, en = res[(src, "new")]
            d, ed = res[(src, "old")]
            r, er = ratio_err(n, en, d, ed)
            ax.errorbar(ctr, r, yerr=er, xerr=xerr, fmt=mark, ms=9, lw=2,
                        color=col, label=lab)
            tot = n.sum() / d.sum()
            txt.append(f"{lab.split()[0]} total {tot:.3f}")
            print(f"  {chan} {src}: integrated new/old = {tot:.4f} "
                  f"({n.sum():.5g} / {d.sum():.5g})")
        ax.axhline(1.0, color=C_GREY, lw=1.6, ls=":")
        ax.set_xlabel(r"measured-jet $p_{T}$ [GeV]")
        ax.set_ylabel("yield ratio  pair_split / legacy")
        ax.set_xscale("log")
        ax.set_xticks([200, 300, 400, 600, 800, 1000, 1500])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(edges[0] * 0.95, edges[-1])
        ax.set_ylim(0.0, 1.55)
        panel_text(ax, f"{chan} channel\n" + "\n".join(txt), fontsize=19)
        ax.legend(loc="lower left", fontsize=19)
        ax.grid(alpha=0.25)
        cms_label(ax, True)
        save(fig, outdir, f"yields_ratio_{chan}.png")

    #### ---- printed table
    print("\n=== new/old reco yield ratio (weighted) ===")
    for chan, res in table.items():
        edges = PT_EDGES[chan]
        print(f"\n{chan}: {'pT bin':>14s} {'MC new':>12s} {'MC old':>12s} "
              f"{'MC n/o':>8s} {'data new':>12s} {'data old':>12s} {'data n/o':>9s}")
        for i in range(len(edges) - 1):
            mn, mo = res[("mc", "new")][0][i], res[("mc", "old")][0][i]
            dn, do = res[("data", "new")][0][i], res[("data", "old")][0][i]
            print(f"{'':>6s} {edges[i]:>6.0f}-{edges[i+1]:<7.0f} "
                  f"{mn:>12.5g} {mo:>12.5g} {mn/mo if mo else np.nan:>8.3f} "
                  f"{dn:>12.5g} {do:>12.5g} {dn/do if do else np.nan:>9.3f}")
        print(f"{'':>6s} {'TOTAL':>14s} "
              f"{res[('mc','new')][0].sum():>12.5g} {res[('mc','old')][0].sum():>12.5g} "
              f"{res[('mc','new')][0].sum()/res[('mc','old')][0].sum():>8.3f} "
              f"{res[('data','new')][0].sum():>12.5g} {res[('data','old')][0].sum():>12.5g} "
              f"{res[('data','new')][0].sum()/res[('data','old')][0].sum():>9.3f}")
    return table


#############################################################################
#### 2. boundary distance D = dphi12 - (pi - 200/ptavg)
#############################################################################
def cmd_boundary(args):
    outdir = Path(args.outdir)
    dbins = np.linspace(0.0, 1.6, 65)
    got = {}
    for src in ("mc", "data"):
        h = [np.zeros(len(dbins) - 1) for _ in BOUNDARY_PT_BINS]
        h2 = [np.zeros(len(dbins) - 1) for _ in BOUNDARY_PT_BINS]
        raw = [np.zeros(len(dbins) - 1) for _ in BOUNDARY_PT_BINS]
        nneg = ntot = 0
        print(f"[boundary] dijet {src} (new)")
        for _, t in stream_table("new", "dijet", src, "reco",
                                 ("ptreco", "phireco", "weight", "jetidx")):
            p = dijet_pairs(t)
            pt1, pt2 = p["ptreco"]
            ph1, ph2 = p["phireco"]
            w = p["weight"][0]
            ptavg = 0.5 * (pt1 + pt2)
            dd = dphi_wrap(ph1, ph2) - boundary_dphi(ptavg)
            nneg += int((dd < 0).sum())
            ntot += len(dd)
            for i, (lo, hi) in enumerate(BOUNDARY_PT_BINS):
                m = (pt1 >= lo) & (pt1 < hi)
                a, _ = np.histogram(dd[m], bins=dbins, weights=w[m])
                b, _ = np.histogram(dd[m], bins=dbins, weights=w[m] ** 2)
                c, _ = np.histogram(dd[m], bins=dbins)
                h[i] += a
                h2[i] += b
                raw[i] += c
        got[src] = (h, h2, raw)
        print(f"  closure: {nneg} of {ntot} selected dijet pairs have D < 0 "
              f"({'PASS' if nneg == 0 else 'FAIL'})")

    for i, (lo, hi) in enumerate(BOUNDARY_PT_BINS):
        fig, (ax, ar) = ratio_single()
        hm, hm2, rm = got["mc"]
        hd, hd2, rd = got["data"]
        nm, nd = hm[i].sum(), hd[i].sum()
        ym, em = hm[i] / nm, np.sqrt(hm2[i]) / nm
        yd, ed = hd[i] / nd, np.sqrt(hd2[i]) / nd
        good = (rm[i] >= MIN_RAW) & (rd[i] >= MIN_RAW)
        step(ax, dbins, ym, color=C_BLUE, lw=2.5, label="QCD MG+Pythia8")
        ax.stairs(ym + em, dbins, baseline=ym - em, fill=True, alpha=0.25,
                  color=C_BLUE, lw=0)
        points(ax, dbins, np.where(rd[i] >= MIN_RAW, yd, np.nan), ed,
               label=DATA_LABEL)
        ax.axvline(0.0, color=C_RED, lw=2.0, ls="--")
        #### x in data coords; 0.030 keeps the rotated text clear of the left
        #### spine (0.012 clipped the label in some eras' renders)
        ax.text(0.030, 0.40, "selection boundary $D=0$", color=C_RED,
                transform=ax.get_xaxis_transform(), rotation=90, va="center",
                fontsize=17)
        ax.set_yscale("log")
        ax.set_ylabel("Fraction of jet pairs / 0.025")
        ax.set_xlim(dbins[0], dbins[-1])
        headroom(ax, max(np.nanmax(ym), np.nanmax(yd)), frac=12.0)
        hi_s = "$\\infty$" if hi >= 1500 else f"{hi:.0f}"
        panel_text(ax, f"dijet, ${lo:.0f} < p_{{T,1}} <$ {hi_s} GeV")
        ax.grid(alpha=0.25)
        cms_label(ax, True)
        #### ratio: both are unit-normalized, so this is a pure SHAPE ratio
        r, er, rel = data_mc_ratio(yd, (np.sqrt(hd2[i]) / nd) ** 2,
                                   ym, (np.sqrt(hm2[i]) / nm) ** 2)
        px, pl = draw_ratio(ar, dbins, [(r, er, good, "o")], rel,
                            xlim=(dbins[0], dbins[-1]))
        ar.set_xlabel(r"$D = \Delta\phi_{12} - (\pi - 200\,\mathrm{GeV}/\bar{p}_T)$")
        h, la = ax.get_legend_handles_labels()
        ax.legend(h + px, la + pl, loc="upper right", fontsize=18)
        save(fig, outdir, f"boundary_distance_dijet_{ptbin_tag(lo, hi)}.png")


#############################################################################
#### 3. dphi12 vs ptavg, 2D, with the boundary curve
#############################################################################
def cmd_dphi2d(args):
    outdir = Path(args.outdir)
    ptb = np.linspace(200., 1200., 100)
    phb = np.linspace(2.0, np.pi, 100)
    H = {}
    for src in ("mc", "data"):
        h = np.zeros((len(ptb) - 1, len(phb) - 1))
        print(f"[dphi2d] dijet {src} (new)")
        for _, t in stream_table("new", "dijet", src, "reco",
                                 ("ptreco", "phireco", "weight", "jetidx")):
            p = dijet_pairs(t)
            pt1, pt2 = p["ptreco"]
            ph1, ph2 = p["phireco"]
            w = p["weight"][0]
            ptavg = 0.5 * (pt1 + pt2)
            a, _, _ = np.histogram2d(ptavg, dphi_wrap(ph1, ph2),
                                     bins=[ptb, phb], weights=w)
            h += a
        H[src] = h

    plt, _ = _mpl()
    from matplotlib.colors import LogNorm
    fig, axes = grid(1, 2)
    for ax, (src, title, isd) in zip(
            axes, (("mc", "QCD MG+Pythia8", False),
                   ("data", DATA_LABEL, True))):
        h = H[src]
        pos = h[h > 0]
        im = ax.pcolormesh(ptb, phb, h.T, cmap="viridis",
                           norm=LogNorm(vmin=max(pos.min(), pos.max() * 1e-6),
                                        vmax=pos.max()))
        x = np.linspace(ptb[0], ptb[-1], 400)
        ax.plot(x, boundary_dphi(x), color=C_RED, lw=3,
                label=r"$\Delta\phi_{12} = \pi - 200\,\mathrm{GeV}/\bar{p}_T$")
        ax.set_xlabel(r"$\bar{p}_T = (p_{T,1}+p_{T,2})/2$ [GeV]")
        ax.set_ylabel(r"$\Delta\phi_{12}$")
        ax.set_ylim(phb[0], phb[-1])
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Weighted jet pairs")
        panel_text(ax, title, color="w")
        ax.legend(loc="lower right", fontsize=18, framealpha=0.9)
    #### mplhep freezes the CMS<->"Simulation" gap as an AXES FRACTION when the
    #### label is stamped; the colorbar then steals ~20% of the axes width and
    #### the two words collide. Let constrained layout settle first, freeze the
    #### geometry, and only then stamp.
    #### reserve the stamp strip AND the band above the frame that the CMS
    #### label will occupy, BEFORE freezing (save() can no longer do either)
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.90))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax, isd in zip(axes, (False, True)):
        cms_label(ax, isd)
    save(fig, outdir, "dphi_vs_ptavg_2d_dijet.png")


#############################################################################
#### 4. orthogonality closure on the joined event tables
#############################################################################
def cmd_orthogonality(args):
    verdicts = []
    for src in ("mc", "data"):
        print(f"[orthogonality] {src} (new)")
        dk, db, dw = load_events("new", "dijet", src)
        tk, tb, tw = load_events("new", "trijet", src)

        def view(a):
            return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()
        do, to = np.argsort(view(dk)), np.argsort(view(tk))
        _, di, ti = np.intersect1d(view(dk)[do], view(tk)[to],
                                   return_indices=True)
        di, ti = do[di], to[ti]
        pd = (db[di] & bit(DIJET_CUTS, "recoTot_seq")) != 0
        pt = (tb[ti] & bit(TRIJET_CUTS, "recoTot_seq")) != 0
        both = int((pd & pt).sum())
        v = (f"{src}: joined {len(di)} events, dijet-selected {int(pd.sum())}, "
             f"trijet-selected {int(pt.sum())}, BOTH = {both} "
             f"({'PASS -- exactly orthogonal' if both == 0 else 'FAIL'})")
        print("  " + v)
        verdicts.append(v)
    print("\n=== orthogonality closure (new production) ===")
    for v in verdicts:
        print("  " + v)
    return verdicts


#############################################################################
#### 5. per-jet kinematics data/MC from the new validation hists
#############################################################################
KIN_VARS = (
    ("pt", r"$p_{T}$ [GeV]", 5, (150., 1500.), True),
    ("eta", r"$\eta$", 1, (-2.6, 2.6), False),
    ("phi", r"$\phi$", 1, (-3.3, 3.3), False),
    ("mass", r"$m_{\mathrm{ungroomed}}$ [GeV]", 2, (0., 300.), True),
    ("msoftdrop", r"$m_{\mathrm{SD}}$ [GeV]", 2, (0., 300.), True),
)


def _rebin(v, e, k):
    if k == 1:
        return v, e
    n = (len(v) // k) * k
    return v[:n].reshape(-1, k).sum(1), e[:n + 1:k]


def rebin_to(v, e, edges):
    """Sum a fine-binned 1D hist into arbitrary (coarser) edges by bin center."""
    ctr = 0.5 * (e[1:] + e[:-1])
    idx = np.digitize(ctr, edges) - 1
    out = np.zeros(len(edges) - 1)
    ok = (idx >= 0) & (idx < len(out))
    np.add.at(out, idx[ok], v[ok])
    return out


def crosscheck_nominal(chan, jets, mincache):
    """minimal_rho (all_syst) vs rho_skim: SAME selection, so the nominal
    data/MC ratios must agree. Anything beyond the per-cent level means the two
    productions are not measuring the same thing and the bands are not
    transferable to the skim-based figures."""
    edges = PT_EDGES[chan]
    print(f"\n  --- nominal cross-check, minimal_rho vs rho_skim ({chan}) ---")
    print(f"  {'pT bin':>14s} {'skim d/MC':>10s} {'minimal d/MC':>13s} "
          f"{'diff':>8s}")
    worst = 0.0
    for j in jets:
        sm, _, e, _ = hist1d_sum("new", chan, "mc", f"pt_{j}", quiet=True)
        sd, _, _, _ = hist1d_sum("new", chan, "data", f"pt_{j}", quiet=True)
        em = mincache[("pt", j, "mc")][2]
        mmv = mincache[("pt", j, "mc")][0]
        mdv = mincache[("pt", j, "data")][0]
        a = rebin_to(sd, e, edges) / np.maximum(rebin_to(sm, e, edges), 1e-30)
        b = rebin_to(mdv, em, edges) / np.maximum(rebin_to(mmv, em, edges), 1e-30)
        for i in range(len(edges) - 1):
            if a[i] <= 0 or b[i] <= 0:
                continue
            d = b[i] / a[i] - 1.0
            worst = max(worst, abs(d))
            print(f"  {edges[i]:>5.0f}-{edges[i+1]:<8.0f} {a[i]:>10.4f} "
                  f"{b[i]:>13.4f} {100*d:>7.2f}%")
    print(f"  worst |difference| = {100*worst:.2f}%  "
          f"({'PASS' if worst < 0.02 else 'CHECK'})")
    return worst


def cmd_kinematics(args):
    outdir = Path(args.outdir)
    syst = bool(getattr(args, "syst", False))
    sfx = "_syst" if syst else ""
    for chan, jets in (("dijet", ("jet1", "jet2")), ("trijet", ("jet3",))):
        prod = "min" if syst else "new"
        cache, varcache = {}, {}
        for var, _, _, _, _ in KIN_VARS:
            for j in jets:
                for src in ("mc", "data"):
                    print(f"[kinematics] {chan} {src} {var}_{j} ({prod})")
                    if syst and src == "mc":
                        _, per = hist_variations(prod, chan, src, f"{var}_{j}",
                                                 quiet=True)
                        varcache[(var, j)] = per
                    cache[(var, j, src)] = hist1d_sum(prod, chan, src,
                                                      f"{var}_{j}", quiet=True)
        #### absolute MC-vs-data normalization, from the pT hist (all jets)
        facs = []
        for j in jets:
            m = cache[("pt", j, "mc")][0].sum()
            d = cache[("pt", j, "data")][0].sum()
            facs.append(d / m)
            print(f"  {chan} {j}: data/MC absolute yield = {d/m:.4f} "
                  f"(data {d:.5g}, MC {m:.5g})")
        fac = float(np.mean(facs))
        if syst:
            crosscheck_nominal(chan, jets, cache)

        for norm in ("shape", "abs"):
            for var, xlab, k, xlim, logy in KIN_VARS:
                fig, (ax, ar) = ratio_single()
                #### pass 1: build the series and find the common scale, so the
                #### sci-notation offset never lands on top of the CMS label
                ser, ymax = [], 0.0
                for j, col, mark in zip(jets, (C_BLUE, C_ORANGE), ("o", "s")):
                    vm, em, e, _ = cache[(var, j, "mc")]
                    vd, ed, _, _ = cache[(var, j, "data")]
                    vm, e2 = _rebin(vm, e, k)
                    em, _ = _rebin(em, e, k)
                    vd, _ = _rebin(vd, e, k)
                    ed, _ = _rebin(ed, e, k)
                    sm, sd = (vm.sum(), vd.sum()) if norm == "shape" else (1., 1.)
                    #### systematic band, computed on the SAME normalization the
                    #### panel shows: shape panels get a shape-only band (each
                    #### variation re-normalized to unit area), the absolute
                    #### panel keeps the yield component in.
                    rsyst = np.zeros_like(vm)
                    if syst:
                        per = {s: (_rebin(v, e, k)[0], va)
                               for s, (v, va) in varcache[(var, j)].items()}
                        tr = area_norm() if norm == "shape" else None
                        nomv, tot, contrib = syst_total(per, transform=tr)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rsyst = np.where(nomv > 0, tot / nomv, 0.0)
                        if var in ("pt", "msoftdrop"):
                            report_sources(f"{chan} {j} {var} [{norm}]", nomv,
                                           contrib, nomv > 0)
                    ser.append((j, col, mark, e2, vm / sm, em / sm ** 2,
                                vd / sd, ed / sd ** 2, rsyst))
                    ymax = max(ymax, np.nanmax(vm / sm), np.nanmax(vd / sd))
                ylab = ("Fraction of jets / bin" if norm == "shape"
                        else "Weighted jets / bin")
                #### fold the power of ten into the LABEL on linear axes
                pw = 0
                if not logy and ymax > 0:
                    pw = 3 * int(np.floor(np.log10(ymax) / 3))
                    if pw:
                        ylab += rf"  ($\times 10^{{{pw}}}$)"
                s = 10.0 ** (-pw)
                rband, rsband, rvals, rerrs, rgood, rmark, wsum = (
                    None, None, [], [], [], [], None)
                for j, col, mark, e2, vm, w2m, vd, w2d, rsyst in ser:
                    ym, eym = vm * s, np.sqrt(w2m) * s
                    yd, eyd = vd * s, np.sqrt(w2d) * s
                    lab = j.replace("jet", "jet ")
                    step(ax, e2, ym, color=col, lw=2.4, label=f"MC, {lab}")
                    sel = np.isfinite(yd) & (yd > 0)
                    ax.errorbar(0.5 * (e2[1:] + e2[:-1])[sel], yd[sel],
                                yerr=eyd[sel], fmt=mark, ms=5,
                                mfc="k" if mark == "o" else "none",
                                color="k", lw=1.3, label=f"Data, {lab}")
                    #### ratio panel: one series per jet, same marker convention
                    r, er, rel = data_mc_ratio(vd, w2d, vm, w2m)
                    #### the hists carry no raw counts; sqrt(sumw2) with the
                    #### relative MC error as the proxy for the <8-entry mask
                    good = (np.isfinite(r) & (vm > 0) & (vd > 0) & (rel < 0.4)
                            & (rsyst < MAX_REL_SYST))
                    rvals.append(r)
                    rerrs.append(er)
                    rgood.append(good)
                    rmark.append(mark)
                    band = np.where(good, rel, 0.0)
                    rband = band if rband is None else np.maximum(rband, band)
                    sband = np.where(good, rsyst, 0.0)
                    rsband = sband if rsband is None else np.maximum(rsband, sband)
                    ww = np.where(good, vm, 0.0)
                    wsum = ww if wsum is None else wsum + ww
                    redges = e2
                ymax *= s
                ax.set_ylabel(ylab)
                ax.set_xlim(*xlim)
                if logy:
                    ax.set_yscale("log")
                    headroom(ax, ymax, frac=25.0)
                else:
                    ax.set_ylim(0, ymax * 1.55)
                ax.grid(alpha=0.25)
                cms_label(ax, True)
                if syst:
                    #### quote the band YIELD-WEIGHTED (a plain median over bins
                    #### is dominated by the near-empty acceptance-edge bins)
                    def _bw(sel):
                        w = wsum * sel
                        return 100 * (rsband * w).sum() / w.sum() if w.sum() else np.nan
                    all1 = np.ones_like(wsum)
                    msg = f"  {chan} {var} [{norm}]: syst band {_bw(all1):.2f}%"
                    if var == "pt":
                        c2 = 0.5 * (redges[1:] + redges[:-1])
                        msg += (f"  |  200-290 GeV {_bw((c2 >= 200) & (c2 < 290)):.2f}%"
                                f"  |  >820 GeV {_bw(c2 >= 820):.2f}%")
                    print(msg)
                proxies, plabs = draw_ratio(
                    ar, redges, list(zip(rvals, rerrs, rgood, rmark)),
                    rband, rsband if syst else None, xlim=xlim)
                ar.set_xlabel(xlab)
                h, la = ax.get_legend_handles_labels()
                ax.legend(h + proxies, la + plabs, loc="upper right",
                          fontsize=15 if syst else 16, ncol=1)
                #### standalone figures must self-identify: channel + norm on
                #### every panel, not only the pT one as in the old grid sheet
                note = (f"{chan} channel, pair_split\n"
                        + ("unit-area normalized" if norm == "shape"
                           else f"absolute; data/MC = {fac:.2f}"))
                panel_text(ax, note, fontsize=18)
                save(fig, outdir, f"kinematics_{chan}_{norm}_{var}{sfx}.png")


#############################################################################
#### 6. m_SD and rho data/MC per measured-jet pT bin
#############################################################################
def cmd_msdrho(args):
    if getattr(args, "syst", False):
        return cmd_msdrho_syst(args)
    outdir = Path(args.outdir)
    specs = (("msd", "mreco_g", np.linspace(0., 250., 51),
              r"$m_{\mathrm{SD}}$ [GeV]"),
             ("rho", None, np.linspace(-6., 0., 49), r"$\log(\rho^2)$"))
    for chan in ("dijet", "trijet"):
        edges = PT_EDGES[chan]
        nb = len(edges) - 1
        acc = {}
        for key, _, bins, _ in specs:
            for src in ("mc", "data"):
                acc[(key, src)] = (np.zeros((nb, len(bins) - 1)),
                                   np.zeros((nb, len(bins) - 1)),
                                   np.zeros((nb, len(bins) - 1)))
        for src in ("mc", "data"):
            print(f"[msdrho] {chan} {src} (new)")
            for _, t in stream_table("new", chan, src, "reco",
                                     ("ptreco", "mreco_g", "weight")):
                pt, msd, w = t["ptreco"], t["mreco_g"], t["weight"]
                ib = np.digitize(pt, edges) - 1
                rr = rho(msd, pt)
                for key, col, bins, _ in specs:
                    x = msd if col else rr
                    sw, sw2, raw = acc[(key, src)]
                    ok = np.isfinite(x)
                    for i in range(nb):
                        m = ok & (ib == i)
                        if not m.any():
                            continue
                        a, _ = np.histogram(x[m], bins=bins, weights=w[m])
                        b, _ = np.histogram(x[m], bins=bins, weights=w[m] ** 2)
                        c, _ = np.histogram(x[m], bins=bins)
                        sw[i] += a
                        sw2[i] += b
                        raw[i] += c

        for key, _, bins, xlab in specs:
            for i in range(nb):
                swm, sw2m, rawm = (x[i] for x in acc[(key, "mc")])
                swd, sw2d, rawd = (x[i] for x in acc[(key, "data")])
                nm, nd = swm.sum(), swd.sum()
                if nm <= 0 or nd <= 0:
                    continue
                fig, (a0, a1) = ratio_single()
                #### area-normalize MC to the data yield IN THIS pT BIN: the
                #### absolute QCD normalization is off by a flat ~20% (see the
                #### kinematics subcommand), which would swamp the shape check
                sc = nd / nm
                ym, eym = swm * sc, np.sqrt(sw2m) * sc
                yd, eyd = swd, np.sqrt(sw2d)
                step(a0, bins, ym, color=C_BLUE, lw=2.4,
                     label=f"QCD MG+Pythia8 ($\\times${sc:.2f})")
                good = (rawm >= MIN_RAW) & (rawd >= MIN_RAW)
                ctr = 0.5 * (bins[1:] + bins[:-1])
                a0.errorbar(ctr[good], yd[good], yerr=eyd[good], fmt="o", ms=5,
                            color="k", lw=1.3, label=DATA_LABEL)
                a0.set_yscale("log")
                pos = ym[ym > 0]
                a0.set_ylim(max(pos.min(), pos.max() * 1e-4), pos.max() * 25)
                a0.set_ylabel("Weighted jets / bin")
                a0.grid(alpha=0.25)
                cms_label(a0, True)
                hi = edges[i + 1]
                hi_s = "$\\infty$" if hi >= 1500 else f"{hi:.0f}"
                panel_text(a0, f"{chan}, ${edges[i]:.0f} < p_T^{{jet}} <$ "
                               f"{hi_s} GeV", fontsize=19)
                r, er, rel = data_mc_ratio(yd, sw2d, ym, sw2m * sc ** 2)
                px, pl = draw_ratio(a1, bins, [(r, er, good, "o")], rel,
                                    xlim=(bins[0], bins[-1]))
                a0.legend(a0.get_legend_handles_labels()[0] + px,
                          a0.get_legend_handles_labels()[1] + pl,
                          loc="upper right", fontsize=17)
                a1.set_xlabel(xlab)
                save(fig, outdir, f"{key}_datamc_{chan}_"
                                  f"{ptbin_tag(edges[i], edges[i + 1])}.png")


def _pt_slices(ptaxis, edges):
    """For each (lo, hi) in `edges`, the hist ptreco bin indices inside it.

    The all_syst hists use the analysis pt axis
    [185, 200, 290, 400, 480, 570, 680, 760, 820, 13000]; the dijet channel
    starts at 200 (its 185-200 bin is empty by construction) and the trijet
    channel merges everything above 400 into one shown bin.
    """
    ctr = 0.5 * (ptaxis[1:] + ptaxis[:-1])
    out = []
    for i in range(len(edges) - 1):
        hi = np.inf if edges[i + 1] >= 1500 else edges[i + 1]
        out.append(np.where((ctr >= edges[i]) & (ctr < hi))[0])
    return out


def _rho_syst_panels(outdir, chan, edges, histname, ftag, rho_xlab,
                     rho_min=None):
    """One data/MC + stat(+)syst figure per measured-jet pT bin, from a 2D
    (ptjet, rhojet) all_syst hist. `ftag` prefixes the output name
    (`rho` = groomed, `rho_u` = ungroomed); the x-window follows the hist's
    own rho axis so the wider ungroomed axis is drawn in full.

    `rho_min` restricts BOTH the drawn window and the normalization: the
    MC-to-data scale and every systematic variation's shape normalization are
    computed from the bins at rho >= rho_min only (Aritra, 2026-08-08 — so
    the unmeasured soft tail cannot set the scale). Output names then carry a
    `_winXpY` suffix."""
    print(f"[msdrho --syst] {chan} {ftag} ({histname}, minimal_rho all_syst)")
    axm, perm = hist_variations("min", chan, "mc", histname)
    _, perd = hist_variations("min", chan, "data", histname)
    ptax, rhoax = axm[0].edges, axm[1].edges
    sl = _pt_slices(ptax, edges)
    for i in range(len(edges) - 1):
        idx = sl[i]
        if len(idx) == 0:
            continue
        #### project this pT slice for every systematic
        perp = {s: (v[idx].sum(0), va[idx].sum(0)) for s, (v, va) in perm.items()}
        swd = perd["nominal"][0][idx].sum(0)
        sw2d = perd["nominal"][1][idx].sum(0)
        swm, sw2m = perp["nominal"]
        nm, nd = swm.sum(), swd.sum()
        if nm <= 0 or nd <= 0:
            continue
        #### normalization window: all bins, or rho >= rho_min when set
        ctr_all = 0.5 * (rhoax[1:] + rhoax[:-1])
        win = (np.ones_like(ctr_all, bool) if rho_min is None
               else ctr_all >= rho_min)
        nm_w, nd_w = swm[win].sum(), swd[win].sum()
        if nm_w <= 0 or nd_w <= 0:
            continue
        fig, (a0, a1) = ratio_single()
        sc = nd_w / nm_w

        def wnorm(v, _win=win, _target=nd_w):
            s = v[_win].sum()
            return v * (_target / s) if s else v

        #### shape-only band: every variation is renormalized to the SAME
        #### data yield the nominal is scaled to, in the SAME window
        nomv, tot, contrib = syst_total(perp, transform=wnorm)
        with np.errstate(divide="ignore", invalid="ignore"):
            rsyst = np.where(nomv > 0, tot / nomv, 0.0)
        #### densities: divide by bin width so the y-axis reads per unit
        #### log(rho^2) rather than per (arbitrary) bin
        bw = np.diff(rhoax)
        ym = swm * sc / bw
        yd, eyd = swd / bw, np.sqrt(sw2d) / bw
        good = ((swm > 0) & (swd > 0)
                & (np.sqrt(sw2m) / np.maximum(swm, 1e-30) < 0.4)
                & (rsyst < MAX_REL_SYST))
        ctr = 0.5 * (rhoax[1:] + rhoax[:-1])
        #### the drawn window: populated bins (+1 bin of margin) -- the
        #### ungroomed rho axis reaches to -10 but fills only above ~-4.
        #### With rho_min set, the window starts exactly at the cut.
        filled = np.nonzero((ym > 0) | (yd > 0))[0]
        xlo = (float(rho_min) if rho_min is not None
               else float(rhoax[max(0, filled[0] - 1)]))
        xhi = float(rhoax[min(len(rhoax) - 1, filled[-1] + 2)])
        vis = (ctr >= xlo) & (ctr <= xhi)
        ymax = max(float(np.nanmax(np.where(vis, ym, 0.0))),
                   float(np.nanmax(np.where(vis & good, yd, 0.0))))
        #### linear y (Aritra, 2026-08-08): inside the measured window the
        #### density spans about a decade, and the shape reads better without
        #### the log compression. The power of ten is folded into the LABEL
        #### so no offset text crowds the CMS header (same as kinematics).
        ylab = r"Weighted jets / unit $\log(\rho^2)$"
        pw = 3 * int(np.floor(np.log10(ymax) / 3)) if ymax > 0 else 0
        if pw:
            ylab += rf"  ($\times 10^{{{pw}}}$)"
        s = 10.0 ** (-pw)
        step(a0, rhoax, ym * s, color=C_BLUE, lw=2.4,
             label=f"QCD MG+Pythia8 ($\\times${sc:.2f})")
        a0.errorbar(ctr[good], yd[good] * s, yerr=eyd[good] * s, fmt="o",
                    ms=5, color="k", lw=1.3, label=DATA_LABEL)
        a0.set_xlim(xlo, xhi)
        a0.set_ylim(0.0, ymax * s * 1.55)
        a0.set_ylabel(ylab)
        a0.grid(alpha=0.25)
        cms_label(a0, True)
        hi = edges[i + 1]
        hi_s = "$\\infty$" if hi >= 1500 else f"{hi:.0f}"
        note = f"{chan}, ${edges[i]:.0f} < p_T^{{jet}} <$ {hi_s} GeV"
        if rho_min is not None:
            note += f"\nnormalized in $\\log(\\rho^2) > {rho_min:g}$"
        panel_text(a0, note, fontsize=19)
        #### sumw2 in density units too (/bw^2) so the ratio errors stay right
        r, er, rel = data_mc_ratio(yd, sw2d / bw ** 2, ym,
                                   sw2m * sc ** 2 / bw ** 2)
        px, pl = draw_ratio(a1, rhoax, [(r, er, good, "o")], rel, rsyst,
                            xlim=(xlo, xhi))
        h, la = a0.get_legend_handles_labels()
        a0.legend(h + px, la + pl, loc="upper right", fontsize=16)
        a1.set_xlabel(rho_xlab)
        m = good & (rsyst > 0)
        if m.any():
            print(f"  {chan} {ftag}, pT {edges[i]:.0f}-{hi_s.strip('$\\infty')or 'inf'}"
                  f": syst band median {100*np.median(rsyst[m]):.2f}%, "
                  f"max {100*rsyst[m].max():.2f}%")
            report_sources(f"{chan} {ftag} pT {edges[i]:.0f}+", nomv, contrib, m)
        wtag = ("" if rho_min is None
                else f"_win{abs(rho_min):g}".replace(".", "p"))
        save(fig, outdir, f"{ftag}_datamc_{chan}_syst_"
                          f"{ptbin_tag(edges[i], edges[i + 1])}{wtag}.png")


def cmd_msdrho_syst(args):
    """The all_syst re-render: rho per measured-jet pT bin from
    ptjet_rhojet_g_reco AND ptjet_rhojet_u_reco (groomed `rho_*` and
    ungroomed `rho_u_*` figures), plus the pT-INCLUSIVE m_SD from the
    per-jet hists.

    minimal_rho carries no pT-differential m_SD hist (there is no
    ptjet_mjet_g_reco in that mode), so the m_SD figure here is pT-inclusive
    and says so on the panel. The pT-differential m_SD data/MC remains the
    skim-table figure `msd_datamc_<chan>.png`, which has no systematics.
    """
    outdir = Path(args.outdir)
    #### groomed and ungroomed rho share the drawing; only the source 2D hist,
    #### the output-name prefix, and the axis label differ
    RHO_SPECS = (
        ("ptjet_rhojet_g_reco", "rho", r"$\log(\rho^2)$, groomed"),
        ("ptjet_rhojet_u_reco", "rho_u", r"$\log(\rho^2)$, ungroomed"),
    )
    for chan, jets in (("dijet", ("jet1", "jet2")), ("trijet", ("jet3",))):
        edges = PT_EDGES[chan]
        #### ---------- rho, per pT bin, from the 2D hists ----------
        for histname, ftag, rho_xlab in RHO_SPECS:
            _rho_syst_panels(outdir, chan, edges, histname, ftag, rho_xlab)

        #### ---------- m_SD, pT-inclusive, from the per-jet hists ----------
        print(f"[msdrho --syst] {chan} m_SD (pT-inclusive)")
        perm_j, perd_j, e_j = {}, None, None
        for j in jets:
            ax_, p_ = hist_variations("min", chan, "mc", f"msoftdrop_{j}",
                                      quiet=True)
            e_j = ax_[0].edges
            for s, (v, va) in p_.items():
                if s not in perm_j:
                    perm_j[s] = [np.zeros_like(v), np.zeros_like(v)]
                perm_j[s][0] += v
                perm_j[s][1] += va
            _, pd_ = hist_variations("min", chan, "data", f"msoftdrop_{j}",
                                     quiet=True)
            if perd_j is None:
                perd_j = [np.zeros_like(pd_["nominal"][0]),
                          np.zeros_like(pd_["nominal"][1])]
            perd_j[0] += pd_["nominal"][0]
            perd_j[1] += pd_["nominal"][1]
        k = 2
        perm_j = {s: (_rebin(v, e_j, k)[0], _rebin(va, e_j, k)[0])
                  for s, (v, va) in perm_j.items()}
        swd, sw2d = _rebin(perd_j[0], e_j, k)[0], _rebin(perd_j[1], e_j, k)[0]
        eb = _rebin(perd_j[0], e_j, k)[1]
        swm, sw2m = perm_j["nominal"]
        sc = swd.sum() / swm.sum()
        nomv, tot, contrib = syst_total(perm_j, transform=area_norm(swd.sum()))
        with np.errstate(divide="ignore", invalid="ignore"):
            rsyst = np.where(nomv > 0, tot / nomv, 0.0)
        fig, cells = ratio_grid(1, ncols=1)
        a0, a1 = cells[0]
        ym, eym = swm * sc, np.sqrt(sw2m) * sc
        yd, eyd = swd, np.sqrt(sw2d)
        good = ((swm > 0) & (swd > 0)
                & (np.sqrt(sw2m) / np.maximum(swm, 1e-30) < 0.4)
                & (rsyst < MAX_REL_SYST))
        step(a0, eb, ym, color=C_BLUE, lw=2.4,
             label=f"QCD MG+Pythia8 ($\\times${sc:.2f})")
        ctr = 0.5 * (eb[1:] + eb[:-1])
        a0.errorbar(ctr[good], yd[good], yerr=eyd[good], fmt="o", ms=5,
                    color="k", lw=1.3, label=DATA_LABEL)
        a0.set_yscale("log")
        pos = ym[ym > 0]
        a0.set_ylim(max(pos.min(), pos.max() * 1e-5), pos.max() * 25)
        a0.set_ylabel("Weighted jets / bin")
        a0.grid(alpha=0.25)
        cms_label(a0, True)
        meas = "jets 1+2" if chan == "dijet" else "jet 3"
        panel_text(a0, f"{chan} channel, {meas}\n"
                       r"$p_T$-inclusive (minimal_rho has no" "\n"
                       r"$p_T$-differential $m_{\mathrm{SD}}$ hist)",
                   fontsize=18)
        r, er, rel = data_mc_ratio(yd, sw2d, ym, sw2m * sc ** 2)
        px, pl = draw_ratio(a1, eb, [(r, er, good, "o")], rel, rsyst,
                            xlim=(0., 300.))
        h, la = a0.get_legend_handles_labels()
        a0.legend(h + px, la + pl, loc="upper right", fontsize=17)
        a0.set_xlim(0., 300.)
        a1.set_xlabel(r"$m_{\mathrm{SD}}$ [GeV]")
        #### quote only the DRAWN window (0-300 GeV); the far tail is off-panel
        m = good & (rsyst > 0) & (ctr <= 300.)
        if m.any():
            print(f"  {chan} m_SD: syst band median "
                  f"{100*np.median(rsyst[m]):.2f}%, max {100*rsyst[m].max():.2f}%")
            report_sources(f"{chan} m_SD", nomv, contrib, m)
        save(fig, outdir, f"msd_datamc_{chan}_syst.png")


#############################################################################
#### 7. |y| edge: 2.5 (legacy) -> 2.4 (pair_split)
#############################################################################
def cmd_etaedge(args):
    outdir = Path(args.outdir)
    bins = np.linspace(0., 2.8, 141)
    series = []
    for prod, ls in (("new", "-"), ("old", "--")):
        for table, cols, col, tag in (
                ("reco", ("ptreco", "etareco", "mreco_u"), None, "reco"),
                ("gen", ("ptgen", "etagen", "mgen_u"), None, "gen")):
            h = np.zeros(len(bins) - 1)
            print(f"[etaedge] dijet mc {prod} {table}")
            for _, t in stream_table(prod, "dijet", "mc", table,
                                     cols + ("weight",)):
                y = np.abs(rapidity(t[cols[0]], t[cols[1]], t[cols[2]]))
                a, _ = np.histogram(y, bins=bins, weights=t["weight"])
                h += a
            series.append((prod, tag, ls, h))

    plt, _ = _mpl()
    fig, ax = plt.subplots(layout="constrained")
    cols = {("new", "reco"): C_BLUE, ("new", "gen"): C_RED,
            ("old", "reco"): C_PURPLE, ("old", "gen"): C_ORANGE}
    for prod, tag, ls, h in series:
        y = h / h.sum()
        lab = f"{'pair_split' if prod == 'new' else 'legacy'}, {tag}"
        step(ax, bins, y, color=cols[(prod, tag)], lw=2.6, ls=ls, label=lab)
        #### where the distribution actually stops
        last = bins[1:][y > 0][-1] if np.any(y > 0) else np.nan
        print(f"  {lab:<20s} highest populated |y| edge = {last:.3f}")
    for x, c, lab in ((2.4, C_BLUE, r"pair_split $|y| < 2.4$"),
                      (2.5, C_PURPLE, r"legacy $|y| < 2.5$")):
        ax.axvline(x, color=c, lw=1.8, ls=":")
        ax.text(x - 0.012, 0.30, lab, color=c, rotation=90, va="center",
                ha="right", fontsize=17,
                transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"$|y|$ of the measured dijet jets")
    ax.set_ylabel("Fraction of jets / 0.02")
    ax.set_xlim(2.0, 2.6)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=18)
    panel_text(ax, "dijet, QCD MG+Pythia8", fontsize=19)
    cms_label(ax, False)
    #### keep the visible window's max off the legend
    vis = [h / h.sum() for _, _, _, h in series]
    m = max(np.max(v[(bins[:-1] >= 2.0) & (bins[:-1] < 2.6)]) for v in vis)
    ax.set_ylim(1e-6, m * 12)
    save(fig, outdir, "eta_edge_dijet.png")


#############################################################################
#### 8. trijet jet-3 rho shape, new vs old selection
#############################################################################
def cmd_sculpting(args):
    outdir = Path(args.outdir)
    bins = np.linspace(-6., 0., 49)
    acc = {}
    for prod in ("new", "old"):
        sw = np.zeros((len(SCULPT_PT_BINS), len(bins) - 1))
        sw2 = np.zeros_like(sw)
        raw = np.zeros_like(sw)
        print(f"[sculpting] trijet mc {prod}")
        for _, t in stream_table(prod, "trijet", "mc", "reco",
                                 ("ptreco", "mreco_g", "weight")):
            r = rho(t["mreco_g"], t["ptreco"])
            ok = np.isfinite(r)
            for i, (lo, hi) in enumerate(SCULPT_PT_BINS):
                m = ok & (t["ptreco"] >= lo) & (t["ptreco"] < hi)
                a, _ = np.histogram(r[m], bins=bins, weights=t["weight"][m])
                b, _ = np.histogram(r[m], bins=bins, weights=t["weight"][m] ** 2)
                c, _ = np.histogram(r[m], bins=bins)
                sw[i] += a
                sw2[i] += b
                raw[i] += c
        acc[prod] = (sw, sw2, raw)

    for i, (lo, hi) in enumerate(SCULPT_PT_BINS):
        fig, (a0, a1) = ratio_single()
        ys = {}
        for prod, col, ls in (("new", C_BLUE, "-"), ("old", C_PURPLE, "--")):
            sw, sw2, raw = (x[i] for x in acc[prod])
            n = sw.sum()
            y, e = sw / n, np.sqrt(sw2) / n
            ys[prod] = (y, e, raw)
            lab = "pair_split" if prod == "new" else "legacy"
            step(a0, bins, y, color=col, lw=2.5, ls=ls, label=f"{lab} selection")
            a0.stairs(y + e, bins, baseline=y - e, fill=True, alpha=0.22,
                      color=col, lw=0)
        a0.set_yscale("log")
        mx = max(np.nanmax(ys[p][0]) for p in ys)
        a0.set_ylim(1e-5, mx * 20)
        a0.set_ylabel("Fraction of jets / bin")
        a0.legend(loc="upper left", fontsize=18)
        a0.grid(alpha=0.25)
        cms_label(a0, False)
        hi_s = "$\\infty$" if hi >= 1500 else f"{hi:.0f}"
        panel_text(a0, f"trijet jet 3, ${lo:.0f} < p_{{T,3}} <$ {hi_s} GeV\n"
                       "QCD MG+Pythia8", fontsize=18, y=0.80)
        yn, en, rn = ys["new"]
        yo, eo, ro = ys["old"]
        good = (rn >= MIN_RAW) & (ro >= MIN_RAW)
        with np.errstate(divide="ignore", invalid="ignore"):
            r, er = ratio_err(yn, en, yo, eo)
        ctr = 0.5 * (bins[1:] + bins[:-1])
        #### the "band" here is the OLD production's stat error, the reference
        #### the new/old shape ratio is measured against
        with np.errstate(divide="ignore", invalid="ignore"):
            relo = np.where(yo > 0, eo / yo, 0.0)
        draw_ratio(a1, bins, [(r, er, good, "o")], relo, ylabel="new / old",
                   xlim=(bins[0], bins[-1]))
        a1.set_xlabel(r"$\log(\rho^2)$, groomed")
        rr = r[good]
        if rr.size:
            print(f"  pT3 {lo:.0f}-{hi:.0f}: shape ratio mean {np.mean(rr):.3f}, "
                  f"rms {np.std(rr):.3f}, range {rr.min():.3f}-{rr.max():.3f}")
        save(fig, outdir,
             f"trijet_rho_shape_new_vs_old_{ptbin_tag(lo, hi)}.png")


#############################################################################
#### 9. fakes and misses vs pT, new vs old
#############################################################################
def cmd_fakesmisses(args):
    outdir = Path(args.outdir)
    edges = PT_DIJET
    out = {}
    for prod in ("new", "old"):
        tot = {}
        for table, ptcol in (("reco", "ptreco"), ("gen", "ptgen")):
            h = np.zeros(len(edges) - 1)
            print(f"[fakesmisses] dijet mc {prod} {table}")
            for _, t in stream_table(prod, "dijet", "mc", table,
                                     (ptcol, "weight")):
                a, _ = np.histogram(t[ptcol], bins=edges, weights=t["weight"])
                h += a
            tot[table] = h
        for ptcol, tag in (("ptreco", "matched_reco"), ("ptgen", "matched_gen")):
            h = np.zeros(len(edges) - 1)
            print(f"[fakesmisses] dijet mc {prod} matched ({ptcol})")
            for _, t in stream_table(prod, "dijet", "mc", "matched",
                                     (ptcol, "weight"), quiet=True):
                a, _ = np.histogram(t[ptcol], bins=edges, weights=t["weight"])
                h += a
            tot[tag] = h
        with np.errstate(divide="ignore", invalid="ignore"):
            out[prod] = {
                "fakes": 1.0 - tot["matched_reco"] / tot["reco"],
                "misses": 1.0 - tot["matched_gen"] / tot["gen"],
            }
        print(f"  {prod}: fakes  " +
              " ".join(f"{v:.3f}" for v in out[prod]["fakes"]))
        print(f"  {prod}: misses " +
              " ".join(f"{v:.3f}" for v in out[prod]["misses"]))

    plt, _ = _mpl()
    fig, axes = grid(1, 2)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    xerr = 0.5 * np.diff(edges)
    for ax, key, ylab, xlab in (
            (axes[0], "fakes", "Fake fraction (reco not matched)",
             r"$p_{T}^{\mathrm{reco}}$ [GeV]"),
            (axes[1], "misses", "Miss fraction (gen not matched)",
             r"$p_{T}^{\mathrm{gen}}$ [GeV]")):
        for prod, col, mark in (("new", C_BLUE, "o"), ("old", C_PURPLE, "s")):
            lab = "pair_split" if prod == "new" else "legacy"
            ax.errorbar(ctr, out[prod][key], xerr=xerr, fmt=mark, ms=9, lw=2,
                        color=col, label=f"{lab} selection")
        ax.set_xscale("log")
        ax.set_xticks([200, 300, 400, 600, 800, 1000, 1500])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(edges[0] * 0.95, edges[-1])
        ax.set_ylim(0, 0.55)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right", fontsize=19)
        ax.grid(alpha=0.25)
        cms_label(ax, False)
        panel_text(ax, "dijet, QCD MG+Pythia8", fontsize=19)
    save(fig, outdir, "fakes_misses_dijet.png")
    return out


#############################################################################
#### 10. systematic breakdown for one representative distribution
#############################################################################
#### weight-systematic families: the two Q2 scale variations are ONE family
WEIGHT_FAMILIES = {
    "Q2 scales": ("Q2muR", "Q2muF"),
    "PDF": ("pdf",), "ISR": ("isr",), "FSR": ("fsr",),
    "Pileup": ("pu",), "L1 prefiring": ("l1prefiring",),
    "Luminosity": ("Luminosity",),
}
N_JES_SHOWN = 4          # largest individual JES sources drawn separately


def cmd_systbreakdown(args):
    """Per-bin relative systematic vs eta for the dijet leading jet, one curve
    per uncertainty GROUP, so which source dominates is readable at a glance.

    Absolute (yield-inclusive) breakdown -- the same band the
    `kinematics_dijet_abs_syst` ratio panel draws. The shape-only counterpart
    removes every flat normalization term (Luminosity, and most of Q2/PDF) and
    is quoted in the run log for comparison.
    """
    outdir = Path(args.outdir)
    name, chan, jet = "eta_jet1", "dijet", "jet 1"
    print(f"[systbreakdown] {chan} {name} (minimal_rho all_syst)")
    axes, per = hist_variations("min", chan, "mc", name)
    edges = axes[0].edges
    nom, tot, contrib = syst_total(per)
    stat = np.sqrt(per["nominal"][1])
    #### only bins that are actually measured: |y| < 2.4 empties the edges
    good = (nom > 0) & (stat / np.maximum(nom, 1e-30) < 0.4)

    def rel(c):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(good, 100.0 * c / np.maximum(nom, 1e-30), np.nan)

    def quad(keys):
        return np.sqrt(sum(contrib[k] ** 2 for k in keys if k in contrib))

    jes_keys = [s for s in contrib if s.startswith("JES_")]
    integ = {k: float(contrib[k][good].sum() / nom[good].sum()) for k in contrib}
    top_jes = sorted(jes_keys, key=lambda k: -integ[k])[:N_JES_SHOWN]

    groups = [("JES total (%d sources)" % len(jes_keys), quad(jes_keys), "-", 3.0)]
    for k in top_jes:
        groups.append((k.replace("JES_", "JES "), contrib[k], "--", 2.0))
    for k in ("JER", "JMS", "JMR"):
        if k in contrib:
            groups.append((k, contrib[k], "-", 2.2))
    for lab, keys in WEIGHT_FAMILIES.items():
        c = quad(keys)
        if np.any(c > 0):
            groups.append((lab, c, "-", 2.2))
    #### legend reads top-down by integrated size
    groups.sort(key=lambda g: -float(g[1][good].sum() / nom[good].sum()))

    plt, _ = _mpl()
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(0.5 * (edges[1:] + edges[:-1]), rel(tot), color="k", lw=4.0,
            label="Total (quadrature)", zorder=5)
    cols = PETROFF_10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
                         "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd",
                         "#5790fc", "#f89c20", "#e42536", "#964a8b"]
    for i, (lab, c, ls, lw) in enumerate(groups):
        ax.plot(0.5 * (edges[1:] + edges[:-1]), rel(c), ls=ls, lw=lw,
                color=cols[i % len(cols)], label=lab)
    ax.plot(0.5 * (edges[1:] + edges[:-1]), rel(stat), ls=":", lw=2.2,
            color="0.35", label="MC stat. unc.")
    ax.set_xlabel(r"$\eta$ of the leading dijet jet")
    ax.set_ylabel("Relative uncertainty [%]")
    ax.set_xlim(-2.6, 2.6)
    top = float(np.nanmax(rel(tot)))
    ax.set_ylim(0, top * 3.0)
    ax.grid(alpha=0.25)
    #### legend on the RIGHT half of the headroom, descriptor on the left half:
    #### a top-centre legend runs straight through the in-frame annotation
    ax.legend(loc="upper right", fontsize=13, ncol=2, framealpha=0.93,
              columnspacing=1.0, handlelength=1.8)
    panel_text(ax, f"{chan} channel, {jet}\nabsolute (yield-inclusive)",
               fontsize=17)
    cms_label(ax, False)
    save(fig, outdir, "syst_breakdown_eta_jet1.png")

    #### ---- the numbers behind the figure
    print(f"\n  === integrated relative size, {name} (absolute) ===")
    print(f"  {'group':<28s} {'integrated':>11s} {'central |eta|<1':>16s} "
          f"{'forward |eta|>1.8':>18s}")
    c2 = 0.5 * (edges[1:] + edges[:-1])
    cen, fwd = good & (np.abs(c2) < 1.0), good & (np.abs(c2) > 1.8)

    def band(c, m):
        #### yield-weighted mean of c/nom == sum(c)/sum(nom) over the bins in m
        return 100 * float(c[m].sum() / nom[m].sum())
    rows = [("TOTAL", tot)] + [(l, c) for l, c, _, _ in groups] + \
           [("MC stat.", stat)]
    for lab, c in rows:
        print(f"  {lab:<28s} {band(c, good):>10.2f}% {band(c, cen):>15.2f}% "
              f"{band(c, fwd):>17.2f}%")
    #### shape-only comparison (flat normalization terms drop out)
    nom_s, tot_s, contrib_s = syst_total(per, transform=area_norm())
    gs = good
    print(f"\n  shape-only total = "
          f"{100*float(tot_s[gs].sum()/nom_s[gs].sum()):.2f}% integrated; "
          "dominant:")
    for v, s in sorted(((float(c[gs].sum() / nom_s[gs].sum()), s)
                        for s, c in contrib_s.items()), reverse=True)[:4]:
        print(f"      {s:<28s} {100*v:6.2f}%")


#############################################################################
CMDS = {
    "yields": cmd_yields,
    "boundary": cmd_boundary,
    "dphi2d": cmd_dphi2d,
    "orthogonality": cmd_orthogonality,
    "kinematics": cmd_kinematics,
    "msdrho": cmd_msdrho,
    "etaedge": cmd_etaedge,
    "sculpting": cmd_sculpting,
    "fakesmisses": cmd_fakesmisses,
    "systbreakdown": cmd_systbreakdown,
}


#### these compare against the LEGACY production, which exists for 2018 only
NEEDS_OLD = ("yields", "etaedge", "sculpting", "fakesmisses")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=list(CMDS) + ["all"])
    ap.add_argument("--outdir", default="pairsplit_validation")
    ap.add_argument("--era", default="2018", choices=sorted(LUMI_BY_ERA),
                    help="which era's pair_split production to read; the "
                         "new/old comparison subcommands "
                         f"({', '.join(NEEDS_OLD)}) are 2018-only")
    ap.add_argument("--syst", action="store_true",
                    help="kinematics/msdrho only: read the all_syst minimal_rho "
                         "production and add the stat (+) syst ratio band "
                         "(writes <name>_syst.png)")
    args = ap.parse_args(argv)
    set_era(args.era)
    todo = list(CMDS) if args.cmd == "all" else [args.cmd]
    if args.cmd == "all":
        #### systbreakdown needs the all_syst production; run it explicitly
        todo = [c for c in todo if c != "systbreakdown"]
    if args.era != "2018":
        dropped = [c for c in todo if c in NEEDS_OLD]
        todo = [c for c in todo if c not in NEEDS_OLD]
        if dropped and args.cmd != "all":
            raise SystemExit(f"{dropped[0]} needs the legacy production, "
                             "which exists for 2018 only")
        if dropped:
            print(f"[era {args.era}] skipping legacy-comparison subcommands: "
                  + ", ".join(dropped))
    if args.syst:
        todo = [c for c in todo if c in ("kinematics", "msdrho")]
        if not todo:
            raise SystemExit("--syst only applies to kinematics and msdrho")
    for c in todo:
        print(f"\n{'='*72}\n### {c}\n{'='*72}")
        CMDS[c](args)


if __name__ == "__main__":
    main()
