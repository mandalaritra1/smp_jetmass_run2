#!/usr/bin/env python
"""AK8 jet multiplicity in the hadronic channels, from the skim event tables.

The `alljets` / `allgenjets` tables are row-per-jet over the FULL FatJet /
GenJetAK8 collections (they are what the event viewer reads), so the per-event
multiplicity is exact -- no selection is folded in. This plots it for

    all skimmed events  |  dijet-selected  |  trijet-selected  |  overlap

at reco, plus the gen multiplicity with and without a pT floor. The floor
matters: GenJetAK8 is stored down to ~100 GeV while FatJet starts at ~170, so
the raw gen multiplicity is systematically higher than reco and any gen-level
jet-counting cut is looser than its reco counterpart unless floored (the same
issue that forced the pT3 > 185 floor on the gen trijet veto -- see
dijet_processor._in_trijet_phase_space).

    python scripts/plot_njet_hadronic.py --outdir DIR
    python scripts/plot_njet_hadronic.py --outdir DIR --data
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.channel_overlap import (  # noqa: E402
    DIJET_CUTS, TRIJET_CUTS, bit, dedupe)

#### reco pT floors to count above (None = every stored jet)
RECO_FLOORS = (None, 200.0, 300.0)
GEN_FLOORS = (None, 170.0, 200.0)


def counts_above(ek, table, n, floors):
    """Per-event jet counts above each pT floor, aligned to the events table."""
    jk = np.stack([np.asarray(table[c].value, dtype=np.int64) for c in
                   ("dataset_id", "run", "lumi", "event")], axis=1)
    jpt = np.asarray(table["pt"].value)

    def view(a):
        return np.ascontiguousarray(a).view([("", np.int64)] * 4).ravel()

    out = {}
    if len(jk) == 0:
        return {f: np.zeros(n, np.int32) for f in floors}
    order = np.argsort(view(ek), kind="stable")
    pos = np.clip(np.searchsorted(view(ek)[order], view(jk)), 0, n - 1)
    hit = view(ek)[order][pos] == view(jk)
    evrow = order[pos]
    for f in floors:
        c = np.zeros(n, np.int32)
        m = hit if f is None else (hit & (jpt > f))
        np.add.at(c, evrow[m], 1)
        out[f] = c
    return out


def load(paths, floors_reco, floors_gen):
    out = {"cutbits": [], "weight": [], "dsid": [], "key": []}
    out |= {f"reco{f}": [] for f in floors_reco}
    out |= {f"gen{f}": [] for f in floors_gen}
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
        out["key"].append(ek)
        out["cutbits"].append(np.asarray(ev["cutbits"].value))
        out["weight"].append(np.asarray(ev["weight"].value))
        out["dsid"].append(np.asarray(ev["dataset_id"].value))
        for f, c in counts_above(ek, sk["alljets"], n, floors_reco).items():
            out[f"reco{f}"].append(c)
        for f, c in counts_above(ek, sk["allgenjets"], n, floors_gen).items():
            out[f"gen{f}"].append(c)
        print(f"  {Path(p).name:<55s} {n:>9d} events")
    return {k: np.concatenate(v) for k, v in out.items()}, datasets


def njet_hist(n, w, nmax=8):
    edges = np.arange(-0.5, nmax + 1.5)
    h, _ = np.histogram(np.clip(n, 0, nmax), bins=edges, weights=w)
    return h, edges


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dijet-mc", default="/Users/aritra/cernbox (2)/dijet_skims")
    ap.add_argument("--trijet-mc", default="/Users/aritra/cernbox (2)/trijet_skims/mc")
    ap.add_argument("--dijet-data", default="/Users/aritra/Downloads/dijet_data_skims")
    ap.add_argument("--trijet-data", default="/Users/aritra/cernbox (2)/trijet_skims/data")
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args(argv)

    dpaths = sorted(Path(args.dijet_data if args.data else args.dijet_mc).glob("*.pkl"))
    tpaths = sorted(Path(args.trijet_data if args.data else args.trijet_mc).glob("*.pkl"))
    print("dijet skims:")
    D, ddsets = load(dpaths, RECO_FLOORS, GEN_FLOORS)
    print("trijet skims:")
    T, tdsets = load(tpaths, RECO_FLOORS, GEN_FLOORS)

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
    #### Use the stored weight as-is. For DATA the stored weight is the
    #### trigger PRESCALE weight (dijet data has no trigger bit; the processor
    #### stores `weights = where(trigsel, psweights, 1)`), and it must be
    #### applied for any shape comparison: raw data counts are dominated by the
    #### unprescaled high-threshold path and so live at much higher pT than the
    #### xs-weighted MC, which is dominated by the 200 GeV turn-on.
    w = D["weight"][di].astype(float)

    #### baseline = everything the skim wrote that cleared npv+MET filters
    base = np.ones(len(di), bool)
    for c in ("npv", "METfilters"):
        base &= (D["cutbits"][di] & bit(DIJET_CUTS, c)) != 0
    passd = (D["cutbits"][di] & bit(DIJET_CUTS, "recoTot_seq")) != 0
    passt = (T["cutbits"][ti] & bit(TRIJET_CUTS, "recoTot_seq")) != 0

    cats = (("all skimmed events", base, "#9c9ca1"),
            ("dijet selected", passd, "#5790fc"),
            ("trijet selected", passt, "#e42536"),
            ("overlap", passd & passt, "#964a8b"))

    label = "2018 JetHT" if args.data else "QCD MG+Pythia8 2018 (xs-weighted)"
    print(f"\n=== AK8 reco jet multiplicity -- {label} ===")
    print(f"{'category':>20s} {'<N>':>6s} " +
          " ".join(f"{f'N={i}':>7s}" for i in range(1, 7)))
    nreco = D[f"reco{None}"][di]
    for name, m, _ in cats:
        ww = w[m]
        tot = ww.sum()
        mean = (nreco[m] * ww).sum() / tot
        fr = [w[m & (nreco == i)].sum() / tot for i in range(1, 7)]
        print(f"{name:>20s} {mean:>6.2f} " + " ".join(f"{x:>7.3f}" for x in fr))

    print(f"\n=== effect of a pT floor on the counted multiplicity ===")
    print(f"{'collection':>22s} {'<N>':>6s} {'f(N>=3)':>9s}")
    for tag, floors, pref in (("reco FatJet", RECO_FLOORS, "reco"),
                              ("gen GenJetAK8", GEN_FLOORS, "gen")):
        for f in floors:
            n = D[f"{pref}{f}"][di]
            fl = "all stored" if f is None else f"pT > {f:.0f}"
            print(f"{tag + ', ' + fl:>22s} {(n * w)[base].sum() / w[base].sum():>6.2f} "
                  f"{w[base & (n >= 3)].sum() / w[base].sum():>9.3f}")

    make_plots(Path(args.outdir), w, D, di, base, cats, args.data)
    return {"njet": D[f"reco{None}"][di], "w": w,
            "base": base, "dijet": passd, "trijet": passt}


def make_plots(outdir, w, D, di, base, cats, is_data):
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.CMS)
    outdir.mkdir(parents=True, exist_ok=True)

    #### (1) reco multiplicity by category
    nreco = D[f"reco{None}"][di]
    fig, ax = plt.subplots(layout="constrained")
    for name, m, c in cats:
        h, edges = njet_hist(nreco[m], w[m])
        hep.histplot(h / h.sum(), edges, ax=ax, histtype="step", lw=2.5,
                     color=c, label=name)
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\mathrm{jets}}$ (AK8, all stored)")
    ax.set_ylabel("Fraction of events")
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(1e-5, ax.get_ylim()[1] * 4)
    ax.legend(loc="upper right", fontsize=18)
    hep.cms.label("Preliminary", data=is_data, loc=0, ax=ax,
                  rlabel="59.8 fb$^{-1}$ (13 TeV)" if is_data else "(13 TeV)")
    stamp(fig)
    f1 = outdir / ("njet_reco_by_channel_data.png" if is_data
                   else "njet_reco_by_channel.png")
    fig.savefig(f1, dpi=110)
    print("wrote", f1)

    #### (2) gen vs reco multiplicity, and the pT floor
    fig, ax = plt.subplots(layout="constrained")
    series = [(f"reco{None}", "reco, all stored ($p_T \\gtrsim 170$)", "#5790fc", "-"),
              (f"gen{None}", "gen, all stored ($p_T \\gtrsim 100$)", "#f89c20", "-"),
              (f"gen{170.0}", "gen, $p_T > 170$ GeV", "#e42536", "--"),
              (f"gen{200.0}", "gen, $p_T > 200$ GeV", "#964a8b", ":")]
    for key, lab, c, ls in series:
        n = D[key][di][base]
        h, edges = njet_hist(n, w[base])
        hep.histplot(h / h.sum(), edges, ax=ax, histtype="step", lw=2.5,
                     color=c, ls=ls, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\mathrm{jets}}$ (AK8)")
    ax.set_ylabel("Fraction of events")
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(1e-5, ax.get_ylim()[1] * 4)
    ax.legend(loc="upper right", fontsize=17)
    hep.cms.label("Preliminary", data=is_data, loc=0, ax=ax,
                  rlabel="59.8 fb$^{-1}$ (13 TeV)" if is_data else "(13 TeV)")
    stamp(fig)
    f2 = outdir / ("njet_gen_vs_reco_data.png" if is_data
                   else "njet_gen_vs_reco.png")
    fig.savefig(f2, dpi=110)
    print("wrote", f2)


def stamp(fig):
    import datetime
    import subprocess
    ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                         capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[2])).stdout.strip()
    fig.text(0.99, 0.002,
             f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {ver}  |  "
             f"inputs: dijet rho_skim10 + trijet rho_skim event tables, 2018",
             ha="right", va="bottom", fontsize=7, color="0.45", family="monospace")


def overlay(outdir):
    """Data/MC SHAPE comparison of the AK8 multiplicity, per channel."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.CMS)
    outdir = Path(outdir)
    print("### MC ###")
    mc = main(["--outdir", str(outdir)])
    print("\n### DATA ###")
    dt = main(["--outdir", str(outdir), "--data"])

    fig, axes = plt.subplots(1, 2, figsize=(21, 10.6), layout="constrained")
    for ax, key, title in ((axes[0], "dijet", "dijet selected"),
                           (axes[1], "trijet", "trijet selected")):
        hm, edges = njet_hist(mc["njet"][mc[key]], mc["w"][mc[key]])
        hd, _ = njet_hist(dt["njet"][dt[key]], dt["w"][dt[key]])
        hm, hd = hm / hm.sum(), hd / hd.sum()
        ctr = 0.5 * (edges[1:] + edges[:-1])
        hep.histplot(hm, edges, ax=ax, histtype="step", lw=2.5, color="#5790fc",
                     label="QCD MG+Pythia8")
        #### prescale-weighted data: the error is sqrt(sum w^2), not sqrt(N)
        wd = dt["w"][dt[key]]
        h2, _ = njet_hist(dt["njet"][dt[key]], wd ** 2)
        ax.errorbar(ctr, hd, yerr=np.sqrt(h2) / wd.sum(),
                    fmt="o", ms=7, color="k", label="Data 2018 (JetHT)")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_{\mathrm{jets}}$ (AK8, all stored)")
        ax.set_ylabel("Fraction of events")
        ax.set_xlim(1.5 if key == "dijet" else 2.5, 6.5)
        ax.set_ylim(1e-5, 8)
        ax.legend(loc="upper right", fontsize=20)
        hep.cms.label("Preliminary", data=True, loc=0, ax=ax, rlabel=title)
        for i, (a, b) in enumerate(zip(hd, hm)):
            if b > 0 and a > 0:
                print(f"  {title:>16s}  N={i}: data/MC = {a / b:.3f}")
    stamp(fig)
    f = outdir / "njet_data_mc_shape.png"
    fig.savefig(f, dpi=110)
    print("wrote", f)


if __name__ == "__main__":
    if "--overlay" in sys.argv:
        i = sys.argv.index("--outdir")
        overlay(sys.argv[i + 1])
    else:
        main()
