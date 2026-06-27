#!/usr/bin/env python3
"""ARC Misc#4 (4.5/4.6) — improved Z+jet event displays.

Fixes the two ARC confusions with the originals:
  (4.5) lepton kinematics now shown explicitly (a per-lepton table: flavour,
        pT, eta, phi, charge, and whether it passes the reco selection).
  (4.6) the Reco Z is drawn ONLY when it is actually reconstructed (>=2 selected
        same-flavour reco leptons forming a valid pair). When fewer than two
        reco leptons are found, NO phantom Z is drawn at the origin / on the lone
        lepton; the panel states "Reco Z: NOT reconstructed (N reco leptons)"
        and the verdict names the failing cut.

Runs on a local DY MC file; picks instructive miss events (gen-fiducial, fails
reco because a lepton is missing) plus one good matched event for contrast.
"""
import os
import sys
sys.path.insert(0, ".")

import awkward as ak
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection
from smp_jetmass_run2.smp_utils import (
    get_z_gen_selection, get_z_reco_selection,
    apply_lepton_separation, apply_lepton_separation_gen,
)
from smp_jetmass_run2.corrections import getRapidity, MET_filters

NanoAODSchema.warn_missing_crossrefs = False
FILE = "/Users/aritra/Projects/omnifold/data/nanoaod/DYJetsToLL_HT400to600_UL18_test.root"
IOV, PTE, PTM = "2018", 40, 29
OUT = "review/figs"


def build(ev):
    sel = PackedSelection()
    # ---- GEN ----
    gl = ev.GenDressedLepton
    is_e, is_m = np.abs(gl.pdgId) == 11, np.abs(gl.pdgId) == 13
    ptcut = (is_e & (gl.pt > PTE)) | (is_m & (gl.pt > PTM))
    ev = ak.with_field(ev, gl[ptcut & (np.abs(gl.eta) < 2.4)], "GenDressedLepton")
    gj = ak.with_field(ev.GenJetAK8, getRapidity(ev.GenJetAK8), "rapidity")
    ev = ak.with_field(ev, gj[(gj.pt > 0) & (np.abs(gj.rapidity) < 2.4)], "GenJetAK8")
    gjc = apply_lepton_separation_gen(ev.GenJetAK8, ev.GenDressedLepton, dr_cut=0.4)
    ev = ak.with_field(ev, gjc[(gjc.pt > 0) & (np.abs(gjc.rapidity) < 2.4)], "GenJetAK8")
    sel.add("oneGenJet", ak.sum(ev.GenJetAK8.pt > 0, axis=1) >= 1)
    sel.add("jet200", ak.sum(ev.GenJetAK8.pt > 200, axis=1) >= 1)
    z_gen = get_z_gen_selection(ev, sel, PTE, PTM, None, None)
    sel.add("z_ptcut_gen", sel.all("twoGen_leptons") & ak.fill_none(z_gen.pt > 90.0, False))
    sel.add("z_mcut_gen", sel.all("twoGen_leptons") & ak.fill_none((z_gen.mass > 71.) & (z_gen.mass < 111.), False))
    kinsel_gen = sel.all("twoGen_leptons", "oneGenJet", "z_ptcut_gen", "z_mcut_gen")
    sel.add("kinsel_gen", kinsel_gen)

    # ---- RECO leptons ----
    eeta = np.abs(ev.Electron.eta)
    ev = ak.with_field(ev, ev.Electron[
        (ev.Electron.pt > PTE) & (eeta < 2.4) & ((eeta < 1.422) | (eeta > 1.566))
        & (ev.Electron.pfRelIso03_all < 0.25) & (ev.Electron.cutBased > 3)
        & (np.abs(ev.Electron.dz) < 0.5) & (np.abs(ev.Electron.dxy) < 0.2)], "Electron")
    ev = ak.with_field(ev, ev.Muon[
        (ev.Muon.pt > PTM) & (np.abs(ev.Muon.eta) < 2.4) & (ev.Muon.pfIsoId > 3)
        & (ev.Muon.tightId == True) & (np.abs(ev.Muon.dz) < 0.5) & (np.abs(ev.Muon.dxy) < 0.2)], "Muon")
    z_reco = get_z_reco_selection(ev, sel, PTE, PTM, None, None)
    has_z = sel.all("twoReco_leptons")
    sel.add("z_ptcut_reco", has_z & ak.fill_none(z_reco.pt > 90, False))
    sel.add("z_mcut_reco", has_z & ak.fill_none((z_reco.mass > 71.) & (z_reco.mass < 111.), False))
    sel.add("kinsel_reco", sel.all("twoReco_leptons", "z_ptcut_reco", "z_mcut_reco"))

    sel.add("npv", ev.PV.npvsGood > 0)
    sel.add("MET", np.logical_and.reduce(np.array([ev.Flag[f] for f in MET_filters[IOV] if f in ev.Flag.fields]), axis=0))

    # ---- RECO jet (analysis AK8: subjets, lepton-cleaned) ----
    fj = ev.FatJet[(ev.FatJet.subJetIdx1 > -1) & (ev.FatJet.subJetIdx2 > -1)]
    fj = apply_lepton_separation(fj, ev.Muon, ev.Electron, dr_cut=0.4)
    fj = fj[(fj.pt > 200) & (np.abs(fj.eta) < 2.5) & (fj.jetId == 6)]
    sel.add("oneRecoJet", ak.num(fj) >= 1)
    ev = ak.with_field(ev, fj, "FatJetSel")
    return ev, sel, z_gen, z_reco


def s1(x, d=np.nan):
    try:
        v = ak.to_numpy(ak.flatten(ak.Array([x]), axis=None))
        return float(v[0]) if len(v) else d
    except Exception:
        return d


def lead(coll, i, n=1):
    c = coll[i]
    return c[:n] if len(c) >= n else c


def render(ev, sel, z_gen, z_reco, iev, tag):
    e = ev[iev]
    flags = {k: bool(sel.all(k)[iev]) for k in
             ["npv", "MET", "twoReco_leptons", "z_ptcut_reco", "z_mcut_reco",
              "kinsel_reco", "oneRecoJet", "kinsel_gen"]}
    allsel_reco = flags["npv"] and flags["MET"] and flags["kinsel_reco"] and flags["oneRecoJet"]

    # leading gen / reco jet by pt
    gjet = lead(ev.GenJetAK8, iev); rjet = lead(ev.FatJetSel, iev)
    gjet_pt, gjet_phi = s1(gjet.pt), s1(gjet.phi)
    rjet_pt, rjet_phi = s1(rjet.pt), s1(rjet.phi)

    # gen Z always defined (2 gen leptons in fiducial); reco Z only if valid
    zg_pt, zg_phi, zg_m = s1(z_gen[iev].pt), s1(z_gen[iev].phi), s1(z_gen[iev].mass)
    n_mu, n_el = len(e.Muon), len(e.Electron)
    is_mm = len(e.GenDressedLepton) >= 1 and abs(s1(e.GenDressedLepton.pdgId)) == 13
    reco_lep_coll, reco_flav = (e.Muon, "muon") if is_mm else (e.Electron, "electron")
    n_reco_lep = len(reco_lep_coll)
    reco_z_ok = flags["twoReco_leptons"]
    zr_pt = zr_phi = zr_m = np.nan
    if reco_z_ok:
        zr_pt, zr_phi, zr_m = s1(z_reco[iev].pt), s1(z_reco[iev].phi), s1(z_reco[iev].mass)

    # ---------- figure ----------
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.5, 1.05], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0], projection="polar")
    axl = fig.add_subplot(gs[0, 1]); axl.axis("off")
    axi = fig.add_subplot(gs[0, 2]); axi.axis("off")

    pts = [p for p in [gjet_pt, rjet_pt, zg_pt, zr_pt] if np.isfinite(p)]
    lep_pts = ak.to_numpy(ak.flatten(ak.Array([e.GenDressedLepton.pt, e.Muon.pt, e.Electron.pt]), axis=None))
    pts += list(lep_pts)
    rmax = max(50, 1.15 * (max(pts) if pts else 100))

    def pt_pts(coll, color, marker, fill, label, line=False):
        try:
            phis = ak.to_numpy(coll.phi); pts_ = ak.to_numpy(coll.pt)
        except Exception:
            return
        for j, (ph, pt_) in enumerate(zip(np.atleast_1d(phis), np.atleast_1d(pts_))):
            if not np.isfinite(pt_):
                continue
            th = ph % (2 * np.pi)
            if line:
                ax.plot([th, th], [0, pt_], color=color, lw=1.0, alpha=0.5)
            ax.scatter([th], [pt_], s=130, marker=marker, label=(label if j == 0 else None),
                       facecolors=(color if fill else "none"), edgecolors=color, linewidths=2, zorder=5)

    pt_pts(e.GenDressedLepton, "royalblue", "o", False, "Gen Dressed Lepton")
    pt_pts(e.Muon, "tab:red", "^", False, "Reco Muon")
    pt_pts(e.Electron, "tab:green", "s", False, "Reco Electron")
    pt_pts(e.GenJetAK8, "brown", "D", True, "Gen Jet")
    pt_pts(ev.FatJet[iev], "coral", "D", False, "Reco Jet")
    if np.isfinite(gjet_pt):
        ax.scatter([gjet_phi % (2*np.pi)], [gjet_pt], s=320, marker="*", color="black", label="Selected Gen Jet", zorder=6)
    if np.isfinite(rjet_pt):
        ax.scatter([rjet_phi % (2*np.pi)], [rjet_pt], s=240, marker="X", color="gold", edgecolors="k", label="Selected Reco Jet", zorder=6)
    # Gen Z always; Reco Z only if reconstructed
    ax.plot([zg_phi % (2*np.pi)]*2, [0, zg_pt], color="magenta", lw=1.2, alpha=0.6)
    ax.scatter([zg_phi % (2*np.pi)], [zg_pt], s=200, marker="p", color="magenta", label="Gen Z", zorder=6)
    if reco_z_ok and np.isfinite(zr_pt):
        ax.plot([zr_phi % (2*np.pi)]*2, [0, zr_pt], color="hotpink", lw=1.2, alpha=0.6, ls="--")
        ax.scatter([zr_phi % (2*np.pi)], [zr_pt], s=200, marker="p", facecolors="none", edgecolors="hotpink", linewidths=2, label="Reco Z", zorder=6)

    ax.set_title(f"Event {iev}", pad=26, fontsize=15)
    ax.set_rmax(rmax); ax.set_rlabel_position(22.5)
    ax.set_xlabel(r"$\phi$ [rad]", labelpad=16)
    ax.set_ylabel(r"$\rho = p_T$ [GeV]", labelpad=24)
    ax.grid(True, alpha=0.6)
    h, l = ax.get_legend_handles_labels()
    axl.legend(h, l, loc="center left", fontsize=10, frameon=True)

    # ---------- info panel: lepton table + Z + verdict ----------
    lines = ["Selection (reco):"]
    for k in ["npv", "MET", "twoReco_leptons", "z_ptcut_reco", "z_mcut_reco", "oneRecoJet"]:
        lines.append(f"  {k:16s}: {flags[k]}")
    lines.append(f"  {'allsel_reco':16s}: {allsel_reco}")
    lines.append(f"  {'allsel_gen':16s}: {flags['kinsel_gen']}")
    lines.append("")
    lines.append("Gen leptons (pT, eta, phi, q):")
    for lp in e.GenDressedLepton:
        fl = "mu" if abs(s1(lp.pdgId)) == 13 else "e"
        q = "-" if s1(lp.pdgId) > 0 else "+"   # pdgId>0 = particle (e-/mu-)
        lines.append(f"  {fl:2s}  {s1(lp.pt):6.1f}  {s1(lp.eta):+5.2f}  {s1(lp.phi):+5.2f}  {q}")
    lines.append(f"Reco leptons found: {n_mu} mu, {n_el} e")
    for lp in e.Muon:
        lines.append(f"  mu  {s1(lp.pt):6.1f}  {s1(lp.eta):+5.2f}  {s1(lp.phi):+5.2f}  {'-' if s1(lp.charge)<0 else '+'}")
    for lp in e.Electron:
        lines.append(f"  e   {s1(lp.pt):6.1f}  {s1(lp.eta):+5.2f}  {s1(lp.phi):+5.2f}  {'-' if s1(lp.charge)<0 else '+'}")
    lines.append("")
    lines.append(f"Gen Z : pT={zg_pt:6.1f}  m={zg_m:5.1f}")
    if reco_z_ok and np.isfinite(zr_m):
        lines.append(f"Reco Z: pT={zr_pt:6.1f}  m={zr_m:5.1f}")
    else:
        lines.append(f"Reco Z: NOT reconstructed")
        lines.append(f"        (only {n_reco_lep} reco {reco_flav}(s); need 2)")
    lines.append("")
    if np.isfinite(gjet_pt) and np.isfinite(rjet_pt):
        dr = float(np.hypot(s1(rjet.eta) - s1(gjet.eta), (rjet_phi - gjet_phi + np.pi) % (2*np.pi) - np.pi))
        lines.append(f"Jet: reco pT={rjet_pt:.1f}, gen pT={gjet_pt:.1f}, dR={dr:.3f}  (matched)")
    # verdict
    if allsel_reco:
        verdict = "PASS reco & gen -> MATCHED (enters response)"
    elif flags["kinsel_gen"] and not reco_z_ok:
        verdict = f"MISS: gen-fiducial but only {n_reco_lep} reco {reco_flav} -> no Z"
    elif flags["kinsel_gen"]:
        verdict = "MISS: gen-fiducial, fails a reco cut (see flags)"
    else:
        verdict = "not gen-fiducial"
    lines.append("")
    lines.append("VERDICT:")
    lines.append(f"  {verdict}")

    axi.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
             fontsize=10.5, transform=axi.transAxes)

    fig.subplots_adjust(left=0.04, right=0.985, top=0.91, bottom=0.09)
    os.makedirs(OUT, exist_ok=True)
    fout = f"{OUT}/event_display_{tag}.png"
    plt.savefig(fout, dpi=130)
    plt.close(fig)
    print(f"  wrote {fout}  (verdict: {verdict})")


def main():
    ev = NanoEventsFactory.from_root({FILE: "Events"}, schemaclass=NanoAODSchema, mode="eager").events()
    ev, sel, z_gen, z_reco = build(ev)

    genfid = sel.all("kinsel_gen", "jet200")
    allreco = sel.all("npv", "MET", "kinsel_reco", "oneRecoJet")
    n_mu = ak.num(ev.Muon); n_el = ak.num(ev.Electron)
    gen_mm = genfid & (ak.num(ev.GenDressedLepton) >= 1) & ak.fill_none(np.abs(ak.firsts(ev.GenDressedLepton.pdgId)) == 13, False)
    gen_ee = genfid & ak.fill_none(np.abs(ak.firsts(ev.GenDressedLepton.pdgId)) == 11, False)

    # instructive cases
    miss_mm_1lep = np.flatnonzero(ak.to_numpy(gen_mm & ~allreco & (n_mu == 1)))
    miss_ee_1lep = np.flatnonzero(ak.to_numpy(gen_ee & ~allreco & (n_el == 1)))
    good = np.flatnonzero(ak.to_numpy(genfid & allreco))
    print(f"miss mm (1 reco mu): {len(miss_mm_1lep)} | miss ee (1 reco e): {len(miss_ee_1lep)} | good matched: {len(good)}")

    picks = []
    if len(miss_mm_1lep): picks.append((int(miss_mm_1lep[0]), "miss_mm_1mu"))
    if len(miss_ee_1lep): picks.append((int(miss_ee_1lep[0]), "miss_ee_1e"))
    if len(good): picks.append((int(good[0]), "good_matched"))
    for iev, tag in picks:
        render(ev, sel, z_gen, z_reco, iev, tag)


if __name__ == "__main__":
    main()
