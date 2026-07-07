#!/usr/bin/env python3
"""Extract gen Z+jet rho_g/rho_u from the pythia_var NanoGEN (nom + cr1/cr2/frag*),
matching the CMS_ZJET_JETMASS gen selection, HT-stitched. -> per-variation npz cache
of (jet_pt, rho_u, rho_g, weight). Reweights w=var/nom are built from these.
"""
import os, glob, sys
from pathlib import Path
import numpy as np
import awkward as ak
import uproot, vector
vector.register_awkward()

BASE = Path("/Users/aritra/cernbox (2)/pythia_var")
OUT = Path("/private/tmp/claude-501/-Users-aritra-Projects-smp-jetmass-run2/844f4080-8bec-48e3-b1b7-5db08922b718/scratchpad")
NFILES = int(os.environ.get("NFILES", "40"))          # files per HT bin (subset)
# standard DYJetsToLL_M-50_HT madgraphMLM xsec [pb] (cancels in var/nom ratio)
XSEC = {"HT-200to400": 48.66, "HT-400to600": 6.968, "HT-600to800": 1.743,
        "HT-800to1200": 0.8052, "HT-1200to2500": 0.1933, "HT-2500toInf": 0.003468}
BR = ["GenDressedLepton_pt", "GenDressedLepton_eta", "GenDressedLepton_phi",
      "GenDressedLepton_mass", "GenDressedLepton_pdgId",
      "GenJetAK8_pt", "GenJetAK8_eta", "GenJetAK8_phi", "GenJetAK8_mass",
      "SubGenJetAK8_pt", "SubGenJetAK8_eta", "SubGenJetAK8_phi", "SubGenJetAK8_mass",
      "genWeight"]


def process(variation):
    pts, rus, rgs, ws = [], [], [], []
    for htdir in sorted(glob.glob(str(BASE / variation / "HT-*"))):
        ht = os.path.basename(htdir)
        files = sorted(glob.glob(f"{htdir}/*.root"))[:NFILES]
        if not files:
            continue
        sumw = 0.0; ev = []
        for f in files:
            try:
                a = uproot.open(f)["Events"].arrays(BR)
            except Exception:
                continue
            sumw += float(ak.sum(a["genWeight"]))
            ev.append(a)
        if not ev:
            continue
        a = ak.concatenate(ev)
        lep = ak.zip({"pt": a.GenDressedLepton_pt, "eta": a.GenDressedLepton_eta,
                      "phi": a.GenDressedLepton_phi, "mass": a.GenDressedLepton_mass,
                      "pdgId": a.GenDressedLepton_pdgId}, with_name="Momentum4D")
        # per-flavour dressed leptons passing pt/eta
        def flav(pid, ptcut):
            m = (abs(lep.pdgId) == pid) & (lep.pt > ptcut) & (abs(lep.eta) < 2.4)
            return lep[m]
        Zcand = []
        for pid, ptcut in [(11, 40.0), (13, 29.0)]:
            L = flav(pid, ptcut)
            has2 = ak.num(L) >= 2
            L2 = L[has2][:, :2]
            os_ = (L2.pdgId[:, 0] * L2.pdgId[:, 1]) < 0
            Z = L2[:, 0] + L2[:, 1]
            good = os_ & (Z.pt > 90) & (Z.mass > 71) & (Z.mass < 111)
            idx = ak.local_index(has2)[has2][good]        # event indices with a Z of this flavour
            Zg = Z[good]
            Zcand.append((np.asarray(idx), Zg))
        # merge flavours: build per-event Z (take first flavour that fired)
        nev = len(a); zpt = np.full(nev, np.nan); zeta = np.zeros(nev); zphi = np.zeros(nev)
        for idx, Zg in Zcand:
            fill = np.isnan(zpt[idx])
            ii = idx[fill]
            zpt[ii] = np.asarray(Zg.pt)[fill]; zphi[ii] = np.asarray(Zg.phi)[fill]
        hasZ = ~np.isnan(zpt)
        if hasZ.sum() == 0:
            continue
        # leading AK8 jet with |eta|<2.4 (rap approx)
        jet = ak.zip({"pt": a.GenJetAK8_pt, "eta": a.GenJetAK8_eta, "phi": a.GenJetAK8_phi,
                      "mass": a.GenJetAK8_mass}, with_name="Momentum4D")
        jet = jet[abs(jet.eta) < 2.4]
        hasj = ak.num(jet) >= 1
        sel = hasZ & np.asarray(hasj)
        j0 = jet[sel][:, 0]
        zpt_s = zpt[sel]; zphi_s = zphi[sel]; w_s = np.asarray(a.genWeight)[sel]
        dphi = np.abs(np.arctan2(np.sin(np.asarray(j0.phi) - zphi_s), np.cos(np.asarray(j0.phi) - zphi_s)))
        jpt = np.asarray(j0.pt); ju = np.asarray(j0.mass)
        asym = np.abs(zpt_s - jpt) / (zpt_s + jpt)
        pas = (dphi > 1.57) & (asym < 0.3) & (jpt >= 200)
        # softdrop mass from the two leading subjets within dR<0.8 of the leading jet
        sub = ak.zip({"pt": a.SubGenJetAK8_pt, "eta": a.SubGenJetAK8_eta,
                      "phi": a.SubGenJetAK8_phi, "mass": a.SubGenJetAK8_mass}, with_name="Momentum4D")
        sub_s = sub[sel]
        dR = j0.deltaR(sub_s)
        near = sub_s[dR < 0.8]
        near = near[ak.argsort(near.pt, ascending=False)]
        two = ak.num(near) >= 2
        msd = np.full(len(jpt), np.nan)
        sd = (near[two][:, 0] + near[two][:, 1]).mass
        msd[np.asarray(two)] = np.asarray(sd)
        pas = pas & np.asarray(two)
        jpt, ju, msd, w_s = jpt[pas], ju[pas], msd[pas], w_s[pas]
        ru = 2 * np.log10(ju / (jpt * 0.8)); rg = 2 * np.log10(msd / (jpt * 0.8))
        wht = XSEC[ht] / sumw
        pts.append(jpt); rus.append(ru); rgs.append(rg); ws.append(np.full(len(jpt), wht) * w_s)
        print(f"  {variation:9s} {ht:14s} files={len(files):3d} sel={len(jpt):6d} w={wht:.3e}", flush=True)
    return (np.concatenate(pts), np.concatenate(rus), np.concatenate(rgs), np.concatenate(ws))


if __name__ == "__main__":
    cache = {}
    for v in ["nom", "cr1", "cr2", "fraghard", "fragsoft"]:
        print(f"=== {v} ===", flush=True)
        pt, ru, rg, w = process(v)
        cache[f"{v}_jet_pt"] = pt; cache[f"{v}_rho_u"] = ru; cache[f"{v}_rho_g"] = rg; cache[f"{v}_weight"] = w
    np.savez(OUT / "pythia_var_cache.npz", **cache)
    print("saved", OUT / "pythia_var_cache.npz")
