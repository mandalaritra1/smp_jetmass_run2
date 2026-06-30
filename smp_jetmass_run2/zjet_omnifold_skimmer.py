"""
ZJetOmniFoldSkimmer
===================
Light AK8 Z(->ll)+jet skimmer -> flat per-event columns (Parquet), for OmniFold /
NLO-ntuple / model-systematic studies.  **Both channels**: Z->mumu and Z->ee.

Selection (simplified, NOT byte-for-byte the analysis -- this is a light skim):
  * tight OS dilepton, 71 < m_ll < 111 GeV
      mm: Muon pt>20, |eta|<2.4, tightId, pfRelIso04_all<0.15
      ee: Electron pt>25, |eta|<2.5 (EE/EB gap 1.4442-1.5660 vetoed), cutBased>=4 (tight)
  * leading AK8 FatJet, |eta|<2.4, jetId>=2, dR(jet,lepton)>0.8, |dphi(jet,Z)|>2, pt>pt_min
  * the jet is matched to GenJetAK8 (gen columns NaN if unmatched)

Per-event output columns (one row per selected reco jet; names chosen to match the
``mass_diagnostic_ntuple`` reco_jet_ntuple so existing loaders work):

  pt, eta, phi, mass, msoftdrop                       (reco AK8 jet)
  gen_pt, gen_eta, gen_phi, gen_mass, gen_msoftdrop   (matched gen jet; NaN if none)
  weight        raw signed genWeight (1.0 for data; NO xs/pu/SF -- apply downstream)
  passes_both   reco jet has a gen match (bool)
  channel       "mm" | "ee"
  event         event number (for half-sample splits)

Channel by primary dataset (avoids data double-counting): a dataset whose name
contains "SingleMuon" is skimmed mm-only; "EGamma"/"SingleElectron" ee-only;
anything else (MC) is skimmed in **both** channels.

Memory model
------------
``output_mode="parquet"`` writes one file per chunk to ``outdir`` (memory-flat,
LPC/shared-FS).  ``output_mode="accumulator"`` returns columns so the *client*
writes one file per dataset (use on casa, where workers don't share the notebook FS).
"""

from __future__ import annotations

import os
import uuid

import awkward as ak
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoAODSchema


RECO_FIELDS = ["pt", "eta", "phi", "mass", "msoftdrop"]
GEN_FIELDS = ["gen_pt", "gen_eta", "gen_phi", "gen_mass", "gen_msoftdrop"]
EXTRA_FIELDS = ["weight", "passes_both", "channel", "event"]
ALL_FIELDS = RECO_FIELDS + GEN_FIELDS + EXTRA_FIELDS


def _np(x):
    """Awkward -> contiguous float64 numpy (None -> NaN)."""
    return ak.to_numpy(ak.fill_none(x, np.nan)).astype(np.float64)


def _gen_softdrop_mass(gen_jet, subgen):
    """Gen soft-drop mass = mass of the two leading SubGenJetAK8 within dR<0.8 of
    ``gen_jet`` (NaN where < 2 subjets)."""
    eta = ak.fill_none(gen_jet.eta, np.nan)
    phi = ak.fill_none(gen_jet.phi, np.nan)
    deta = eta - subgen.eta
    dphi = (phi - subgen.phi + np.pi) % (2.0 * np.pi) - np.pi
    dr = np.sqrt(deta**2 + dphi**2)
    near = subgen[dr < 0.8]
    near = near[ak.argsort(near.pt, axis=1, ascending=False)]
    pair = near[:, :2]
    summed = pair.sum(axis=1)
    return ak.where(ak.num(near, axis=1) >= 2, summed.mass, np.nan)


def _channels_for(dataset: str):
    if "SingleMuon" in dataset:
        return ["mm"]
    if "EGamma" in dataset or "SingleElectron" in dataset:
        return ["ee"]
    return ["mm", "ee"]               # MC: both


class ZJetOmniFoldSkimmer(processor.ProcessorABC):
    def __init__(self, output_mode="parquet", outdir=None, pt_min=150.0):
        if output_mode not in ("parquet", "accumulator"):
            raise ValueError("output_mode must be 'parquet' or 'accumulator'")
        if output_mode == "parquet" and not outdir:
            raise ValueError("outdir is required for output_mode='parquet'")
        self.output_mode = output_mode
        self.outdir = outdir
        self.pt_min = pt_min

    # ------------------------------------------------------------------
    def _tight_leptons(self, events, channel):
        if channel == "mm":
            mu = events.Muon
            return mu[(mu.pt > 20.0) & (np.abs(mu.eta) < 2.4) & mu.tightId
                      & (mu.pfRelIso04_all < 0.15)]
        el = events.Electron
        ae = np.abs(el.eta)
        gap = (ae < 1.4442) | ((ae > 1.5660) & (ae < 2.5))
        return el[(el.pt > 25.0) & gap & (el.cutBased >= 4)]

    def _select_channel(self, events, channel, is_mc):
        """Return (cols dict, n_selected, n_matched) for one channel, or (None,0,0)."""
        lep = self._tight_leptons(events, channel)
        keep = (ak.num(lep) >= 2) & (ak.sum(lep.charge, axis=1) == 0)
        events, lep = events[keep], lep[keep]
        if len(events) == 0:
            return None, 0, 0
        Z = lep[:, 0] + lep[:, 1]
        zok = (Z.mass > 71.0) & (Z.mass < 111.0)
        events, Z, lep = events[zok], Z[zok], lep[zok]

        fj = events.FatJet
        base = fj[(fj.pt > 0.0) & (np.abs(fj.eta) < 2.4) & (fj.jetId >= 2)]
        clean = base[ak.all(base.metric_table(lep) > 0.8, axis=2)]
        hasjet = ak.num(clean) >= 1
        events, Z, clean = events[hasjet], Z[hasjet], clean[hasjet]
        if len(events) == 0:
            return None, 0, 0
        lead = clean[:, 0]
        ok = (np.abs(lead.delta_phi(Z)) > 2.0) & (lead.pt > self.pt_min)
        events, lead = events[ok], lead[ok]
        n_sel = len(lead)
        if n_sel == 0:
            return None, 0, 0

        cols = {
            "pt": _np(lead.pt), "eta": _np(lead.eta), "phi": _np(lead.phi),
            "mass": _np(lead.mass), "msoftdrop": _np(lead.msoftdrop),
            "channel": np.full(n_sel, channel, dtype=object),
            "event": ak.to_numpy(events.event).astype(np.uint64),
        }
        if is_mc:
            mg = lead.matched_gen
            matched = ak.to_numpy(ak.fill_none(lead.genJetAK8Idx >= 0, False)).astype(bool)
            cols["gen_pt"] = _np(mg.pt); cols["gen_eta"] = _np(mg.eta)
            cols["gen_phi"] = _np(mg.phi); cols["gen_mass"] = _np(mg.mass)
            cols["gen_msoftdrop"] = ak.to_numpy(
                _gen_softdrop_mass(mg, events.SubGenJetAK8)).astype(np.float64)
            cols["weight"] = _np(events.genWeight)
            cols["passes_both"] = matched
            n_matched = int(matched.sum())
        else:
            nan = np.full(n_sel, np.nan)
            for f in GEN_FIELDS:
                cols[f] = nan.copy()
            cols["weight"] = np.ones(n_sel)
            cols["passes_both"] = np.zeros(n_sel, dtype=bool)
            n_matched = 0
        return cols, n_sel, n_matched

    # ------------------------------------------------------------------
    def process(self, events):
        NanoAODSchema.warn_missing_crossrefs = False
        dataset = events.metadata.get("dataset", "unknown")
        n_read = len(events)
        is_mc = "GenJetAK8" in events.fields

        parts = []
        n_sel = n_matched = 0
        for ch in _channels_for(dataset):
            cols, ns, nm = self._select_channel(events, ch, is_mc)
            n_sel += ns; n_matched += nm
            if cols is not None:
                parts.append(cols)

        if not parts:
            return self._wrap(dataset, None,
                              {"n_read": n_read, "n_selected": 0, "n_matched": 0}, None)

        merged = {k: np.concatenate([p[k] for p in parts]) for k in ALL_FIELDS}
        counts = {"n_read": n_read, "n_selected": n_sel, "n_matched": n_matched}

        if self.output_mode == "parquet":
            out_dir = os.path.join(self.outdir, dataset)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"part-{uuid.uuid4().hex}.parquet")
            ak.to_parquet(ak.Array(merged), path)
            return self._wrap(dataset, None, counts, path)
        return self._wrap(dataset, merged, counts, None)

    # ------------------------------------------------------------------
    def _wrap(self, dataset, cols, counts, wrote):
        out = {dataset: dict(counts)}
        out[dataset]["files"] = [wrote] if wrote else []
        if self.output_mode == "accumulator" and cols is not None:
            out[dataset]["columns"] = {
                k: processor.column_accumulator(np.asarray(v)) for k, v in cols.items()
            }
        return out

    def postprocess(self, accumulator):
        return accumulator
