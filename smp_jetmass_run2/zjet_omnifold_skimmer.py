"""
ZJetOmniFoldSkimmer
===================
Minimal AK8 Z(->mumu)+jet skimmer for OmniFold unfolding tests.

Adapted from ``zjet_minimal_processor.ZJetMinimalProcessor``.  The selection is
the same (tight Z->mumu, recoil jet, lepton cleaning, back-to-back), but:

  * the jet is the leading **AK8 FatJet** matched to **GenJetAK8** (not AK4), and
  * the output is **per-event flat columns** written to Parquet, not histograms.

Stored columns (one row per event passing the *reco* selection):

  reco_pt, reco_eta, reco_phi, reco_mass, reco_msoftdrop
  gen_pt,  gen_eta,  gen_phi,  gen_mass,  gen_msoftdrop   (NaN if unmatched)
  genWeight                                               (raw, signed)
  matched, dataset                                        (bookkeeping)

``matched`` lets a downstream loader keep only 1:1 reco/gen pairs (what OmniFold
needs) while still recording reco-only "fakes" for later efficiency studies.
Gen-level "misses" (gen jet with no reco) are *not* captured here -- that needs
a separate gen-driven loop and is left as a follow-up.

Memory model
------------
``output_mode="parquet"`` writes one Parquet file per chunk straight to
``outdir`` and returns only counts, so memory is flat regardless of total event
count (no giant column_accumulator merge).  ``output_mode="accumulator"`` keeps
the columns in memory and returns them for small local/debug runs.
"""

from __future__ import annotations

import os
import uuid

import awkward as ak
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoAODSchema


# Columns produced by the skimmer (kept in one place so the runner/merger agree).
RECO_FIELDS = ["reco_pt", "reco_eta", "reco_phi", "reco_mass", "reco_msoftdrop"]
GEN_FIELDS = ["gen_pt", "gen_eta", "gen_phi", "gen_mass", "gen_msoftdrop"]
EXTRA_FIELDS = ["genWeight", "matched"]
ALL_FIELDS = RECO_FIELDS + GEN_FIELDS + EXTRA_FIELDS


def _np(x):
    """Awkward -> contiguous float64 numpy (handles option/None via NaN)."""
    return ak.to_numpy(ak.fill_none(x, np.nan)).astype(np.float64)


def _gen_softdrop_mass(gen_jet, subgen):
    """Reconstruct gen soft-drop mass = mass of the two leading SubGenJetAK8
    within dR<0.8 of ``gen_jet`` (per event).  NaN where < 2 subjets are found.

    ``gen_jet`` is the (option-typed) matched GenJetAK8 for the lead reco jet;
    ``subgen`` is ``events.SubGenJetAK8``.  Both already carry PtEtaPhiM Lorentz
    behaviour from NanoAODSchema.
    """
    eta = ak.fill_none(gen_jet.eta, np.nan)
    phi = ak.fill_none(gen_jet.phi, np.nan)
    deta = eta - subgen.eta
    dphi = (phi - subgen.phi + np.pi) % (2.0 * np.pi) - np.pi
    dr = np.sqrt(deta**2 + dphi**2)
    near = subgen[dr < 0.8]
    near = near[ak.argsort(near.pt, axis=1, ascending=False)]
    pair = near[:, :2]
    summed = pair.sum(axis=1)
    msd = ak.where(ak.num(near, axis=1) >= 2, summed.mass, np.nan)
    return msd


class ZJetOmniFoldSkimmer(processor.ProcessorABC):
    """Skim Z(->mumu)+AK8-jet events into flat per-event columns for OmniFold.

    Parameters
    ----------
    output_mode : {"parquet", "accumulator"}
        "parquet": write one file per chunk to ``outdir`` (memory-flat).
        "accumulator": return columns in the accumulator (small/debug runs).
    outdir : str | None
        Output directory for ``output_mode="parquet"``.  Files are written to
        ``{outdir}/{dataset}/part-{uuid}.parquet``.
    pt_min : float
        Minimum leading-jet pT (GeV).  Loose by design so the spectrum is not
        sculpted before unfolding.
    """

    def __init__(
        self,
        output_mode: str = "parquet",
        outdir: str | None = None,
        pt_min: float = 150.0,
    ):
        if output_mode not in ("parquet", "accumulator"):
            raise ValueError("output_mode must be 'parquet' or 'accumulator'")
        if output_mode == "parquet" and not outdir:
            raise ValueError("outdir is required for output_mode='parquet'")
        self.output_mode = output_mode
        self.outdir = outdir
        self.pt_min = pt_min

    # ------------------------------------------------------------------
    def process(self, events):
        NanoAODSchema.warn_missing_crossrefs = False
        dataset = events.metadata.get("dataset", "unknown")
        n_read = len(events)

        # ---- 1. Muon selection (tight Z->mumu) -----------------------
        mu = events.Muon
        tight_mu = mu[
            (mu.pt > 20.0)
            & (np.abs(mu.eta) < 2.4)
            & mu.tightId
            & (mu.pfRelIso04_all < 0.15)
        ]
        has_os = (ak.num(tight_mu) >= 2) & (ak.sum(tight_mu.charge, axis=1) == 0)
        events = events[has_os]
        tight_mu = tight_mu[has_os]

        Z = tight_mu[:, 0] + tight_mu[:, 1]
        z_ok = (Z.mass > 71.0) & (Z.mass < 111.0)
        events = events[z_ok]
        Z = Z[z_ok]
        tight_mu = tight_mu[z_ok]

        # ---- 2. Reco AK8 jet selection + lepton cleaning -------------
        fatjets = events.FatJet
        base = fatjets[
            (fatjets.pt > 0.0)
            & (np.abs(fatjets.eta) < 2.4)
            & (fatjets.jetId >= 2)  # AK8 tight ID bit
        ]
        clean = base[ak.all(base.metric_table(tight_mu) > 0.8, axis=2)]

        has_jet = ak.num(clean) >= 1
        events = events[has_jet]
        Z = Z[has_jet]
        clean = clean[has_jet]
        lead = clean[:, 0]

        dphi_ok = np.abs(lead.delta_phi(Z)) > 2.0
        events = events[dphi_ok]
        lead = lead[dphi_ok]

        pt_ok = lead.pt > self.pt_min
        events = events[pt_ok]
        lead = lead[pt_ok]

        n_sel = len(lead)
        is_mc = "GenJetAK8" in events.fields

        # ---- 3. Build output columns --------------------------------
        cols: dict[str, np.ndarray] = {
            "reco_pt": _np(lead.pt),
            "reco_eta": _np(lead.eta),
            "reco_phi": _np(lead.phi),
            "reco_mass": _np(lead.mass),
            "reco_msoftdrop": _np(lead.msoftdrop),
        }

        if is_mc:
            mg = lead.matched_gen  # option-typed GenJetAK8 (None if unmatched)
            matched = ak.fill_none(lead.genJetAK8Idx >= 0, False)
            cols["gen_pt"] = _np(mg.pt)
            cols["gen_eta"] = _np(mg.eta)
            cols["gen_phi"] = _np(mg.phi)
            cols["gen_mass"] = _np(mg.mass)
            cols["gen_msoftdrop"] = ak.to_numpy(
                _gen_softdrop_mass(mg, events.SubGenJetAK8)
            ).astype(np.float64)
            cols["genWeight"] = _np(events.genWeight)
            cols["matched"] = ak.to_numpy(matched).astype(bool)
            n_matched = int(np.sum(cols["matched"]))
        else:
            nan = np.full(n_sel, np.nan)
            for f in GEN_FIELDS:
                cols[f] = nan.copy()
            cols["genWeight"] = np.ones(n_sel)
            cols["matched"] = np.zeros(n_sel, dtype=bool)
            n_matched = 0

        counts = {
            "n_read": n_read,
            "n_selected": n_sel,
            "n_matched": n_matched,
        }

        # ---- 4. Emit -------------------------------------------------
        if n_sel == 0:
            return self._wrap(dataset, cols, counts, wrote=None)

        if self.output_mode == "parquet":
            out_dir = os.path.join(self.outdir, dataset)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"part-{uuid.uuid4().hex}.parquet")
            ak.to_parquet(ak.Array(cols), path)
            return self._wrap(dataset, None, counts, wrote=path)

        return self._wrap(dataset, cols, counts, wrote=None)

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
