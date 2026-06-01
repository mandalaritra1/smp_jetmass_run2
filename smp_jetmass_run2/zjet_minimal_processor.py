"""
ZJetMinimalProcessor
====================
Bare-bones Z+jet processor for diagnosing purity/stability.

NO corrections, NO systematics, NO reweighting.
Goal: produce a 2D response matrix (reco jet mass vs gen jet mass) in a given
jet pT bin using only the minimal cuts needed to select Z+jet events.

If purity/stability look bad here, the problem is in the event selection or
binning — not in the corrections stack.
"""

import awkward as ak
import numpy as np
import hist
from coffea import processor
from coffea.nanoevents import NanoAODSchema


class ZJetMinimalProcessor(processor.ProcessorABC):
    """
    Minimal Z(->mumu)+jet processor.

    Parameters
    ----------
    pt_lo, pt_hi : float
        Reco jet pT range to select (GeV).  Default: 290–400 GeV.
    mass_bins : list[float]
        Bin edges for the jet mass axis (applied to both reco and gen).
    """

    def __init__(
        self,
        pt_lo: float = 290.0,
        pt_hi: float = 400.0,
        mass_bins: list | None = None,
    ):
        self.pt_lo = pt_lo
        self.pt_hi = pt_hi
        self.mass_bins = mass_bins or [
            20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
        ]

    # ------------------------------------------------------------------
    def process(self, events):
        NanoAODSchema.warn_missing_crossrefs = False

        # ---- 1. Muon selection (tight Z→μμ) --------------------------
        mu = events.Muon
        tight_mu = mu[
            (mu.pt > 20.0)
            & (np.abs(mu.eta) < 2.4)
            & mu.tightId
            & (mu.pfRelIso04_all < 0.15)
        ]

        # Require ≥2 tight muons with opposite charge summing to zero
        has_os   = (ak.num(tight_mu) >= 2) & (ak.sum(tight_mu.charge, axis=1) == 0)
        events   = events[has_os]
        tight_mu = tight_mu[has_os]

        # Build Z candidate from the two leading tight muons
        mu1 = tight_mu[:, 0]
        mu2 = tight_mu[:, 1]
        Z   = mu1 + mu2

        # Z mass window: 71–111 GeV
        z_ok    = (Z.mass > 71.0) & (Z.mass < 111.0)
        events  = events[z_ok]
        Z       = Z[z_ok]
        mu1     = mu1[z_ok]
        mu2     = mu2[z_ok]
        tight_mu = tight_mu[z_ok]

        # ---- 2. Reco jet selection with explicit lepton cleaning ------
        jets = events.Jet
        base_jets = jets[
            (jets.pt > 30.0)
            & (np.abs(jets.eta) < 2.4)
            & (jets.jetId >= 2)           # tight jet ID
        ]

        # Lepton cleaning: remove jets within ΔR < 0.4 of any tight muon.
        # metric_table returns shape (n_events, n_jets, n_muons); ak.all over
        # axis=2 requires ALL muons to be > 0.4 away.
        clean_jets = base_jets[
            ak.all(base_jets.metric_table(tight_mu) > 0.4, axis=2)
        ]

        # Require ≥1 clean jet; take the leading one
        has_jet   = ak.num(clean_jets) >= 1
        events    = events[has_jet]
        Z         = Z[has_jet]
        mu1       = mu1[has_jet]
        mu2       = mu2[has_jet]
        tight_mu  = tight_mu[has_jet]
        clean_jets = clean_jets[has_jet]
        lead_jet  = clean_jets[:, 0]

        # Back-to-back with Z: |ΔΦ(jet, Z)| > 2.7
        dphi_ok  = np.abs(lead_jet.delta_phi(Z)) > 2.7
        events   = events[dphi_ok]
        Z        = Z[dphi_ok]
        tight_mu = tight_mu[dphi_ok]
        lead_jet = lead_jet[dphi_ok]

        # Jet pT bin selection
        pt_ok    = (lead_jet.pt > self.pt_lo) & (lead_jet.pt < self.pt_hi)
        events   = events[pt_ok]
        Z        = Z[pt_ok]
        tight_mu = tight_mu[pt_ok]
        lead_jet = lead_jet[pt_ok]

        # ---- 3. Histograms -------------------------------------------
        mass_axis_reco = hist.axis.Variable(
            self.mass_bins, name="reco_mass", label="Reco Jet Mass [GeV]"
        )
        mass_axis_gen = hist.axis.Variable(
            self.mass_bins, name="gen_mass", label="Gen Jet Mass [GeV]"
        )

        response_hist = hist.Hist(mass_axis_reco, mass_axis_gen)
        reco_mass_1d  = hist.Hist(mass_axis_reco)
        gen_mass_1d   = hist.Hist(
            hist.axis.Variable(self.mass_bins, name="gen_mass", label="Gen Jet Mass [GeV]")
        )

        is_mc = hasattr(events, "GenJet")

        if is_mc:
            # ---- 4. Gen-level lepton cleaning ------------------------
            gen_jets = events.GenJet
            base_gen_jets = gen_jets[
                (gen_jets.pt > 30.0)
                & (np.abs(gen_jets.eta) < 2.4)
            ]

            # Select gen-level muons: last-copy status flag (bit 13), pT > 10 GeV
            gen_parts = events.GenPart
            gen_mu = gen_parts[
                (np.abs(gen_parts.pdgId) == 13)
                & (gen_parts.pt > 10.0)
                & ((gen_parts.statusFlags >> 13) & 1 == 1)  # isLastCopy
            ]

            # Clean gen jets: remove those within ΔR < 0.4 of any gen muon.
            # metric_table shape: (n_events, n_gen_jets, n_gen_muons)
            clean_gen_jets = base_gen_jets[
                ak.all(base_gen_jets.metric_table(gen_mu) > 0.4, axis=2)
            ]

            # ---- 5. Gen-reco matching with gen-cleaning verification --
            # Step A: NanoAOD built-in cross-reference
            matched_gen_raw = lead_jet.matched_gen   # None when unmatched

            # Step B: keep only matches where the gen jet also survives
            #         the gen-level lepton cleaning.
            # Build a per-event boolean mask: is the matched gen jet present
            # in clean_gen_jets?  We compare by index in the original GenJet array.
            # NanoAOD genJetIdx gives the index into events.GenJet; we check
            # whether that index is in the set of surviving gen-jet indices.
            matched_gen_idx = lead_jet.genJetIdx      # -1 when unmatched

            # Indices of gen jets that survived lepton cleaning (per event)
            # ak.local_index gives the position in the *base* (pt>30, |eta|<2.4)
            # collection, not the original GenJet array.  Use the original array
            # index instead via ak.local_index on the original collection.
            orig_gen_idx = ak.local_index(gen_jets, axis=1)  # 0,1,2,... per event

            # Surviving gen jets have these original indices
            surviving_idx = orig_gen_idx[
                (gen_jets.pt > 30.0)
                & (np.abs(gen_jets.eta) < 2.4)
                & ak.all(gen_jets.metric_table(gen_mu) > 0.4, axis=2)
            ]

            # For each event, check if matched_gen_idx appears in surviving_idx
            gen_match_survives = ak.any(
                surviving_idx == matched_gen_idx[:, np.newaxis], axis=1
            )

            # Unmatched events (genJetIdx == -1) trivially fail the survival check
            has_raw_match    = matched_gen_idx >= 0
            has_clean_match  = has_raw_match & gen_match_survives

            # ---- 6. Diagnostic: matching fraction before/after cleaning
            n_total      = len(lead_jet)
            n_raw_match  = int(ak.sum(has_raw_match))
            n_clean_match = int(ak.sum(has_clean_match))
            print(
                f"[chunk] events={n_total}  "
                f"raw_match={n_raw_match} ({n_raw_match/max(n_total,1):.3f})  "
                f"clean_match={n_clean_match} ({n_clean_match/max(n_total,1):.3f})  "
                f"lost_to_gen_lep_cleaning={n_raw_match - n_clean_match}"
            )

            # ---- 7. Fill histograms -----------------------------------
            w_all = np.sign(ak.to_numpy(events.genWeight))

            # Response: only events with a clean gen match
            matched_reco_j = lead_jet[has_clean_match]
            matched_gen_j  = matched_gen_raw[has_clean_match]
            w_matched      = w_all[ak.to_numpy(has_clean_match)]

            response_hist.fill(
                reco_mass=ak.to_numpy(matched_reco_j.mass),
                gen_mass=ak.to_numpy(matched_gen_j.mass),
                weight=w_matched,
            )

            reco_mass_1d.fill(
                reco_mass=ak.to_numpy(lead_jet.mass),
                weight=w_all,
            )

            gen_mass_1d.fill(
                gen_mass=ak.to_numpy(matched_gen_raw[has_clean_match].mass),
                weight=w_all[ak.to_numpy(has_clean_match)],
            )

        else:
            # Data: reco only
            reco_mass_1d.fill(reco_mass=ak.to_numpy(lead_jet.mass))

        return {
            "response":  response_hist,
            "reco_mass": reco_mass_1d,
            "gen_mass":  gen_mass_1d,
            "n_events":  len(events),
        }

    def postprocess(self, accumulator):
        return accumulator
