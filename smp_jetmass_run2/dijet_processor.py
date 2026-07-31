 #### This file contains the processors for dijet and trijet hist selections. Plotting and resulting studies are in separate files.
#### LMH

import argparse

import awkward as ak
import numpy as np
import os
import re
import pandas as pd
import zlib

from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from collections import defaultdict
from .corrections import *
from .hist_utils import util_binning, register_hist, fill_hist
from copy import deepcopy
import hist
import time


class Log:
    def __init__(self, mode="info"):
        self.mode = mode
    def info(self, *msg):
        if self.mode in ["info", "debug"]:
            print("[INFO]", *msg)
    def debug(self, *msg):
        if self.mode == "debug":
            print("[DEBUG]", *msg)


#### currently only for MC --> makes hists and response matrix
class DijetProcessor(processor.ProcessorABC):
    '''
    Processor to run a dijet jet mass cross section analysis.
    With "do_gen == True", will perform GEN selection and create response matrices.
    Will always plot RECO level quantities.

    Ported from GluonJetMass python/dijetProcessor.py to the zjet QJetMassProcessor
    coding conventions. The event selection is byte-for-byte identical to the
    original; the intentional changes are:
      - HEM handling (weight-based for 2018 MC, hard veto for 2018 data);
      - canonical systematic-axis names (PUSF->pu, L1prefiring->l1prefiring,
        PDF->pdf, ISR->isr, FSR->fsr);
      - unfolding-hist fill semantics follow zjet, not GluonJetMass: gen hists
        are the FULL gen truth (all genTot_seq events, no matched_gen
        requirement) and reco hists the FULL reco spectrum (all recoTot_seq
        events, no matched_reco requirement), so the unfolder's subtraction
        misses = gen - response / fakes = reco - response is complete;
      - gen hists are also filled per weight systematic with theory-only
        partial weights (isr/fsr/pdf/Q2 move the gen truth, detector
        systematics leave it nominal), mirroring zjet.
    '''
    #### Weight systematics that genuinely change the gen-level prediction
    #### (theory reweightings, nominal weight 1.0). Under these the gen truth,
    #### the response and the misses must move consistently; every other weight
    #### systematic is detector-side and leaves the gen truth nominal
    #### (mirrors zjet's _theory_gen_systs; dijet keeps the split Q2muF/Q2muR).
    _THEORY_GEN_SYSTS = frozenset({
        "isrUp", "isrDown", "fsrUp", "fsrDown", "pdfUp", "pdfDown",
        "Q2muFUp", "Q2muFDown", "Q2muRUp", "Q2muRDown",
    })

    def __init__(self, do_gen=True, mode="minimal", debug=False,
                 jet_systematics=None, systematics=None,
                 ptcut=200., ycut=2.5, jk=False, jk_range=None,
                 trijet_veto=True):
        # should have separate **lower** ptcut for gen
        self.do_gen = do_gen
        self._mode = mode
        #### Trijet-priority orthogonality veto (2026-07-31): dijet drops any
        #### event inside the trijet measurement phase space, so the two
        #### hadronic channels never share an event. trijet_veto=False restores
        #### the GluonJetMass-inherited selection (used by the cutflow
        #### regression). Study: scripts/channel_orthogonality.py.
        self.trijet_veto = trijet_veto
        # diagnostics filled only in validation/full (mirrors zjet's mode gating)
        self.do_minimal = self._mode not in ("validation", "full")
        self.ptcut = ptcut
        self.ycut = ycut  # rapidity
        self.jk = jk
        self.jk_range = jk_range
        # jackknife active either via the jk flag or the *_jk modes (mirrors zjet)
        self._do_jk = bool(self.jk) or self._mode in ("mass_jk", "rho_jk")
        # Herwig/Pythia rho reweighting (mirrors zjet's "reweight_pythia_rho").
        # Only rho is available for dijet (no mass herwig inputs). Requires gen.
        self._do_reweight = self._mode == "reweight_pythia_rho"
        #### Flat per-jet ntuple for binning studies (rho resolution vs pt/eta,
        #### purity/stability at arbitrary edges). Nominal jet systematic only,
        #### MC only. Selections are untouched -- this is a pure extra output.
        #### "rho_skim" keeps every event; "rho_skimN" keeps 1 event in N
        #### (deterministic on the event number, so the three tables stay
        #### consistent and the choice is reproducible across chunks/reruns)
        #### and scales the stored weight by N so normalization is preserved.
        self._do_skim = self._mode.startswith("rho_skim")
        self._skim_prescale = 1
        if self._do_skim and self._mode != "rho_skim":
            try:
                self._skim_prescale = int(self._mode[len("rho_skim"):])
            except ValueError:
                raise ValueError(f"Bad skim mode '{self._mode}': expected "
                                 "'rho_skim' or 'rho_skim<N>', e.g. rho_skim20")
            if self._skim_prescale < 1:
                raise ValueError(f"Skim prescale must be >= 1, got {self._skim_prescale}")
        if jet_systematics is None:
            jet_systematics = ['nominal', 'JERUp', 'JERDown', 'JMSUp', 'JMSDown',
                               'JMRUp', 'JMRDown']
        if self._do_jk:
            # protect against doing unc for jk --> only need nominal and memory intensive
            jet_systematics = ["nominal"]
        self.jet_systematics = jet_systematics
        self.systematics = systematics
        self.logging = Log(mode="debug" if debug else "info")
        self.logging.info(f"Dijet do_gen={self.do_gen} mode={self._mode} do_jk={self._do_jk}")

        #### Binning: single source of truth (channel-aware). jet radius for rho.
        b = util_binning(channel="dijet")
        self._jetR    = b.jetR
        dataset_axis  = b.dataset_axis
        syst_cat      = b.syst_axis
        jk_axis       = b.jackknife_axis
        pt_bin        = b.ptreco_axis
        pt_gen_bin    = b.ptgen_axis
        mass_bin      = b.mreco_axis
        mass_gen_bin  = b.mgen_axis
        rho_bin       = b.mreco_over_pt_axis
        rho_gen_bin   = b.mgen_over_pt_axis
        rho_g_bin     = b.mreco_over_pt_g_axis
        rho_gen_g_bin = b.mgen_over_pt_g_axis
        #### diagnostic-only axes (validation/full)
        jet_cat   = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat= hist.axis.StrCategory([], growth=True, name="partonFlav", label="Parton Flavour")
        fine_mass_bin = hist.axis.Regular(500, 0.0, 1000.0, name="mass", label=r"mass [GeV]")
        fine_pt_bin   = hist.axis.Regular(400, 0.0, 8000.0, name="pt", label=r"$p_T$ [GeV]")
        eta_bin   = hist.axis.Regular(25, -4.0, 4.0, name="eta", label=r"$\eta$")
        frac_axis = hist.axis.Regular(400, 0, 2.5, name="frac", label="Fraction")
        phi_axis  = hist.axis.Regular(25, -np.pi, np.pi, name="phi", label=r"$\phi$")

        #### jk axis only present in jackknife modes (mirrors zjet)
        jk_axes = [jk_axis] if self._do_jk else []

        # NOTE: keep this a PLAIN dict (not processor.dict_accumulator). coffea
        # reduces a plain-dict accumulator with recursive MutableMapping merging,
        # which correctly handles the nested plain `cutflow` dict; a dict_accumulator
        # would instead do `cutflow += cutflow` and crash on multi-chunk reduction.
        # Hist / column_accumulator / defaultdict_accumulator leaves are Addable.
        self.hists = {}
        self.hists['cutflow'] = {}
        self.hists['jkflow'] = processor.defaultdict_accumulator(int)
        #### DATA only: log (run, lumi, event) of finally selected events so the
        #### three channels (zjet/dijet/trijet) can be checked for orthogonality.
        if not self.do_gen:
            self.hists['event_id'] = processor.dict_accumulator({
                'run': processor.column_accumulator(np.array([], dtype=np.int64)),
                'luminosityBlock': processor.column_accumulator(np.array([], dtype=np.int64)),
                'event': processor.column_accumulator(np.array([], dtype=np.int64)),
            })

        #### Binning-study ntuple: three flat per-jet tables mirroring the three
        #### unfolding-hist families, so purity/stability/fakes/misses can be
        #### recomputed at ANY bin edges (the hists can only ever be merged).
        ####   matched -> final_seq  (response matrix)
        ####   reco    -> recoTot_seq (reco spectrum, fakes included)
        ####   gen     -> genTot_seq  (gen spectrum, misses included)
        #### DATA fills only `reco` (and the event-display tables); `matched` and
        #### `gen` are still BOOKED so a data pkl and an MC pkl have an identical
        #### accumulator schema and merge/read back without special-casing --
        #### they just stay empty. For data final_seq == recoTot_seq (see the
        #### `sel.add` in the data branch), so the reco table means the same
        #### thing in both and data/MC can be compared bin-for-bin.
        if self._do_skim:
            tables = {
                t: processor.dict_accumulator(
                    {c: processor.column_accumulator(np.array([], dtype=d))
                     for c, d in cols}
                )
                for t, cols in (
                    ("matched", self._SKIM_RECO_COLS + self._SKIM_GEN_COLS
                                + self._SKIM_COMMON_COLS),
                    ("reco",    self._SKIM_RECO_COLS + self._SKIM_COMMON_COLS),
                    ("gen",     self._SKIM_GEN_COLS + self._SKIM_COMMON_COLS),
                )
            }
            #### dataset is stored as a 4-byte crc32 id, not a per-row string:
            #### an object column of N identical strings costs ~90 B/row once
            #### pickled, more than every physics column put together. The
            #### id -> name map is this set.
            tables['datasets'] = processor.set_accumulator()
            #### Event-display tables: EVERY AK8 jet (not just the leading two)
            #### plus a per-event cut bitmask, so an efficiency loss can be traced
            #### to the jet that caused it. Prescaled _SKIM_EVENT_FACTOR harder
            #### than the physics tables -- these are for looking at events, not
            #### for statistics (the cutflow is already exact).
            for t, cols in (("events",     self._SKIM_EVENT_COLS),
                            ("alljets",    self._SKIM_ALLJET_COLS),
                            ("allgenjets", self._SKIM_ALLGENJET_COLS)):
                tables[t] = processor.dict_accumulator(
                    {c: processor.column_accumulator(np.array([], dtype=d))
                     for c, d in cols}
                )
            self.hists['skim'] = processor.dict_accumulator(tables)

        mass_modes = ("minimal", "mass_jk", "validation", "full")
        #### rho_skim also books the standard rho hists: they are cheap and give
        #### a free closure test of the ntuple against the histogram path.
        rho_modes  = ("minimal_rho", "rho_jk", "reweight_pythia_rho", "validation", "full")

        #### Mass unfolding inputs
        if self._mode in mass_modes:
            register_hist(self.hists, 'ptjet_mjet_u_reco', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin])
            register_hist(self.hists, 'ptjet_mjet_g_reco', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin])
            if self.do_gen:
                register_hist(self.hists, 'ptjet_mjet_u_gen', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, mass_gen_bin])
                register_hist(self.hists, 'ptjet_mjet_g_gen', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, mass_gen_bin])
                register_hist(self.hists, 'response_matrix_u', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin])
                register_hist(self.hists, 'response_matrix_g', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin])

        #### Rho unfolding inputs
        if self._mode in rho_modes or self._do_skim:
            register_hist(self.hists, 'ptjet_rhojet_u_reco', [dataset_axis, syst_cat, *jk_axes, pt_bin, rho_bin])
            register_hist(self.hists, 'ptjet_rhojet_g_reco', [dataset_axis, syst_cat, *jk_axes, pt_bin, rho_g_bin])
            #### Full-stat diagnostic for the historical m_g > 2 GeV floor.
            #### Separate keys keep the adopted no-floor result untouched.
            register_hist(self.hists, 'ptjet_rhojet_g_reco_mfloor2', [dataset_axis, syst_cat, *jk_axes, pt_bin, rho_g_bin])
            #### Event-clustered covariance of the reco (pt, rho) spectrum:
            #### V_ij = sum_e w_e^2 n_{e,i} n_{e,j}. Two jets per event make the
            #### input covariance non-diagonal (~10% off-diag correlations, up to
            #### ~15% same-bin variance inflation on the UL18 QCD test file) --
            #### this is EXACT for a Poisson event count, no bootstrap needed.
            #### Filled nominal-only; 4D (pt_i, rho_i, pt_j, rho_j) by value so
            #### the axes' own flow handles out-of-range jets. Channel-specific:
            #### zjet fills one jet/event (its V is exactly diagonal sumw2);
            #### trijet will adopt it with its fill-semantics port. NB scales as
            #### scale^2 under XS normalization -- see postprocess.
            if not self._do_jk:
                def _axis_copy(ax, name):
                    return hist.axis.Variable(ax.edges, name=name, label=ax.label)
                register_hist(self.hists, 'reco_cov_rho_u', [dataset_axis, syst_cat,
                    _axis_copy(pt_bin, 'ptreco_i'), _axis_copy(rho_bin, 'mpt_reco_i'),
                    _axis_copy(pt_bin, 'ptreco_j'), _axis_copy(rho_bin, 'mpt_reco_j')])
                register_hist(self.hists, 'reco_cov_rho_g', [dataset_axis, syst_cat,
                    _axis_copy(pt_bin, 'ptreco_i'), _axis_copy(rho_g_bin, 'mpt_reco_i'),
                    _axis_copy(pt_bin, 'ptreco_j'), _axis_copy(rho_g_bin, 'mpt_reco_j')])
                register_hist(self.hists, 'reco_cov_rho_g_mfloor2', [dataset_axis, syst_cat,
                    _axis_copy(pt_bin, 'ptreco_i'), _axis_copy(rho_g_bin, 'mpt_reco_i'),
                    _axis_copy(pt_bin, 'ptreco_j'), _axis_copy(rho_g_bin, 'mpt_reco_j')])
            if self.do_gen:
                register_hist(self.hists, 'ptjet_rhojet_u_gen', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, rho_gen_bin])
                register_hist(self.hists, 'ptjet_rhojet_g_gen', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, rho_gen_g_bin])
                register_hist(self.hists, 'ptjet_rhojet_g_gen_mfloor2', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, rho_gen_g_bin])
                register_hist(self.hists, 'response_matrix_rho_u', [dataset_axis, syst_cat, *jk_axes, pt_bin, pt_gen_bin, rho_bin, rho_gen_bin])
                register_hist(self.hists, 'response_matrix_rho_g', [dataset_axis, syst_cat, *jk_axes, pt_bin, pt_gen_bin, rho_g_bin, rho_gen_g_bin])
                register_hist(self.hists, 'response_matrix_rho_g_mfloor2', [dataset_axis, syst_cat, *jk_axes, pt_bin, pt_gen_bin, rho_g_bin, rho_gen_g_bin])

        #### Diagnostics (validation / full only)
        if self._mode in ("validation", "full"):
            register_hist(self.hists, 'misses_u', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, mass_gen_bin])
            register_hist(self.hists, 'misses_g', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, mass_gen_bin])
            register_hist(self.hists, 'fakes_u', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin])
            register_hist(self.hists, 'fakes_g', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin])
            register_hist(self.hists, 'alljet_ptreco_mreco', [dataset_axis, jet_cat, parton_cat, mass_bin, pt_bin])
            register_hist(self.hists, 'btag_eta', [dataset_axis, jet_cat, parton_cat, frac_axis, eta_bin])
            register_hist(self.hists, 'HT_nocuts', [dataset_axis, syst_cat, fine_pt_bin])
            register_hist(self.hists, 'HT_wXS', [dataset_axis, syst_cat, fine_pt_bin])
            register_hist(self.hists, 'HT_aftercuts', [dataset_axis, syst_cat, fine_pt_bin])
            register_hist(self.hists, 'MET_over_sumET_pt_reco', [dataset_axis, syst_cat, frac_axis, pt_bin])
            register_hist(self.hists, 'MET_pt_reco', [dataset_axis, syst_cat, fine_pt_bin, pt_bin])
            register_hist(self.hists, 'jet_eta_phi_precuts', [dataset_axis, syst_cat, phi_axis, eta_bin])
            register_hist(self.hists, 'jet_eta_phi_preveto', [dataset_axis, syst_cat, phi_axis, eta_bin])
            register_hist(self.hists, 'jet_pt_eta_phi', [dataset_axis, syst_cat, pt_bin, phi_axis, eta_bin])
            register_hist(self.hists, 'asymm_gen', [dataset_axis, syst_cat, pt_gen_bin, frac_axis])
            register_hist(self.hists, 'asymm_reco', [dataset_axis, syst_cat, pt_bin, frac_axis])
            register_hist(self.hists, 'm_u_jet_reco_over_gen', [dataset_axis, pt_gen_bin, mass_gen_bin, frac_axis])
            register_hist(self.hists, 'm_g_jet_reco_over_gen', [dataset_axis, pt_gen_bin, mass_gen_bin, frac_axis])

    @property
    def accumulator(self):
        return self.hists

    def _rho(self, mass, pt):
        #### rho = 2*log10(m/(pt*R)) -- matches the zjet QJetMassProcessor definition
        #### (includes the jet radius R), not the bare 2*log10(m/pt).
        return 2 * np.log10(mass / (pt * self._jetR))

    #############################################################
    #### Binning-study skim ("rho_skim" mode)
    #############################################################
    #### One row per JET. Column sets are shared between the three tables so a
    #### single loader can stack them; `weight` is raw genWeight*SF here and gets
    #### the xs*lumi/sumw scale in postprocess (XS-in-postprocess is a hard
    #### repo invariant -- see AGENTS.md).
    _SKIM_RECO_COLS = (
        ("ptreco",  np.float32), ("etareco", np.float32), ("phireco", np.float32),
        ("mreco_u", np.float32), ("mreco_g", np.float32),
        ("nconst",  np.int16),   ("nsub",    np.int8),
        ("zg",      np.float32), ("Rg",      np.float32),
    )
    _SKIM_GEN_COLS = (
        ("ptgen",  np.float32), ("etagen", np.float32), ("phigen", np.float32),
        ("mgen_u", np.float32), ("mgen_g", np.float32),
    )
    _SKIM_COMMON_COLS = (
        ("weight", np.float64), ("dataset_id", np.uint32), ("jetidx", np.int8),
        ("pu_rho", np.float32), ("npv", np.int16),
    )

    #### Event-display tables. (run, lumi, event, dataset_id) is the join key
    #### between `events` and the two per-jet tables -- these are ROW-PER-JET
    #### over the FULL collection, so the third/fourth jet that broke the
    #### dijet topology is visible instead of silently cut away.
    _SKIM_EVENT_KEY = (
        ("run", np.uint32), ("lumi", np.uint32), ("event", np.int64),
        ("dataset_id", np.uint32),
    )
    _SKIM_EVENT_COLS = _SKIM_EVENT_KEY + (
        ("weight", np.float64), ("npv", np.int16), ("pu_rho", np.float32),
        ("njet_reco", np.int16), ("njet_gen", np.int16),
        ("ht_reco", np.float32), ("ht_gen", np.float32),
        ("met", np.float32),
        #### bit i of cutbits == event passed _SKIM_CUTS[i]
        ("cutbits", np.uint32),
    )
    _SKIM_ALLJET_COLS = _SKIM_EVENT_KEY + (
        ("jetidx", np.int16), ("pt", np.float32), ("eta", np.float32),
        ("phi", np.float32), ("mass", np.float32), ("msoftdrop", np.float32),
        ("jetId", np.int16), ("nconst", np.int16),
    )
    _SKIM_ALLGENJET_COLS = _SKIM_EVENT_KEY + (
        ("jetidx", np.int16), ("pt", np.float32), ("eta", np.float32),
        ("phi", np.float32), ("mass", np.float32),
    )
    #### Order is the bit order in `cutbits` -- APPEND only, never reorder, or
    #### previously written skims decode wrong.
    _SKIM_CUTS = (
        "npv", "METfilters",
        "twoRecoJet", "recoRap2p5", "recodphi2", "recoAsym0p3",
        "muonIso0p4", "jetId", "hemveto", "recoTot_seq",
        "twoGenJet", "genRap2p5", "dphiGen2", "genAsym0p3", "genTot_seq",
        "matched_gen", "matched_reco", "final_seq",
        #### appended 2026-07-31 (trijet-priority orthogonality veto; True =
        #### event NOT in the trijet phase space). Skims written before then
        #### leave these bits 0.
        "vetoTrijetReco", "vetoTrijetGen",
    )
    #### The event tables are for LOOKING at events, not for statistics, so they
    #### are prescaled this much harder than the physics tables on top of the
    #### mode's own prescale (rho_skim10 -> 1 event in 1000 here).
    _SKIM_EVENT_FACTOR = 100

    @staticmethod
    def _skim_dataset_id(dataset):
        return np.uint32(zlib.crc32(dataset.encode()))

    @staticmethod
    def _skim_f32(x):
        #### None -> NaN so downstream masks are a single np.isfinite
        return ak.to_numpy(ak.fill_none(x, np.nan)).astype(np.float32)

    def _skim_subjet_cols(self, jets, n):
        #### zg/Rg need the SubJet cross-reference, which does not always survive
        #### the corrected-jet rebuild. Degrade to NaN rather than crash -- these
        #### are diagnostic columns, not unfolding inputs.
        nsub = np.full(n, -1, dtype=np.int8)
        zg = np.full(n, np.nan, dtype=np.float32)
        rg = np.full(n, np.nan, dtype=np.float32)
        try:
            sj = jets.subjets
            nsub = ak.to_numpy(ak.num(sj, axis=1)).astype(np.int8)
            sj2 = ak.pad_none(sj, 2, clip=True)
            p1, p2 = sj2[:, 0], sj2[:, 1]
            a, b = self._skim_f32(p1.pt), self._skim_f32(p2.pt)
            with np.errstate(invalid="ignore"):
                zg = (np.minimum(a, b) / (a + b)).astype(np.float32)
            rg = self._skim_f32(p1.delta_r(p2))
        except Exception as exc:
            self.logging.debug(f"rho_skim: subjet columns unavailable ({exc})")
        return nsub, zg, rg

    def _fill_skim(self, out, table, dataset, weights, evt,
                   reco=None, gen=None, gen_groomed=None):
        """Append one block of per-jet rows to out['skim'][table].

        `weights`/`reco`/`gen` are already flattened to 2 jets per event, in
        (leading, subleading) order -- the same ordering every hist fill uses.
        """
        n = len(weights)
        if n == 0:
            return
        #### prescale on the EVENT number so both jets of an event survive or
        #### neither, and so the decision is identical in all three tables
        keep = None
        if self._skim_prescale > 1:
            ev = ak.to_numpy(evt.event).astype(np.int64)
            keep = np.repeat(ev % self._skim_prescale == 0, 2)
            if not keep.any():
                return
        cols = {
            "weight":     np.asarray(weights, dtype=np.float64) * self._skim_prescale,
            "dataset_id": np.full(n, self._skim_dataset_id(dataset), dtype=np.uint32),
            "jetidx":     np.tile(np.array([0, 1], dtype=np.int8), n // 2),
        }
        for name, src, dtype in (("pu_rho", "fixedGridRhoFastjetAll", np.float32),
                                 ("npv", "PV", np.int16)):
            try:
                v = evt.PV.npvsGood if src == "PV" else evt[src]
                cols[name] = np.repeat(ak.to_numpy(ak.fill_none(v, -1)), 2).astype(dtype)
            except Exception:
                cols[name] = np.full(n, -1, dtype=dtype)
        if reco is not None:
            nsub, zg, rg = self._skim_subjet_cols(reco, n)
            cols.update(
                ptreco=self._skim_f32(reco.pt), etareco=self._skim_f32(reco.eta),
                phireco=self._skim_f32(reco.phi), mreco_u=self._skim_f32(reco.mass),
                mreco_g=self._skim_f32(reco.msoftdrop),
                nconst=ak.to_numpy(ak.fill_none(reco.nConstituents, -1)).astype(np.int16),
                nsub=nsub, zg=zg, Rg=rg,
            )
        if gen is not None:
            cols.update(
                ptgen=self._skim_f32(gen.pt), etagen=self._skim_f32(gen.eta),
                phigen=self._skim_f32(gen.phi), mgen_u=self._skim_f32(gen.mass),
                mgen_g=self._skim_f32(gen_groomed.mass),
            )
        out['skim']['datasets'].add(dataset)
        acc = out['skim'][table]
        for name, arr in cols.items():
            arr = np.asarray(arr)
            acc[name] += processor.column_accumulator(arr if keep is None else arr[keep])

    def _fill_skim_event(self, out, dataset, weights, evt, sel):
        """Append the event-display block: one `events` row per event plus one
        `alljets`/`allgenjets` row per jet in the FULL collections.

        Filled once per chunk at the point where every cut in _SKIM_CUTS is
        defined, so `cutbits` records exactly where each event died.
        """
        n = len(evt)
        if n == 0:
            return
        stride = self._skim_prescale * self._SKIM_EVENT_FACTOR
        ev = ak.to_numpy(evt.event).astype(np.int64)
        keep = ev % stride == 0
        if not keep.any():
            return
        did = self._skim_dataset_id(dataset)

        def key(counts=None):
            #### (run, lumi, event, dataset_id) -- repeated per jet when counts given
            k = {"run": ak.to_numpy(evt.run).astype(np.uint32)[keep],
                 "lumi": ak.to_numpy(evt.luminosityBlock).astype(np.uint32)[keep],
                 "event": ev[keep],
                 "dataset_id": np.full(int(keep.sum()), did, dtype=np.uint32)}
            if counts is None:
                return k
            return {c: np.repeat(v, counts) for c, v in k.items()}

        #### cut bitmask. A cut absent from this path (gen cuts in data) stays 0.
        cutbits = np.zeros(n, dtype=np.uint32)
        for i, cut in enumerate(self._SKIM_CUTS):
            if cut not in sel.names:
                continue
            cutbits |= (ak.to_numpy(sel.all(cut)).astype(np.uint32) << np.uint32(i))

        def num(coll):
            try:
                return ak.to_numpy(ak.num(evt[coll], axis=1)).astype(np.int64)
            except Exception:
                return np.zeros(n, dtype=np.int64)

        def ht(coll):
            try:
                return self._skim_f32(ak.sum(evt[coll].pt, axis=-1))
            except Exception:
                return np.full(n, np.nan, dtype=np.float32)

        try:
            met = self._skim_f32(evt.MET.pt)
        except Exception:
            met = np.full(n, np.nan, dtype=np.float32)
        try:
            purho = ak.to_numpy(ak.fill_none(evt.fixedGridRhoFastjetAll, -1)).astype(np.float32)
        except Exception:
            purho = np.full(n, -1, dtype=np.float32)

        ev_cols = dict(
            key(),
            weight=np.asarray(weights, dtype=np.float64)[keep] * stride,
            npv=ak.to_numpy(ak.fill_none(evt.PV.npvsGood, -1)).astype(np.int16)[keep],
            pu_rho=purho[keep],
            njet_reco=num("FatJet")[keep].astype(np.int16),
            njet_gen=num("GenJetAK8")[keep].astype(np.int16),
            ht_reco=ht("FatJet")[keep], ht_gen=ht("GenJetAK8")[keep],
            met=met[keep], cutbits=cutbits[keep],
        )

        blocks = [("events", ev_cols)]
        for table, coll, extra in (("alljets", "FatJet", True),
                                   ("allgenjets", "GenJetAK8", False)):
            try:
                jets = evt[coll][keep]
            except Exception:
                continue
            counts = ak.to_numpy(ak.num(jets, axis=1)).astype(np.int64)
            flat = ak.flatten(jets, axis=1)
            if len(flat) == 0:
                continue
            cols = dict(
                key(counts),
                #### index WITHIN the event, so jetidx>1 is the extra-jet activity
                jetidx=ak.to_numpy(ak.flatten(ak.local_index(jets, axis=1))).astype(np.int16),
                pt=self._skim_f32(flat.pt), eta=self._skim_f32(flat.eta),
                phi=self._skim_f32(flat.phi), mass=self._skim_f32(flat.mass),
            )
            if extra:
                cols.update(
                    msoftdrop=self._skim_f32(flat.msoftdrop),
                    jetId=ak.to_numpy(ak.fill_none(flat.jetId, -1)).astype(np.int16),
                    nconst=ak.to_numpy(ak.fill_none(flat.nConstituents, -1)).astype(np.int16),
                )
            blocks.append((table, cols))

        out['skim']['datasets'].add(dataset)
        for table, cols in blocks:
            acc = out['skim'][table]
            for name, arr in cols.items():
                acc[name] += processor.column_accumulator(np.asarray(arr))

    def _scale_skim_weights(self, accumulator, iov_of):
        #### Same xs*lumi*1000/sumw scale the hists get, applied per dataset to
        #### the flat `weight` column (postprocess only walks hist.Hist objects).
        skim = accumulator.get('skim')
        if skim is None:
            return
        names = sorted(skim.get('datasets', set()))
        for table, acc in skim.items():
            #### alljets/allgenjets carry no weight -- they join to `events` on
            #### (run, lumi, event, dataset_id) and inherit its weight.
            if table == 'datasets' or 'weight' not in acc:
                continue
            w = np.asarray(acc['weight'].value, dtype=np.float64)
            if len(w) == 0:
                continue
            ids = np.asarray(acc['dataset_id'].value, dtype=np.uint32)
            for d in names:
                scale = getXSweight(d, iov_of(d))
                if scale is None or scale == 1.0:
                    self.logging.info(f"rho_skim[{table}]: no scale for {d}")
                    continue
                w[ids == self._skim_dataset_id(d)] *= scale
            acc['weight'] = processor.column_accumulator(w)

    def _herwig_rho_weights(self, pt, ungroomed_mass, groomed_mass):
        #### Per-jet Herwig/Pythia weights vs rho for the dijet channel,
        #### mirroring zjet's QJetMassProcessor._get_herwig_reweights (rho mode).
        #### `pt`/`mass` are (possibly awkward) flat per-jet arrays; jets whose
        #### rho is non-finite (e.g. msoftdrop<0) get a neutral weight of 1.
        pt = ak.to_numpy(pt)
        rho_u = ak.to_numpy(self._rho(ungroomed_mass, pt))
        rho_g = ak.to_numpy(self._rho(groomed_mass, pt))
        w_u = get_herwig_weight_u(mode="rho", channel="dijet").weight_array(pt, rho_u)
        w_g = get_herwig_weight_g(mode="rho", channel="dijet").weight_array(pt, rho_g)
        return np.nan_to_num(w_u, nan=1.0), np.nan_to_num(w_g, nan=1.0)

    def _rapidity(self, p4):
        #### Preserve the original GluonJetMass selection helper exactly.
        return 0.5 * np.log((p4.energy + p4.pz) / (p4.energy - p4.pz))

    def _weight_variations(self, weights_obj):
        if self.systematics is None:
            return list(weights_obj.variations)
        return [syst for syst in self.systematics if syst != "nominal" and syst in weights_obj.variations]

    def _fill_reco_cov(self, out, dataset, name, ptvals, rhovals, ok, w_event):
        """Accumulate the event-clustered reco covariance V_ij = sum_e w_e^2 n_ei n_ej.

        ptvals/rhovals/ok are flat per-jet arrays (2 jets per event, event-major
        order, as produced by ak.flatten(FatJet[:,:2])); w_event is the
        per-event weight. For every ordered pair (a, b) of an event's jets --
        including a == b, so a same-bin pair adds 4 w^2 on the diagonal -- fill
        (bin_a, bin_b) with w^2. Fills by VALUE into the 4D hist so binning and
        flow stay consistent with the spectrum fills by construction.
        """
        if name not in out:
            return
        pt2 = np.asarray(ptvals, dtype=float).reshape(-1, 2)
        rho2 = np.asarray(rhovals, dtype=float).reshape(-1, 2)
        ok2 = (np.asarray(ok).reshape(-1, 2)) & np.isfinite(rho2)
        w2 = np.asarray(w_event, dtype=float) ** 2
        for a in (0, 1):
            for b in (0, 1):
                m = ok2[:, a] & ok2[:, b]
                fill_hist(out, name, dataset=dataset, systematic="nominal",
                          ptreco_i=pt2[m, a], mpt_reco_i=rho2[m, a],
                          ptreco_j=pt2[m, b], mpt_reco_j=rho2[m, b],
                          weight=w2[m])

    def process(self, events):
        out = self.hists
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        self.logging.debug("Filename: ", filename)
        self.logging.debug("Dataset: ", dataset)
        if "madgraph" in dataset and "pythia" in dataset:
            mctype="pythiaMG"
        elif "herwig" in dataset:
            mctype='herwig'
        elif "_pythia" in dataset:
            mctype="pythia"
        else:
            mctype="data"    
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'HIPM', dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')
        if 'QCD' in dataset:
            datastr = 'QCD_'+mctype+IOV
        elif "WJets" in dataset:
            datastr = "WJets_"+mctype+IOV
        elif "ZJets" in dataset:
            datastr = 'ZJets_'+mctype+IOV
        elif "TTJets" in dataset:
            datastr = 'TTJets_'+mctype+IOV
        else: datastr = mctype+IOV
        self.logging.debug(datastr)
        out['cutflow'][datastr] = defaultdict(int)
        out['cutflow'][datastr]['nEvents initial ' + dataset] += (len(events.FatJet))
        out['cutflow'][datastr]['nEvents initial (all datasets)'] += (len(events.FatJet))
        if (self.do_gen):
            firstidx = filename.find( "store/mc/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            ht_bin = fname_toks[ fname_toks.index("mc") + 2]
            if "LHEWeight" in events.fields:
                weights = events["LHEWeight"].originalXWGTUP
            else:
                weights = events.genWeight
            out['cutflow'][datastr]['sumw for '+ht_bin] += np.sum(weights)
        index_list = np.arange(len(events))
        ###### Choose number of slices to break data into for jackknife method
        if self._do_jk:
            self.logging.debug("Self.jk ", self.jk)
            range_max = 10
        else: range_max=1
            
            #####################################
            #### Loop through JK slices
            #####################################
        if self.jk_range == None:
            jk_inds = range(0,range_max)
        else:
            jk_inds = range(self.jk_range[0], self.jk_range[1])
            self.logging.debug("Jk indices we'll loop over ", jk_inds)
        for jk_index in jk_inds:
            if self._do_jk:
                self.logging.debug("Now doing jackknife {}".format(jk_index))
                self.logging.debug("Len of events before jk selection ", len(events))
            else:
                jk_index=-1
            #### jk axis is only present on hists in jackknife modes; pass it only then
            jkkw = {'jk': jk_index} if self._do_jk else {}
            # print("range max ", range_max)
            jk_sel = ak.where(index_list%range_max == jk_index, False, True)
            #####################################
            #### Apply JK selection
            #####################################
            events_jk = events[jk_sel]
            # print("Len of events after jk selection ", len(events_jk))
            del jk_sel
            #### only consider pfmuons w/ similar selection to aritra for later jet isolation
            events_jk = ak.with_field(events_jk, 
                                      events_jk.Muon[(events_jk.Muon.mediumId > 0)
                                      &(np.abs(events_jk.Muon.eta) < 2.5)
                                      &(events_jk.Muon.pfIsoId > 1) ], 
                                      "Muon")
            #### require at least one fat jet and one subjet so corrections do not fail
            FatJet=events_jk.FatJet
            FatJet["p4"] = ak.with_name(events_jk.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            if ak.sum(ak.num(FatJet.pt)>0)<1:
                self.logging.debug("No fat jet pts at all")
                return out
            if self.do_gen:
                era = None
                GenJetAK8 = events_jk.GenJetAK8
                GenJetAK8['p4']= ak.with_name(events_jk.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            else:
                firstidx = filename.find("store/data/")
                fname2 = filename[firstidx:]
                fname_toks = fname2.split("/")
                era = fname_toks[ fname_toks.index("data") + 1]
                # print("IOV, era ", IOV, era)
            #####################################
            #### Apply jet corrections
            #####################################
            self.logging.debug("starting jet corrections")
            corrected_fatjets = GetJetCorrections(FatJet, events_jk, era, IOV, isData=not self.do_gen)
            corrected_fatjets = corrected_fatjets[corrected_fatjets.subJetIdx1 > -1]
            self.logging.debug(" Uncorrected subjet mass", events_jk.SubJet.mass)
            self.logging.debug(" Uncorrected jet mass", events_jk.FatJet.mass)
            self.logging.debug("ak sum of subjet mass ", ak.sum(events_jk.SubJet.mass))
            self.logging.debug("ak sum of fat jet mass", ak.sum( events_jk.FatJet.mass))
            if ak.sum(ak.num(events_jk.SubJet.mass)>0)<1:
                self.logging.debug("No subjet pts")
                return out
            self.logging.debug("starting softdrop mass correction")
            corrected_subjets = GetJetCorrections(events_jk.SubJet, events_jk, era, IOV, isData = not self.do_gen, mode = 'AK4')
            corrected_fatjets['msoftdrop'] =   (corrected_subjets[corrected_fatjets.subJetIdx1] + corrected_subjets[corrected_fatjets.subJetIdx2]).mass 
            
            #####################################
            #### Fill plots to compare jet correction techniques
            #####################################
            # if not self.jk:
            #     # corrected_fatjets_ak8 = corrected_fatjets.copy()
            #     # corrected_fatjets_ak8 = GetCorrectedSDMass(corrected_fatjets, events_jk, era, IOV, isData=not self.do_gen, useSubjets = False)
            #     fill_hist(out, "sdmass_orig", dataset=dataset, **jkkw, pt=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].pt, axis=1), mass=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].msoftdrop, axis=1))
            #     fill_hist(out, "mass_orig", dataset=dataset,**jkkw, pt=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].pt, axis=1), mass=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].mass, axis=1))
            #     fill_hist(out, "sdmass_ak4corr", dataset=dataset, **jkkw, pt=ak.flatten(corrected_fatjets[(ak.num(corrected_fatjets) > 1)][:,:2].pt, axis=1), mass=ak.flatten(corrected_fatjets[(ak.num(corrected_fatjets) > 1)][:,:2].msoftdrop, axis=1))
                # fill_hist(out, "sdmass_ak8corr", dataset=dataset, **jkkw, pt=ak.flatten(corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 1)][:,:2].pt, axis=1), mass=ak.flatten(corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 1)][:,:2].msoftdrop, axis=1))

            self.logging.debug("successfully corrected jets")
                    
            #####################################
            #### Loop over each jet correction
            #####################################
            # print("Final jet corrs to run over: ", jet_corrs)
            for jetsyst in self.jet_systematics:
                self.logging.debug("Doing analysis for corr ", jetsyst)
                # The HEM/JER/JMR/JMS/JES branches below are gated on self.do_gen;
                # for data only 'nominal' assigns corr_jets_final. Skip everything
                # else explicitly rather than falling through to an UnboundLocalError.
                if not self.do_gen and jetsyst != 'nominal':
                    continue
                #####################################
                #### For each jet correction, we need to add JMR and JMS corrections on top (except if we're doing data).
                #####################################
                if jetsyst == 'nominal':
                    if not self.do_gen:
                        self.logging.debug("Doing nominal data")
                        corr_jets_final = deepcopy(corrected_fatjets)
                    else:
                        corr_jets_final = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets))
                elif 'JER' in jetsyst and self.do_gen:
                    if "Up" in jetsyst:
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets.JER.up))
                        corr_jets_final['msoftdrop'] = (corrected_subjets.JER.up[corrected_fatjets.subJetIdx1] + corrected_subjets.JER.up[corrected_fatjets.subJetIdx2]).mass
                    else: 
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets.JER.down))
                        corr_jets_final['msoftdrop'] = (corrected_subjets.JER.down[corrected_fatjets.subJetIdx1] + corrected_subjets.JER.down[corrected_fatjets.subJetIdx2]).mass
                elif "JMR" in jetsyst and self.do_gen:
                    if "Up" in jetsyst:
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets), var = "up")
                    else: 
                        corr_jets_final  =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets), var = "down")
                elif "JMS" in jetsyst and self.do_gen:
                    if "Up" in jetsyst:
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets, var = "up"))
                    else:
                        corr_jets_final =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets, var = "down"))
                elif "JES" in jetsyst and self.do_gen:
                    if jetsyst[-2:]=="Up":
                        field = jetsyst[:-2]
                        corr_jets_final =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets[field].up))
                        corr_jets_final['msoftdrop'] = (corrected_subjets[field].up[corrected_fatjets.subJetIdx1] + corrected_subjets[field].up[corrected_fatjets.subJetIdx2]).mass
                    elif jetsyst[-4:]=="Down":
                        field = jetsyst[:-4]
                        corr_jets_final =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets[field].down))
                        corr_jets_final['msoftdrop'] = (corrected_subjets[field].down[corrected_fatjets.subJetIdx1] + corrected_subjets[field].down[corrected_fatjets.subJetIdx2]).mass
                self.logging.debug(corr_jets_final)
                #################################################################
                #### sort corrected jets by pt before being put into events object
                #################################################################

                sortJets_ind = ak.argsort(corr_jets_final.pt, ascending=False)
                corr_jets_sorted = corr_jets_final[sortJets_ind]
                events_corr = ak.with_field(events_jk, corr_jets_sorted, "FatJet")  
                del corr_jets_sorted, corr_jets_final
                out['cutflow'][datastr]['nEvents initial '+jetsyst] += (len(events_corr.FatJet))
                
                ###################################
                ######### INITIALIZE WEIGHTS AND SELECTION
                ##################################
                sel = PackedSelection()
                self.logging.debug("mctype ", mctype, " gen? ", self.do_gen)
                
                ###############
                #### For data: apply lumimask and require at least one jet to apply jet trigger prescales
                ##############
                if self.do_gen:
                    #### XS*lumi/sumw normalization is applied in postprocess (zjet-style),
                    #### so fill with the raw generator weight here.
                    if "LHEWeight" in events_corr.fields:
                        weights = events_corr.LHEWeight.originalXWGTUP
                    else:
                        weights = events_corr.genWeight
                    
                ###############
                #### For data: apply lumimask to events and weights, get trigger prescaled weights, and save selection
                ##############
                else:
                    lumi_mask = getLumiMask(IOV)(events_corr.run, events_corr.luminosityBlock)
                    events_corr = events_corr[lumi_mask]
                    if "ver2" in dataset:
                        trigsel, psweights, HLT_cutflow_initial, HLT_cutflow_final = applyPrescales(events_corr, trigger= "PFJet", year = IOV)
                    else:
                        trigsel, psweights, HLT_cutflow_initial, HLT_cutflow_final = applyPrescales(events_corr, year = IOV)
                    #### adding trigger values to cutflow
                    for path in HLT_cutflow_initial:
                        out['cutflow'][datastr][path+" inital"] += HLT_cutflow_initial[path]
                        out['cutflow'][datastr][path+" final"] += HLT_cutflow_final[path]
                    out['cutflow'][datastr]['nEvents after good lumi sel'] += (len(events_corr.FatJet))
                    
                    psweights=ak.where(ak.is_none(psweights), 1.0, psweights)
                    trigsel=ak.where(ak.is_none(trigsel), False, trigsel)
                    weights = ak.where(trigsel, psweights, 1.0)
                    sel.add("trigsel", trigsel)
                    out['cutflow'][datastr]['nEvents after trigger sel'] += (ak.sum(sel.all("trigsel")))
                if self.do_gen:
                    sel.add("npv", events_corr.PV.npvsGood > 0)
                else:
                    sel.add("npv", sel.all("trigsel") & (events_corr.PV.npvsGood > 0))

                ###################################
                ##### Get weights object ready
                ###################################
                self.logging.debug("Weights ", weights)
                #### storeIndividual so partial_weight can build the theory-only
                #### gen-truth weights (zjet parity)
                weights_obj = Weights(len(weights), storeIndividual=True)
                weights_obj.add('initWeight', weight=weights)
                if self.do_gen:
                    #### Apply L1 prefiring weights
                    if "L1PreFiringWeight" in events_corr.fields:
                        prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(IOV, events_corr)
                        weights_obj.add("l1prefiring", weight=prefiringNom, weightUp=prefiringUp, weightDown=prefiringDown,)
                    #### Apply Pileup reweighting and get up and down uncertainties
                    puNom, puUp, puDown = get_pu_weights(events_corr, IOV)
                    weights_obj.add("pu", weight=puNom, weightUp=puUp, weightDown=puDown,)
                    #### Get luminosity uncertainties (nominal weight is 1.0)
                    lumiNom, lumiUp, lumiDown = GetLumiUnc(events_corr, IOV)
                    weights_obj.add("Luminosity", weight=lumiNom, weightUp=lumiUp, weightDown=lumiDown) 
                    #### Get q2 and pdf uncs (not available in pythia+pythia files)
                    if 'herwig' in dataset or 'madgraph' in dataset:
                        pdfNom, pdfUp, pdfDown = GetPDFweights(events_corr)
                        weights_obj.add("pdf", weight=pdfNom, weightUp=pdfUp, weightDown=pdfDown)
                        q2muFNom, q2muFUp, q2muFDown = GetQ2muF(events_corr)
                        weights_obj.add("Q2muF", weight=q2muFNom, weightUp=q2muFUp,weightDown=q2muFDown) 
                        q2muRNom, q2muRUp, q2muRDown = GetQ2muR(events_corr)
                        weights_obj.add("Q2muR", weight=q2muRNom, weightUp=q2muRUp, weightDown=q2muRDown) 
                    if "PSWeight" in events_corr.fields and 'herwig' not in dataset:                
                        ISRNom, ISRUp, ISRDown = GetPSWeights(events_corr, shower="ISR")
                        weights_obj.add("isr", weight=ISRNom, weightUp=ISRUp, weightDown=ISRDown,)
                        FSRNom, FSRUp, FSRDown = GetPSWeights(events_corr, shower="FSR")
                        weights_obj.add("fsr", weight=FSRNom, weightUp=FSRUp, weightDown=FSRDown,)
                ###################################
                #### Add MET filters
                ###################################
                METsel = np.array([events_corr.Flag[MET_filters[IOV][i]] for i in range(len(MET_filters[IOV])) if MET_filters[IOV][i] in events_corr.Flag.fields])
                METsel = np.logical_and.reduce(METsel, axis=0) ## a passing event should pass "ALL" the MET filters
                sel.add("METfilters", METsel)
                self.logging.debug("Nevents before met filters ", len(events_corr), " nevents after met filters ", np.sum(METsel))
                
                #####################################
                #### Begin GEN selections
                ####################################
                
                if self.do_gen:
                    self.logging.debug("DOING GEN")
                    #### Select events with at least 2 jets
                    if not self.do_minimal:
                        HT = ak.sum(events_corr.GenJetAK8.pt, axis=-1)
                        # fill_hist(out, "njet_gen", dataset=dataset, syst = jetsyst, n=ak.num(events_corr.GenJetAK8), 
                        #                  weight = weights )
                        self.logging.debug("Len of HT ", len(HT))
                        self.logging.debug("HT (sum of jet pts) ", HT)
                        self.logging.debug("len of events ", len(events_jk))
                        fill_hist(out, "HT_nocuts", dataset=dataset, systematic=jetsyst, pt=HT)
                        fill_hist(out, "HT_wXS", dataset=dataset, systematic=jetsyst, pt=HT, weight=weights)
                    #### pt_cut_gen = ak.all(events_corr.GenJetAK8[:,:2].pt > 200., axis = -1) ### 80% of reco pt cut --> for now removing pt cut
                    sel.add("twoGenJet", (ak.num(events_corr.GenJetAK8) > 1))
                    GenJetAK8 = events_corr.GenJetAK8
                    GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                    rap_cut_gen = ak.all(np.abs(self._rapidity(GenJetAK8[:,:2].p4)) < self.ycut, axis = -1)
                    rap_sel = ak.where(sel.all("twoGenJet"), rap_cut_gen, False)
                    sel.add("genRap2p5", rap_sel)
                    sel.add("genRap_seq", sel.all("twoGenJet", "genRap2p5"))
                    # if not self.jk:
                    #     fill_hist(out, "jet_rap_gen", dataset=dataset, syst = jetsyst, rapidity=ak.flatten(getRapidity(GenJetAK8[sel.all("twoGenJet")][:,:2].p4), axis=1), weight=np.repeat(weights[sel.all("twoGenJet")], 2))
                    #     fill_hist(out, "jet_phi_gen", dataset=dataset, systematic=jetsyst, phi=ak.flatten(GenJetAK8[sel.all("twoGenJet")][:,:2].phi, axis=1), weight=np.repeat(weights[sel.all("twoGenJet")], 2))  
                    
                    if jetsyst == "nominal": out['cutflow'][datastr]['nEvents after gen rapidity selection '] += (len(events_corr[sel.all("genRap_seq")].FatJet))
                    #### get dphi selection -- require two jets in addition to cuts to prevent breaking anything
                    genjet1 = ak.firsts(events_corr.GenJetAK8[:,0:])
                    genjet2 = ak.firsts(events_corr.GenJetAK8[:,1:])
                    dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
                    dphi12_gen_sel = ak.where(sel.all("twoGenJet"), dphi12_gen > 2., False)
                    sel.add("dphiGen2", dphi12_gen_sel)
                    #### get pt assym selection
                    asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
                    asymm_gen_sel = ak.where(sel.all("twoGenJet"), asymm_gen < 0.3, False)
                    #### get gen sd mass so that we can check non-zero values
                    genjet1 = ak.firsts(events_corr.GenJetAK8[:,0:])
                    genjet2 = ak.firsts(events_corr.GenJetAK8[:,1:])
                    groomed_genjet0 = get_gen_sd_mass_jet(genjet1, events_corr.SubGenJetAK8)
                    groomed_genjet1 = get_gen_sd_mass_jet(genjet2, events_corr.SubGenJetAK8)
                    groomed_gen_dijet = ak.concatenate([ak.unflatten(groomed_genjet0, 1),  ak.unflatten(groomed_genjet1, 1)], axis=1)
                    self.logging.debug("Length of groomed gen jets", len(groomed_gen_dijet), " len of events ", len(events_corr))
                    sel.add("genAsym0p3", asymm_gen_sel)
                    sel.add("genDphi_seq", sel.all("dphiGen2", "genRap_seq"))
                    #### Trijet-priority veto, gen side: mirrors the reco veto
                    #### below (>=3 jets, |y(jet3)| < ycut, min pairwise dphi of
                    #### the three leading > 1.0) with the SAME pt3 > 185 floor
                    #### -- GenJetAK8 is stored to lower pt than FatJet, so an
                    #### unfloored gen veto would be far looser than reco and
                    #### turn ~11% of the response into fakes (floored: ~1.5%
                    #### one-sided migration). Floor = the trijet pt axis low
                    #### edge, 185-200 sink included.
                    if self.trijet_veto:
                        genjet3 = ak.firsts(GenJetAK8[:, 2:])
                        gdphimin3 = ak.min([dphi12_gen,
                                            np.abs(genjet1.delta_phi(genjet3)),
                                            np.abs(genjet2.delta_phi(genjet3))], axis=0)
                        in_trijet_gen = ak.fill_none(ak.where(
                            ak.num(events_corr.GenJetAK8) > 2,
                            (np.abs(self._rapidity(genjet3.p4)) < self.ycut)
                            & (gdphimin3 > 1.0) & (genjet3.pt > 185.),
                            False), False)
                        sel.add("vetoTrijetGen", ~in_trijet_gen)
                    else:
                        sel.add("vetoTrijetGen", ak.ones_like(weights, dtype=bool))
                    sel.add("genTot_seq", sel.all("genRap_seq", "dphiGen2", "genAsym0p3", "vetoTrijetGen")& ~ak.is_none(events_corr.GenJet[:,:2].mass) & ~ak.is_none(groomed_gen_dijet[:,:2].mass))
                    #### N-1 plots
                    # if not self.jk and jetsyst=="nominal":
                    #     fill_hist(out, "asymm_gen", dataset=dataset, systematic=jetsyst,ptgen=events_corr[sel.all("twoGenJet")].GenJetAK8[:,0].pt, frac=asymm_gen[sel.all("twoGenJet")], weight=weights[sel.all("twoGenJet")])  
                    #     fill_hist(out, "dphi_gen", dataset=dataset, systematic=jetsyst, dphi=dphi12_gen[sel.all("twoGenJet")], weight=weights[sel.all("twoGenJet")])

                #####################################
                #### Reco Jet Selection
                #################################### 

                #### If there are no events passing MET filters end here to avoid errors
                if ak.sum(METsel) < 1 :
                    self.logging.debug("No events passing MET filters.")
                    return out
                
                
                #### Apply pt and rapidity cuts
                # if not self.jk:
                #     fill_hist(out, "njet_reco", dataset=dataset, syst = jetsyst, n=ak.to_numpy(ak.num(events_corr[sel.all("npv")].FatJet), allow_missing=True), 
                #                          weight = weights[sel.all("npv")])
                # pt_cut_reco = ak.all(events_corr.FatJet[:,:2].pt > 200., axis = -1)  ###remove for now
                sel.add("twoRecoJet", (ak.num(events_corr.FatJet) > 1))
                sel.add("twoRecoJet_seq",  sel.all('npv', 'METfilters', "twoRecoJet"))
                FatJet = events_corr.FatJet
                FatJet["p4"] = ak.with_name(FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                rap_cut_reco = ak.all(np.abs(self._rapidity(FatJet[:,:2].p4)) < self.ycut, axis = -1)
                rap_sel = ak.where(sel.all("twoRecoJet_seq"), rap_cut_reco, False)
                sel.add("recoRap2p5", rap_sel)
                sel.add("recoRap_seq", sel.all("twoRecoJet_seq", "recoRap2p5")) 
                self.logging.debug("Nevents after rap ", len(events_corr[sel.all("recoRap_seq")]))
                # if not self.jk and jetsyst=="nominal":
                #     fill_hist(out, "jet_rap_reco", dataset=dataset, syst = jetsyst, rapidity=ak.to_numpy(ak.flatten(getRapidity(FatJet[sel.all("twoRecoJet_seq")][:,:2].p4)), allow_missing=True),
                #                              weight=np.repeat(weights[sel.all("twoRecoJet_seq")], 2))
                #     fill_hist(out, "jet_phi_reco", dataset=dataset, systematic=jetsyst, phi=ak.flatten(FatJet[sel.all("twoRecoJet_seq")][:,:2].phi, axis=1), weight=np.repeat(weights[sel.all("twoRecoJet_seq")], 2)) 
                #### get dphi and pt asymm selections
                jet1 = ak.firsts(events_corr.FatJet[:,0:])
                jet2 = ak.firsts(events_corr.FatJet[:,1:])
                dphi12 = (np.abs(jet1.delta_phi(jet2)) > 2.)
                dphi12_sel = ak.where(sel.all("twoRecoJet_seq"), dphi12, False)
                sel.add("recodphi2", dphi12_sel)
                sel.add("recodphi_seq", sel.all("recodphi2", "recoRap_seq"))
                asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
                # if not self.jk and jetsyst=="nominal":
                    # fill_hist(out, "dphi_reco", dataset=dataset, systematic=jetsyst, dphi =dphi12[sel.all("twoRecoJet_seq")], weight=weights[sel.all("twoRecoJet_seq")])
                    # fill_hist(out, "asymm_reco", dataset=dataset, systematic=jetsyst, ptreco=events_corr[sel.all("twoRecoJet_seq")].FatJet[:,0].pt, frac=asymm[sel.all("twoRecoJet_seq")], weight=weights[sel.all("twoRecoJet_seq")])
                asymm_reco_sel = ak.where(sel.all("twoRecoJet_seq"), asymm < 0.3, False)
                sel.add("recoAsym0p3", asymm_reco_sel)
                sel.add("recoAsym_seq", sel.all("recoAsym0p3", "recodphi_seq"))
                #### Check that nearest pfmuon and is at least dR > 0.4 away
                muonIso = ak.all((events_corr.FatJet[:,:2].delta_r(events_corr.FatJet[:,:2].nearest(events_corr.Muon)) > 0.4), axis = -1)
                muon_sel = ak.where(sel.all("twoRecoJet_seq"), muonIso, False)
                sel.add("muonIso0p4", muon_sel)
                ##### apply jetid selection
                jetid_sel = ak.where(sel.all("twoRecoJet_seq"), ak.all(events_corr.FatJet[:,:2].jetId > 2, axis=-1), False)
                sel.add("jetId", jetid_sel)
                #### Trijet-priority orthogonality veto: drop events the trijet
                #### channel measures. Kinematic form (>=3 AK8 jets, |y(jet3)| <
                #### ycut, min pairwise dphi of the three leading > 1.0) shown
                #### equivalent to the full trijet reco selection on the skim
                #### join -- jet-3 jetId/muIso add nothing; pt3 > 185 restricts
                #### the veto to the trijet measurement phase space (its pt
                #### axis floor, 185-200 sink included). Study:
                #### scripts/channel_orthogonality.py.
                if self.trijet_veto:
                    jet3 = ak.firsts(FatJet[:, 2:])
                    dphimin3 = ak.min([np.abs(jet1.delta_phi(jet2)),
                                       np.abs(jet1.delta_phi(jet3)),
                                       np.abs(jet2.delta_phi(jet3))], axis=0)
                    in_trijet = ak.fill_none(ak.where(
                        ak.num(events_corr.FatJet) > 2,
                        (np.abs(self._rapidity(jet3.p4)) < self.ycut)
                        & (dphimin3 > 1.0) & (jet3.pt > 185.),
                        False), False)
                    sel.add("vetoTrijetReco", ~in_trijet)
                else:
                    sel.add("vetoTrijetReco", ak.ones_like(weights, dtype=bool))
                if jetsyst == "nominal":
                    out['cutflow'][datastr]['nEvents vetoed as trijet (reco)'] += int(
                        ak.sum(sel.all("recoAsym_seq", "jetId", "muonIso0p4") & ~sel.all("vetoTrijetReco")))
                #### Fill eta phi map with pre cut reco values to check 
                if not self.do_minimal and jetsyst=='nominal': 
                        fill_hist(out, "jet_eta_phi_precuts", dataset=dataset, systematic=jetsyst, phi=ak.flatten(events_corr[sel.all("twoRecoJet")].FatJet[:,:2].phi, axis=-1), eta=ak.flatten(events_corr[sel.all("twoRecoJet")].FatJet[:,:2].eta, axis=-1), weight=np.repeat(weights[sel.all("twoRecoJet")], 2))  
                ####################################
                ### Apply HEM veto
                ####################################
                if IOV == '2018':
                    if self.do_gen:
                        #### MC 2018: HEM as a flat lumi-fraction weight (not a veto)
                        hem_weight = HEMVeto(events_corr.FatJet, events_corr.run, isMC=True, year=IOV)
                        weights_obj.add("HEM", hem_weight)
                        sel.add('hemveto', ak.ones_like(weights, dtype=bool))
                    else:
                        #### data 2018: hard HEM veto
                        hemveto = HEMVeto(events_corr.FatJet, events_corr.run, isMC=False, year=IOV)
                        sel.add('hemveto', hemveto)
                else:
                    sel.add('hemveto', ak.ones_like(weights, dtype=bool))
                ####  Get Final RECO selection
                sel.add("recoTot_seq", sel.all("recoAsym_seq", "jetId", "muonIso0p4", "hemveto", "vetoTrijetReco") & ~ak.is_none(events_corr.FatJet[:,:2].mass) & ~ak.is_none(events_corr.FatJet[:,:2].msoftdrop))
                if (len(events_corr[sel.all("recoTot_seq")]) < 1): 
                    self.logging.debug("no events passing reco sel")
                    return out
                
                ################
                #### Find fakes, misses, and underflow and remove them to get final selection
                ###############
                
                if self.do_gen:
                    matches = ak.all(events_corr.GenJetAK8[:,:2].delta_r(events_corr.GenJetAK8[:,:2].nearest(events_corr.FatJet[:,:2])) < 0.4, axis = -1) 
                    sel.add("matched_gen", matches)
                    ################
                    #### Misses include events failing DR matching, events passing gen cut but failing the reco cut, and events missing a reco level mass or sdmass value
                    ################
                    misses = ~matches | sel.require(genTot_seq=True, recoTot_seq=False)
                    sel.add("misses", misses )
                    miss_sel = misses & sel.all("genTot_seq")
                    # print("Nevents after removing misses ", ak.sum(sel.all("recoTot_seq", "removeMisses")))
                    # print("Number of misses ", ak.sum(miss_sel))
                    if len(weights[miss_sel])>0 and not self.do_minimal:
                        if jetsyst == "nominal": out['cutflow'][datastr]['misses_u'] += (len(events_corr[miss_sel].GenJetAK8))
                        ###### Applying misses selection to gen jets and getting sd mass
                        genjet1 = ak.firsts(events_corr[miss_sel].GenJetAK8[:,0:])
                        genjet2 = ak.firsts(events_corr[miss_sel].GenJetAK8[:,1:])
                        groomed_genjet0 = get_gen_sd_mass_jet(genjet1, events_corr[miss_sel].SubGenJetAK8)
                        groomed_genjet1 = get_gen_sd_mass_jet(genjet2, events_corr[miss_sel].SubGenJetAK8)
                        groomed_gen_dijet = ak.concatenate([ak.unflatten(groomed_genjet0, 1),  ak.unflatten(groomed_genjet1, 1)], axis=1)
                        groomed_gen_dijet = ak.flatten(groomed_gen_dijet, axis=1)
                        miss_dijets = ak.flatten(events_corr[miss_sel].GenJetAK8[:,:2], axis=1)
                        miss_weights = np.repeat(weights[miss_sel], 2)
                        # print("Len of missed dijets ", len(miss_dijets), " and weights ", len(miss_weights))
                        fill_hist(out, "misses_u", dataset=dataset, systematic=jetsyst, **jkkw, ptgen = miss_dijets.pt, mgen = miss_dijets.mass, weight = miss_weights)
                        fill_hist(out, "misses_g", dataset=dataset, systematic=jetsyst, **jkkw, ptgen = miss_dijets.pt, mgen = groomed_gen_dijet.mass, weight = miss_weights)
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "matched_gen")])<1: 
                        self.logging.debug("No events after all selections and removing misses")
                        return out
                    #### Fakes include events missing a gen level mass or sdmass value, events failing index dr matching, and events passing reco cut but failing the gen cut
                    matches = ~ak.any(ak.is_none(events_corr.FatJet[:,:2].matched_gen.pt, axis=-1), axis=1)
                    sel.add("matched_reco", matches)
                    # print("Matches ", matches)
                    # print("Number of nones (fakes) ", ak.sum(~matches))
                    # print("matched_gen nones ", ak.sum(ak.is_none(events_corr.FatJet[:,:2].matched_gen.pt, axis=-1)))
                    fakes = ~matches | sel.require(genTot_seq=False, recoTot_seq=True)
                    fakes = ak.where(sel.all("recoTot_seq"), fakes, False)
                    sel.add("fakes", fakes)
                    if len(weights[fakes])>0 and not self.do_minimal:
                        fake_dijets = ak.flatten(events_corr[fakes].FatJet[:,:2], axis=1)
                        fake_weights = np.repeat(weights_obj.weight()[fakes], 2)
                        fill_hist(out, "fakes_u", dataset=dataset, systematic=jetsyst, **jkkw, ptreco = fake_dijets[~ak.is_none(fake_dijets.mass)].pt, mreco = fake_dijets[~ak.is_none(fake_dijets.mass)].mass, weight = fake_weights[~ak.is_none(fake_dijets.mass)])
                        fill_hist(out, "fakes_g", dataset=dataset, systematic=jetsyst, **jkkw, ptreco = fake_dijets[~ak.is_none(fake_dijets.msoftdrop)].pt, mreco = fake_dijets[~ak.is_none(fake_dijets.msoftdrop)].msoftdrop, weight = fake_weights[~ak.is_none(fake_dijets.msoftdrop)])
                        if jetsyst=="nominal":
                            for syst in self._weight_variations(weights_obj):
                                # print("Weight variation: ", syst)
                                fake_weights = np.repeat(weights_obj.weight(syst)[fakes], 2)
                                fill_hist(out, "fakes_u", dataset=dataset, systematic=syst, **jkkw, ptreco = fake_dijets[~ak.is_none(fake_dijets.mass)].pt, mreco = fake_dijets[~ak.is_none(fake_dijets.mass)].mass, weight = fake_weights[~ak.is_none(fake_dijets.mass)])
                                fill_hist(out, "fakes_g", dataset=dataset, systematic=syst, **jkkw, ptreco = fake_dijets[~ak.is_none(fake_dijets.msoftdrop)].pt, mreco = fake_dijets[~ak.is_none(fake_dijets.msoftdrop)].msoftdrop, weight = fake_weights[~ak.is_none(fake_dijets.msoftdrop)])

                    if jetsyst == "nominal": out['cutflow'][datastr]['fakes_u'] += (len(events_corr[fakes].FatJet))
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "matched_gen", "matched_reco")])<1: 
                        self.logging.debug("No events after all selections and removing fakes & misses")
                        return out
                    #######################
                    #### Make final selection and fill gen plots
                    #######################
                    sel.add("final_seq", sel.all("genTot_seq", "recoTot_seq", "matched_gen", "matched_reco"))

                    #### Event-display tables. Filled over ALL events reaching
                    #### this point (no selection applied) so the ones that FAILED
                    #### are in the file too -- that is the whole point: cutbits
                    #### says where each died, alljets says which jet did it.
                    ####
                    #### CAVEAT (survivorship bias at CHUNK level): the guards at
                    #### ~902/966/999/1024 `return out` early when a chunk has no
                    #### event passing METfilters / recoTot_seq / matched, so a
                    #### chunk with ZERO survivors contributes no display rows at
                    #### all. Per-event this is unbiased, but a whole-chunk wipeout
                    #### is invisible here -- that is why HT100to200 has 0 rows
                    #### while its cutflow is also 0. Use the CUTFLOW, never these
                    #### tables, for any efficiency number; these are for looking
                    #### at individual events.
                    if self._do_skim and jetsyst == "nominal":
                        self._fill_skim_event(out, dataset, weights, events_corr, sel)

                    #### GEN truth = ALL events passing the gen selection, matched or not
                    #### (zjet semantics). The unfolder derives misses = gen - response
                    #### projection, so requiring matched_gen here (as GluonJetMass did)
                    #### silently drops the dR-unmatched component of the misses.
                    gen_weights = np.repeat(weights[sel.all("genTot_seq")], 2)
                    genjet0 = events_corr[sel.all("genTot_seq")].GenJetAK8[:,0]
                    genjet1 = events_corr[sel.all("genTot_seq")].GenJetAK8[:,1]
                    gen_dijet_truth = ak.concatenate([ak.unflatten(genjet0, 1),  ak.unflatten(genjet1, 1)], axis=1)
                    gen_dijet_truth = ak.flatten(gen_dijet_truth, axis=1)
                    groomed_genjet0 = get_gen_sd_mass_jet(genjet0, events_corr[sel.all("genTot_seq")].SubGenJetAK8)
                    groomed_genjet1 = get_gen_sd_mass_jet(genjet1, events_corr[sel.all("genTot_seq")].SubGenJetAK8)
                    groomed_gen_dijet_truth = ak.concatenate([ak.unflatten(groomed_genjet0, 1),  ak.unflatten(groomed_genjet1, 1)], axis=1)
                    groomed_gen_dijet_truth = ak.flatten(groomed_gen_dijet_truth, axis=1)
                    #### Without the matched_gen requirement the groomed gen mass can be
                    #### None per jet (no SubGenJetAK8 near an unmatched jet) -- the
                    #### event-level is_none in genTot_seq does not catch per-jet Nones.
                    #### Mask groomed fills explicitly (zjet masks ~is_none(mgen) too).
                    ok_g_truth = ak.to_numpy(ak.fill_none(~ak.is_none(groomed_gen_dijet_truth.mass), False))
                    groomed_mgen_truth = ak.to_numpy(ak.fill_none(groomed_gen_dijet_truth.mass, np.nan))
                    if self._do_skim and jetsyst == "nominal":
                        self._fill_skim(out, "gen", dataset, gen_weights,
                                        events_corr[sel.all("genTot_seq")],
                                        gen=gen_dijet_truth,
                                        gen_groomed=groomed_gen_dijet_truth)
                    fill_hist(out, "ptjet_mjet_u_gen", dataset=dataset, systematic=jetsyst, **jkkw, ptgen=gen_dijet_truth.pt, mgen=gen_dijet_truth.mass, weight=gen_weights )
                    fill_hist(out, "ptjet_mjet_g_gen", dataset=dataset, systematic=jetsyst, **jkkw, ptgen=gen_dijet_truth.pt[ok_g_truth], mgen=groomed_mgen_truth[ok_g_truth], weight=gen_weights[ok_g_truth] )
                    #### rho reweight: weight gen rho hists by the gen-rho Herwig/Pythia ratio
                    gen_rw_u, gen_rw_g = gen_weights, gen_weights
                    if self._do_reweight:
                        hw_u, hw_g = self._herwig_rho_weights(gen_dijet_truth.pt, gen_dijet_truth.mass, groomed_mgen_truth)
                        gen_rw_u, gen_rw_g = gen_weights * hw_u, gen_weights * hw_g
                    fill_hist(out, "ptjet_rhojet_u_gen", dataset=dataset, systematic=jetsyst, **jkkw, ptgen=gen_dijet_truth.pt, mpt_gen=self._rho(gen_dijet_truth.mass, gen_dijet_truth.pt), weight=gen_rw_u )
                    fill_hist(out, "ptjet_rhojet_g_gen", dataset=dataset, systematic=jetsyst, **jkkw, ptgen=gen_dijet_truth.pt[ok_g_truth], mpt_gen=self._rho(groomed_mgen_truth, ak.to_numpy(gen_dijet_truth.pt))[ok_g_truth], weight=gen_rw_g[ok_g_truth] )
                    if jetsyst == "nominal":
                        floor_gen_g = ok_g_truth & (groomed_mgen_truth > 2.0)
                        fill_hist(out, "ptjet_rhojet_g_gen_mfloor2", dataset=dataset, systematic=jetsyst, **jkkw,
                                  ptgen=gen_dijet_truth.pt[floor_gen_g],
                                  mpt_gen=self._rho(groomed_mgen_truth, ak.to_numpy(gen_dijet_truth.pt))[floor_gen_g],
                                  weight=gen_rw_g[floor_gen_g])
                    #######################
                    #### Check jec's with selection
                    #######################
                    # #jets of interest
                    # joi = (ak.all(events_corr.FatJet[:,:2].mass < 50., axis=-1) & ak.all(events_corr.FatJet[:,:2].pt <290., axis=-1))
                    # jets_to_print = events_corr[sel.all("final_seq") &joi].FatJet[:8,0]
                    # avail_srcs = [unc_src[4:] for unc_src in jets_to_print.fields if "JES_" in unc_src]
                    # print(jets_to_print.fields)
                    # print(avail_srcs)
                    # d = {
                    #     "reco_pt_leadingJERC" : jets_to_print.pt,
                    #     "gen_pt_leading" : jets_to_print.pt,
                    #     "reco_ptRAW_leading": events_jk[sel.all("final_seq")&joi].FatJet[:8,0].pt,
                    #     "gen/reco": events_corr[sel.all("final_seq")&joi].GenJetAK8[:8,0].pt/jets_to_print.pt,
                    #     "reco_rho_leading": np.log((jets_to_print.mass/jets_to_print.pt)**2),
                    #     "reco_eta_leading" : jets_to_print.eta,
                    #     "reco_phi_leading": jets_to_print.phi,
                    #     # "reco_pt_subleading" : events_corr[sel.all("final_seq")&joi].FatJet[:8,1].pt,
                    #     # "gen_pt_subleading" : events_corr[sel.all("final_seq")&joi].GenJetAK8[:8,1].pt,
                    #     # "reco_ptRAW_subleading" : events_jk[sel.all("final_seq")&joi].FatJet[:8,1].pt,
                    #     # "reco_rho_subleading" : np.log((events_corr[sel.all("final_seq")&joi].FatJet[:8,1].mass/events_corr[sel.all("final_seq")&joi].FatJet[:8,1].pt)**2),
                    #     # "reco_eta_subleading" : events_corr[sel.all("final_seq")&joi].FatJet[:8,1].eta,
                    #     # "reco_phi_subleading" : events_corr[sel.all("final_seq")&joi].FatJet[:8,1].phi, 
                    # }
                    # if jetsyst == "nominal":
                    #     for src in avail_srcs:
                    #         print(src)
                    #         d["raw/jec"] = events_jk[sel.all("final_seq")&joi].FatJet[:8,0].pt/jets_to_print.pt_jec
                    #         d["reco_pt_leadingJEC"]  = jets_to_print.pt_jec
                    #         d["reco_nominal_jec_factor"] =  jets_to_print.jet_energy_correction
                    #         d[src+" sf up"] = jets_to_print["jet_energy_uncertainty_"+src][:,0]
                    #         d[src+" sf dn"] = jets_to_print["jet_energy_uncertainty_"+src][:,1]
                    #         d[src+" pt up"] = jets_to_print["JES_"+src].up.pt
                    #         d[src+" pt dn"] = jets_to_print["JES_"+src].down.pt
                    # pd.set_option('display.max_rows', None)
                    # df = pd.DataFrame.from_dict(d, orient="index")
                    # print(df.to_string())
                    # display(df)
                    
                else:
                    sel.add("final_seq", sel.all("recoTot_seq"))
                    #### Data event-display fill, the mirror of the MC one above.
                    #### Placed here, before the `final_seq` early return, and the
                    #### gen bits of `cutbits` simply stay 0 (no gen cut is in
                    #### `sel.names` on this path).
                    if self._do_skim and jetsyst == "nominal":
                        self._fill_skim_event(out, dataset, weights, events_corr, sel)

                #######################
                #### Apply final selections and jet veto maps
                #######################
                if len(events_corr[sel.all("final_seq")])<1:
                        self.logging.debug("no more events after final selection")
                        return out
                final_weights = weights_obj.weight()[sel.all("final_seq")]
                self.logging.debug("Weights after all selections ", final_weights)
                #### define final dijet object for filling
                dijet = ak.flatten(events_corr[sel.all("final_seq")].FatJet[:,:2], axis=1)
                ##### define final dijet weights for filling nominal MC or data
                dijet_weights = np.repeat(final_weights, 2)

                #### Orthogonality log: record (run, lumi, event) of finally selected
                #### DATA events (one row per event). data forces nominal, so no dup.
                if not self.do_gen:
                    _fin = events_corr[sel.all("final_seq")]
                    out['event_id']['run'] += processor.column_accumulator(ak.to_numpy(_fin.run).astype(np.int64))
                    out['event_id']['luminosityBlock'] += processor.column_accumulator(ak.to_numpy(_fin.luminosityBlock).astype(np.int64))
                    out['event_id']['event'] += processor.column_accumulator(ak.to_numpy(_fin.event).astype(np.int64))

                #### Make eta phi plot to check effects of cuts
                # if not self.jk: fill_hist(out, "jet_eta_phi_preveto", dataset=dataset, systematic=jetsyst, phi=dijet.phi, eta=dijet.eta, weight=dijet_weights)      
                #### Apply jet veto map
                # jet1 = events_corr.FatJet[:,0]
                # jet2 = events_corr.FatJet[:,1]
                # veto = ApplyVetoMap(IOV, jet1, mapname='jetvetomap') & ApplyVetoMap(IOV, jet2, mapname='jetvetomap')
                # if len(events_corr[veto])<1:
                #         print("no more events after jet veto")
                #         return out
                # events_corr = events_corr[veto]
                # weights = weights[veto]
                
                negMSD = ak.flatten(events_corr[sel.all("final_seq")].FatJet[:,:2].msoftdrop<0, axis=1)
                if jetsyst == "nominal": out['cutflow'][datastr]['nEvents failing softdrop condition'] += ak.sum(negMSD)
                
                ##################
                #### Apply final selections to GEN and fill any plots requiring gen, including resp. matrices
                ##################
                
                if self.do_gen:
                    #### RECO spectrum = ALL events passing the reco selection, fakes
                    #### included (zjet semantics). The unfolder derives fakes =
                    #### reco - response projection, so requiring matched_reco here
                    #### (as GluonJetMass did) drops the dR-unmatched fakes.
                    reco_weights = np.repeat(weights_obj.weight()[sel.all("recoTot_seq")], 2)
                    reco_dijet = ak.flatten(events_corr[sel.all("recoTot_seq")].FatJet[:,:2], axis=1)
                    #### JMR smearing leaves dR-unmatched jets with an undefined (None)
                    #### mass (the smear factor needs matched_gen.mass), so mask per jet
                    #### and per mass type -- the same masking the explicit fakes fills
                    #### use. Only these jets drop out of the reco spectra.
                    ok_reco_u = ak.to_numpy(ak.fill_none(~ak.is_none(reco_dijet.mass), False))
                    ok_reco_g = ak.to_numpy(ak.fill_none(~ak.is_none(reco_dijet.msoftdrop), False))
                    mreco_u_filled = ak.to_numpy(ak.fill_none(reco_dijet.mass, np.nan))
                    mreco_g_filled = ak.to_numpy(ak.fill_none(reco_dijet.msoftdrop, np.nan))
                    ptreco_np = ak.to_numpy(reco_dijet.pt)
                    if self._do_skim and jetsyst == "nominal":
                        self._fill_skim(out, "reco", dataset, reco_weights,
                                        events_corr[sel.all("recoTot_seq")],
                                        reco=reco_dijet)
                    genjet0 = events_corr[sel.all("final_seq")].FatJet[:,0].matched_gen
                    genjet1 = events_corr[sel.all("final_seq")].FatJet[:,1].matched_gen
                    # print("Number of mismatched leading dijets", ak.sum(events_corr[sel.all("final_seq")].GenJetAK8[:,0].pt!=genjet0.pt))
                    # print("Number of jets j0 matched to j1", ak.sum(events_corr[sel.all("final_seq")].GenJetAK8[:,0].mass==genjet1.mass))
                    # print("Number of mismatched subleading dijets", ak.sum(events_corr[sel.all("final_seq")].GenJetAK8[:,1].pt!=genjet1.pt))
                    # print("Number of jets j1 matched to j0", ak.sum(events_corr[sel.all("final_seq")].GenJetAK8[:,1].mass==genjet0.mass))
                    groomed_genjet0 = get_gen_sd_mass_jet(genjet0, events_corr[sel.all("final_seq")].SubGenJetAK8)
                    groomed_genjet1 = get_gen_sd_mass_jet(genjet1, events_corr[sel.all("final_seq")].SubGenJetAK8)
                    groomed_gen_dijet = ak.concatenate([ak.unflatten(groomed_genjet0, 1),  ak.unflatten(groomed_genjet1, 1)], axis=1)
                    groomed_gen_dijet = ak.flatten(groomed_gen_dijet, axis=1)
                    gen_dijet = ak.concatenate([ak.unflatten(genjet0, 1),  ak.unflatten(genjet1, 1)], axis=1)
                    gen_dijet = ak.flatten(gen_dijet, axis=1)
                    if self._do_skim and jetsyst == "nominal":
                        self._fill_skim(out, "matched", dataset, dijet_weights,
                                        events_corr[sel.all("final_seq")],
                                        reco=dijet, gen=gen_dijet,
                                        gen_groomed=groomed_gen_dijet)
                    # print("Number of weird dijets ", len(weird_dijets))
                    if not self.do_minimal and jetsyst=="nominal":
                        HT = ak.sum(events_corr[sel.all("final_seq")].GenJetAK8.pt, axis=-1)
                        #### plots for checking MET/sumET
                        fill_hist(out, "MET_over_sumET_pt_reco", dataset=dataset,systematic=jetsyst, frac=events_corr[sel.all("final_seq")].MET.pt/events_corr[sel.all("final_seq")].MET.sumEt, ptreco=events_corr[sel.all("final_seq")].FatJet[:,0].pt, weight=final_weights)
                        fill_hist(out, "MET_pt_reco", dataset=dataset,systematic=jetsyst, pt=events_corr[sel.all("final_seq")].MET.pt, ptreco=events_corr[sel.all("final_seq")].FatJet[:,0].pt, weight=final_weights)
                        fill_hist(out, "HT_aftercuts", dataset=dataset, systematic=jetsyst, pt=HT, weight=final_weights)
                        # fill_hist(out, "ptreco_mreco_fine_u", dataset=dataset,systematic=jetsyst, **jkkw, pt=dijet.pt, mass=dijet.mass, weight=dijet_weights )
                        # fill_hist(out, "ptreco_mreco_fine_g", dataset=dataset,systematic=jetsyst, **jkkw, pt=dijet.pt, mass=dijet.msoftdrop, weight=dijet_weights)
                    #### Final MC plots -- filling nominal weights
                    fill_hist(out, "response_matrix_u", dataset=dataset, systematic=jetsyst, **jkkw,  ptreco=dijet.pt, mreco=dijet.mass,
                                                  ptgen=gen_dijet.pt, mgen=gen_dijet.mass,
                                                  weight=dijet_weights)
                    fill_hist(out, "response_matrix_g", dataset=dataset, systematic=jetsyst, **jkkw,ptreco=dijet.pt, mreco=dijet.msoftdrop,
                                                  ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight=dijet_weights)
                    #### rho reweight: weight rho response/reco by the reco-rho Herwig/Pythia ratio
                    resp_rw_u, resp_rw_g = dijet_weights, dijet_weights
                    reco_rw_u, reco_rw_g = reco_weights, reco_weights
                    if self._do_reweight:
                        hw_resp_u, hw_resp_g = self._herwig_rho_weights(dijet.pt, dijet.mass, dijet.msoftdrop)
                        resp_rw_u, resp_rw_g = dijet_weights * hw_resp_u, dijet_weights * hw_resp_g
                        hw_reco_u, hw_reco_g = self._herwig_rho_weights(reco_dijet.pt, mreco_u_filled, mreco_g_filled)
                        reco_rw_u, reco_rw_g = reco_weights * hw_reco_u, reco_weights * hw_reco_g
                    fill_hist(out, "response_matrix_rho_u", dataset=dataset, systematic=jetsyst, **jkkw,  mpt_reco=self._rho(dijet.mass, dijet.pt), mpt_gen=self._rho(gen_dijet.mass, gen_dijet.pt), ptreco=dijet.pt, ptgen=gen_dijet.pt, weight=resp_rw_u)
                    fill_hist(out, "response_matrix_rho_g", dataset=dataset, systematic=jetsyst, **jkkw, mpt_reco=self._rho(dijet.msoftdrop, dijet.pt), mpt_gen=self._rho(groomed_gen_dijet.mass, gen_dijet.pt), ptreco=dijet.pt, ptgen=gen_dijet.pt, weight=resp_rw_g)
                    if jetsyst == "nominal":
                        floor_response_g = (
                            ak.to_numpy(ak.fill_none(dijet.msoftdrop, np.nan)) > 2.0
                        ) & (
                            ak.to_numpy(ak.fill_none(groomed_gen_dijet.mass, np.nan)) > 2.0
                        )
                        fill_hist(out, "response_matrix_rho_g_mfloor2", dataset=dataset, systematic=jetsyst, **jkkw,
                                  mpt_reco=self._rho(dijet.msoftdrop[floor_response_g], dijet.pt[floor_response_g]),
                                  mpt_gen=self._rho(groomed_gen_dijet.mass[floor_response_g], gen_dijet.pt[floor_response_g]),
                                  ptreco=dijet.pt[floor_response_g], ptgen=gen_dijet.pt[floor_response_g],
                                  weight=resp_rw_g[floor_response_g])
                    fill_hist(out, "ptjet_mjet_u_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=ptreco_np[ok_reco_u], mreco=mreco_u_filled[ok_reco_u], weight=reco_weights[ok_reco_u] )
                    fill_hist(out, "ptjet_mjet_g_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=ptreco_np[ok_reco_g], mreco=mreco_g_filled[ok_reco_g], weight=reco_weights[ok_reco_g] )
                    fill_hist(out, "ptjet_rhojet_u_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=ptreco_np[ok_reco_u], mpt_reco=self._rho(mreco_u_filled, ptreco_np)[ok_reco_u], weight=reco_rw_u[ok_reco_u])
                    fill_hist(out, "ptjet_rhojet_g_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=ptreco_np[ok_reco_g], mpt_reco=self._rho(mreco_g_filled, ptreco_np)[ok_reco_g], weight=reco_rw_g[ok_reco_g] )
                    if jetsyst == "nominal":
                        floor_reco_g = ok_reco_g & (mreco_g_filled > 2.0)
                        fill_hist(out, "ptjet_rhojet_g_reco_mfloor2", dataset=dataset, systematic=jetsyst, **jkkw,
                                  ptreco=ptreco_np[floor_reco_g],
                                  mpt_reco=self._rho(mreco_g_filled, ptreco_np)[floor_reco_g],
                                  weight=reco_rw_g[floor_reco_g])
                    #### event-clustered covariance of the reco spectra (nominal only)
                    if jetsyst == "nominal":
                        cov_w_ev = weights_obj.weight()[sel.all("recoTot_seq")]
                        self._fill_reco_cov(out, dataset, "reco_cov_rho_u", ptreco_np,
                                            self._rho(mreco_u_filled, ptreco_np), ok_reco_u, cov_w_ev)
                        self._fill_reco_cov(out, dataset, "reco_cov_rho_g", ptreco_np,
                                            self._rho(mreco_g_filled, ptreco_np), ok_reco_g, cov_w_ev)
                        self._fill_reco_cov(out, dataset, "reco_cov_rho_g_mfloor2", ptreco_np,
                                            self._rho(mreco_g_filled, ptreco_np),
                                            ok_reco_g & (mreco_g_filled > 2.0), cov_w_ev)
                    if not self.do_minimal:
                        fill_hist(out, "jet_pt_eta_phi", dataset=dataset, systematic=jetsyst, ptreco=dijet.pt, phi=dijet.phi, eta=dijet.eta, weight=dijet_weights)
                        
                        fill_hist(out, "m_u_jet_reco_over_gen", dataset=dataset, ptgen=gen_dijet.pt, mgen=gen_dijet.mass, frac = dijet.mass/gen_dijet.mass, 
                                                           weight = dijet_weights)
                        fill_hist(out, "m_g_jet_reco_over_gen", dataset=dataset, ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, 
                                                           frac = dijet.msoftdrop/groomed_gen_dijet.mass, weight = dijet_weights)
                    #### if nominal, have to fill final plots for all variations
                    if jetsyst == "nominal":
                        for syst in self._weight_variations(weights_obj):
                            #### define weights based on which systematic variation considered
                            dijet_weights = np.repeat(weights_obj.weight(syst)[sel.all("final_seq")], 2)
                            reco_weights = np.repeat(weights_obj.weight(syst)[sel.all("recoTot_seq")], 2)
                            #### fill nominal, up, and down variations for each         
                            fill_hist(out, "response_matrix_u", dataset=dataset,systematic=syst, **jkkw, ptreco=dijet.pt, mreco=dijet.mass,
                                                          ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight= dijet_weights)
                            fill_hist(out, "response_matrix_g", dataset=dataset,systematic=syst, **jkkw, ptreco=dijet.pt, mreco=dijet.msoftdrop,
                                                          ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight= dijet_weights)
                            #### reuse the per-jet reco-rho Herwig/Pythia ratios from above
                            resp_rw_u_v, resp_rw_g_v = dijet_weights, dijet_weights
                            reco_rw_u_v, reco_rw_g_v = reco_weights, reco_weights
                            if self._do_reweight:
                                resp_rw_u_v, resp_rw_g_v = dijet_weights * hw_resp_u, dijet_weights * hw_resp_g
                                reco_rw_u_v, reco_rw_g_v = reco_weights * hw_reco_u, reco_weights * hw_reco_g
                            fill_hist(out, "response_matrix_rho_u", dataset=dataset, systematic=syst, **jkkw, mpt_reco=self._rho(dijet.mass, dijet.pt), mpt_gen=self._rho(gen_dijet.mass, gen_dijet.pt), ptreco=dijet.pt, ptgen=gen_dijet.pt, weight= resp_rw_u_v)
                            fill_hist(out, "response_matrix_rho_g", dataset=dataset, systematic=syst, **jkkw, mpt_reco=self._rho(dijet.msoftdrop, dijet.pt), mpt_gen=self._rho(groomed_gen_dijet.mass, gen_dijet.pt), ptreco=dijet.pt, ptgen=gen_dijet.pt, weight= resp_rw_g_v)
                            fill_hist(out, "ptjet_mjet_u_reco", dataset=dataset,systematic=syst, **jkkw, ptreco=ptreco_np[ok_reco_u], mreco=mreco_u_filled[ok_reco_u], weight=reco_weights[ok_reco_u] )
                            fill_hist(out, "ptjet_mjet_g_reco", dataset=dataset,systematic=syst, **jkkw, ptreco=ptreco_np[ok_reco_g], mreco=mreco_g_filled[ok_reco_g], weight=reco_weights[ok_reco_g] )
                            fill_hist(out, "ptjet_rhojet_u_reco", dataset=dataset, systematic=syst, **jkkw, ptreco=ptreco_np[ok_reco_u], mpt_reco=self._rho(mreco_u_filled, ptreco_np)[ok_reco_u], weight=reco_rw_u_v[ok_reco_u])
                            fill_hist(out, "ptjet_rhojet_g_reco", dataset=dataset, systematic=syst, **jkkw, ptreco=ptreco_np[ok_reco_g], mpt_reco=self._rho(mreco_g_filled, ptreco_np)[ok_reco_g], weight=reco_rw_g_v[ok_reco_g] )
                            #### GEN truth per weight systematic (zjet parity): only the
                            #### theory weights enter (nominal 1.0, so the nominal gen
                            #### spectrum is unchanged) and only theory variations shift
                            #### it; detector systematics pass modifier=None.
                            _gen_modifier = syst if syst in self._THEORY_GEN_SYSTS else None
                            _gen_include = [n for n in ('initWeight', 'isr', 'fsr', 'pdf', 'Q2muF', 'Q2muR')
                                            if n in weights_obj._weights]
                            gen_syst_weights = np.repeat(
                                weights_obj.partial_weight(include=_gen_include, modifier=_gen_modifier)[sel.all("genTot_seq")], 2)
                            gen_rw_u_v, gen_rw_g_v = gen_syst_weights, gen_syst_weights
                            if self._do_reweight:
                                #### reuse the per-jet gen-rho ratios from the nominal gen fill
                                gen_rw_u_v, gen_rw_g_v = gen_syst_weights * hw_u, gen_syst_weights * hw_g
                            fill_hist(out, "ptjet_mjet_u_gen", dataset=dataset, systematic=syst, **jkkw, ptgen=gen_dijet_truth.pt, mgen=gen_dijet_truth.mass, weight=gen_syst_weights)
                            fill_hist(out, "ptjet_mjet_g_gen", dataset=dataset, systematic=syst, **jkkw, ptgen=gen_dijet_truth.pt[ok_g_truth], mgen=groomed_mgen_truth[ok_g_truth], weight=gen_syst_weights[ok_g_truth])
                            fill_hist(out, "ptjet_rhojet_u_gen", dataset=dataset, systematic=syst, **jkkw, ptgen=gen_dijet_truth.pt, mpt_gen=self._rho(gen_dijet_truth.mass, gen_dijet_truth.pt), weight=gen_rw_u_v)
                            fill_hist(out, "ptjet_rhojet_g_gen", dataset=dataset, systematic=syst, **jkkw, ptgen=gen_dijet_truth.pt[ok_g_truth], mpt_gen=self._rho(groomed_mgen_truth, ak.to_numpy(gen_dijet_truth.pt))[ok_g_truth], weight=gen_rw_g_v[ok_g_truth])
                        #################
                        #### Gluon purity plots
                        #################
                        jet1flav = getJetFlavors(events_corr[sel.all("final_seq")].FatJet[:,0])
                        jet2flav = getJetFlavors(events_corr[sel.all("final_seq")].FatJet[:,1])
                        genjet0 = events_corr[sel.all("final_seq")].FatJet[:,0].matched_gen
                        genjet1 = events_corr[sel.all("final_seq")].FatJet[:,1].matched_gen
                        
                        jets = {"jet1":jet1flav, "jet2":jet2flav}
                        if not self.do_minimal:
                            for flavor in jet1flav.keys():
                                for jetname, jetobj in jets.items():
                                    jetobj[flavor] = jetobj[flavor][~ak.is_none(jetobj[flavor])]
                                    #### no weights --> figure out later?
                                    fill_hist(out, "alljet_ptreco_mreco", dataset=dataset, jetNumb = jetname, partonFlav = flavor, 
                                                                    mreco = jetobj[flavor].mass, 
                                                                    ptreco = jetobj[flavor].pt)
                                    fill_hist(out, "btag_eta", dataset=dataset, jetNumb = jetname, partonFlav = flavor, 
                                                         frac = jetobj[flavor].btagDeepB, eta = jetobj[flavor].eta)
                        out['cutflow'][datastr]['nGluonJets'] += (len(jet1flav["Gluon"])+len(jet2flav["Gluon"]))
                        out['cutflow'][datastr]['nJets'] += (len(events_corr[sel.all("final_seq")].FatJet[:,0])+len(events_corr[sel.all("final_seq")].FatJet[:,1]))

                ###############
                ##### If running over DATA only fill final reco plots
                ###############
                
                else:
                    #### Data `reco` table. For data final_seq == recoTot_seq, so
                    #### `dijet`/`dijet_weights` here are exactly the objects the
                    #### MC path passes as reco_dijet/reco_weights -- same content,
                    #### same two-jets-per-event ordering. postprocess returns
                    #### early for data, so _scale_skim_weights never runs and the
                    #### weight column keeps the trigger-prescale weight as-is.
                    if self._do_skim and jetsyst == "nominal":
                        self._fill_skim(out, "reco", dataset, dijet_weights,
                                        events_corr[sel.all("recoTot_seq")],
                                        reco=dijet)

                    fill_hist(out, "ptjet_mjet_u_reco", dataset=dataset,systematic=jetsyst, **jkkw, ptreco=dijet.pt, mreco=dijet.mass,
                                               weight=dijet_weights)
                    fill_hist(out, "ptjet_mjet_g_reco", dataset=dataset,systematic=jetsyst, **jkkw, ptreco=dijet.pt, mreco=dijet.msoftdrop,
                                               weight=dijet_weights )
                    fill_hist(out, "ptjet_rhojet_u_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=dijet.pt, mpt_reco=self._rho(dijet.mass, dijet.pt), weight=dijet_weights)
                    fill_hist(out, "ptjet_rhojet_g_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=dijet.pt, mpt_reco=self._rho(dijet.msoftdrop, dijet.pt), weight=dijet_weights )
                    if jetsyst == "nominal":
                        _data_mg = ak.to_numpy(ak.fill_none(dijet.msoftdrop, np.nan))
                        _data_floor_g = _data_mg > 2.0
                        fill_hist(out, "ptjet_rhojet_g_reco_mfloor2", dataset=dataset, systematic=jetsyst, **jkkw,
                                  ptreco=dijet.pt[_data_floor_g],
                                  mpt_reco=self._rho(dijet.msoftdrop[_data_floor_g], dijet.pt[_data_floor_g]),
                                  weight=dijet_weights[_data_floor_g])
                    #### event-clustered covariance of the data reco spectra
                    #### (prescale weights enter as w^2)
                    if jetsyst == "nominal":
                        _pt_np = ak.to_numpy(dijet.pt)
                        _ok_all = np.ones(len(_pt_np), dtype=bool)
                        self._fill_reco_cov(out, dataset, "reco_cov_rho_u", _pt_np,
                                            ak.to_numpy(self._rho(dijet.mass, dijet.pt)), _ok_all, final_weights)
                        self._fill_reco_cov(out, dataset, "reco_cov_rho_g", _pt_np,
                                            ak.to_numpy(self._rho(dijet.msoftdrop, dijet.pt)), _ok_all, final_weights)
                        self._fill_reco_cov(out, dataset, "reco_cov_rho_g_mfloor2", _pt_np,
                                            ak.to_numpy(self._rho(dijet.msoftdrop, dijet.pt)),
                                            ak.to_numpy(ak.fill_none(dijet.msoftdrop, np.nan)) > 2.0,
                                            final_weights)
                    if not self.do_minimal and jetsyst=="nominal":
                        HT = ak.sum(events_corr[sel.all("final_seq")].FatJet.pt, axis=-1)
                        #### plots for checking MET/sumET
                        fill_hist(out, "MET_over_sumET_pt_reco", dataset=dataset,systematic=jetsyst, frac=events_corr[sel.all("final_seq")].MET.pt/events_corr[sel.all("final_seq")].MET.sumEt, ptreco=events_corr[sel.all("final_seq")].FatJet[:,0].pt, weight=final_weights)
                        fill_hist(out, "MET_pt_reco", dataset=dataset,systematic=jetsyst, pt=events_corr[sel.all("final_seq")].MET.pt, ptreco=events_corr[sel.all("final_seq")].FatJet[:,0].pt, weight=final_weights)
                        #### plots for checking whether jet veto is needed
                        fill_hist(out, "jet_pt_eta_phi", dataset=dataset,systematic=jetsyst, ptreco=dijet.pt, phi=dijet.phi, eta=dijet.eta, weight=dijet_weights)
                        fill_hist(out, "HT_aftercuts", dataset=dataset, systematic=jetsyst, pt=HT, weight=final_weights)
                        # fill_hist(out, "ptreco_mreco_fine_u", dataset=dataset,systematic=jetsyst, **jkkw, pt=dijet.pt, mass=dijet.mass, 
                        #                             weight=dijet_weights )
                        # fill_hist(out, "ptreco_mreco_fine_g", dataset=dataset,systematic=jetsyst, **jkkw, pt=dijet.pt, mass=dijet.msoftdrop, 
                        #                             weight=dijet_weights )
                if (jetsyst == "nominal"): 
                    for name in sel.names:
                        out["cutflow"][datastr][name] += sel.all(name).sum()
                self.logging.debug("NUMBER OF FINAL EVENTS ", len(events_corr))
                del events_corr, weights
            del events_jk
        out['cutflow'][datastr]['chunks'] += 1
        return out    
    def postprocess(self, accumulator):
        #### Apply MC xs*lumi*1000/sumw normalization here (zjet-style), per dataset.
        #### Data carries no genWeight scaling. The `dataset` axis holds the full
        #### dataset name so we can infer the IOV and look up xs/sumw.
        if not self.do_gen:
            return accumulator

        def _iov(ds):
            if re.findall(r'APV', ds) or re.findall(r'HIPM', ds):
                return '2016APV'
            if re.findall(r'UL18', ds) or re.findall(r'UL2018', ds):
                return '2018'
            if re.findall(r'UL17', ds) or re.findall(r'UL2017', ds):
                return '2017'
            return '2016'

        self._scale_skim_weights(accumulator, _iov)

        for key, h in accumulator.items():
            if not isinstance(h, hist.Hist):
                continue
            axnames = [ax.name for ax in h.axes]
            if 'dataset' not in axnames:
                continue
            ds_axis = h.axes['dataset']
            view = h.view(flow=True)
            for ds in list(ds_axis):
                scale = getXSweight(ds, _iov(ds))
                if scale is None:
                    scale = 1.0
                idx = ds_axis.index(ds)
                if key.startswith('reco_cov_'):
                    #### covariance hists are second-moment quantities: the
                    #### VALUE is V_ij = sum w^2 n_i n_j and scales as scale^2
                    view['value'][idx] *= scale * scale
                    view['variance'][idx] *= scale ** 4
                else:
                    view['value'][idx] *= scale
                    view['variance'][idx] *= scale * scale
        return accumulator

##### TO DO #####
#make mass vs pt and response matrix (pt_gen, mass_gen, pt_reco, mass_reco)
# Add eta/phi/delta_r/pt cuts fully
# Make 2 eta collections --> high eta (>1.7) and central (<1.7)
# add same cuts on GenJetAK8
# find misses --> need to do deltaR matching by hand --> if no reco miss
# do Rivet routine
# make central (eta < 1.7) and high eta bins (1.7 < eta < 2.5)
# try AK4 jets to give low pT ??
# remove phi cuts --  why past me why?? do you mean try with and without phi cuts?
