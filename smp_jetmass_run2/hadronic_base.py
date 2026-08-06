"""Shared machinery for the hadronic (dijet / trijet) processors.

Everything here was byte-identical (or identical up to the jets-per-event
fan-out, parametrized below) between dijet_processor.py and
trijet_processor.py before the 2026-07-31 extraction. The selection logic and
histogram fills stay in the channel processors on purpose: they are audited
line-by-line against GluonJetMass (repo invariant #1), and a parametrized
process() would make that audit harder, not easier.

Subclasses must provide (attributes referenced here):
    self.hists, self.mode, self.logging, self.do_gen, self.systematics,
    self.ycut, self._jetR, self._skim_prescale, self._SKIM_CUTS,
    self._SKIM_EVENT_FACTOR,
and may override the per-jet skim fan-out:
    _SKIM_JETS_PER_EVENT   rows per event in the physics skim tables
                           (dijet: 2 -- leading+subleading; trijet: 1 -- jet 3)
    _SKIM_JETIDX           index-within-event of those rows
"""
from __future__ import annotations

import re
import zlib

import awkward as ak
import numpy as np
import hist

from coffea import processor

from .corrections import getXSweight
from .hist_utils import fill_hist


#### The 27 split JES sources GetJetCorrections builds into the corrected jets
#### (corrections.py _get_jet_factory, uncertainties_key=None). Kept in the same
#### order and under the same 'JES_<source>{Up,Down}' names zjet_processor uses,
#### because the dijet/trijet JES branch recovers the field with jetsyst[:-2] /
#### jetsyst[:-4] -- rename these and the lookup silently misses.
JES_SOURCES = ["AbsoluteMPFBias", "AbsoluteScale", "AbsoluteStat", "FlavorQCD",
               "Fragmentation", "PileUpDataMC", "PileUpPtBB", "PileUpPtEC1",
               "PileUpPtEC2", "PileUpPtHF", "PileUpPtRef", "RelativeFSR",
               "RelativeJEREC1", "RelativeJEREC2", "RelativeJERHF",
               "RelativePtBB", "RelativePtEC1", "RelativePtEC2", "RelativePtHF",
               "RelativeBal", "RelativeSample", "RelativeStatEC",
               "RelativeStatFSR", "RelativeStatHF", "SinglePionECAL",
               "SinglePionHCAL", "TimePtEta"]

#### nominal + JER/JMS/JMR up/down: the cheap set, and what the minimal_syst
#### profile pins explicitly.
BASE_JET_SYSTEMATICS = ['nominal', 'JERUp', 'JERDown', 'JMSUp', 'JMSDown',
                        'JMRUp', 'JMRDown']

#### the full hadronic jet-systematic set: the base set plus both directions of
#### every split JES source (61 entries), matching what zjet_processor assembles
#### when jet_systematics is None.
FULL_JET_SYSTEMATICS = BASE_JET_SYSTEMATICS + [
    f"JES_{src}{direction}" for src in JES_SOURCES for direction in ("Up", "Down")]


class Log:
    def __init__(self, mode="info"):
        self.mode = mode
    def info(self, *msg):
        if self.mode in ["info", "debug"]:
            print("[INFO]", *msg)
    def debug(self, *msg):
        if self.mode == "debug":
            print("[DEBUG]", *msg)


class HadronicProcessorBase(processor.ProcessorABC):
    #### per-jet fan-out of the physics skim tables; dijet overrides to 2 jets
    _SKIM_JETS_PER_EVENT = 1
    _SKIM_JETIDX = np.array([2], dtype=np.int8)

    @property
    def accumulator(self):
        return self.hists

    def _rho(self, mass, pt):
        #### rho = 2*log10(m/(pt*R)) -- matches the zjet QJetMassProcessor definition
        #### (includes the jet radius R), not the bare 2*log10(m/pt).
        return 2 * np.log10(mass / (pt * self._jetR))

    def _rapidity(self, p4):
        #### Preserve the original GluonJetMass selection helper exactly.
        return 0.5 * np.log((p4.energy + p4.pz) / (p4.energy - p4.pz))

    def _weight_variations(self, weights_obj):
        if self.systematics is None:
            return list(weights_obj.variations)
        return [syst for syst in self.systematics if syst != "nominal" and syst in weights_obj.variations]

    def _in_trijet_phase_space(self, jets, pt3_floor=185.0):
        """True where the event lies in the trijet measurement phase space:
        >=3 AK8 jets, |y(jet3)| < ycut, min pairwise dphi of the three leading
        > 1.0, and pt3 above the trijet pt-axis floor (185, its 185-200 sink
        bin included). Shown on the skim join to reproduce the full trijet
        reco selection among dijet-selected events (jet-3 jetId/muIso add
        nothing) -- see scripts/channel_orthogonality.py. `jets` must carry a
        `p4` field (PtEtaPhiMLorentzVector); works for FatJet and GenJetAK8
        alike, which is what keeps the dijet reco and gen vetoes mirrored.
        """
        j1 = ak.firsts(jets[:, 0:])
        j2 = ak.firsts(jets[:, 1:])
        j3 = ak.firsts(jets[:, 2:])
        dphimin3 = ak.min([np.abs(j1.delta_phi(j2)),
                           np.abs(j1.delta_phi(j3)),
                           np.abs(j2.delta_phi(j3))], axis=0)
        return ak.fill_none(ak.where(
            ak.num(jets) > 2,
            (np.abs(self._rapidity(j3.p4)) < self.ycut)
            & (dphimin3 > 1.0) & (j3.pt > pt3_floor),
            False), False)

    def _fill_jet_kinematics(self, out, dataset, jetsyst, jets, weights_obj, mask):
        """Per-jet RECO kinematics for the always-on validation hists.

        `jets` maps the hist suffix ("jet1"/"jet2" for dijet, "jet3" for the
        trijet measured jet) to the already-selected jet record; `mask` is the
        per-EVENT reco selection those jets were sliced with, and the per-event
        weights are read out of `weights_obj` under it (one row per event, NOT
        the two-rows-per-event dijet fan-out). Naming/axes follow zjet's
        pt_jet0/eta_jet0/phi_jet0/mass_jet0.

        Filled for EVERY jet systematic -- the caller loops over
        self.jet_systematics and the varied jets arrive here already smeared,
        so `jetsyst` just labels them -- and, under nominal jets, additionally
        once per weight variation, exactly the pattern ptjet_mjet_u_reco uses.
        These are cheap 1D hists and carrying the full systematic set is what
        puts uncertainty bands on the kinematic data/MC ratios.

        Weights are the raw generator weights -- the xs*lumi/sumw scale is
        applied in postprocess like every other hist, never here. Masses can be
        None for dR-unmatched jets under JMR smearing, so the mass fills are
        masked with ~ak.is_none exactly as the spectra are.
        """
        #### The kinematics do not depend on the weight variation: convert to
        #### numpy once and reuse across the systematic loop below.
        kin = {}
        for name, jet in jets.items():
            masses = []
            for hname, vals in (("mass", jet.mass), ("msoftdrop", jet.msoftdrop)):
                ok = np.asarray(ak.to_numpy(ak.fill_none(~ak.is_none(vals), False)))
                m = np.asarray(ak.to_numpy(ak.fill_none(vals, np.nan)))
                masses.append((hname, ok, m))
            kin[name] = (ak.to_numpy(jet.pt), ak.to_numpy(jet.eta),
                         ak.to_numpy(jet.phi), masses)
        #### nominal jets additionally carry every weight variation; a varied
        #### jet systematic is filled with the nominal weights only.
        systs = [jetsyst]
        if jetsyst == "nominal":
            systs += self._weight_variations(weights_obj)
        for syst in systs:
            w = np.asarray(ak.to_numpy(
                weights_obj.weight(None if syst == jetsyst else syst)[mask]),
                dtype=np.float64)
            for name, (pt, eta, phi, masses) in kin.items():
                fill_hist(out, f"pt_{name}", dataset=dataset, systematic=syst,
                          pt=pt, weight=w)
                fill_hist(out, f"eta_{name}", dataset=dataset, systematic=syst,
                          eta=eta, weight=w)
                fill_hist(out, f"phi_{name}", dataset=dataset, systematic=syst,
                          phi=phi, weight=w)
                for hname, ok, m in masses:
                    fill_hist(out, f"{hname}_{name}", dataset=dataset, systematic=syst,
                              mass=m[ok], weight=w[ok])

    #############################################################
    #### rho_skim machinery (tables shared by both channels)
    #############################################################

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

    def _fill_skim(self, out, table, dataset, weights, evt,
                   reco=None, gen=None, gen_groomed=None):
        """Append one block of per-jet rows to out['skim'][table].

        `weights`/`reco`/`gen` are already flattened to _SKIM_JETS_PER_EVENT
        rows per event (dijet: leading+subleading, the same ordering every
        hist fill uses; trijet: the measured third jet), aligned with `evt`.
        """
        k = self._SKIM_JETS_PER_EVENT
        n = len(weights)
        if n == 0:
            return
        #### prescale on the EVENT number so all rows of an event survive or
        #### none, and so the decision is identical in all three tables
        keep = None
        if self._skim_prescale > 1:
            ev = ak.to_numpy(evt.event).astype(np.int64)
            keep = np.repeat(ev % self._skim_prescale == 0, k)
            if not keep.any():
                return
        cols = {
            "weight":     np.asarray(weights, dtype=np.float64) * self._skim_prescale,
            "dataset_id": np.full(n, self._skim_dataset_id(dataset), dtype=np.uint32),
            #### index WITHIN the event of the measured jet(s)
            "jetidx":     np.tile(self._SKIM_JETIDX, n // k),
        }
        for name, src, dtype in (("pu_rho", "fixedGridRhoFastjetAll", np.float32),
                                 ("npv", "PV", np.int16)):
            try:
                v = evt.PV.npvsGood if src == "PV" else evt[src]
                cols[name] = np.repeat(ak.to_numpy(ak.fill_none(v, -1)), k).astype(dtype)
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
                #### index WITHIN the event, so rows beyond the measured jet(s)
                #### are the rest of the topology
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

        #### the flat skim tables are not hist.Hist, so they need the same scale
        #### applied explicitly (XS-in-postprocess is a hard repo invariant)
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
                    #### VALUE is V_ij = sum w^2 n_i n_j and scales as scale^2.
                    #### No-op for trijet (no cov hists) -- safe to share.
                    view['value'][idx] *= scale * scale
                    view['variance'][idx] *= scale ** 4
                else:
                    view['value'][idx] *= scale
                    view['variance'][idx] *= scale * scale
        return accumulator
