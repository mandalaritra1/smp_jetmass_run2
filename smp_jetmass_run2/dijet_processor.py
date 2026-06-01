 #### This file contains the processors for dijet and trijet hist selections. Plotting and resulting studies are in separate files.
#### LMH

import argparse

import awkward as ak
import numpy as np
import os
import re
import pandas as pd

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
    original; the only intentional change is HEM handling (weight-based for 2018
    MC, hard veto for 2018 data) and canonical systematic-axis names
    (PUSF->pu, L1prefiring->l1prefiring, PDF->pdf, ISR->isr, FSR->fsr).
    '''
    def __init__(self, do_gen=True, mode="minimal", debug=False,
                 jet_systematics=None, systematics=None,
                 ptcut=200., ycut=2.5, jk=False, jk_range=None):
        # should have separate **lower** ptcut for gen
        self.do_gen = do_gen
        self._mode = mode
        # diagnostics filled only in validation/full (mirrors zjet's mode gating)
        self.do_minimal = self._mode not in ("validation", "full")
        self.ptcut = ptcut
        self.ycut = ycut  # rapidity
        self.jk = jk
        self.jk_range = jk_range
        # jackknife active either via the jk flag or the *_jk modes (mirrors zjet)
        self._do_jk = bool(self.jk) or self._mode in ("mass_jk", "rho_jk")
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

        self.hists = processor.dict_accumulator({})
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

        mass_modes = ("minimal", "mass_jk", "validation", "full")
        rho_modes  = ("minimal_rho", "rho_jk", "validation", "full")

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
        if self._mode in rho_modes:
            register_hist(self.hists, 'ptjet_rhojet_u_reco', [dataset_axis, syst_cat, *jk_axes, pt_bin, rho_bin])
            register_hist(self.hists, 'ptjet_rhojet_g_reco', [dataset_axis, syst_cat, *jk_axes, pt_bin, rho_bin])
            if self.do_gen:
                register_hist(self.hists, 'ptjet_rhojet_u_gen', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, rho_gen_bin])
                register_hist(self.hists, 'ptjet_rhojet_g_gen', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, rho_gen_bin])
                register_hist(self.hists, 'response_matrix_rho_u', [dataset_axis, syst_cat, *jk_axes, pt_bin, pt_gen_bin, rho_bin, rho_gen_bin])
                register_hist(self.hists, 'response_matrix_rho_g', [dataset_axis, syst_cat, *jk_axes, pt_bin, pt_gen_bin, rho_bin, rho_gen_bin])

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

    def _rapidity(self, p4):
        #### Preserve the original GluonJetMass selection helper exactly.
        return 0.5 * np.log((p4.energy + p4.pz) / (p4.energy - p4.pz))

    def _weight_variations(self, weights_obj):
        if self.systematics is None:
            return list(weights_obj.variations)
        return [syst for syst in self.systematics if syst != "nominal" and syst in weights_obj.variations]

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
                weights_obj = Weights(len(weights))
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
                    sel.add("genTot_seq", sel.all("genRap_seq", "dphiGen2", "genAsym0p3")& ~ak.is_none(events_corr.GenJet[:,:2].mass) & ~ak.is_none(groomed_gen_dijet[:,:2].mass))
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
                sel.add("recoTot_seq", sel.all("recoAsym_seq", "jetId", "muonIso0p4", "hemveto") & ~ak.is_none(events_corr.FatJet[:,:2].mass) & ~ak.is_none(events_corr.FatJet[:,:2].msoftdrop))
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
                    
                    gen_weights = np.repeat(weights[sel.all("genTot_seq","matched_gen")], 2)
                    genjet0 = events_corr[sel.all("genTot_seq","matched_gen")].GenJetAK8[:,0]
                    genjet1 = events_corr[sel.all("genTot_seq","matched_gen")].GenJetAK8[:,1]
                    gen_dijet = ak.concatenate([ak.unflatten(genjet0, 1),  ak.unflatten(genjet1, 1)], axis=1)
                    gen_dijet = ak.flatten(gen_dijet, axis=1)
                    groomed_genjet0 = get_gen_sd_mass_jet(genjet0, events_corr[sel.all("genTot_seq","matched_gen")].SubGenJetAK8)
                    groomed_genjet1 = get_gen_sd_mass_jet(genjet1, events_corr[sel.all("genTot_seq","matched_gen")].SubGenJetAK8)
                    groomed_gen_dijet = ak.concatenate([ak.unflatten(groomed_genjet0, 1),  ak.unflatten(groomed_genjet1, 1)], axis=1)
                    groomed_gen_dijet = ak.flatten(groomed_gen_dijet, axis=1)
                    fill_hist(out, "ptjet_mjet_u_gen", dataset=dataset, systematic=jetsyst, **jkkw, ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight=gen_weights )
                    fill_hist(out, "ptjet_mjet_g_gen", dataset=dataset, systematic=jetsyst, **jkkw, ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight=gen_weights )
                    fill_hist(out, "ptjet_rhojet_u_gen", dataset=dataset, systematic=jetsyst, **jkkw, ptgen=gen_dijet.pt, mpt_gen=self._rho(gen_dijet.mass, gen_dijet.pt), weight=gen_weights )
                    fill_hist(out, "ptjet_rhojet_g_gen", dataset=dataset, systematic=jetsyst, **jkkw, ptgen=gen_dijet.pt, mpt_gen=self._rho(groomed_gen_dijet.mass, gen_dijet.pt), weight=gen_weights )
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
                    reco_weights = np.repeat(weights_obj.weight()[sel.all("recoTot_seq", "matched_reco")], 2)
                    reco_dijet = ak.flatten(events_corr[sel.all("recoTot_seq", "matched_reco")].FatJet[:,:2], axis=1)
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
                    fill_hist(out, "response_matrix_rho_u", dataset=dataset, systematic=jetsyst, **jkkw,  mpt_reco=self._rho(dijet.mass, dijet.pt), mpt_gen=self._rho(gen_dijet.mass, gen_dijet.pt), ptreco=dijet.pt, ptgen=gen_dijet.pt, weight=dijet_weights)
                    fill_hist(out, "response_matrix_rho_g", dataset=dataset, systematic=jetsyst, **jkkw, mpt_reco=self._rho(dijet.msoftdrop, dijet.pt), mpt_gen=self._rho(groomed_gen_dijet.mass, gen_dijet.pt), ptreco=dijet.pt, ptgen=gen_dijet.pt, weight=dijet_weights)
                    fill_hist(out, "ptjet_mjet_u_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=reco_dijet.pt, mreco=reco_dijet.mass, weight=reco_weights )
                    fill_hist(out, "ptjet_mjet_g_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=reco_dijet.pt, mreco=reco_dijet.msoftdrop, weight=reco_weights )
                    fill_hist(out, "ptjet_rhojet_u_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=reco_dijet.pt, mpt_reco=self._rho(reco_dijet.mass, reco_dijet.pt), weight=reco_weights)
                    fill_hist(out, "ptjet_rhojet_g_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=reco_dijet.pt, mpt_reco=self._rho(reco_dijet.msoftdrop, reco_dijet.pt), weight=reco_weights )
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
                            reco_weights = np.repeat(weights_obj.weight(syst)[sel.all("recoTot_seq", "matched_reco")], 2)
                            #### fill nominal, up, and down variations for each         
                            fill_hist(out, "response_matrix_u", dataset=dataset,systematic=syst, **jkkw, ptreco=dijet.pt, mreco=dijet.mass,
                                                          ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight= dijet_weights)
                            fill_hist(out, "response_matrix_g", dataset=dataset,systematic=syst, **jkkw, ptreco=dijet.pt, mreco=dijet.msoftdrop,
                                                          ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight= dijet_weights)
                            fill_hist(out, "response_matrix_rho_u", dataset=dataset, systematic=syst, **jkkw, mpt_reco=self._rho(dijet.mass, dijet.pt), mpt_gen=self._rho(gen_dijet.mass, gen_dijet.pt), ptreco=dijet.pt, ptgen=gen_dijet.pt, weight= dijet_weights)
                            fill_hist(out, "response_matrix_rho_g", dataset=dataset, systematic=syst, **jkkw, mpt_reco=self._rho(dijet.msoftdrop, dijet.pt), mpt_gen=self._rho(groomed_gen_dijet.mass, gen_dijet.pt), ptreco=dijet.pt, ptgen=gen_dijet.pt, weight= dijet_weights)
                            fill_hist(out, "ptjet_mjet_u_reco", dataset=dataset,systematic=syst, **jkkw, ptreco=reco_dijet.pt, mreco=reco_dijet.mass, weight= reco_weights )
                            fill_hist(out, "ptjet_mjet_g_reco", dataset=dataset,systematic=syst, **jkkw, ptreco=reco_dijet.pt, mreco=reco_dijet.msoftdrop, weight= reco_weights )                 
                            fill_hist(out, "ptjet_rhojet_u_reco", dataset=dataset, systematic=syst, **jkkw, ptreco=reco_dijet.pt, mpt_reco=self._rho(reco_dijet.mass, reco_dijet.pt), weight=reco_weights)
                            fill_hist(out, "ptjet_rhojet_g_reco", dataset=dataset, systematic=syst, **jkkw, ptreco=reco_dijet.pt, mpt_reco=self._rho(reco_dijet.msoftdrop, reco_dijet.pt), weight=reco_weights )
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
                    
                    fill_hist(out, "ptjet_mjet_u_reco", dataset=dataset,systematic=jetsyst, **jkkw, ptreco=dijet.pt, mreco=dijet.mass, 
                                               weight=dijet_weights)
                    fill_hist(out, "ptjet_mjet_g_reco", dataset=dataset,systematic=jetsyst, **jkkw, ptreco=dijet.pt, mreco=dijet.msoftdrop,
                                               weight=dijet_weights )
                    fill_hist(out, "ptjet_rhojet_u_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=dijet.pt, mpt_reco=self._rho(dijet.mass, dijet.pt), weight=dijet_weights)
                    fill_hist(out, "ptjet_rhojet_g_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=dijet.pt, mpt_reco=self._rho(dijet.msoftdrop, dijet.pt), weight=dijet_weights )
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
