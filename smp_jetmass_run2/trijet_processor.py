#### This file contains the processors for trijet hist selections. Plotting and resulting studies are in separate files.
#### LMH

import awkward as ak
import numpy as np
import re
import hist
from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
from collections import defaultdict
#### import our python packages
from .corrections import *
from .hist_utils import util_binning, register_hist, fill_hist
from copy import deepcopy


class Log:
    def __init__(self, mode="info"):
        self.mode = mode
    def info(self, *msg):
        if self.mode in ["info", "debug"]:
            print("[INFO]", *msg)
    def debug(self, *msg):
        if self.mode == "debug":
            print("[DEBUG]", *msg)


##### TO DO #####
# do Rivet routine
# get_gen_sd_mass_jet, getJetFlavors, applyBTag now live in corrections.py

#bcut options: b_loose (apply loose bTag threshold to only hardest jet), bb_loose (apply loose bTag to leading two jets),
#              b_med(apply medium bTag to only the hardest jet), bb_med (apply medium bTag to leading two jets)

class TrijetProcessor(processor.ProcessorABC):
    '''
    Processor to run a trijet jet mass cross section analysis.
    Ported from GluonJetMass python/trijetProcessor.py to the zjet QJetMassProcessor
    coding conventions. The event selection is byte-for-byte identical to the
    original; the only intentional change is HEM handling (weight-based for 2018
    MC, hard veto for 2018 data) and canonical systematic-axis names
    (PUSF->pu, L1prefiring->l1prefiring, PDF->pdf, ISR->isr, FSR->fsr).
    '''
    def __init__(self, do_gen=True, mode="minimal", debug=False,
                 jet_systematics=None, systematics=None,
                 ycut=2.5, btag='None', jk=False, jk_range=None):

        self.do_gen = do_gen
        self._mode = mode
        self.do_minimal = self._mode not in ("validation", "full")
        self.ycut = ycut
        self.btag = btag
        self.jk = jk
        self.jk_range = jk_range
        self._do_jk = bool(self.jk) or self._mode in ("mass_jk", "rho_jk")
        if jet_systematics is None:
            jet_systematics = ['nominal', 'JERUp', 'JERDown', 'JMSUp', 'JMSDown',
                               'JMRUp', 'JMRDown']
        if self._do_jk:
            # protect against doing unc and extra plots for jk --> only need nominal and memory intensive
            jet_systematics = ["nominal"]
        self.jet_systematics = jet_systematics
        self.systematics = systematics
        self.logging = Log(mode="debug" if debug else "info")
        self.logging.info(f"Trijet do_gen={self.do_gen} mode={self._mode} do_jk={self._do_jk}")
        
        #### Define axes for hists
        b = util_binning(channel="trijet")
        self._jetR = b.jetR
        dataset_axis = b.dataset_axis
        syst_cat = b.syst_axis
        jk_axis = b.jackknife_axis
        pt_bin = b.ptreco_axis
        pt_gen_bin = b.ptgen_axis
        mass_bin = b.mreco_axis
        mass_gen_bin = b.mgen_axis
        rho_bin = b.mreco_over_pt_axis
        rho_gen_bin = b.mgen_over_pt_axis
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat = hist.axis.StrCategory([],growth=True,name="partonFlav", label="Parton Flavour")
        fine_mass_bin = hist.axis.Regular(500, 0.0, 1000.0, name="mass", label=r"mass [GeV]")
        fine_pt_bin = hist.axis.Regular(400, 0.0, 8000.0, name="pt", label=r"$p_T$ [GeV]")
        eta_bin = hist.axis.Regular(25, -4., 4., name="eta", label=r"$\eta$")
        frac_axis = hist.axis.Regular(400, 0., 2.5, name="frac", label="Fraction")
        phi_axis = hist.axis.Regular(25, -np.pi, np.pi, name="phi", label=r"$\phi$")
        jk_axes = [jk_axis] if self._do_jk else []

        self.hists = processor.dict_accumulator({})
        self.hists['cutflow'] = {}
        self.hists['jkflow'] = processor.defaultdict_accumulator(int)

        mass_modes = ("minimal", "mass_jk", "validation", "full")
        rho_modes = ("minimal_rho", "rho_jk", "validation", "full")

        if self._mode in mass_modes:
            register_hist(self.hists, 'ptjet_mjet_u_reco', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin])
            register_hist(self.hists, 'ptjet_mjet_g_reco', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin])
            if self.do_gen:
                register_hist(self.hists, 'ptjet_mjet_u_gen', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, mass_gen_bin])
                register_hist(self.hists, 'ptjet_mjet_g_gen', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, mass_gen_bin])
                register_hist(self.hists, 'response_matrix_u', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin])
                register_hist(self.hists, 'response_matrix_g', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin])

        if self._mode in rho_modes:
            register_hist(self.hists, 'ptjet_rhojet_u_reco', [dataset_axis, syst_cat, *jk_axes, pt_bin, rho_bin])
            register_hist(self.hists, 'ptjet_rhojet_g_reco', [dataset_axis, syst_cat, *jk_axes, pt_bin, rho_bin])
            if self.do_gen:
                register_hist(self.hists, 'ptjet_rhojet_u_gen', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, rho_gen_bin])
                register_hist(self.hists, 'ptjet_rhojet_g_gen', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, rho_gen_bin])
                register_hist(self.hists, 'response_matrix_rho_u', [dataset_axis, syst_cat, *jk_axes, pt_bin, pt_gen_bin, rho_bin, rho_gen_bin])
                register_hist(self.hists, 'response_matrix_rho_g', [dataset_axis, syst_cat, *jk_axes, pt_bin, pt_gen_bin, rho_bin, rho_gen_bin])

        if self._mode in ("validation", "full"):
            register_hist(self.hists, 'misses_u', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, mass_gen_bin])
            register_hist(self.hists, 'misses_g', [dataset_axis, syst_cat, *jk_axes, pt_gen_bin, mass_gen_bin])
            register_hist(self.hists, 'fakes_u', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin])
            register_hist(self.hists, 'fakes_g', [dataset_axis, syst_cat, *jk_axes, pt_bin, mass_bin])
            register_hist(self.hists, 'alljet_ptreco_mreco', [dataset_axis, jet_cat, parton_cat, mass_bin, pt_bin])
            register_hist(self.hists, 'btag_eta', [dataset_axis, jet_cat, parton_cat, frac_axis, eta_bin])
            register_hist(self.hists, 'MET_over_sumET_pt_reco', [dataset_axis, syst_cat, frac_axis, pt_bin])
            register_hist(self.hists, 'MET_pt_reco', [dataset_axis, syst_cat, fine_pt_bin, pt_bin])
            register_hist(self.hists, 'HT_nocuts', [dataset_axis, syst_cat, fine_pt_bin])
            register_hist(self.hists, 'HT_wXS', [dataset_axis, syst_cat, fine_pt_bin])
            register_hist(self.hists, 'HT_aftercuts', [dataset_axis, syst_cat, fine_pt_bin])
            register_hist(self.hists, 'asymm_reco', [dataset_axis, syst_cat, pt_bin, frac_axis])
            register_hist(self.hists, 'asymm_gen', [dataset_axis, syst_cat, pt_gen_bin, frac_axis])
            register_hist(self.hists, 'jet_eta_phi_precuts', [dataset_axis, syst_cat, phi_axis, eta_bin])
            register_hist(self.hists, 'jet_eta_phi_preveto', [dataset_axis, syst_cat, phi_axis, eta_bin])
            register_hist(self.hists, 'jet_pt_eta_phi', [dataset_axis, syst_cat, pt_bin, phi_axis, eta_bin])
            register_hist(self.hists, 'ptreco_mreco_fine_u', [dataset_axis, syst_cat, *jk_axes, fine_pt_bin, fine_mass_bin])
            register_hist(self.hists, 'ptreco_mreco_fine_g', [dataset_axis, syst_cat, *jk_axes, fine_pt_bin, fine_mass_bin])
            register_hist(self.hists, 'm_u_jet_reco_over_gen', [dataset_axis, pt_gen_bin, mass_gen_bin, frac_axis])
            register_hist(self.hists, 'm_g_jet_reco_over_gen', [dataset_axis, pt_gen_bin, mass_gen_bin, frac_axis])
    
    @property
    def accumulator(self):
        return self.hists

    def _rho(self, mass, pt):
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
        #####################################
        #### Make loop for running 1/10 of dataset for jackknife
        #####################################
        if 'QCD' in dataset:
            datastr = 'QCD_'+mctype+IOV
        elif "WJets" in dataset:
            datastr = "WJets_"+mctype+IOV
        elif "ZJets" in dataset:
            datastr = 'ZJets_'+mctype+IOV
        elif "TTJets" in dataset:
            datastr = 'TTJets_'+mctype+IOV
        else: datastr = mctype+IOV
        datastr = mctype+IOV
        self.logging.debug("Filename: ", filename)
        self.logging.debug("Dataset: ", dataset)
        ####################################
        #### Inititalize cutflow table
        ###################################            
        
        out['cutflow'][datastr] = defaultdict(int)
        out['cutflow'][datastr]['nEvents initial'] += (len(events.FatJet))
        out['cutflow']['trigger_init'] = defaultdict(int)
        out['cutflow']['trigger_final'] = defaultdict(int)
        
        if (self.do_gen):
            firstidx = filename.find( "store/mc/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            # year = fname_toks[ fname_toks.index("mc") + 1]
            ht_bin = fname_toks[ fname_toks.index("mc") + 2]
            if "LHEWeight" in events.fields:
                weights = events["LHEWeight"].originalXWGTUP
            else:
                weights = events.genWeight
            out['cutflow'][datastr]['sumw for '+ht_bin] += np.sum(weights)
            ## Flag used for number of events

        
        index_list = np.arange(len(events))
        ###### Choose number of slices to break data into for jackknife method
        if self._do_jk:
            self.logging.debug("Self.jk ", self.jk)
            range_max = 10
        else: range_max=1
            
        if self.jk_range == None:
            jk_inds = range(0,range_max)
        else:
            jk_inds = range(self.jk_range[0], self.jk_range[1])
        for jk_index in jk_inds:
            self.logging.debug("Event indices ", index_list)
            if self._do_jk:
                self.logging.debug("Now doing jackknife {}".format(jk_index))
                self.logging.debug("Len of events before jk selection ", len(events))
            else:
                jk_index=-1
            jkkw = {'jk': jk_index} if self._do_jk else {}
            self.logging.debug(index_list%range_max == jk_index)
            jk_sel = ak.where(index_list%range_max == jk_index, False, True)
            ######## Select portion for jackknife and ensure that all jets have a softdrop mass so sd mass correction does not fail
            events_jk = events[jk_sel]
            del jk_sel
            #### only consider pfmuons w/ similar selection to aritra for later jet isolation
            events_jk = ak.with_field(events_jk, 
                                      events_jk.Muon[(events_jk.Muon.mediumId > 0)
                                      &(np.abs(events_jk.Muon.eta) < 2.5)
                                      &(events_jk.Muon.pfIsoId > 1) ], 
                                      "Muon")
            FatJet=events_jk.FatJet
            FatJet["p4"] = ak.with_name(events_jk.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            self.logging.debug(FatJet)
            #### Make sure there is at least one jet to run over
            if ak.sum(ak.num(FatJet.pt)>0)<1:
                self.logging.debug("No fat jet pts at all")
                return out
            if self.do_gen:
                era = None
                GenJetAK8 = events_jk.GenJetAK8
                GenJetAK8['p4']= ak.with_name(events_jk.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            else:
                firstidx = filename.find( "store/data/" )
                fname2 = filename[firstidx:]
                fname_toks = fname2.split("/")
                era = fname_toks[ fname_toks.index("data") + 1]
                self.logging.debug("IOV ", IOV, ", era ", era)
            self.logging.debug("starting jet corrections")
            #### Rewrite getjetcorrections to correct all jets and correct sd mass at the same time
            corrected_fatjets = GetJetCorrections(FatJet, events_jk, era, IOV, isData=not self.do_gen)
            corrected_fatjets = corrected_fatjets[corrected_fatjets.subJetIdx1 > -1]
            if ak.sum(ak.num(events_jk.SubJet.mass)>0)<1:
                self.logging.debug("No subjets")
                return out
            corrected_subjets = GetJetCorrections(events_jk.SubJet, events_jk, era, IOV, isData = not self.do_gen, mode = 'AK4')
            corrected_fatjets['msoftdrop'] =   (corrected_subjets[corrected_fatjets.subJetIdx1] + corrected_subjets[corrected_fatjets.subJetIdx2]).mass 

            # if not self.do_minimal:
            #     fill_hist(out, "sdmass_orig", dataset=dataset, **jkkw, ptreco=events_jk[(ak.num(events_jk.FatJet) > 2)].FatJet[:,2].pt, mreco=events_jk[(ak.num(events_jk.FatJet) > 2)].FatJet[:,2].msoftdrop)
            #     fill_hist(out, "sdmass_ak4corr", dataset=dataset, **jkkw, ptreco=corrected_fatjets[(ak.num(corrected_fatjets) > 2)][:,2].pt, mreco=corrected_fatjets[(ak.num(corrected_fatjets) > 2)][:,2].msoftdrop)
                # corrected_fatjets_ak8 = corrected_fatjets
                # corrected_fatjets_ak8 = GetCorrectedSDMass(corrected_fatjets, events_jk, era, IOV, isData=not self.do_gen, useSubjets = False)
                # fill_hist(out, "sdmass_ak8corr", dataset=dataset, **jkkw, ptreco=corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 2)][:,2].pt, mreco=corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 2)][:,2].msoftdrop)

            for jetsyst in self.jet_systematics:
                #####################################
                #### For each jet correction, we need to add JMR and JMS corrections on top (except if we're doing data).
                #####################################
                self.logging.debug("Jet syst running over: ", jetsyst)
                # The HEM/JER/JMR/JMS/JES branches below are gated on self.do_gen;
                # for data only 'nominal' assigns corr_jets_final. Skip everything
                # else explicitly rather than falling through to an UnboundLocalError.
                if not self.do_gen and jetsyst != 'nominal':
                    continue
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
                #################################################################
                #### sort corrected jets by pt before being put into events object
                #################################################################

                sortJets_ind = ak.argsort(corr_jets_final.pt, ascending=False)
                corr_jets_sorted = corr_jets_final[sortJets_ind]
                events_corr = ak.with_field(events_jk, corr_jets_sorted, "FatJet")  
                del corr_jets_sorted, corr_jets_final
                ###################################
                ######### INITIALIZE WEIGHTS AND SELECTION
                ##################################
                sel = PackedSelection()
                if (jetsyst == "nominal"): out['cutflow'][datastr]['nEvents initial'] += (len(events.FatJet))
                self.logging.debug("mctype ", mctype, " gen? ", self.do_gen)
                if self.do_gen:
                    if "LHEWeight" in events_corr.fields: 
                        weights = events_corr.LHEWeight.originalXWGTUP
                    else:
                        weights = events_corr.genWeight
                else:
                    ############
                    ### Doing data -- apply lumimask and require at least one jet to apply jet trigger prescales
                    ############
                    lumi_mask = getLumiMask(IOV)(events_corr.run, events_corr.luminosityBlock)
                    
                    events_corr = events_corr[lumi_mask]
                    weights = np.ones(len(events_corr))
                    if "ver2" in dataset:
                        trigsel, psweights, HLTflow_init, HLTflow_final = applyPrescales(events_corr, trigger= "PFJet", year = IOV)
                    else:
                        trigsel, psweights, HLTflow_init, HLTflow_final = applyPrescales(events_corr, year = IOV)
                    for path in HLTflow_init:
                        out['cutflow']['trigger_init'][path] += HLTflow_init[path]
                        out['cutflow']['trigger_final'][path] += HLTflow_final[path]
                    psweights=ak.where(ak.is_none(psweights), 1.0, psweights)
                    trigsel=ak.where(ak.is_none(trigsel), False, trigsel)
                    weights = ak.where(trigsel, psweights, weights)
                    sel.add("trigsel", trigsel)
                    if len(events_corr[trigsel])<1:
                        self.logging.debug("No events after golden json")
                        return out
                    if (jetsyst == "nominal"): 
                        out['cutflow'][datastr]['nEvents after trigger sel '] += (ak.sum(sel.all("trigsel")))
                        self.logging.debug("ADDED TRIGGER TO CUTFLOW FOR NOM FOR KEY ", dataset)
                if self.do_gen:
                    sel.add("npv", events_corr.PV.npvsGood > 0)
                else:
                    sel.add("npv", sel.all("trigsel") & (events_corr.PV.npvsGood > 0))
                ###################################
                ##### Intitialize and fill weights object
                ###################################
                
                weights_obj = Weights(len(weights))
                self.logging.debug("weights ", weights)
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
                                        #### Apply L1 prefiring weights
                    if "PSWeight" in events_corr.fields and 'herwig' not in dataset:
                        ISRNom, ISRUp, ISRDown = GetPSWeights(events_corr, shower="ISR")
                        weights_obj.add("isr", weight=ISRNom, weightUp=ISRUp, weightDown=ISRDown)
                        FSRNom, FSRUp, FSRDown = GetPSWeights(events_corr, shower="FSR")
                        weights_obj.add("fsr", weight=FSRNom, weightUp=FSRUp, weightDown=FSRDown)
                ###################################
                #### Apply MET filters
                ###################################
                METsel = np.array([events_corr.Flag[MET_filters[IOV][i]] for i in range(len(MET_filters[IOV])) if MET_filters[IOV][i] in events_corr.Flag.fields])
                METsel = np.logical_and.reduce(METsel, axis=0) ## a passing event should pass "ALL" the MET filters
                sel.add("METfilters", METsel)
                if ak.sum(METsel) < 1 :
                    self.logging.debug("No events passing MET filters.")
                    return out
                    
                #####################################
                #### Gen Jet Selection
                #################################### 
                if self.do_gen:
                    self.logging.debug("DOING GEN")
                    # if not self.do_minimal:
                    #     HT = ak.sum(events_corr.GenJetAK8.pt, axis=-1)
                        # fill_hist(out, "HT_nocuts", dataset=dataset, systematic=jetsyst, pt=HT)
                        # fill_hist(out, "HT_wXS", dataset=dataset, systematic=jetsyst, pt=HT, weight=weights)
                    # pt_cut_gen = genjet.pt > 200. #### removing
                    sel.add("triGenJet", (ak.num(events_corr.GenJetAK8) > 2)) # & pt_cut_gen ) ####removing to be consistent w/ aritra
                    GenJetAK8 = events_corr.GenJetAK8
                    GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                    genjet = ak.firsts(GenJetAK8[:,2:])  
                    rap_cut_gen = ak.where(sel.all("triGenJet"), np.abs(self._rapidity(genjet.p4)) < self.ycut, False)
                    sel.add("rapGen", rap_cut_gen)
                    # if not self.do_minimal:
                    #     fill_hist(out, "jet_rap_gen", dataset=dataset, syst = jetsyst, rapidity=getRapidity(GenJetAK8[sel.all("triGenJet")][:,2].p4), weight=weights[sel.all("triGenJet")])
                    #     fill_hist(out, "jet_phi_gen", dataset=dataset, systematic=jetsyst, phi=GenJetAK8[sel.all("triGenJet")][:,2].phi, weight=weights[sel.all("triGenJet")])  
                    #### get dphi and pt asymm selections                     
                    genjet1 = ak.firsts(events_corr.GenJetAK8[:,0:])
                    genjet2 = ak.firsts(events_corr.GenJetAK8[:,1:])
                    ##### genjet3 is already defined genjet
                    #### calculate dphi_min
                    dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
                    dphi13_gen = np.abs(genjet1.delta_phi(genjet))
                    dphi23_gen = np.abs(genjet2.delta_phi(genjet))
                    dphimin_gen = ak.min([dphi12_gen, dphi13_gen, dphi23_gen], axis = 0)
                    dphimin_gen_sel = ak.where(sel.all("triGenJet"), dphimin_gen > 1.0, False)
                    asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
                    sel.add("dphiGen", dphimin_gen_sel)
                    self.logging.debug("Finding min dphi done")
                    # if not self.do_minimal and jetsyst=='nominal':
                    #     fill_hist(out, "asymm_gen", dataset=dataset, systematic=jetsyst, ptgen=events_corr[sel.all("triGenJet")].GenJetAK8[:,2].pt, frac = asymm_gen[sel.all("triGenJet")], weight=weights[sel.all("triGenJet")])
                        # fill_hist(out, "dphimin_gen", dataset=dataset, systematic=jetsyst, dphi = dphimin_gen[sel.all("triGenJet")], weight = weights[sel.all("triGenJet")])
                    gensubjets = events_corr.SubGenJetAK8
                    groomed_genjet = get_gen_sd_mass_jet(ak.firsts(GenJetAK8[:,2:]), gensubjets)
                    sel.add("genTot_seq", sel.all("triGenJet", "dphiGen", "rapGen") & ~ak.is_none(genjet.mass) & ~ak.is_none(groomed_genjet.mass))
                    if (len(events_corr[sel.all("genTot_seq")]) < 1): 
                        self.logging.debug("No gen jets selected")
                        return out        
        
                #####################################
                #### Reco Jet Selection
                ####################################
            
                # if not self.do_minimal and jetsyst=='nominal':
                #     fill_hist(out, "njet_reco", dataset=dataset, syst = jetsyst, n=ak.to_numpy(ak.num(events_corr[sel.all("npv")].FatJet), allow_missing=True), 
                #                          weight = ak.to_numpy(weights[sel.all("npv")], allow_missing=True) )
                FatJet = events_corr.FatJet
                FatJet["p4"] = ak.with_name(events_corr.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                jet = ak.firsts(FatJet[:,2:])
                # pt_cut_reco = jet.pt > 200.    ### applying this just through binning of hist now
                sel.add("triRecoJet", (ak.num(events_corr.FatJet) > 2))
                sel.add("triRecoJet_seq", sel.all('npv', 'METfilters', 'triRecoJet'))
                rap_cut = np.abs(self._rapidity(jet.p4)) < self.ycut
                rap_sel = ak.where(sel.all("triRecoJet_seq"), rap_cut, False)
                sel.add("recoRap2p5", rap_sel)
                sel.add("recoRap_seq", sel.all("triRecoJet_seq", "recoRap2p5")) 
                self.logging.debug("nevents after rap cut ", ak.sum(sel.all("recoRap_seq")))
                # if not self.do_minimal and jetsyst=='nominal':
                #     fill_hist(out, "jet_rap_reco", dataset=dataset, syst = jetsyst, rapidity=ak.to_numpy(getRapidity(FatJet[sel.all("triRecoJet_seq")][:,2].p4), allow_missing=True), weight=weights[sel.all("triRecoJet_seq")])
                #     fill_hist(out, "jet_phi_reco", dataset=dataset, systematic=jetsyst, phi=FatJet[sel.all("triRecoJet_seq")][:,2].phi, weight=weights[sel.all("triRecoJet_seq")]) 

                #### ak.first fills empty values with none --> ak.singletons 
                jet1 = ak.firsts(events_corr.FatJet[:,0:])
                jet2 = ak.firsts(events_corr.FatJet[:,1:])
                jet3 = ak.firsts(events_corr.FatJet[:,2:])
                dphi12 = np.abs(jet1.delta_phi(jet2))
                dphi13 = np.abs(jet1.delta_phi(jet3))
                dphi23 = np.abs(jet2.delta_phi(jet3))
                dphimin = ak.min([dphi12, dphi13, dphi23], axis = 0)
                dphi_sel = ak.where(sel.all("triRecoJet_seq"), (dphimin > 1.0), False)
                sel.add("recodphimin", dphi_sel)
                sel.add("recodphi_seq", sel.all("recodphimin", "recoRap_seq"))
                asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
                if not self.do_minimal:
                    # fill_hist(out, "dphimin_reco", dataset=dataset, systematic=jetsyst, dphi = dphimin[sel.all("triRecoJet_seq")], weight=weights[sel.all("triRecoJet_seq")])
                    fill_hist(out, "asymm_reco", dataset=dataset, systematic=jetsyst, ptreco=events_corr[sel.all("triRecoJet_seq")].FatJet[:,2].pt, frac = asymm[sel.all("triRecoJet_seq")], weight=weights[sel.all("triRecoJet_seq")])
                #### Check that nearest pfmuon and is at least dR > 0.4 away
                # Get the nearest muon to that jet
                muon_sel =  ak.where(sel.all("triRecoJet_seq"), ak.all(jet3.delta_r(ak.singletons(jet3).nearest(events_corr.Muon))>0.4, axis=-1), False)
                sel.add("muonIso0p4", muon_sel)
                jetid_sel = ak.where(sel.all("triRecoJet_seq"), (jet3.jetId > 2), False)
                sel.add("jetId", jetid_sel)
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
                sel.add("recoTot_seq", sel.all("recodphi_seq", "jetId", "muonIso0p4", "hemveto") & ~ak.is_none(jet3.mass) & ~ak.is_none(jet3.msoftdrop))
                #### Check eta phi map pre cuts
                if not self.do_minimal and jetsyst=='nominal': 
                        fill_hist(out, 'jet_eta_phi_precuts', dataset=dataset, systematic=jetsyst, phi=events_corr[sel.all("triRecoJet")].FatJet[:,2].phi, eta=events_corr[sel.all("triRecoJet")].FatJet[:,2].eta, weight=weights[sel.all("triRecoJet")])                
                if (len(events_corr[sel.all("recoTot_seq")]) < 1): 
                    self.logging.debug("no events passing reco sel")
                    return out 
                ################
                #### Find fakes, misses, and underflow and remove them to get final selection
                ###############
                
                if self.do_gen:
                    jet = ak.firsts(events_corr.FatJet[:,2:])
                    matches = genjet.delta_r(jet) < 0.4
                    sel.add("matched_gen", matches)
                    misses = ~matches | sel.require(genTot_seq=True, recoTot_seq=False)
                    sel.add("misses", misses )
                    self.logging.debug("Number of misses ", ak.sum(misses))
                    miss_sel = misses & sel.all("genTot_seq")
                    self.logging.debug("Number of misses w/ 3 jets ", ak.sum(miss_sel))
                    if ak.sum(miss_sel) > 0 and not self.do_minimal:
                        if jetsyst == "nominal": 
                            out['cutflow'][datastr]['misses'] += (len(events_corr[miss_sel].GenJetAK8))
                        self.logging.debug("Number of none missed jets ", ak.sum(ak.is_none(GenJetAK8[miss_sel][:,2])))
                        ###### Applying misses selection to gen jets and getting sd mass
                        miss_jets = events_corr[miss_sel].GenJetAK8[:,2]
                        groomed_missjet = get_gen_sd_mass_jet(miss_jets, events_corr[miss_sel].SubGenJetAK8)
                        miss_weights = weights[miss_sel]
                        fill_hist(out, "misses_u", dataset=dataset, systematic=jetsyst, **jkkw, ptgen = miss_jets[~ak.is_none(miss_jets.mass)].pt, mgen = miss_jets[~ak.is_none(miss_jets.mass)].mass, weight = miss_weights[~ak.is_none(miss_jets.mass)])
                        fill_hist(out, "misses_g", dataset=dataset, systematic=jetsyst, **jkkw, ptgen = miss_jets[~ak.is_none(groomed_missjet.mass)].pt, mgen = groomed_missjet[~ak.is_none(groomed_missjet.mass)].mass, weight = miss_weights[~ak.is_none(groomed_missjet.mass)])
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "matched_gen")])<1: 
                        self.logging.debug("No events after all selections and removing misses")
                        return out
                    #### Fakes include events missing a reco mass or sdmass value, events failing index dr matching, and events passing reco cut but failing the gen cut
                    matches = ~ak.is_none(jet.matched_gen)
                    sel.add("matched_reco", matches)
                    self.logging.debug("matched gen jets to reco ", matches)
                    fakes = ~matches | sel.require(genTot_seq=False, recoTot_seq=True)
                    sel.add("fakes", fakes)
                    fake_sel = sel.all("recoTot_seq") & fakes
                    if len(weights[fake_sel])>0 and not self.do_minimal:
                        fake_jets = events_corr[fake_sel].FatJet[:,2]
                        fake_weights = weights_obj.weight()[fake_sel]
                        fill_hist(out, "fakes_u", dataset=dataset, systematic=jetsyst, **jkkw, ptreco = fake_jets[~ak.is_none(fake_jets.mass)].pt, mreco = fake_jets[~ak.is_none(fake_jets.mass)].mass, weight = fake_weights[~ak.is_none(fake_jets.mass)])
                        fill_hist(out, "fakes_g", dataset=dataset, systematic=jetsyst, **jkkw, ptreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].pt, mreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].msoftdrop, weight = fake_weights[~ak.is_none(fake_jets.msoftdrop)])
                        if jetsyst=="nominal":
                            for syst in self._weight_variations(weights_obj):
                                self.logging.debug("Weight variation: ", syst)
                                fake_weights = weights_obj.weight(syst)[fake_sel]
                                fill_hist(out, "fakes_u", dataset=dataset, systematic=syst, **jkkw, ptreco = fake_jets[~ak.is_none(fake_jets.mass)].pt, mreco = fake_jets[~ak.is_none(fake_jets.mass)].mass, weight = fake_weights[~ak.is_none(fake_jets.mass)])
                                fill_hist(out, "fakes_g", dataset=dataset, systematic=syst, **jkkw, ptreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].pt, mreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].msoftdrop, weight = fake_weights[~ak.is_none(fake_jets.msoftdrop)])
                    if (jetsyst == "nominal"): 
                        out['cutflow'][datastr]['fakes'] += (len(events_corr[fakes].FatJet))
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "matched_reco", "matched_gen")])<1: 
                        self.logging.debug("No events after all selections and removing fakes & misses")
                        return out
                    ##############
                    ### Make final selection and fill gen truth plots
                    ###############
                    sel.add("final_seq", sel.all("genTot_seq", "recoTot_seq", "matched_reco", "matched_gen"))

                    genjet = events_corr[sel.all("genTot_seq","matched_gen")].GenJetAK8[:,2]
                    groomed_genjet = get_gen_sd_mass_jet(genjet, events_corr[sel.all("genTot_seq","matched_gen")].SubGenJetAK8)
                    gen_weights = weights[sel.all("genTot_seq","matched_gen")]
                    fill_hist(out, 'ptjet_mjet_u_gen', dataset=dataset, systematic=jetsyst, **jkkw, ptgen=genjet.pt, mgen=genjet.mass, weight=gen_weights )
                    fill_hist(out, 'ptjet_mjet_g_gen', dataset=dataset, systematic=jetsyst, **jkkw, ptgen=genjet.pt, mgen=groomed_genjet.mass, weight=gen_weights )
                    fill_hist(out, 'ptjet_rhojet_u_gen', dataset=dataset, systematic=jetsyst, **jkkw, ptgen=genjet.pt, mpt_gen=self._rho(genjet.mass, genjet.pt), weight=gen_weights )
                    fill_hist(out, 'ptjet_rhojet_g_gen', dataset=dataset, systematic=jetsyst, **jkkw, ptgen=genjet.pt, mpt_gen=self._rho(groomed_genjet.mass, genjet.pt), weight=gen_weights )
                    #######################
                else:
                    sel.add("final_seq", sel.all("recoTot_seq"))

                #######################
                #### Apply final selections and jet veto map
                #######################
                if len(events_corr[sel.all("final_seq")])<1:
                        self.logging.debug("no more events after final sel")
                        return out
                
                #### Check eta phi map after cuts but before jet veto
                #### Make eta phi plot to check effects of cuts
                # if not self.do_minimal: fill_hist(out, 'jet_eta_phi_preveto', dataset=dataset, systematic=jetsyst, phi=events_corr.FatJet[:,2].phi, eta=events_corr.FatJet[:,2].eta, weight=weights)      
                
                #### Apply jet veto map
                # jet = events_corr.FatJet[:,2]
                # veto = ApplyVetoMap(IOV, jet, mapname='jetvetomap')
                # events_corr = events_corr[veto]
                # weights = weights[veto]
                # if len(events_corr)<1:
                #         print("no more events after jet veto")
                #         return out
 
                #######################
                #### Get final jets and weights and fill final plots
                #######################
                final_weights = weights_obj.weight()[sel.all("final_seq")]
                jet = events_corr[sel.all("final_seq")].FatJet[:,2]
                    
                ##################
                #### Apply final selections to GEN and fill any plots requiring gen, including resp. matrices
                ##################
                
                if self.do_gen:
                    ##### define reco-only jets to find ptjet_mjet_*_reco to get fakes
                    recojet = events_corr[sel.all("recoTot_seq", "matched_reco")].FatJet[:,2]
                    reco_weights = weights_obj.weight()[sel.all("recoTot_seq", "matched_reco")]
                    #### define gen jets with final selection for response matrix
                    final_events = events_corr[sel.all("final_seq")]
                    genjet = final_events.FatJet[:,2].matched_gen
                    groomed_genjet = get_gen_sd_mass_jet(genjet, final_events.SubGenJetAK8)
                    weird_jets = final_events[(final_events.GenJetAK8[:,2].mass < 20.) & (final_events.FatJet[:,2].mass >20.)]
                    if jetsyst == "nominal": out['cutflow'][datastr]['nEvents weird (mreco>20, mgen<20) ungroomed'] += len(weird_jets)
                    #### plots to check backgrounds and eta phi dists
                    if not self.do_minimal and jetsyst=='nominal':
                        #### plots for checking MET/sumET --> potentially need cut <0.3job
                        fill_hist(out, "MET_over_sumET_pt_reco", dataset=dataset,systematic=jetsyst, frac=events_corr[sel.all("final_seq")].MET.pt/events_corr[sel.all("final_seq")].MET.sumEt, ptreco=jet.pt, weight=final_weights )
                        fill_hist(out, "MET_pt_reco", dataset=dataset,systematic=jetsyst, pt=events_corr[sel.all("final_seq")].MET.pt, ptreco=jet.pt, weight=final_weights )
                        #### plots for checking whether jet veto map is needed
                        HT = ak.sum(events_corr[sel.all("final_seq")].GenJetAK8.pt, axis=-1)
                        fill_hist(out, "HT_aftercuts", dataset=dataset, systematic=jetsyst, pt=HT, weight=final_weights)
                        # fill_hist(out, "ptreco_mreco_fine_u", dataset=dataset,systematic=jetsyst, **jkkw, pt=jet.pt, mass=jet.mass, weight=reco_weights )
                        # fill_hist(out, "ptreco_mreco_fine_g", dataset=dataset,systematic=jetsyst, **jkkw, pt=jet.pt, mass=jet.msoftdrop, weight=reco_weights )
                        
                    fill_hist(out, "response_matrix_u", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=jet.pt, ptgen=genjet.pt, 
                                                  mreco=jet.mass, mgen=genjet.mass, weight = final_weights)
                    fill_hist(out, "response_matrix_g", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=jet.pt, ptgen=genjet.pt, 
                                                  mreco=jet.msoftdrop, mgen=groomed_genjet.mass, weight = final_weights )
                    fill_hist(out, "response_matrix_rho_u", dataset=dataset, systematic=jetsyst, **jkkw,  mpt_reco=self._rho(jet.mass, jet.pt), mpt_gen=self._rho(genjet.mass, genjet.pt), ptreco=jet.pt, ptgen=genjet.pt, weight=final_weights)
                    fill_hist(out, "response_matrix_rho_g", dataset=dataset, systematic=jetsyst, **jkkw,  mpt_reco=self._rho(jet.msoftdrop, jet.pt), mpt_gen=self._rho(groomed_genjet.mass, genjet.pt), ptreco=jet.pt, ptgen=genjet.pt, weight=final_weights)
                    fill_hist(out, "ptjet_mjet_u_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=recojet.pt, mreco=recojet.mass, weight=reco_weights)
                    fill_hist(out, "ptjet_mjet_g_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=recojet.pt, mreco=recojet.msoftdrop, weight=reco_weights )
                    fill_hist(out, "ptjet_rhojet_u_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=recojet.pt, mpt_reco=self._rho(recojet.mass, recojet.pt), weight=reco_weights)
                    fill_hist(out, "ptjet_rhojet_g_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=recojet.pt,mpt_reco=self._rho(recojet.msoftdrop, recojet.pt), weight=reco_weights )
                    if not self.do_minimal:
                        fill_hist(out, "jet_pt_eta_phi", dataset=dataset, systematic=jetsyst, ptreco=jet.pt, phi=jet.phi, eta=jet.eta, weight=final_weights)
                        fill_hist(out, 'm_u_jet_reco_over_gen', dataset=dataset, ptgen=genjet.pt, mgen=genjet.mass, frac = jet.mass/genjet.mass, 
                                                           weight = final_weights)
                        fill_hist(out, 'm_g_jet_reco_over_gen', dataset=dataset, ptgen=genjet.pt, mgen=groomed_genjet.mass, 
                                                           frac = jet.msoftdrop/groomed_genjet.mass, weight = final_weights)
                    if jetsyst=="nominal":
                        for syst in self._weight_variations(weights_obj):
                            self.logging.debug("Weight variation: ", syst)
                            reco_weights = weights_obj.weight(syst)[sel.all("recoTot_seq", "matched_reco")]
                            final_weights = weights_obj.weight(syst)[sel.all("final_seq")]
                            #fill nominal, up, and down variations for each          
                            fill_hist(out, "response_matrix_u", dataset=dataset, systematic=syst, **jkkw,ptreco=jet.pt, mreco=jet.mass, ptgen=genjet.pt, mgen=genjet.mass, weight=final_weights)
                            fill_hist(out, "response_matrix_g", dataset=dataset, systematic=syst, **jkkw, ptreco=jet.pt, mreco=jet.msoftdrop, ptgen=genjet.pt, mgen=groomed_genjet.mass, weight=final_weights)
                            fill_hist(out, "response_matrix_rho_u", dataset=dataset, systematic=syst, **jkkw,  mpt_reco=self._rho(jet.mass, jet.pt), mpt_gen=self._rho(genjet.mass, genjet.pt), ptreco=jet.pt, ptgen=genjet.pt, weight=final_weights)
                            fill_hist(out, "response_matrix_rho_g", dataset=dataset, systematic=syst, **jkkw,  mpt_reco=self._rho(jet.msoftdrop, jet.pt),mpt_gen=self._rho(groomed_genjet.mass, genjet.pt), ptreco=jet.pt, ptgen=genjet.pt, weight=final_weights)
                            fill_hist(out, "ptjet_mjet_u_reco", dataset=dataset, systematic=syst, **jkkw, ptreco=recojet.pt, 
                                                          mreco=recojet.mass, weight=reco_weights )
                            fill_hist(out, "ptjet_mjet_g_reco", dataset=dataset, systematic=syst, **jkkw, ptreco=recojet.pt, 
                                                       mreco=recojet.msoftdrop, weight=reco_weights )
                            fill_hist(out, "ptjet_rhojet_u_reco", dataset=dataset, systematic=syst, **jkkw, ptreco=recojet.pt, mpt_reco=self._rho(recojet.mass, recojet.pt), weight=reco_weights)
                            fill_hist(out, "ptjet_rhojet_g_reco", dataset=dataset, systematic=syst, **jkkw, ptreco=recojet.pt,mpt_reco=self._rho(recojet.msoftdrop, recojet.pt), weight=reco_weights )
                            if not self.do_minimal:
                                fill_hist(out, "jet_pt_eta_phi", dataset=dataset, systematic=syst, ptreco=jet.pt, phi=jet.phi, eta=jet.eta, weight=final_weights)
                        #### Gluon purity plots        
                        jet1flav = getJetFlavors(events_corr[sel.all("final_seq")].FatJet[:,0])
                        jet2flav = getJetFlavors(events_corr[sel.all("final_seq")].FatJet[:,1])
                        jet3flav = getJetFlavors(events_corr[sel.all("final_seq")].FatJet[:,2])
                        genjet1 = events_corr[sel.all("final_seq")].FatJet[:,0].matched_gen
                        genjet2 = events_corr[sel.all("final_seq")].FatJet[:,1].matched_gen
                        jet3 = events_corr[sel.all("final_seq")].FatJet[:,2]
                        jet3_bb = jet3[(np.abs(genjet1.partonFlavour) == 5) & (np.abs(genjet2.partonFlavour) == 5)]
                        jet3_b = jet3[(np.abs(genjet1.partonFlavour) == 5)]
                        jet3_jetbb_flav = getJetFlavors(jet3_bb)
                        jet3_jetb_flav = getJetFlavors(jet3_b)
                        
                        jets = {"jet1":jet1flav, "jet2":jet2flav,  "jet3":jet3flav, "jet3_bb":jet3_jetbb_flav, "jet3_b":jet3_jetb_flav}
                        if not self.do_minimal:
                            for flavor in jet1flav.keys():
                                for jetname, jetobj in jets.items():
                                    jetobj[flavor] = jetobj[flavor][~ak.is_none(jetobj[flavor])]
                                    fill_hist(out, 'alljet_ptreco_mreco', dataset=dataset, jetNumb = jetname, partonFlav = flavor, 
                                                                    mreco = jetobj[flavor].mass, 
                                                                    ptreco = jetobj[flavor].pt)
                                    fill_hist(out, 'btag_eta', dataset=dataset, jetNumb = jetname, partonFlav = flavor, 
                                                         frac = jetobj[flavor].btagDeepB, eta = jetobj[flavor].eta )
                        out['cutflow'][datastr]['nGluonJets'] += (len(jet3flav["Gluon"])+len(jet1flav["Gluon"])+len(jet2flav["Gluon"]))
                        out['cutflow'][datastr]['nJets'] += (len(events_corr[sel.all("final_seq")].FatJet[:,0])+len(events_corr[sel.all("final_seq")].FatJet[:,1])+len(events_corr[sel.all("final_seq")].FatJet[:,2]))
                        out['cutflow'][datastr]['nSoftestGluonJets'] += (len(jet3flav["Gluon"]))
                        out['cutflow'][datastr]['nSoftestGluonJets_b'] += (len(jet3_jetb_flav["Gluon"]))
                        out['cutflow'][datastr]['nSoftestGluonJets_bb'] += (len(jet3_jetbb_flav["Gluon"]))
                        out['cutflow'][datastr]['nSoftestJets_b'] += (len(jet3_b))
                        out['cutflow'][datastr]['nSoftestJets_bb'] += (len(jet3_bb))
                        out['cutflow'][datastr]['n3Jets'] += (len(events_corr[sel.all("final_seq")].FatJet[:,2].pt))
                ###############
                ##### If running over DATA fill only final reco plots
                ###############
                
                else:
                    fill_hist(out, "ptjet_mjet_u_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=jet.pt, mreco=jet.mass, weight=final_weights  )
                    fill_hist(out, "ptjet_mjet_g_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=jet.pt, mreco=jet.msoftdrop, weight=final_weights  )
                    fill_hist(out, "ptjet_rhojet_u_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=jet.pt, mpt_reco=self._rho(jet.mass, jet.pt), weight=final_weights)
                    fill_hist(out, "ptjet_rhojet_g_reco", dataset=dataset, systematic=jetsyst, **jkkw, ptreco=jet.pt,mpt_reco=self._rho(jet.msoftdrop, jet.pt), weight=final_weights )
                    if not self.do_minimal:
                        # fill_hist(out, "ptreco_mreco_fine_u", dataset=dataset,systematic=jetsyst, **jkkw, pt=jet.pt, mass=jet.mass, weight=reco_weights  )
                        # fill_hist(out, "ptreco_mreco_fine_g", dataset=dataset,systematic=jetsyst, **jkkw, pt=jet.pt, mass=jet.msoftdrop, weight=reco_weights  )
                        fill_hist(out, "MET_over_sumET_pt_reco", dataset=dataset,systematic=jetsyst, frac=events_corr[sel.all("final_seq")].MET.pt/events_corr[sel.all("final_seq")].MET.sumEt, ptreco=jet.pt, weight=final_weights  )
                        fill_hist(out, "MET_pt_reco", dataset=dataset,systematic=jetsyst, pt=events_corr[sel.all("final_seq")].MET.pt, ptreco=jet.pt, weight=final_weights )
                        fill_hist(out, "jet_pt_eta_phi", dataset=dataset, systematic=jetsyst, ptreco=jet.pt, phi=jet.phi, eta=jet.eta, weight=final_weights )
                        HT = ak.sum(events_corr[sel.all("final_seq")].FatJet.pt, axis=-1)
                        fill_hist(out, "HT_aftercuts", dataset=dataset, systematic=jetsyst, pt=HT, weight=final_weights )
                self.logging.debug("final jets ", jet)
                self.logging.debug("final jet pt ", jet.pt)
                self.logging.debug("final number of events added to hists " , len(events_corr))
                if (jetsyst == "nominal"):
                    for name in sel.names:
                        out["cutflow"][datastr][name] += sel.all(name).sum()
                        self.logging.debug("ADDED ", name, " TO CUTFLOW")
                negMSD = jet.msoftdrop<0.
                self.logging.debug("Number of negative softdrop values ", ak.sum(negMSD) )
                if (jetsyst == "nominal"): 
                    out['cutflow'][datastr]['nEvents failing softdrop condition'] += ak.sum(negMSD)
                    self.logging.debug("ADDED NEG SD EVENTS TO CUTFLOW")
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
