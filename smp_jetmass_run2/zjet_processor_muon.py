import awkward as ak
import numpy as np
import time
import uproot
import hist
import vector
import time
import os
import pandas as pd
import time
from coffea import util, processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict
import gc
import tokenize as tok
import re
import copy


from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from .hist_utils import *
from .smp_utils import *


from .corrections import *

class Log:
    def __init__(self, mode="info"):
        self.mode = mode
    def info(self, msg):
        if self.mode in ["info", "debug"]:
            print("[INFO]", msg)
    def debug(self, msg):
        if self.mode == "debug":
            print("[DEBUG]", msg)


class ZplusJetProcessor(processor.ProcessorABC):
        def __init__(self, do_gen = True, mode = "minimal",  debug = False, jet_systematics = None, systematics = None):

            self._do_gen = do_gen
            self._mode = mode
            self._debug = debug
            self._do_reweight = False
            self._do_jk = False

            ## List jet systematics
            if jet_systematics == None:
                self.jet_systematics = ['nominal', 'JERUp', 'JERDown', 'JMSUp', 'JMSDown', 'JMRUp', 'JMRDown']
                jes_systematics = ['JES_AbsoluteMPFBiasUp', 'JES_AbsoluteMPFBiasDown', 'JES_AbsoluteScaleUp', 'JES_AbsoluteScaleDown',
                    'JES_AbsoluteStatUp', 'JES_AbsoluteStatDown', 'JES_FlavorQCDUp', 'JES_FlavorQCDDown', 'JES_FragmentationUp',
                    'JES_FragmentationDown', 'JES_PileUpDataMCUp', 'JES_PileUpDataMCDown', 'JES_PileUpPtBBUp', 'JES_PileUpPtBBDown',
                    'JES_PileUpPtEC1Up', 'JES_PileUpPtEC1Down', 'JES_PileUpPtEC2Up', 'JES_PileUpPtEC2Down', 'JES_PileUpPtHFUp', 'JES_PileUpPtHFDown', 
                    'JES_PileUpPtRefUp', 'JES_PileUpPtRefDown', 'JES_RelativeFSRUp', 'JES_RelativeFSRDown', 'JES_RelativeJEREC1Up',
                    'JES_RelativeJEREC1Down', 'JES_RelativeJEREC2Up', 'JES_RelativeJEREC2Down', 'JES_RelativeJERHFUp', 'JES_RelativeJERHFDown',
                    'JES_RelativePtBBUp', 'JES_RelativePtBBDown', 'JES_RelativePtEC1Up', 'JES_RelativePtEC1Down', 'JES_RelativePtEC2Up', 'JES_RelativePtEC2Down',
                    'JES_RelativePtHFUp', 'JES_RelativePtHFDown', 'JES_RelativeBalUp', 'JES_RelativeBalDown', 'JES_RelativeSampleUp', 'JES_RelativeSampleDown', 
                    'JES_RelativeStatECUp', 'JES_RelativeStatECDown', 'JES_RelativeStatFSRUp', 'JES_RelativeStatFSRDown', 'JES_RelativeStatHFUp', 'JES_RelativeStatHFDown',
                    'JES_SinglePionECALUp', 'JES_SinglePionECALDown', 'JES_SinglePionHCALUp', 'JES_SinglePionHCALDown', 'JES_TimePtEtaUp', 'JES_TimePtEtaDown']
                self.jet_systematics = self.jet_systematics + jes_systematics
            else:
                self.jet_systematics = jet_systematics


            if systematics == None:
                self.systematics = ['nominal', 'puUp', 'puDown' , 'elerecoUp', 'elerecoDown', 
                                        'eleidUp', 'eleidDown', 'eletrigUp', 'eletrigDown', 'murecoUp', 'murecoDown', 
                                        'muidUp', 'muidDown', 'mutrigUp', 'mutrigDown', 'muisoUp', 'muisoDown',
                                        'pdfUp', 'pdfDown', 'q2Up', 'q2Down', 'isrUp', 'isrDown', 'fsrUp', 'fsrDown',
                                        'l1prefiringUp', 'l1prefiringDown'] 
            else:
                self.systematics = systematics
            
    
            self.lepptcuts = [40,29] # [ele, mu]

            if self._debug:
                self.logging = Log(mode="debug")
            else:
                self.logging = Log(mode="info")

            binning = util_binning()

            # Define axes
            ptreco_axis = binning.ptreco_axis
            mreco_axis = binning.mreco_axis
            ptgen_axis = binning.ptgen_axis     
            mgen_axis = binning.mgen_axis
    
            dataset_axis = binning.dataset_axis
            lep_axis = binning.lep_axis
            n_axis = binning.n_axis
            mass_axis = binning.mass_axis
            zmass_axis = binning.zmass_axis
            pt_axis = binning.pt_axis
            frac_axis = binning.frac_axis
            dr_axis = binning.dr_axis
            dr_fine_axis = binning.dr_fine_axis
            dphi_axis = binning.dphi_axis    
            syst_axis = binning.syst_axis
            eta_axis = binning.eta_axis
            phi_axis = binning.phi_axis
            ptfine_axis = binning.ptfine_axis
            mreco_over_pt_axis = binning.mreco_over_pt_axis
            mgen_over_pt_axis = binning.mgen_over_pt_axis
            
            ptgen_axis_fine  = binning.ptgen_axis_fine 
    
            
            mptreco_axis = binning.mreco_over_pt_axis
            mptgen_axis = binning.mgen_over_pt_axis

            ht_axis = hist.axis.StrCategory([],growth = True, name = "ht_bin", label = "h_T bin")
            channel_axis = hist.axis.StrCategory([],growth = True, name = "channel", label = "Channel")
            
            weight_axis = hist.axis.Regular(100, 0, 10, name="corrWeight", label=r"Weight")
            met_pt_axis = hist.axis.Regular(20, 0, 200, name = "pt", label = r"$p_T$")
    
            mcut_reco_u_axis = binning.mcut_reco_u_axis
            mcut_reco_g_axis = binning.mcut_reco_g_axis
    
            mcut_gen_u_axis = binning.mcut_gen_u_axis
            mcut_gen_g_axis = binning.mcut_gen_g_axis

            # introduce accumulator
            self.hists = processor.dict_accumulator()

            if self._mode == "validation":
                register_hist(self.hists, "pt_mu0", [dataset_axis, pt_axis, syst_axis])
                register_hist(self.hists, "pt_Z", [dataset_axis, pt_axis, syst_axis])
                register_hist(self.hists, "pt_el0", [dataset_axis, pt_axis, syst_axis])
                self.hists["sumw"] = processor.defaultdict_accumulator(float)
                self.hists["nev"] = processor.defaultdict_accumulator(int)
                self.hists["cutflow"] = processor.defaultdict_accumulator(int)
                self.logging.debug(f"Registered Histograms {self.hists.keys()}")


        @property
        def accumulator(self):
            return self.hists

        def process(self, events_all):
            self.logging.debug(f"Systematics {self.systematics}")
            self.logging.debug(f"Current Mode {self._mode}")
            t0 = time.time()
            dataset = events_all.metadata["dataset"]
            filename = events_all.metadata['filename']
            if "SingleMuon" in filename:
                self._do_gen = False
            elif "pythia" in filename:
                self._do_gen = True
            self.logging.info(f"Starting processing for dataset: {dataset} and file: {filename}")

        

            self.logging.debug(f"Total events in chunk: {len(events_all)}")

            IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'UL2016APV', dataset)))
                else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
                else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
                else '2016')

            self.IOV = IOV

            self.hists["cutflow"][f"{dataset}_all"] += len(events_all)
            self.hists["nev"][dataset] += len(events_all)
            if self._do_gen:
                self.hists["sumw"][dataset] += ak.sum(events_all.genWeight)
            else:
                self.hists["sumw"][dataset] += len(events_all)

            
            index_list = np.arange(len(events_all))

            for jk_index in range(0, 10):
                if not self._do_jk:
                    events1 = events_all
                    self.logging.debug("Jackknife resampling not enabled, processing all events together.")
                if self._do_jk:
                    jk_sel = ak.where( (index_list % 10) == jk_index, False, True)
                    #self.logging.debug("JK index ", jk_index, " events dropped ", ak.sum(~jk_sel) )
                    events1 = events_all[jk_sel]
                    del jk_sel

                sel = PackedSelection(dtype='uint64')

                if (self._do_gen):
                    era = None
                    firstidx = filename.find( "store/mc/" )
                    fname2 = filename[firstidx:]
                    fname_toks = fname2.split("/")
                    year = fname_toks[ fname_toks.index("mc") + 1]
                    ht_bin = fname_toks[ fname_toks.index("mc") + 2]
                    ht_bin_tokens = ht_bin.split("_")
                    ht_bin = ht_bin_tokens[ht_bin_tokens.index("DYJetsToLL")+2]
                    

                    ## Flag used for number of events
                    herwig = False
                    if 'herwig' in filename: herwig = True
                    self.herwig = herwig

                ## Finding event variables
                if not self._do_gen:
                    firstidx = filename.find( "store/data/" )
                    fname2 = filename[firstidx:]
                    fname_toks = fname2.split("/")
                    era = fname_toks[ fname_toks.index("data") + 1]
                    channel = fname_toks[ fname_toks.index('NANOAOD') - 1]
                    ht_bin = 'all'
                    self.lumimasks = getLumiMaskRun2()
                    lumi_mask = self.lumimasks[IOV](events1.run, events1.luminosityBlock)

                    events1 = events1[lumi_mask]

                    self.logging.info("Channel {}".format(channel))
                    if "UL2016" in dataset: 
                        if channel == "SingleMuon":
                            trigsel = events1.HLT.IsoMu24  
                        else:
                            self.logging.info("Doing {channel} in {dataset}")
                            trigsel = events1.HLT.Ele27_WPTight_Gsf | events1.HLT.Photon175
                    elif "UL2017" in dataset:
                        if channel == "SingleMuon":
                            trigsel = events1.HLT.IsoMu27  
                        else:
                            self.logging.info(f"Doing {channel} in {dataset}")
                            trigsel = events1.HLT.Ele35_WPTight_Gsf | events1.HLT.Photon200
                    elif "UL2018" in dataset:
                        if channel == "SingleMuon":
                            trigsel = events1.HLT.IsoMu24  
                        else:
                            self.logging.info("Doing {channel} in {dataset}")
                            trigsel = events1.HLT.Ele32_WPTight_Gsf | events1.HLT.Photon200
                    else:
                        raise Exception("Dataset is incorrect, should have 2016, 2017, 2018: ", dataset)
                    sel.add("trigsel", trigsel)    

                    self.logging.debug(f"Trigger Selection {ak.sum(sel.require(trigsel = True))}")
                else:
                    self.logging.info(f"year: {year}, ht_bin: {ht_bin}, herwig: {herwig}")

                events0 = events1

                weights = Weights(size = len(events0), storeIndividual = True) #initialize weights class
                weights.add("unity", np.ones(len(events0)))
                self.logging.debug("Weights initialized")
                
                if len(events0) <1:
                    return self.hists

                self.logging.info("Entering RECO selection")


                ## ALL RECO SELECTIONS
                
                #npv cut
                sel.add('npv', events0.PV.npvsGood > 0)

                ## Applying MET Filters

                #####################################
                ### MET Filters ####################
                #####################################

                self.logging.info("Applying MET filters")

                MET_filters = {'2016APV':["goodVertices",
                                    "globalSuperTightHalo2016Filter",
                                    "HBHENoiseFilter",
                                    "HBHENoiseIsoFilter",
                                    "EcalDeadCellTriggerPrimitiveFilter",
                                    "BadPFMuonFilter",
                                    "BadPFMuonDzFilter",
                                    "eeBadScFilter",
                                    "hfNoisyHitsFilter"],
                        '2016'   :["goodVertices",
                                    "globalSuperTightHalo2016Filter",
                                    "HBHENoiseFilter",
                                    "HBHENoiseIsoFilter",
                                    "EcalDeadCellTriggerPrimitiveFilter",
                                    "BadPFMuonFilter",
                                    "BadPFMuonDzFilter",
                                    "eeBadScFilter",
                                    "hfNoisyHitsFilter"],
                        '2017'   :["goodVertices",
                                    "globalSuperTightHalo2016Filter",
                                    "HBHENoiseFilter",
                                    "HBHENoiseIsoFilter",
                                    "EcalDeadCellTriggerPrimitiveFilter",
                                    "BadPFMuonFilter",
                                    "BadPFMuonDzFilter",
                                    "hfNoisyHitsFilter",
                                    "eeBadScFilter",
                                    "ecalBadCalibFilter"],
                        '2018'   :["goodVertices",
                                    "globalSuperTightHalo2016Filter",
                                    "HBHENoiseFilter",
                                    "HBHENoiseIsoFilter",
                                    "EcalDeadCellTriggerPrimitiveFilter",
                                    "BadPFMuonFilter",
                                    "BadPFMuonDzFilter",
                                    "hfNoisyHitsFilter",
                                    "eeBadScFilter",
                                    "ecalBadCalibFilter"]}

                selectEvents = np.array([events0.Flag[MET_filters[IOV][i]] for i in range(len(MET_filters[IOV]))
                        if MET_filters[IOV][i] in events0.Flag.fields])
                selectEvents = np.logical_and.reduce(selectEvents, axis=0) ## a passing event should pass "ALL" the MET filters
                


                self.logging.debug("MET Filter applied")
                sel.add("MET", selectEvents)
                sel.add("MET_seq", sel.all('npv', 'MET')) # This is RECO only
                ## introducing weights
                if self._do_gen:
                    pu_nom, pu_up, pu_down = get_pu_weights(events0, IOV)
                    self.logging.debug(f"PU weights (nom, up, down) : {pu_nom[:10]}")
                    
                    weights.add(name = "pu", weight = pu_nom, weightUp = pu_up, weightDown = pu_down)
                    pdf_nom, pdf_up, pdf_down = GetPDFweights(events0)
                    self.logging.debug(f"pdf weights (nom, up, down) : {pdf_nom[:10]}")
                    weights.add(name = "pdf", weight = pdf_nom, weightUp = pdf_up, weightDown = pdf_down)
                    l1prefire_nom, l1prefire_up, l1prefire_down = GetL1PreFiringWeight(IOV, events0)
                    self.logging.debug(f"L1 prefiring weights (nom, up, down) : {l1prefire_nom[:10]}")
                    weights.add(name = "l1prefiring", weight = l1prefire_nom, weightUp = l1prefire_up, weightDown = l1prefire_down)
                    q2_nom, q2_up, q2_down = GetQ2weights(events0)
                    self.logging.debug(f"Q2 weights (nom, up, down) : {q2_nom[:10]}")
                    weights.add(name = "q2", weight = q2_nom, weightUp = q2_up, weightDown = q2_down)
    
                    isr_nom, isr_up, isr_down = GetPSweights(events0, "ISR")
                    weights.add(name = "isr", weight = isr_nom, weightUp = isr_up, weightDown = isr_down)
    
                    fsr_nom, fsr_up, fsr_down = GetPSweights(events0, "FSR")
                    weights.add(name = "fsr", weight = fsr_nom, weightUp = fsr_up, weightDown = fsr_down)
                # introducing electrons 
                eta = np.abs(events0.Electron.eta)
                events0 = ak.with_field(
                    events0,
                    events0.Electron[
                        (events0.Electron.pt > self.lepptcuts[0])
                        & (eta < 2.5)
                        & ((eta < 1.422) | (eta > 1.566))      # exclude ECAL crack
                        & (events0.Electron.pfRelIso03_all < 0.25)  # suppressing isolation cut here
                        & (events0.Electron.cutBased > 3)      # tight: 4, medium: 3
                        & (np.abs(events0.Electron.dz) < 0.5)
                        & (np.abs(events0.Electron.dxy) < 0.2)
                    ],
                    "Electron",
                )

                print("Number of electron", ak.num(events0.Electron, axis = 1))

                events0 = ak.with_field(
                        events0,
                        events0.Muon[(events0.Muon.pt > self.lepptcuts[1]) 
                                    &(np.abs(events0.Muon.eta) < 2.5)
                                    &(events0.Muon.pfIsoId > 3) #tight iso, pfIso04 < 0.2 , 2 = loose, 3 = medium # supressing isolation cut here
                                    #&(events0.Muon.miniIsoId > 1)
                                    &(events0.Muon.tightId	 == True)
                                    & (np.abs(events0.Muon.dz) < 0.5) ## dz < 0.2 cm to prevent pileup
                                    & (np.abs(events0.Muon.dxy) < 0.2)
                                    #&(events0.Muon.looseId	 == True)
                        
                        ],
                        "Muon"
                    )
                
                self.logging.debug("Leptons Selected")

                z_reco = get_z_reco_selection(events0, sel, self.lepptcuts[0], self.lepptcuts[1], None, None)
                has_z = sel.require(twoReco_leptons=True)
                z_ptcut_reco  = has_z & (z_reco.pt > 90)
                z_mcut_reco   = has_z & (z_reco.mass > 71.) & (z_reco.mass < 111.)
                
                sel.add("z_ptcut_reco", z_ptcut_reco)
                sel.add("z_mcut_reco",  z_mcut_reco)
                z_ptcut_reco = z_reco.pt > 90
                z_mcut_reco = (z_reco.mass > 71.) & (z_reco.mass < 111.)
                sel.add("z_ptcut_reco", z_ptcut_reco & (sel.require(twoReco_leptons = True) ))
                sel.add("z_mcut_reco", z_mcut_reco & (sel.require(twoReco_leptons = True) ))

                self.logging.debug("Z Object Created")

                events_j = events0
                
                presort_recojets = events_j.FatJet[
                        (events_j.FatJet.mass > 0) 
                        &(events_j.FatJet.pt > 0) 
                        &(np.abs(events_j.FatJet.eta) < 2.5) 
                        &(events_j.FatJet.jetId == 6) 
                    ]
                ## Sorting jets by pt
                index = ak.argsort(presort_recojets.pt, ascending=False)
                
                recojets  = presort_recojets[index]
                #recojets = presort_recojets
                events_j = ak.with_field(events_j, recojets, "FatJet")

                ## adding lepton separation
                recojets = apply_lepton_separation(
                                events_j.FatJet,
                                events_j.Muon,
                                events_j.Electron,
                                dr_cut=0.4,
                            )
                recojets = recojets[recojets.subJetIdx1 > -1]
                #recojets = events_j.FatJet
                hem_sel = HEMVeto(recojets, events_j.run)
                

                sel.add("oneRecoJet", 
                            ak.sum( (events_j.FatJet.pt > 0) & (np.abs(events_j.FatJet.eta) < 2.5)  & (events_j.FatJet.jetId == 6) & hem_sel, axis=1 ) >= 1
                        )
                jetR = 0.8
                

                reco_jet, z_jet_dphi_reco = get_dphi( z_reco, recojets )

                z_jet_dr_reco = reco_jet.delta_r(z_reco)
                z_jet_dphi_reco_values = z_jet_dphi_reco

                ####### MAKE PRESEL PLOTS ######
                #filter_sel = sel.all('npv', 'oneRecoJet')
                reco_exists = ~ak.is_none(reco_jet.mass)

                
                #####################################
                ### Reco event topology sel
                #####################################
                z_jet_dphi_sel_reco = (z_jet_dphi_reco > 1.57)  #& (sel.require(twoReco_leptons = True))#np.pi * 0.5
                z_pt_asym_reco = np.abs(z_reco.pt - reco_jet.pt) / (z_reco.pt + reco_jet.pt)
                z_pt_frac_reco = reco_jet.pt / z_reco.pt
                z_pt_asym_sel_reco = (z_pt_asym_reco < 0.3) #& (sel.require(twoReco_leptons = True))
                sel.add("z_jet_dphi_sel_reco", z_jet_dphi_sel_reco)
                sel.add("z_pt_asym_sel_reco", z_pt_asym_sel_reco)


                
                kinsel_reco = sel.require(twoReco_leptons=True,oneRecoJet=True,z_ptcut_reco=True,z_mcut_reco=True)#, jetid = True)
                sel.add("kinsel_reco", kinsel_reco)
                #print("Leading RECO Jet matched muon ID", reco_jet.muonIdx3SJ)
                #print("Z-Jet dphi cut ",  sel.require(kinsel_reco= True, z_jet_dphi_sel_reco= True,trigsel= True ).sum())
                #print("Z-Jet pt-asymmetry cut ",  sel.require(kinsel_reco= True,z_jet_dphi_sel_reco= True, z_pt_asym_sel_reco= True,trigsel= True ).sum())
                
                
                toposel_reco = sel.require( z_pt_asym_sel_reco=True, z_jet_dphi_sel_reco=True)
                sel.add("toposel_reco", toposel_reco)
                if self._do_gen:
                    #is_matched_reco = reco_jet.delta_r(gen_jet) < 0.4
                    #sel.add("is_matched_reco", is_matched_reco)

                    allsel_reco = sel.all("npv", "MET", "kinsel_reco", "toposel_reco")#, "is_matched_reco" )
                    sel.add("allsel_reco", allsel_reco)
                    #is_matched_gen = gen_jet.delta_r(reco_jet) < 0.4
                    #sel.add("is_matched_gen", is_matched_gen)
                    #allsel_gen = sel.all("kinsel_gen", "toposel_gen" , "is_matched_gen" )
                    #sel.add("allsel_gen", allsel_gen)
                    #sel.add("fakes", sel.require(allsel_reco = True, allsel_gen = False))
                else:
                    allsel_reco = sel.all("npv", "MET", "kinsel_reco", "toposel_reco", "trigsel" )
                    sel.add("allsel_reco", allsel_reco)
                channels = ['mm', 'ee']
                for channel in channels:
                    if channel == 'mm':
                        sel_reco = sel.require(allsel_reco = True, twoReco_mm = True)
                        weights_reco = weights.weight()[sel_reco]
                    else:
                        sel_reco = sel.require(allsel_reco = True, twoReco_ee = True)
                        weights_reco = weights.weight()[sel_reco]
                    
                    reco_jet_meas = reco_jet[sel_reco]
                    ptreco = reco_jet_meas.pt
                    ptreco_g = reco_jet_meas.pt
                    mreco = reco_jet_meas.mass
                    mreco_g = reco_jet_meas.msoftdrop
                    #mreco_g2 = reco_jet_meas.msoftdrop_orig
    
                    ptreco = ptreco[~ak.is_none(mreco)]
                    mreco = mreco[~ak.is_none(mreco)]
                    mreco_g = mreco_g[~ak.is_none(mreco_g)]
                    #mreco_g2 = mreco_g2[~ak.is_none(mreco_g2)]
                    ptreco_g = ptreco_g[~ak.is_none(mreco_g)]
                    weights_reco_g = weights_reco[~ak.is_none(mreco_g)]
                    weights_reco = weights_reco[~ak.is_none(mreco)]
    
                    # mpt_reco = 2*np.log10(mreco/(ptreco*jetR))
                    # mpt_reco_g = 2*np.log10(mreco_g/(ptreco*jetR))
    
                    events_j_meas = events_j[sel_reco]
                    z_reco_meas = z_reco[sel_reco]
                    if channel == "mm":
                        pt_mu0 = events_j_meas.Muon[:, 0].pt
                        eta_mu0 = events_j_meas.Muon[:, 0].eta
                        phi_mu0 = events_j_meas.Muon[:, 0].phi
                        pt_mu1 = events_j_meas.Muon[:, 1].pt
                        eta_mu1 = events_j_meas.Muon[:, 1].eta
                        phi_mu1 = events_j_meas.Muon[:, 1].phi
                        
                        q0 = events_j_meas.Muon[:, 0].charge
                        q1 = events_j_meas.Muon[:, 1].charge
                        
                        pt_mupos = ak.where(q0 > 0, pt_mu0, pt_mu1)
                        pt_muneg = ak.where(q0 < 0, pt_mu0, pt_mu1)
                        
                        eta_mupos = ak.where(q0 > 0, eta_mu0, eta_mu1)
                        eta_muneg = ak.where(q0 < 0, eta_mu0, eta_mu1)
                        
                        phi_mupos = ak.where(q0 > 0, phi_mu0, phi_mu1)
                        phi_muneg = ak.where(q0 < 0, phi_mu0, phi_mu1)
                    if channel == "ee":
                        pt_el0 = events_j_meas.Electron[:, 0].pt
                        eta_el0 = events_j_meas.Electron[:, 0].eta
                        phi_el0 = events_j_meas.Electron[:, 0].phi
                        pt_el1 = events_j_meas.Electron[:, 1].pt
                        eta_el1 = events_j_meas.Electron[:, 1].eta
                        phi_el1 = events_j_meas.Electron[:, 1].phi

                        q0 = events_j_meas.Electron[:, 0].charge
                        q1 = events_j_meas.Electron[:, 1].charge
                        
                        pt_elpos = ak.where(q0 > 0, pt_el0, pt_el1)
                        pt_elneg = ak.where(q0 < 0, pt_el0, pt_el1)
                    # if channel == "ee":
                    #     pt_el0 = events_j_meas.Electron[:, 0].pt
                    #     eta_el0 = events_j_meas.Electron[:, 0].eta
                    #     phi_el0 = events_j_meas.Electron[:, 0].phi
                    #     pt_el1 = events_j_meas.Electron[:, 1].pt
                    #     eta_el1 = events_j_meas.Electron[:, 1].eta
                    #     phi_el1 = events_j_meas.Electron[:, 1].phi
                    
                    pt_Z = z_reco_meas.pt
                    eta_Z = z_reco_meas.eta
                    phi_Z = z_reco_meas.phi
                    mass_Z = z_reco_meas.mass
                    nJet = ak.num(events_j_meas.FatJet, axis = 1)
                    
                    dr = z_jet_dr_reco[sel_reco]
                    dphi = z_jet_dphi_reco[sel_reco]
                    ptasym = z_pt_asym_reco[sel_reco]
                    jet_syst = 'nominal'
                    if channel == "mm":
                        fill_hist(self.hists, "pt_mu0", dataset = dataset, pt = pt_mupos, systematic = jet_syst, weight = weights_reco )
                        fill_hist(self.hists, "eta_mu0", dataset = dataset, eta = eta_mu0, systematic = jet_syst, weight = weights_reco )
                        fill_hist(self.hists, "phi_mu0", dataset = dataset, phi = phi_mu0, systematic = jet_syst, weight = weights_reco )
                        fill_hist(self.hists, "pt_mu1", dataset = dataset, pt = pt_mu1, systematic = jet_syst, weight = weights_reco )
                        fill_hist(self.hists, "eta_mu1", dataset = dataset, eta = eta_mu1, systematic = jet_syst, weight = weights_reco )
                        fill_hist(self.hists, "phi_mu1", dataset = dataset, phi = phi_mu1, systematic = jet_syst, weight = weights_reco )


                                                    ## leading and sub-leading electron
                    if channel == "ee":
                        fill_hist(self.hists, "pt_el0", dataset = dataset, pt = pt_elpos, systematic = jet_syst, weight = weights_reco )
                        fill_hist(self.hists, "eta_el0", dataset = dataset, eta = eta_el0, systematic = jet_syst, weight = weights_reco )
                        fill_hist(self.hists, "phi_el0", dataset = dataset, phi = phi_el0, systematic = jet_syst, weight = weights_reco )
                        fill_hist(self.hists, "pt_el1", dataset = dataset, pt = pt_el1, systematic = jet_syst, weight = weights_reco )
                        fill_hist(self.hists, "eta_el1", dataset = dataset, eta = eta_el1, systematic = jet_syst, weight = weights_reco )
                        fill_hist(self.hists, "phi_el1", dataset = dataset, phi = phi_el1, systematic = jet_syst, weight = weights_reco )
                    fill_hist(self.hists, "pt_Z", dataset = dataset, pt = pt_Z, systematic = jet_syst, weight = weights_reco )
                    fill_hist(self.hists, "eta_Z", dataset = dataset, eta = eta_Z, systematic = jet_syst, weight = weights_reco )
                    fill_hist(self.hists, "phi_Z", dataset = dataset, phi = phi_Z, systematic = jet_syst, weight = weights_reco )
                    fill_hist(self.hists, "mass_Z", dataset = dataset, mass = mass_Z, systematic = jet_syst, weight = weights_reco )
                
                if self._do_jk == False:
                    self.logging.info("Breaking at first iteration since JK is disabled")
                    break
            return self.hists    
        def postprocess(self, accumulator):
            return accumulator
            
  

            
            
                
            
    

            
