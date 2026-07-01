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
import os



from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from .hist_utils import *
from . import smp_utils as zjet_smp_utils
from .smp_utils import *


from .corrections import *


# get_herwig_weight_g = PtRhoWeighter('correctionFiles/spline_groomed.npz')
# get_herwig_weight_u = PtRhoWeighter('correctionFiles/spline_ungroomed.npz')

class Log:
    def __init__(self, mode="info"):
        self.mode = mode
    def info(self, msg):
        if self.mode in ["info", "debug"]:
            print("[INFO]", msg)
    def warning(self, msg):
        print("[WARNING]", msg)
    def debug(self, msg):
        if self.mode == "debug":
            print("[DEBUG]", msg)
        

class QJetMassProcessor(processor.ProcessorABC):
    '''
    Processor to run a Z+jets jet mass cross section analysis. 
    With "do_gen == True", will perform GEN selection and create response matrices. 
    Will always plot RECO level quantities. 
    '''
    def __init__(self, do_gen = True, mode = "minimal",  debug = False, jet_systematics = None, systematics = None):
        '''
        Args:
            do_gen (bool): whether to run gen-level analysis and create response matrices
            mode (str): "minimal" or "full". "minimal" runs a smaller set of histograms for testing purposes. 
                        "full" runs the full set of histograms for publication.
            testing (bool): whether to run in testing mode (faster, less data)
        '''
        self._do_gen = do_gen
        self._mode = mode
        self._debug = debug
        self._do_reweight = False
        self._do_jk = False
        self._do_reco_jet_ntuple = mode == "mass_diagnostic_ntuple"
        self._split2 = False  # 2-way (50:50) disjoint event split for the fine-binning test

        if mode == "minimal":
            self._do_jk = False
        elif mode in ("reweight_pythia", "reweight_pythia_rho", "reweight_data_prior_rho"):
            self._do_reweight = True
        elif mode == "jk_mc" or mode == "jk_data" or mode == "rho_jk" or mode == "mass_jk":
            self._do_jk = True
        elif mode == "minimal_rho_fine_split":
            # two independent halves (index % 2) carried on the jk axis: far cheaper
            # than the 10-replica jackknife, enough for the fine-binning closure test
            self._do_jk = True
            self._split2 = True
        elif mode == "full":
            self._do_jk = False
        else:
            self._do_jk = False
        
        
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

        if self._do_reco_jet_ntuple:
            self.jet_systematics = ["nominal"]
            self.systematics = ["nominal"]
        

        self.lepptcuts = [40,29] # [ele, mu]
        
        if self._debug:
            self.logging = Log(mode="debug")
        else:
            self.logging = Log(mode="info")

        # minimal_rho_fine[_split]: 4x-finer rho axes (48 gen / 96 reco, still #reco = 2x #gen)
        # for the fine-bin / fine-then-rebin unfolding study; everything else as minimal_rho.
        binning = util_binning(
            rho_refine=4 if self._mode in ("minimal_rho_fine", "minimal_rho_fine_split") else 1
        )

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
        mass_ratio_axis = binning.mass_ratio_axis
        m_u_reco_5gev_axis = binning.m_u_reco_5gev_axis
        m_g_reco_5gev_axis = binning.m_g_reco_5gev_axis
        m_u_gen_cov_axis = binning.m_u_gen_cov_axis
        m_g_gen_cov_axis = binning.m_g_gen_cov_axis
        mpt_u_gen_cov_axis = binning.mpt_u_gen_cov_axis
        mpt_g_gen_cov_axis = binning.mpt_g_gen_cov_axis
        m_u_reco_cov_axis = binning.m_u_reco_cov_axis
        m_g_reco_cov_axis = binning.m_g_reco_cov_axis
        mpt_u_reco_cov_axis = binning.mpt_u_reco_cov_axis
        mpt_g_reco_cov_axis = binning.mpt_g_reco_cov_axis
        dr_axis = binning.dr_axis
        dr_fine_axis = binning.dr_fine_axis
        dphi_axis = binning.dphi_axis    
        syst_axis = binning.syst_axis
        eta_axis = binning.eta_axis
        phi_axis = binning.phi_axis
        ptfine_axis = binning.ptfine_axis
        mreco_over_pt_axis = binning.mreco_over_pt_axis
        mgen_over_pt_axis = binning.mgen_over_pt_axis
        y_axis = binning.y_axis
        ptlong_axis = binning.ptlong_axis
        
        ptgen_axis_fine  = binning.ptgen_axis_fine 

        
        mptreco_axis = binning.mreco_over_pt_axis
        mptgen_axis = binning.mgen_over_pt_axis

        

        ht_axis = hist.axis.StrCategory([],growth = True, name = "ht_bin", label = "h_T bin")
        channel_axis = hist.axis.StrCategory([],growth = True, name = "channel", label = "Channel")
        parton_flavor_axis = hist.axis.StrCategory(
            ["quark", "gluon", "other"],
            name="parton_flavor",
            label="Initiating parton flavor",
        )
        
        weight_axis = hist.axis.Regular(100, 0, 10, name="corrWeight", label=r"Weight")
        met_pt_axis = hist.axis.Regular(20, 0, 200, name = "pt", label = r"$p_T$")

        mcut_reco_u_axis = binning.mcut_reco_u_axis
        mcut_reco_g_axis = binning.mcut_reco_g_axis

        mcut_gen_u_axis = binning.mcut_gen_u_axis
        mcut_gen_g_axis = binning.mcut_gen_g_axis
        #### weight to check what is causing this
        
        #self.gen_binning = binning.gen_binning
        #self.reco_binning = binning.reco_binning

        self.hists = processor.dict_accumulator()

        if self._mode == "minimal" or self._mode == "reweight_pythia"  :
            register_hist(self.hists, "ptjet_mjet_u_reco", [dataset_axis,channel_axis, ptreco_axis, mreco_axis, syst_axis])
            register_hist(self.hists, "ptjet_mjet_g_reco", [dataset_axis,channel_axis, ptreco_axis, mreco_axis, syst_axis])
            register_hist(self.hists, "m_g_over_m_u_reco", [dataset_axis, channel_axis, ptreco_axis, mass_ratio_axis, syst_axis])
            register_hist(self.hists, "m_g_vs_m_u_reco", [dataset_axis, channel_axis, m_u_reco_5gev_axis, m_g_reco_5gev_axis, syst_axis])
            register_hist(self.hists, "m_g_over_m_u_raw_reco", [dataset_axis, channel_axis, ptreco_axis, mass_ratio_axis, syst_axis])
            register_hist(self.hists, "m_g_vs_m_u_raw_reco", [dataset_axis, channel_axis, m_u_reco_5gev_axis, m_g_reco_5gev_axis, syst_axis])
            # Z pT/mass (reco) for data/MC validation; filled at process() for data
            # + MC (gated only on jet_syst=="nominal"). Top level so data gets it too.
            register_hist(self.hists, "ptz_mz_reco", [dataset_axis, zmass_axis, pt_axis])

            if self._do_gen:
                register_hist(self.hists, "response_matrix_u", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis])
                register_hist(self.hists, "response_matrix_g", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis])
                register_hist(self.hists, "ptjet_mjet_u_gen", [dataset_axis,channel_axis, ptgen_axis, mgen_axis, syst_axis])
                register_hist(self.hists, "ptjet_mjet_g_gen", [dataset_axis,channel_axis, ptgen_axis, mgen_axis, syst_axis])

        if self._mode in ("minimal_rho", "minimal_rho_fine", "reweight_pythia_rho", "reweight_data_prior_rho"):
            register_hist(self.hists, "ptjet_rhojet_u_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])
            register_hist(self.hists, "ptjet_rhojet_g_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])
            register_hist(self.hists, "m_g_over_m_u_reco", [dataset_axis, channel_axis, ptreco_axis, mass_ratio_axis, syst_axis])
            register_hist(self.hists, "m_g_vs_m_u_reco", [dataset_axis, channel_axis, m_u_reco_5gev_axis, m_g_reco_5gev_axis, syst_axis])
            register_hist(self.hists, "m_g_over_m_u_raw_reco", [dataset_axis, channel_axis, ptreco_axis, mass_ratio_axis, syst_axis])
            register_hist(self.hists, "m_g_vs_m_u_raw_reco", [dataset_axis, channel_axis, m_u_reco_5gev_axis, m_g_reco_5gev_axis, syst_axis])
            #register_hist(self.hists, "ptjet_rhojet_g_reco2", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])
            # Z pT and Z mass (reco) for data/MC validation; filled at line ~1558 for
            # data + MC (gated only on jet_syst=="nominal"). Registered at top level so
            # data (do_gen=False) also gets it.
            register_hist(self.hists, "ptz_mz_reco", [dataset_axis, zmass_axis, pt_axis])

            if self._do_gen:
                register_hist(self.hists, "response_matrix_rho_u", [dataset_axis, ptreco_axis, mreco_over_pt_axis, ptgen_axis, mgen_over_pt_axis, syst_axis])
                register_hist(self.hists, "response_matrix_rho_g", [dataset_axis, ptreco_axis, mreco_over_pt_axis, ptgen_axis, mgen_over_pt_axis, syst_axis])
                register_hist(self.hists, "ptjet_rhojet_u_gen", [dataset_axis, ptgen_axis, mgen_over_pt_axis, syst_axis])
                register_hist(self.hists, "ptjet_rhojet_g_gen", [dataset_axis, ptgen_axis, mgen_over_pt_axis, syst_axis])

        if self._mode == "minimal_rho_fine_split":
            # Same 6 rho hists as minimal_rho but with the 'jk' axis carrying the
            # disjoint 50:50 split (jk=0,1) -- two independent halves in one pass,
            # for the fine-binning closure/bias test (cheaper than the 10x jackknife).
            register_hist(self.hists, "ptjet_rhojet_u_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis, binning.jackknife_axis, syst_axis])
            register_hist(self.hists, "ptjet_rhojet_g_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis, binning.jackknife_axis, syst_axis])
            if self._do_gen:
                register_hist(self.hists, "response_matrix_rho_u", [dataset_axis, ptreco_axis, mreco_over_pt_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "response_matrix_rho_g", [dataset_axis, ptreco_axis, mreco_over_pt_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "ptjet_rhojet_u_gen", [dataset_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "ptjet_rhojet_g_gen", [dataset_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis])


                #register_hist(self.hists, 'm_u_jet_reco_over_gen', [dataset_axis, ptgen_axis, mgen_axis, frac_axis])
                #register_hist(self.hists, 'm_g_jet_reco_over_gen', [dataset_axis, ptgen_axis, mgen_axis, frac_axis])

        if self._mode == "mass_cov":
            # Joint gen-level (groomed mass) x (ungroomed mass) per pT bin, filled
            # from the SAME events -> gives the statistical correlation between the
            # groomed and ungroomed mass spectra (ARC Scope#3 / simultaneous fit).
            register_hist(self.hists, "ptjet_mjet_g_vs_u_reco",
                          [dataset_axis, channel_axis, ptreco_axis, m_g_reco_cov_axis, m_u_reco_cov_axis, syst_axis])
            register_hist(self.hists, "ptjet_rhojet_g_vs_u_reco",
                          [dataset_axis, channel_axis, ptreco_axis, mpt_g_reco_cov_axis, mpt_u_reco_cov_axis, syst_axis])
            if self._do_gen:
                register_hist(self.hists, "ptjet_mjet_g_vs_u_gen",
                              [dataset_axis, channel_axis, ptgen_axis, m_g_gen_cov_axis, m_u_gen_cov_axis, syst_axis])
                register_hist(self.hists, "ptjet_rhojet_g_vs_u_gen",
                              [dataset_axis, channel_axis, ptgen_axis, mpt_g_gen_cov_axis, mpt_u_gen_cov_axis, syst_axis])

        if self._mode == "validation":
            register_hist(self.hists, "pt_mupos", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "eta_mupos", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_mupos", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "pt_muneg", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "eta_muneg", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_muneg", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "y_mupos", [dataset_axis, y_axis, syst_axis])
            register_hist(self.hists, "y_muneg", [dataset_axis, y_axis, syst_axis])


            
            register_hist(self.hists, "pt_elpos", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "eta_elpos", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_elpos", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "pt_elneg", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "eta_elneg", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_elneg", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "y_elpos", [dataset_axis, y_axis, syst_axis])
            register_hist(self.hists, "y_elneg", [dataset_axis, y_axis, syst_axis])
            
            register_hist(self.hists, "nJets", [dataset_axis, n_axis, syst_axis])
            
            register_hist(self.hists, "pt_Z", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "eta_Z", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_Z", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "mass_Z", [dataset_axis, zmass_axis, syst_axis])
            register_hist(self.hists, "y_Z", [dataset_axis, y_axis, syst_axis])
            
            
            register_hist(self.hists, "pt_jet0", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "pt_flavor_jet0_gen", [dataset_axis, pt_axis, n_axis, syst_axis])
            register_hist(self.hists, "y_flavor_jet0_gen", [dataset_axis, y_axis, n_axis, syst_axis])
            register_hist(self.hists, "mass_flavor_jet0_gen", [dataset_axis, mass_axis, parton_flavor_axis, syst_axis])
            register_hist(self.hists, "mpt_flavor_jet0_gen", [dataset_axis, mgen_over_pt_axis, parton_flavor_axis, syst_axis])
            register_hist(self.hists, "eta_jet0", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_jet0", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "mass_jet0", [dataset_axis, mass_axis, syst_axis])
            register_hist(self.hists, "y_jet0", [dataset_axis, y_axis, syst_axis])
            register_hist(self.hists, "eta_phi_jet_reco", [dataset_axis, eta_axis, phi_axis])

            #### MET pt / phi (reco) -- data vs MC cross-check that the 2018 HEM
            #### treatment (flat lumi-fraction weight for MC, hard veto for data)
            #### is applied correctly: without it, lost energy in the HEM sector
            #### (phi in [-1.57, -0.87]) shows up as an excess of fake MET pointing
            #### opposite the hole. Filled from events_j_meas.MET (PF Type-1 MET).
            register_hist(self.hists, "met_pt",  [dataset_axis, met_pt_axis, syst_axis])
            register_hist(self.hists, "met_phi", [dataset_axis, phi_axis, syst_axis])
            #### same MET after the UL MET-xy (phi) correction -- removes the
            #### nPV-dependent phi-modulation so data and MC agree; before/after
            #### pair demonstrates the residual is modulation, not HEM.
            register_hist(self.hists, "met_pt_xy",  [dataset_axis, met_pt_axis, syst_axis])
            register_hist(self.hists, "met_phi_xy", [dataset_axis, phi_axis, syst_axis])
            #### AK4 jet occupancy for the HEM cross-check (ARC): the AK8 jet is
            #### HEM-vetoed, but AK4 jets are not -- a suppressed HEM sector in
            #### data (vs MC) shows up here. Wide eta axis to reach the HEM strip
            #### (down to eta ~ -3), which the analysis eta_axis (+/-2.5) misses.
            ak4_eta_axis = hist.axis.Regular(50, -5.0, 5.0, name="eta", label=r"AK4 jet $\eta$")
            ak4_phi_axis = hist.axis.Regular(50, -np.pi, np.pi, name="phi", label=r"AK4 jet $\phi$")
            register_hist(self.hists, "ak4_eta_phi_reco", [dataset_axis, ak4_eta_axis, ak4_phi_axis])
            register_hist(self.hists, "ak4_phi_hemeta",   [dataset_axis, ak4_phi_axis])
            #### "after" distributions with the AK4-in-HEM event veto applied
            #### (data: hard veto; 2018 MC: flat lumi-fraction weight, mirroring
            #### the AK8 HEMVeto). Paired with met_pt/met_phi/mass_jet0 ("before")
            #### to size how much the veto improves data/MC and whether it biases
            #### the jet mass.
            register_hist(self.hists, "met_pt_ak4veto",    [dataset_axis, met_pt_axis, syst_axis])
            register_hist(self.hists, "met_phi_ak4veto",   [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "mass_jet0_ak4veto", [dataset_axis, mass_axis, syst_axis])
            #### joint MET x (leading-jet pt) and MET x (jet mass): keep the
            #### per-event correlation the 1D hists throw away, so the MET tail
            #### can be shown to be jet-resolution-driven (rises with jet pt) and
            #### the jet mass shown insensitive to a MET<50/>50 split (offline).
            register_hist(self.hists, "met_ptjet_reco", [dataset_axis, met_pt_axis, ptreco_axis, syst_axis])
            register_hist(self.hists, "mass_met_reco",  [dataset_axis, mass_axis, met_pt_axis, syst_axis])

            register_hist(self.hists, "ptasym_presel", [dataset_axis, frac_axis])
            register_hist(self.hists, "ptasym", [dataset_axis, frac_axis, syst_axis])
            register_hist(self.hists, "dr", [dataset_axis, dr_axis, syst_axis])
            register_hist(self.hists, "dphi", [dataset_axis, dphi_axis, syst_axis])

            register_hist(self.hists, "ptjet_rhojet_u_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])
            register_hist(self.hists, "ptjet_rhojet_g_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])
            register_hist(self.hists, "m_g_over_m_u_reco", [dataset_axis, channel_axis, ptreco_axis, mass_ratio_axis, syst_axis])
            register_hist(self.hists, "m_g_vs_m_u_reco", [dataset_axis, channel_axis, m_u_reco_5gev_axis, m_g_reco_5gev_axis, syst_axis])
            register_hist(self.hists, "m_g_over_m_u_raw_reco", [dataset_axis, channel_axis, ptreco_axis, mass_ratio_axis, syst_axis])
            register_hist(self.hists, "m_g_vs_m_u_raw_reco", [dataset_axis, channel_axis, m_u_reco_5gev_axis, m_g_reco_5gev_axis, syst_axis])
            
            #register_hist(self.hists, "ht", [dataset_axis, ptlong_axis, ht_axis ])
            #register_hist(self.hists, "eta_phi_jet_reco", [dataset_axis, eta_axis, phi_axis])
            
            # register_hist(self.hists, "ht_AK4", [dataset_axis, pt_axis, ht_axis ])
            # register_hist(self.hists, "ht_reco", [dataset_axis, pt_axis, ht_axis ])
            # register_hist(self.hists, "ht_reco_AK4", [dataset_axis, pt_axis, ht_axis ])
            # register_hist(self.hists, "pt_mu0", [dataset_axis, pt_axis, syst_axis])
            # register_hist(self.hists, "pt_Z", [dataset_axis, pt_axis, syst_axis])
            # register_hist(self.hists, "pt_el0", [dataset_axis, pt_axis, syst_axis])
  
            
            

            

                
        if self._mode == "rho_jk":
            register_hist(self.hists, "ptjet_rhojet_u_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis,  binning.jackknife_axis , syst_axis,])
            register_hist(self.hists, "ptjet_rhojet_g_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis,  binning.jackknife_axis , syst_axis, ])
            #register_hist(self.hists, "ptjet_rhojet_g_reco2", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])

            if self._do_gen:
                register_hist(self.hists, "response_matrix_rho_u", [dataset_axis, ptreco_axis, mreco_over_pt_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis,])
                register_hist(self.hists, "response_matrix_rho_g", [dataset_axis, ptreco_axis, mreco_over_pt_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis,])
                register_hist(self.hists, "ptjet_rhojet_u_gen", [dataset_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "ptjet_rhojet_g_gen", [dataset_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis])
                

                register_hist(self.hists, 'm_u_jet_reco_over_gen', [dataset_axis, ptgen_axis, mgen_axis, frac_axis, binning.jackknife_axis, syst_axis,])
                register_hist(self.hists, 'm_g_jet_reco_over_gen', [dataset_axis, ptgen_axis, mgen_axis, frac_axis, binning.jackknife_axis, syst_axis,])

        if self._mode == "mass_jk":
            register_hist(self.hists, "ptjet_mjet_u_reco", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, binning.jackknife_axis, syst_axis])
            register_hist(self.hists, "ptjet_mjet_g_reco", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, binning.jackknife_axis, syst_axis])
            register_hist(self.hists, "m_g_over_m_u_reco", [dataset_axis, channel_axis, ptreco_axis, mass_ratio_axis, binning.jackknife_axis, syst_axis])
            register_hist(self.hists, "m_g_vs_m_u_reco", [dataset_axis, channel_axis, m_u_reco_5gev_axis, m_g_reco_5gev_axis, binning.jackknife_axis, syst_axis])
            register_hist(self.hists, "m_g_over_m_u_raw_reco", [dataset_axis, channel_axis, ptreco_axis, mass_ratio_axis, binning.jackknife_axis, syst_axis])
            register_hist(self.hists, "m_g_vs_m_u_raw_reco", [dataset_axis, channel_axis, m_u_reco_5gev_axis, m_g_reco_5gev_axis, binning.jackknife_axis, syst_axis])

            if self._do_gen:
                register_hist(self.hists, "response_matrix_u", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "response_matrix_g", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "ptjet_mjet_u_gen", [dataset_axis, channel_axis, ptgen_axis, mgen_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "ptjet_mjet_g_gen", [dataset_axis, channel_axis, ptgen_axis, mgen_axis, binning.jackknife_axis, syst_axis])


                
        if self._mode == "jk_mc":
            register_hist(self.hists, "jk_response_matrix_u", [dataset_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, binning.jackknife_axis])
            register_hist(self.hists, "jk_response_matrix_g", [dataset_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, binning.jackknife_axis])
        if self._mode == "jk_data":
            register_hist(self.hists, "jk_ptjet_mjet_g_reco", [dataset_axis,ptreco_axis, mreco_axis, binning.jackknife_axis])
            register_hist(self.hists, "jk_ptjet_mjet_u_reco", [dataset_axis,ptreco_axis, mreco_axis, binning.jackknife_axis])

        if self._do_reco_jet_ntuple:
            self._register_reco_jet_ntuple()

        
        self.hists["sumw"] = processor.defaultdict_accumulator(float)
        self.hists["nev"] = processor.defaultdict_accumulator(int)
        self.hists["cutflow"] = processor.defaultdict_accumulator(int)
        self.logging.debug(f"Registered Histograms {self.hists.keys()}")

    def _reweight_mode(self):
        if self._mode == "reweight_pythia":
            return "mass"
        if self._mode == "reweight_pythia_rho":
            return "rho"
        if self._mode == "reweight_data_prior_rho":
            return "data_prior_rho"
        return None

    def _is_mass_reweight(self):
        return self._reweight_mode() == "mass"

    def _is_rho_reweight(self):
        return self._reweight_mode() in ("rho", "data_prior_rho")

    def _is_data_prior_rho_reweight(self):
        return self._reweight_mode() == "data_prior_rho"

    @staticmethod
    def _reweight_numpy(values):
        try:
            return ak.to_numpy(ak.fill_none(values, np.nan))
        except Exception:
            return np.asarray(values, dtype=float)

    def _reweight_value(self, mass, pt, jetR):
        mode = self._reweight_mode()
        mass = self._reweight_numpy(mass)
        pt = self._reweight_numpy(pt)
        if mode == "mass":
            return mass
        if mode in ("rho", "data_prior_rho"):
            return 2 * np.log10(mass / (pt * jetR))
        raise ValueError(f"Reweighting requested for unsupported mode '{self._mode}'.")

    def _get_herwig_reweights(self, pt, ungroomed_mass, groomed_mass, jetR):
        mode = self._reweight_mode()
        if mode is None:
            raise ValueError("Herwig reweights requested outside reweight modes.")

        ungroomed_value = self._reweight_value(ungroomed_mass, pt, jetR)
        groomed_value = self._reweight_value(groomed_mass, pt, jetR)
        pt = self._reweight_numpy(pt)
        if mode == "data_prior_rho":
            data_prior_weight_g = get_data_prior_rho_weight_g().weight_array(pt, groomed_value)
            data_prior_weight_u = get_data_prior_rho_weight_u().weight_array(pt, ungroomed_value)
            return (
                np.nan_to_num(data_prior_weight_u, nan=1.0, posinf=1.0, neginf=1.0),
                np.nan_to_num(data_prior_weight_g, nan=1.0, posinf=1.0, neginf=1.0),
            )

        herwig_weight_g = get_herwig_weight_g(mode=mode).weight_array(pt, groomed_value)
        herwig_weight_u = get_herwig_weight_u(mode=mode).weight_array(pt, ungroomed_value)
        return herwig_weight_u, herwig_weight_g

    def _register_reco_jet_ntuple(self):
        float_columns = [
            "weight",
            "pt",
            "mass",
            "msoftdrop",
            "eta",
            "phi",
            "rapidity",
            "pt_nanoaod",
            "mass_nanoaod",
            "msoftdrop_nanoaod",
            "pt_raw",
            "mass_raw",
            "msoftdrop_raw",
            "msoftdrop_raw_fatjet",
            "gen_pt",
            "gen_eta",
            "gen_phi",
            "gen_mass",
            "gen_rapidity",
            "gen_msoftdrop",
            "reco_gen_dr",
        ]
        uint64_columns = ["event"]
        uint32_columns = ["run", "luminosityBlock"]
        uint8_columns = ["has_gen_match", "passes_gen_selection", "passes_both"]
        object_columns = ["dataset", "channel", "systematic"]

        columns = {}
        for name in float_columns:
            columns[name] = processor.column_accumulator(np.array([], dtype=np.float64))
        for name in uint64_columns:
            columns[name] = processor.column_accumulator(np.array([], dtype=np.uint64))
        for name in uint32_columns:
            columns[name] = processor.column_accumulator(np.array([], dtype=np.uint32))
        for name in uint8_columns:
            columns[name] = processor.column_accumulator(np.array([], dtype=np.uint8))
        for name in object_columns:
            columns[name] = processor.column_accumulator(np.array([], dtype=object))

        self.hists["reco_jet_ntuple"] = processor.dict_accumulator(columns)

    @staticmethod
    def _to_numpy_column(values, dtype=np.float64):
        if isinstance(values, np.ndarray):
            return np.asarray(values, dtype=dtype)
        return np.asarray(ak.to_numpy(ak.fill_none(values, np.nan)), dtype=dtype)

    @staticmethod
    def _to_numpy_flag_column(values):
        if isinstance(values, np.ndarray):
            return np.asarray(values, dtype=np.uint8)
        return np.asarray(ak.to_numpy(ak.fill_none(values, False)), dtype=np.uint8)

    def _fill_reco_jet_ntuple(
        self,
        dataset,
        channel,
        systematic,
        events,
        reco_jet,
        weight,
        gen_jet=None,
        groomed_gen_jet=None,
        has_gen_match=None,
        passes_gen_selection=None,
        passes_both=None,
    ):
        if "reco_jet_ntuple" not in self.hists:
            return

        nrows = len(reco_jet.pt)
        if nrows == 0:
            return

        ntuple = self.hists["reco_jet_ntuple"]
        ntuple["dataset"] += processor.column_accumulator(np.full(nrows, dataset, dtype=object))
        ntuple["channel"] += processor.column_accumulator(np.full(nrows, channel, dtype=object))
        ntuple["systematic"] += processor.column_accumulator(np.full(nrows, systematic, dtype=object))
        ntuple["weight"] += processor.column_accumulator(self._to_numpy_column(weight))

        ntuple["run"] += processor.column_accumulator(self._to_numpy_column(events.run, dtype=np.uint32))
        ntuple["luminosityBlock"] += processor.column_accumulator(
            self._to_numpy_column(events.luminosityBlock, dtype=np.uint32)
        )
        ntuple["event"] += processor.column_accumulator(self._to_numpy_column(events.event, dtype=np.uint64))

        ntuple["pt"] += processor.column_accumulator(self._to_numpy_column(reco_jet.pt))
        ntuple["mass"] += processor.column_accumulator(self._to_numpy_column(reco_jet.mass))
        ntuple["msoftdrop"] += processor.column_accumulator(self._to_numpy_column(reco_jet.msoftdrop))
        ntuple["eta"] += processor.column_accumulator(self._to_numpy_column(reco_jet.eta))
        ntuple["phi"] += processor.column_accumulator(self._to_numpy_column(reco_jet.phi))
        ntuple["rapidity"] += processor.column_accumulator(self._to_numpy_column(reco_jet.rapidity))

        ntuple["pt_nanoaod"] += processor.column_accumulator(self._to_numpy_column(reco_jet.pt_nanoaod))
        ntuple["mass_nanoaod"] += processor.column_accumulator(self._to_numpy_column(reco_jet.mass_nanoaod))
        ntuple["msoftdrop_nanoaod"] += processor.column_accumulator(
            self._to_numpy_column(reco_jet.msoftdrop_nanoaod)
        )

        ntuple["pt_raw"] += processor.column_accumulator(self._to_numpy_column(reco_jet.pt_raw_diagnostic))
        ntuple["mass_raw"] += processor.column_accumulator(self._to_numpy_column(reco_jet.mass_raw_diagnostic))
        ntuple["msoftdrop_raw"] += processor.column_accumulator(
            self._to_numpy_column(reco_jet.msoftdrop_raw_diagnostic)
        )
        ntuple["msoftdrop_raw_fatjet"] += processor.column_accumulator(
            self._to_numpy_column(reco_jet.msoftdrop_raw_fatjet_diagnostic)
        )

        if gen_jet is None:
            gen_pt = gen_eta = gen_phi = gen_mass = gen_rapidity = np.full(nrows, np.nan)
            reco_gen_dr = np.full(nrows, np.nan)
        else:
            gen_pt = self._to_numpy_column(gen_jet.pt)
            gen_eta = self._to_numpy_column(gen_jet.eta)
            gen_phi = self._to_numpy_column(gen_jet.phi)
            gen_mass = self._to_numpy_column(gen_jet.mass)
            gen_rapidity = self._to_numpy_column(gen_jet.rapidity)
            reco_gen_dr = self._to_numpy_column(reco_jet.delta_r(gen_jet))

        if groomed_gen_jet is None:
            gen_msoftdrop = np.full(nrows, np.nan)
        else:
            gen_msoftdrop = self._to_numpy_column(groomed_gen_jet.mass)

        ntuple["gen_pt"] += processor.column_accumulator(gen_pt)
        ntuple["gen_eta"] += processor.column_accumulator(gen_eta)
        ntuple["gen_phi"] += processor.column_accumulator(gen_phi)
        ntuple["gen_mass"] += processor.column_accumulator(gen_mass)
        ntuple["gen_rapidity"] += processor.column_accumulator(gen_rapidity)
        ntuple["gen_msoftdrop"] += processor.column_accumulator(gen_msoftdrop)
        ntuple["reco_gen_dr"] += processor.column_accumulator(reco_gen_dr)

        flag_defaults = {
            "has_gen_match": has_gen_match,
            "passes_gen_selection": passes_gen_selection,
            "passes_both": passes_both,
        }
        for name, values in flag_defaults.items():
            if values is None:
                column = np.zeros(nrows, dtype=np.uint8)
            else:
                column = self._to_numpy_flag_column(values)
            ntuple[name] += processor.column_accumulator(column)

    @staticmethod
    def _postprocess_scale_for_dataset(dataset, sumw):
        dataset = str(dataset)
        # 2016 UL: preVFP (APV) = 19.52 fb-1, postVFP = 16.81 fb-1.
        # UL16NanoAODv9 = postVFP, UL16NanoAODAPVv9 = preVFP/APV.
        lumi_db = {
            "UL16NanoAODv9": 16.81,
            "UL16NanoAODAPVv9": 19.52,
            "UL17NanoAODv9": 41.48,
            "UL18NanoAODv9": 59.83,
        }

        if dataset.startswith(("SingleMuon", "EGamma", "SingleElectron")):
            return 1.0

        def scale_from(xs, iov):
            sw = float(sumw[dataset])
            if sw == 0.0:
                return None
            return (xs * lumi_db[iov] * 1000.0) / sw

        if "pythia" in dataset:
            xsdb = {
                "HT-70to100": 140.0,
                "HT-100to200": 139.2,
                "HT-200to400": 38.4,
                "HT-400to600": 5.174,
                "HT-600to800": 1.258,
                "HT-800to1200": 0.5598,
                "HT-1200to2500": 0.1305,
                "HT-2500toInf": 0.002997,
            }
            ht_bin = dataset.split("_")[-1]
            iov = dataset.split("_")[-2]
            xs = xsdb[ht_bin]
            scale = scale_from(xs, iov)
            if scale is None:
                return None
            return scale * 1.1297638966

        if "herwig" in dataset:
            iov = dataset.split("_")[-2]
            return scale_from(5.036, iov)

        if "powheg" in dataset:
            iov = dataset.split("_")[-2]
            return scale_from(1976.0, iov)

        if dataset.startswith("inclusive_"):
            # NLO amcatnloFXFX DY M-50; xsdb GenXSecAnalyzer value (UL ~6404 pb).
            # NLO matrix element, so no LO->NNLO k-factor (unlike the pythia path).
            iov = dataset.split("_")[-1]
            return scale_from(6404.0, iov)

        if dataset.startswith("ptz_"):
            # PtZ-binned NLO amcatnloFXFX DY (LHEFilterPtZ, MatchEWPDG20). Each bin
            # normalized to its own XSDB cross section (pb) -> the 6 bins stitch into
            # the ptZ spectrum. NLO ME, so no k-factor. Name: ptz_<bin>_<iov>.
            ptz_xsdb = {
                "0To50": 1485.0,
                "50To100": 397.4,
                "100To250": 97.2,
                "250To400": 3.701,
                "400To650": 0.5086,
                "650ToInf": 0.04728,
            }
            tokens = dataset.split("_")
            ptz_bin, iov = tokens[1], tokens[-1]
            xs = ptz_xsdb.get(ptz_bin)
            if xs is None:
                return None
            return scale_from(xs, iov)

        if ("st" in dataset) or ("ST" in dataset):
            xsdb = {
                "st_tW_antitop": 34.97,
                "st_tW_top": 34.91,
                "ST_t-channel_antitop": 67.93,
                "ST_t-channel_top": 69.09,
            }
            if dataset.startswith("st_tW_antitop"):
                xs = xsdb["st_tW_antitop"]
            elif dataset.startswith("st_tW_top"):
                xs = xsdb["st_tW_top"]
            elif dataset.startswith("ST_t-channel_antitop"):
                xs = xsdb["ST_t-channel_antitop"]
            else:
                xs = xsdb["ST_t-channel_top"]
            iov = dataset.split("_")[-1]
            return scale_from(xs, iov)

        if "ww" in dataset:
            iov = dataset.split("_")[-1]
            return scale_from(75.95, iov)

        if "wz" in dataset:
            iov = dataset.split("_")[-1]
            return scale_from(27.6, iov)

        if "zz" in dataset:
            iov = dataset.split("_")[-1]
            return scale_from(12.17, iov)

        if "ttjets" in dataset:
            iov = dataset.split("_")[-1]
            return scale_from(471.7, iov)

        return None

    def _scale_reco_jet_ntuple_weights(self, ntuple, sumw):
        if "weight" not in ntuple or "dataset" not in ntuple:
            return

        weights = np.asarray(ntuple["weight"].value, dtype=np.float64)
        if len(weights) == 0:
            return

        datasets = np.asarray(ntuple["dataset"].value, dtype=object)
        scaled_weights = weights.copy()
        for dataset in np.unique(datasets):
            scale = self._postprocess_scale_for_dataset(dataset, sumw)
            if scale is None:
                self.logging.info(f"Ntuple dataset {dataset} is not listed, no scaling done")
                continue
            if scale == 1.0:
                continue
            scaled_weights[datasets == dataset] *= scale
            self.logging.info(f"Scaled reco_jet_ntuple weights for dataset {dataset} by {scale:.6f}")

        ntuple["weight"] = processor.column_accumulator(scaled_weights)

    @staticmethod
    def _with_reco_mass_diagnostic_fields(fatjets):
        fatjets = ak.with_field(
            fatjets,
            fatjets.pt,
            "pt_nanoaod",
        )
        fatjets = ak.with_field(
            fatjets,
            fatjets.mass,
            "mass_nanoaod",
        )
        fatjets = ak.with_field(
            fatjets,
            fatjets.msoftdrop,
            "msoftdrop_nanoaod",
        )
        fatjets = ak.with_field(
            fatjets,
            (1 - fatjets.rawFactor) * fatjets.pt,
            "pt_raw_diagnostic",
        )
        fatjets = ak.with_field(
            fatjets,
            (1 - fatjets.rawFactor) * fatjets.mass,
            "mass_raw_diagnostic",
        )
        fatjets = ak.with_field(
            fatjets,
            (1 - fatjets.rawFactor) * fatjets.msoftdrop,
            "msoftdrop_raw_fatjet_diagnostic",
        )
        return fatjets

    @staticmethod
    def _mass_from_pt_eta_phi_m(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2):
        px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
        py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
        pz = pt1 * np.sinh(eta1) + pt2 * np.sinh(eta2)
        energy1 = np.sqrt((pt1 * np.cosh(eta1)) ** 2 + mass1**2)
        energy2 = np.sqrt((pt2 * np.cosh(eta2)) ** 2 + mass2**2)
        mass2_pair = (energy1 + energy2) ** 2 - px**2 - py**2 - pz**2
        return np.sqrt(np.maximum(mass2_pair, 0))

    @classmethod
    def _with_raw_softdrop_mass_from_subjets(cls, fatjets, subjets):
        raw_subjets = ak.with_field(
            subjets,
            (1 - subjets.rawFactor) * subjets.pt,
            "pt_raw_diagnostic",
        )
        raw_subjets = ak.with_field(
            raw_subjets,
            (1 - subjets.rawFactor) * subjets.mass,
            "mass_raw_diagnostic",
        )

        subjet1 = raw_subjets[fatjets.subJetIdx1]
        subjet2 = raw_subjets[fatjets.subJetIdx2]
        raw_softdrop_mass = cls._mass_from_pt_eta_phi_m(
            subjet1.pt_raw_diagnostic,
            subjet1.eta,
            subjet1.phi,
            subjet1.mass_raw_diagnostic,
            subjet2.pt_raw_diagnostic,
            subjet2.eta,
            subjet2.phi,
            subjet2.mass_raw_diagnostic,
        )
        return ak.with_field(
            fatjets,
            raw_softdrop_mass,
            "msoftdrop_raw_diagnostic",
        )

    @staticmethod
    def _with_ak8_scaled_softdrop_mass(fatjets, raw_reference=None):
        raw_source = fatjets if raw_reference is None else raw_reference
        raw_mass = raw_source.mass_raw_diagnostic
        raw_softdrop_mass = raw_source.msoftdrop_raw_diagnostic
        corrected_mass = fatjets.mass
        valid = (
            ak.fill_none(np.isfinite(raw_mass), False)
            & ak.fill_none(np.isfinite(raw_softdrop_mass), False)
            & ak.fill_none(np.isfinite(corrected_mass), False)
            & ak.fill_none(raw_mass > 0, False)
        )
        safe_raw_mass = ak.where(valid, raw_mass, 1.0)
        scaled_softdrop_mass = ak.where(
            valid,
            raw_softdrop_mass * corrected_mass / safe_raw_mass,
            np.nan,
        )
        return ak.with_field(fatjets, scaled_softdrop_mass, "msoftdrop")

    def _fill_groomed_over_ungroomed_reco(
        self,
        dataset,
        channel,
        ptreco,
        ungroomed_mass,
        groomed_mass,
        weight,
        systematic,
        mass_ratio_name="m_g_over_m_u_reco",
        mass_2d_name="m_g_vs_m_u_reco",
        **extra_axes,
    ):
        fill_mass_ratio = mass_ratio_name in self.hists
        fill_mass_2d = mass_2d_name in self.hists
        if not fill_mass_ratio and not fill_mass_2d:
            return

        valid_mass_ratio = ak.fill_none(
            ~ak.is_none(ungroomed_mass)
            & ~ak.is_none(groomed_mass)
            & np.isfinite(ungroomed_mass)
            & np.isfinite(groomed_mass)
            & (ungroomed_mass > 0),
            False,
        )
        mass_ratio = groomed_mass[valid_mass_ratio] / ungroomed_mass[valid_mass_ratio]
        finite_mass_ratio = ak.fill_none(np.isfinite(mass_ratio), False)

        valid_ungroomed_mass = ungroomed_mass[valid_mass_ratio][finite_mass_ratio]
        valid_groomed_mass = groomed_mass[valid_mass_ratio][finite_mass_ratio]
        valid_weight = weight[valid_mass_ratio][finite_mass_ratio]

        if fill_mass_ratio:
            fill_hist(
                self.hists,
                mass_ratio_name,
                dataset=dataset,
                channel=channel,
                ptreco=ptreco[valid_mass_ratio][finite_mass_ratio],
                mass_ratio=mass_ratio[finite_mass_ratio],
                systematic=systematic,
                weight=valid_weight,
                **extra_axes,
            )

        if fill_mass_2d:
            fill_hist(
                self.hists,
                mass_2d_name,
                dataset=dataset,
                channel=channel,
                m_u_reco=valid_ungroomed_mass,
                m_g_reco=valid_groomed_mass,
                systematic=systematic,
                weight=valid_weight,
                **extra_axes,
            )

    @property
    def accumulator(self):
        return self.hists

    def process(self, events_all):
        self.logging.debug(f"Systematics {self.systematics}")
        self.logging.debug(f"Current Mode {self._mode}")
        t0 = time.time()
        dataset = events_all.metadata["dataset"]
        filename = events_all.metadata['filename']
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

        #print(f"Binning : {events_all.Generator.binvar}")
        n_jk = 2 if self._split2 else 10  # 50:50 split = 2 disjoint halves; jackknife = 10
        for jk_index in range(0, n_jk): ## loops if do_jk is enabled, otherwise breaks at 0
            if not self._do_jk:
                events1 = events_all
                self.logging.debug("Jackknife resampling not enabled, processing all events together.")
            if self._do_jk:
                if self._split2:
                    jk_sel = (index_list % 2) == jk_index  # KEEP one disjoint half (~50% each)
                else:
                    jk_sel = ak.where( (index_list % 10) == jk_index, False, True)  # delete one-tenth
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
                try:
                    ht_bin = ht_bin_tokens[ht_bin_tokens.index("DYJetsToLL")+2]
                except:
                    ht_bin = 'inclusive'
                

                ## Flag used for number of events
                herwig = False
                if 'herwig' in filename: herwig = True
                self.herwig = herwig
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
            if self._do_gen:
                self.logging.info(f"year: {year}, ht_bin: {ht_bin}, herwig: {herwig}")
            else:
                self.logging.info(f"channel: {channel}")
            # year = '2018'


            # ht_bin = 'all'
            # herwig = False
            #channel = 'SingleMuon'
                # lumi_mask = self.lumimasks[IOV](events_all.run, events_all.luminosityBlock)
                # # print("RUN", events_all.run )
                # # print("Lumi block", events_all.luminosityBlock )
                # # print("lumi_mask", lumi_mask)
                # print("Len of evnets after mask", len(events_all[lumi_mask]) )
                # events_all = events_all[lumi_mask]

            # Trigger selection
        
            if not self._do_gen:
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


            events0 = events1 # fix because hassle to change all events1 to events0 below

            weights = Weights(size = len(events0), storeIndividual = True) #initialize weights class
            self.logging.debug("Weights initialized")
            if len(events0) <1:
                return self.hists
            # Store GEN weights or ones based on Simulation/Data
            if self._do_gen:
                
                
                weights.add("genWeight", events0.genWeight)
                if "LHE" in events0.fields:
                    ht_value = events0.LHE.HT
                    print("ht bin", ht_bin)
                    fill_hist(self.hists, 'ht', dataset = dataset, pt = ht_value, ht_bin = ht_bin, weight = weights.weight())

                    # ht_value_ak4 = ak.sum(events0.GenJet[events0.GenJet.pt > 10].pt, axis = 1)
                    
                    # fill_hist(self.hists, 'ht_AK4', dataset = dataset, pt = ht_value_ak4, ht_bin = ht_bin, weight = weights.weight())

                    del ht_value, #ht_value_ak4
            else:
                weights = Weights(size = len(events0), storeIndividual = True)
                weights.add("unity", np.ones(len(events0)))

            


            
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
                fsr_nom, fsr_up, fsr_down = GetPSweights(events0, "FSR")
                try:
                    weights.add(name = "isr", weight = isr_nom, weightUp = isr_up, weightDown = isr_down)
                    weights.add(name = "fsr", weight = fsr_nom, weightUp = fsr_up, weightDown = fsr_down)
                except Exception as e:
                    # e.g. Herwig samples have no/odd PSWeight branch -> isr/fsr not
                    # registered. The systematics loop filters on weights.variations,
                    # so this just means ISR/FSR variations are skipped for this sample.
                    self.logging.warning(f"Could not add ISR/FSR PS weights, skipping them: {e}")

                
            
            # GEN Selection
            if self._do_gen:
                self.logging.info("Entering GEN selection")

                # Apply selection on GEN particles
                # Apply different pt cuts for electrons and muons
                is_electron = np.abs(events0.GenDressedLepton.pdgId) == 11
                is_muon = np.abs(events0.GenDressedLepton.pdgId) == 13

                pt_cut = (
                    (is_electron & (events0.GenDressedLepton.pt > self.lepptcuts[0])) |
                    (is_muon & (events0.GenDressedLepton.pt > self.lepptcuts[1]))
                )
                
                eta_cut = np.abs(events0.GenDressedLepton.eta) < 2.4
                
                events0 = ak.with_field(
                    events0,
                    events0.GenDressedLepton[pt_cut & eta_cut],
                    "GenDressedLepton"
                )

                ## Adding rapidity
                events0 = ak.with_field(
                    events0,
                    ak.with_field(
                        events0.GenJetAK8,
                        getRapidity(events0.GenJetAK8),
                        "rapidity"
                    ),
                    "GenJetAK8"
                )
                
                events0 = ak.with_field(
                                    events0,
                                    events0.GenJetAK8[(events0.GenJetAK8.pt > 0)
                                                & (np.abs(events0.GenJetAK8.rapidity) < 2.4)
                                    ],
                                    "GenJetAK8"
                )
                
                genjets_clean = apply_lepton_separation_gen(
                                events0.GenJetAK8,
                                events0.GenDressedLepton,
                                dr_cut=0.4
                            )
                events0 = ak.with_field(
                            events0,
                            genjets_clean[
                                (genjets_clean.pt > 0)
                                & (np.abs(genjets_clean.rapidity) < 2.4)
                            ],
                            "GenJetAK8"
                        )


                sel.add("oneGenJet", 
                        ak.sum( (events0.GenJetAK8.pt > 0) & (np.abs(events0.GenJetAK8.rapidity) < 2.4), axis=1 ) >= 1
                    )
                sel.add("oneGenJet_pt200", 
                        ak.sum( (events0.GenJetAK8.pt > 200) & (np.abs(events0.GenJetAK8.rapidity) < 2.4), axis=1 ) >= 1
                    )


                z_gen = get_z_gen_selection(events0, sel, self.lepptcuts[0], self.lepptcuts[1], None, None)
                z_ptcut_gen = sel.all("twoGen_leptons") & ak.fill_none(z_gen.pt > 90.0, False)
                z_mcut_gen = sel.all("twoGen_leptons") & ak.fill_none(
                    (z_gen.mass > 71.0) & (z_gen.mass < 111.0),
                    False,
                )

                sel.add("z_ptcut_gen", z_ptcut_gen)
                sel.add("z_mcut_gen", z_mcut_gen)

                ######## dr between two leptons ########
                twoGen_ee_sel = sel.require(twoGen_ee = True)
                twoGen_mm_sel = sel.require(twoGen_mm = True)


                gen_jet, z_jet_dphi_gen = zjet_smp_utils.get_dphi( z_gen, events0.GenJetAK8 )
                z_jet_dr_gen = gen_jet.delta_r(z_gen)

                gensubjets = events0.SubGenJetAK8
                groomed_gen_jet, groomedgensel = get_groomed_jet(gen_jet, gensubjets, False)

                #####################################
                ### Gen event topology selection
                #####################################        
                z_pt_asym_gen = np.abs(z_gen.pt - gen_jet.pt) / (z_gen.pt + gen_jet.pt)
                z_pt_frac_gen = gen_jet.pt / z_gen.pt
                z_pt_asym_sel_gen =  z_pt_asym_gen < 0.3
                z_jet_dphi_sel_gen = z_jet_dphi_gen > 1.57 #2.8 #np.pi * 0.5

                sel.add("z_jet_dphi_sel_gen", z_jet_dphi_sel_gen)
                sel.add("z_pt_asym_sel_gen", z_pt_asym_sel_gen)

                # sel.add("z_jet_dphi_sel_gen_seq", sel.all("z_mcut_gen_seq", "z_jet_dphi_sel_gen") )
                # sel.add( "z_pt_asym_sel_gen_seq", sel.all("z_jet_dphi_sel_gen_seq", "z_pt_asym_sel_gen") )

                kinsel_gen = sel.require(twoGen_leptons=True,oneGenJet=True,z_ptcut_gen=True,z_mcut_gen=True)
                sel.add("kinsel_gen", kinsel_gen)
                toposel_gen = sel.require( z_pt_asym_sel_gen=True, z_jet_dphi_sel_gen=True)
                sel.add("toposel_gen", toposel_gen)

            

            # ----------------- RECO Selection ---------------------------
            self.logging.info("Entering RECO selection")
            
                
            # npv cut
            sel.add('npv', events0.PV.npvsGood >0)

            selectEvents = np.array([events0.Flag[MET_filters[IOV][i]] for i in range(len(MET_filters[IOV]))
                            if MET_filters[IOV][i] in events0.Flag.fields])
            selectEvents = np.logical_and.reduce(selectEvents, axis=0) ## a passing event should pass "ALL" the MET filters

            self.logging.debug("MET Filter applied")

            if ak.sum(selectEvents) < 1 :
                print("No event passed the MET filters.")
                return self.hists

            sel.add("MET", selectEvents)
            sel.add("MET_seq", sel.all('npv', 'MET')) # This is RECO only

            eta = np.abs(events0.Electron.eta)
            
            events0 = ak.with_field(
                events0,
                events0.Electron[
                    (events0.Electron.pt > self.lepptcuts[0])
                    & (eta < 2.4)
                    & ((eta < 1.422) | (eta > 1.566))      # exclude ECAL crack
                    & (events0.Electron.pfRelIso03_all < 0.25)  # suppressing isolation cut here
                    & (events0.Electron.cutBased > 3)      # tight: 4, medium: 3
                    & (np.abs(events0.Electron.dz) < 0.5)
                    & (np.abs(events0.Electron.dxy) < 0.2)
                ],
                "Electron",
            )


                
            events0 = ak.with_field(
                    events0,
                    events0.Muon[(events0.Muon.pt > self.lepptcuts[1]) 
                                &(np.abs(events0.Muon.eta) < 2.4)
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



            # Guard: only evaluate Z kinematics where a valid dilepton pair exists
            has_z = sel.require(twoReco_leptons=True)
            z_ptcut_reco  = has_z & (z_reco.pt > 90)
            z_mcut_reco   = has_z & (z_reco.mass > 71.) & (z_reco.mass < 111.)
            
            sel.add("z_ptcut_reco", z_ptcut_reco)
            sel.add("z_mcut_reco",  z_mcut_reco)
            # z_ptcut_reco = z_reco.pt > 90
            # z_mcut_reco = (z_reco.mass > 71.) & (z_reco.mass < 111.)
            # sel.add("z_ptcut_reco", z_ptcut_reco & (sel.require(twoReco_leptons = True) ))
            # sel.add("z_mcut_reco", z_mcut_reco & (sel.require(twoReco_leptons = True) ))

            self.logging.debug("Z Object Created")
            #### dr reco plots ###
            sel_nominal = copy.deepcopy(sel)
            
            twoReco_ee_sel = sel.require(twoReco_ee = True)
            twoReco_mm_sel = sel.require(twoReco_mm = True)
            twoReco_ll_sel = sel.require(twoReco_leptons = True)

            ## Storing jec variation in the main 'Events' class
            
            fatjets_with_mass_diagnostics = self._with_reco_mass_diagnostic_fields(events0.FatJet)
            corr_jets = GetJetCorrections(fatjets_with_mass_diagnostics, events0, era, IOV, isData = not self._do_gen, mode='AK8')  ###### correcting FatJet.mass
            corr_jets = corr_jets[(corr_jets.subJetIdx1 > -1) & (corr_jets.subJetIdx2 > -1)]
            corr_jets = self._with_raw_softdrop_mass_from_subjets(corr_jets, events0.SubJet)
            corr_jets = self._with_ak8_scaled_softdrop_mass(corr_jets)
            del fatjets_with_mass_diagnostics

            self.logging.debug("Jet Corrections Applied")
            self.logging.debug(f"Available Jet systematics {self.jet_systematics}")
            
            
            
            for jet_syst in self.jet_systematics: # Start loop over jet systematics
                self.logging.debug(f"Processing jet systematic: {jet_syst}")
                        # Apply jet corrections and show leading changes for quick inspection
                if jet_syst != 'nominal':
                    del sel
                sel = copy.deepcopy(sel_nominal)

                # corr_fatjets = GetJetCorrections(
                #     events0.FatJet,
                #     events0,
                #     era=dataset,
                #     IOV=IOV,
                #     isData=not self._do_gen,
                #     mode='AK8',
                # )
                
                # fatjet_pt_after = ak.to_numpy(ak.flatten(corr_fatjets.pt, axis=None))
                self.logging.debug(f"FatJet pt before correction: {events0[ak.num(events0.FatJet, axis = 1) > 0].FatJet.pt}" )
                # self.logging.debug(f"FatJet pt after correction: {fatjet_pt_after[:5]}" )
                

                if jet_syst == "nominal":
                    if self._do_gen:
                        events_j = ak.with_field(events0, jmrsf(IOV, jmssf(IOV, corr_jets)), "FatJet")
                    else:
                        events_j = ak.with_field(events0, corr_jets , "FatJet")
                    self.logging.debug(f"FatJet pt after correction: {events_j[ak.num(events_j.FatJet, axis = 1) > 0].FatJet.pt}" )
                elif jet_syst == "JERUp":
                    corr_jets_obj = self._with_ak8_scaled_softdrop_mass(corr_jets.JER.up, corr_jets)
                    
                    events_j = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                elif jet_syst == "JERDown":
                    corr_jets_obj = self._with_ak8_scaled_softdrop_mass(corr_jets.JER.down, corr_jets)
                    
                    events_j = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                elif jet_syst == "JMSUp":
                    events_j = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets, var = "up")) , "FatJet")
                elif jet_syst == "JMSDown":
                    events_j = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets, var = "down")) , "FatJet")
                elif jet_syst == "JMRUp":
                    events_j = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets), var = "up") , "FatJet")
                elif jet_syst == "JMRDown":
                    events_j = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets), var = "down") , "FatJet")
                
                elif (jet_syst[-2:]=="Up" and "JES" in jet_syst):
                    #print(jet_syst)
                    field = jet_syst[:-2]
                    #print(field)
                    corr_jets_obj = self._with_ak8_scaled_softdrop_mass(corr_jets[field].up, corr_jets)
                    
                    events_j = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                    
                elif (jet_syst[-4:]=="Down" and "JES" in jet_syst):
                    field = jet_syst[:-4]
                    corr_jets_obj = self._with_ak8_scaled_softdrop_mass(corr_jets[field].down, corr_jets)
                    
                    events_j = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                
                else:
                    print("{} is not considered".format(jet_syst))

                ####### ADDIND RAPIDITY ############


                events_j = ak.with_field(
                    events_j,
                    ak.with_field(
                        events_j.FatJet,
                        getRapidity(events_j.FatJet),
                        "rapidity"
                    ),
                    "FatJet"
                )

                
                events_j = ak.with_field(
                    events_j,
                    ak.with_field(
                        events_j.Muon,
                        getRapidity(events_j.Muon),
                        "rapidity"
                    ),
                    "Muon"
                )
                
                events_j = ak.with_field(
                    events_j,
                    ak.with_field(
                        events_j.Electron,
                        getRapidity(events_j.Electron),
                        "rapidity"
                    ),
                    "Electron"
                )
                #events_j.FatJet['y'] = compute_rapidity(events_j.FatJet)


                # events_j = ak.with_field(
                #             events_j,
                #             ak.with_field(events_j.FatJet, compute_rapidity(events_j.FatJet), "y"),
                #             "FatJet"
                #         )
                # events_j = ak.with_field(
                #             events_j,
                #             ak.with_field(events_j.Electron, compute_rapidity(events_j.Electron), "y"),
                #             "Electron"
                #         )
                
                
                presort_recojets = events_j.FatJet[
                    (events_j.FatJet.mass > 0) 
                    &(events_j.FatJet.pt > 0) 
                    &(np.abs(events_j.FatJet.rapidity) < 2.4) 
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
                #recojets = events_j.FatJet
                # if self._do_gen:
                #     hem_weight = HEMVeto(FatJets, runs, isMC=self._do_gen, year = IOV)
                # hem_sel = HEMVeto(recojets, events_j.run) #is not in HEM 15/16 region
                
                hem_weight = None
                if self._do_gen and IOV == "2018":
                    hem_weight = HEMVeto(events_j.FatJet, events_j.run, isMC=True)
                    hem_sel = ak.ones_like(events_j.event)
                else:
                    hem_sel = HEMVeto(recojets, events_j.run) #is not in HEM 15/16 region
                
                # sel.add("oneRecoJet", 
                #             ak.sum( (events_j.FatJet.pt > 0) & (np.abs(events_j.FatJet.eta) < 2.5)  & (events_j.FatJet.jetId == 6) & hem_sel, axis=1 ) >= 1
                #         )

                sel.add("oneRecoJet",
                    ak.sum((recojets.pt > 0) & (np.abs(recojets.eta) < 2.5) & (recojets.jetId == 6) & hem_sel, axis=1) >= 1 )


                reco_jet, z_jet_dphi_reco = zjet_smp_utils.get_dphi( z_reco, recojets )
                
            
                #reco_jet = ak.firsts(recojets)


                ## Muon correction
                rocc = get_rocc_corrections()
                rng = np.random.default_rng(seed=12345)
                
                if self._do_gen:
                    k = rocc.ak_kSmearMC(events_j.Muon, events_j.Muon.nTrackerLayers, rng=rng)
                else:
                    
                    k = rocc.ak_kScaleDT(events_j.Muon)
                
                mu_corr = ak.with_field(events_j.Muon, events_j.Muon.pt * k, "pt")
                events_j = ak.with_field(events_j, mu_corr, "Muon")

                ### Sorting by pt
                mu = events_j.Muon

                index = ak.argsort(mu.pt, axis=1, ascending=False)
                mu_sorted = mu[index]
                
                # Replace the Muon collection in events_j
                events_j = ak.with_field(events_j, mu_sorted, "Muon")


            
                #print("Special event reco_jet object pt", reco_jet[sel_spl].pt)
                #print("Special event FatJets object pt, eta, jetid", events0[sel_spl].FatJet.pt, events0[sel_spl].FatJet.eta, events0[sel_spl].FatJet.jetId)

                #print("Special event FatJets object pt, eta, jetid", recojets[sel_spl].pt, recojets[sel_spl].eta, recojets[sel_spl].jetId)
                #reco_jet, dr = find_closest_dr( gen_jet, recojets )
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
                if jet_syst == "nominal":
                    ## N-1 for pt asym cut
                    if self._do_gen:
                        sel_n1_ptasym = sel.all("npv" , "MET" , "kinsel_reco", "z_jet_dphi_sel_reco") 
                    else:
                        sel_n1_ptasym = sel.all("npv" , "MET" , "kinsel_reco","trigsel", "z_jet_dphi_sel_reco" ) 
    
                    if ak.sum(sel_n1_ptasym)>0:
                        self.logging.debug(f"Sum of this sel is {ak.sum(sel_n1_ptasym)}")
                        self.logging.debug(f"Len? {len(z_pt_asym_reco[sel_n1_ptasym])}  ")
                        
                        #print(z_pt_asym_reco[sel_n1_ptasym])
                        fill_hist(self.hists, "ptasym_presel", dataset = dataset, frac = z_pt_asym_reco[sel_n1_ptasym], weight = weights.weight()[sel_n1_ptasym])
                
                
                if self._do_gen:
                    import matplotlib.pyplot as plt
                    #plt.hist(reco_jet.delta_r(gen_jet), bins = 100, range = (0,1))
                    #plt.show()
                    is_matched_reco = reco_jet.delta_r(gen_jet) < 0.4
                    sel.add("is_matched_reco", is_matched_reco)

                    allsel_reco = sel.all("npv", "MET", "kinsel_reco", "toposel_reco")#, "is_matched_reco" )
                    #plt.hist(reco_jet[allsel_reco].delta_r(gen_jet[allsel_reco]), bins = 1000, range = (0,4))
                    #plt.show()
                    sel.add("allsel_reco", allsel_reco)
                    is_matched_gen = gen_jet.delta_r(reco_jet) < 0.4
                    sel.add("is_matched_gen", is_matched_gen)
                    allsel_gen = sel.all("kinsel_gen", "toposel_gen" , "is_matched_gen" )
                    sel.add("allsel_gen", allsel_gen)
                    #sel.add("fakes", sel.require(allsel_reco = True, allsel_gen = False))
                else:
                    allsel_reco = sel.all("npv", "MET", "kinsel_reco", "toposel_reco", "trigsel" )
                    sel.add("allsel_reco", allsel_reco)

            




                if (self._do_gen) and (jet_syst == "nominal"):

                    self.logging.debug("Padded Electron/Muon collections to minimum length 2 per event")
                    add_lepton_weights(events_j, twoReco_ee_sel, twoReco_mm_sel, weights, IOV)  

                    self.logging.info("Lepton weights added")      
                sel_reco = sel.require(allsel_reco = True)
                if jet_syst == "nominal":
                    fill_hist(self.hists, "ptz_mz_reco", dataset =  dataset, mass = z_reco[sel_reco].mass, pt = z_reco[sel_reco].pt, weight = weights.weight()[sel_reco]) 

                #####-###-----------PLOTTING EVENTS ----___##


                
                # ------------------------------------------------------------
                # Small helpers
                # ------------------------------------------------------------
                # def _to_1d_numpy(x):
                #     if x is None:
                #         return np.array([])
                #     return ak.to_numpy(ak.flatten(ak.fill_none(x, []), axis=None))
                
                # def _wrap_phi(phi):
                #     """Wrap phi into [0, 2pi) for polar plotting."""
                #     phi = np.asarray(phi)
                #     return np.mod(phi, 2 * np.pi)
                
                # def _event_collection(ev, name):
                #     return getattr(ev, name) if name in ev.fields else None
                
                # def _plot_collection_polar(
                #     ax,
                #     coll,
                #     label,
                #     color=None,
                #     marker="o",
                #     open_marker=False,
                #     annotate_pt=False,
                #     max_r=None,
                #     lw=1.5,
                #     alpha=0.95,
                # ):
                #     """
                #     Plot a single-event collection on polar plane:
                #       theta = phi
                #       r     = pt
                #     """
                #     if coll is None or len(coll) == 0:
                #         return
                
                #     pt  = _to_1d_numpy(coll.pt)
                #     phi = _wrap_phi(_to_1d_numpy(coll.phi))
                
                #     if len(pt) == 0:
                #         return
                
                #     # scatter
                #     if open_marker:
                #         ax.scatter(phi, pt, marker=marker, s=65, facecolors="none",
                #                    edgecolors=color, linewidths=lw, alpha=alpha, label=label)
                #     else:
                #         ax.scatter(phi, pt, marker=marker, s=45, color=color,
                #                    linewidths=lw, alpha=alpha, label=label)
                
                #     # radial lines from origin
                #     for p, ph in zip(pt, phi):
                #         ax.plot([ph, ph], [0, p], color=color, alpha=0.35, lw=1.2)
                
                #     if annotate_pt:
                #         for p, ph in zip(pt, phi):
                #             ax.text(ph, p + 0.02 * (max_r if max_r is not None else np.max(pt) + 1),
                #                     f"{p:.0f}", fontsize=7, ha="center", va="bottom")
                
                # def _plot_single_object_polar(
                #     ax,
                #     obj,
                #     label,
                #     color=None,
                #     marker="D",
                #     open_marker=False,
                #     lw=2.0,
                #     alpha=1.0,
                # ):
                #     """
                #     Plot a single object (not a collection) that has pt, phi fields.
                #     Works for ak.Record or one-element arrays.
                #     """
                #     if obj is None:
                #         return
                
                #     try:
                #         pt = ak.to_numpy(ak.atleast_1d(obj.pt))
                #         phi = _wrap_phi(ak.to_numpy(ak.atleast_1d(obj.phi)))
                #     except Exception:
                #         return
                
                #     if len(pt) == 0:
                #         return
                
                #     p = float(pt[0])
                #     ph = float(phi[0])
                
                #     ax.plot([ph, ph], [0, p], color=color, alpha=0.5, lw=1.5)
                
                #     if open_marker:
                #         ax.scatter([ph], [p], marker=marker, s=70, facecolors="none",
                #                    edgecolors=color, linewidths=lw, alpha=alpha, label=label)
                #     else:
                #         ax.scatter([ph], [p], marker=marker, s=60, color=color,
                #                    linewidths=lw, alpha=alpha, label=label)
                
                # def _compute_event_z_from_leptons(coll):
                #     """
                #     Approximate Z direction from first two leptons in a collection using vector sum in transverse plane.
                #     Returns (pt, phi) or None.
                #     """
                #     if coll is None or len(coll) < 2:
                #         return None
                
                #     try:
                #         pt  = ak.to_numpy(coll.pt)
                #         phi = ak.to_numpy(coll.phi)
                #     except Exception:
                #         return None
                
                #     if len(pt) < 2:
                #         return None
                
                #     px = pt[0] * np.cos(phi[0]) + pt[1] * np.cos(phi[1])
                #     py = pt[0] * np.sin(phi[0]) + pt[1] * np.sin(phi[1])
                
                #     z_pt = np.hypot(px, py)
                #     z_phi = np.arctan2(py, px)
                #     return {"pt": z_pt, "phi": z_phi}
                
                # def _plot_z_candidate(ax, zcand, label, color, ls="--", marker="p"):
                #     if zcand is None:
                #         return
                #     ph = _wrap_phi(np.array([zcand["phi"]]))[0]
                #     pt = zcand["pt"]
                #     ax.plot([ph, ph], [0, pt], color=color, alpha=0.7, lw=1.6, ls=ls)
                #     ax.scatter([ph], [pt], marker=marker, s=70, facecolors="none",
                #                edgecolors=color, linewidths=1.8, label=label)
                # def _mask_value(mask, iev):
                #     if mask is None:
                #         return "NA"
                #     try:
                #         return bool(mask[iev])
                #     except Exception:
                #         return "NA"
                
                # # ------------------------------------------------------------
                # # Main plotting function
                # # ------------------------------------------------------------

                # def add_event_debug_panel(
                #     ax,
                #     iev,
                #     allsel_reco,
                #     allsel_gen,
                #     npv_mask,
                #     MET_mask,
                #     kinsel_reco_mask,
                #     toposel_reco_mask,
                #     kinsel_gen_mask,
                #     toposel_gen_mask,
                #     is_matched_gen,
                #     reco_jet,
                #     gen_jet,
                #     z_reco,
                #     z_gen,
                # ):
                #     lines = []
                
                #     # --- selection bits ---
                #     lines.append("Selection:")
                #     lines.append(f" npv           : {_mask_value(npv_mask, iev)}")
                #     lines.append(f" MET           : {_mask_value(MET_mask, iev)}")
                #     lines.append(f" kinsel_reco   : {_mask_value(kinsel_reco_mask, iev)}")
                #     lines.append(f" toposel_reco  : {_mask_value(toposel_reco_mask, iev)}")
                #     lines.append(f" kinsel_gen    : {_mask_value(kinsel_gen_mask, iev)}")
                #     lines.append(f" toposel_gen   : {_mask_value(toposel_gen_mask, iev)}")
                #     lines.append(f" matched       : {_mask_value(is_matched_gen_mask, iev)}")
                #     lines.append(f" allsel_reco   : {_mask_value(allsel_reco, iev)}")
                #     lines.append(f" allsel_gen    : {_mask_value(allsel_gen, iev)}")
                
                #     # --- object info ---
                #     try:
                #         rj = reco_jet[iev]
                #         gj = gen_jet[iev]
                #         zr = z_reco[iev]
                #         zg = z_gen[iev]
                
                #         lines.append("Reco jet:")
                #         lines.append(f" pt = {float(rj.pt):.1f}")
                #         lines.append(f" eta= {float(rj.eta):.2f}")
                #         lines.append(f" phi= {float(rj.phi):.2f}")
                #         lines.append("")
                
                #         lines.append("Gen jet:")
                #         lines.append(f" pt = {float(gj.pt):.1f}")
                #         lines.append(f" eta= {float(gj.eta):.2f}")
                #         lines.append(f" phi= {float(gj.phi):.2f}")
                #         lines.append("")
                
                #         lines.append("Reco Z:")
                #         lines.append(f" pt = {float(zr.pt):.1f}")
                #         lines.append(f" m  = {float(zr.mass):.1f}")
                #         lines.append("")
                
                #         lines.append("Gen Z:")
                #         lines.append(f" pt = {float(zg.pt):.1f}")
                #         lines.append(f" m  = {float(zg.mass):.1f}")
                #         lines.append("")
                
                #         lines.append("Derived:")
                #         lines.append(f" dphi Z_reco-jet = {abs(float(zr.delta_phi(rj))):.3f}")
                #         lines.append(f" dphi Z_gen-jet  = {abs(float(zg.delta_phi(gj))):.3f}")
                #         lines.append(f" dR reco-gen jet = {float(rj.delta_r(gj)):.3f}")
                #         lines.append(f" pt balance reco = {float(rj.pt/zr.pt):.3f}")
                #         lines.append(f" pt balance gen  = {float(gj.pt/zg.pt):.3f}")
                
                #     except:
                #         lines.append("Object info missing")
                
                #     text = "\n".join(lines)
                
                #     ax.text(
                #         0.0,
                #         1.0,
                #         text,
                #         fontsize=12,
                #         verticalalignment="top",
                #         horizontalalignment="left",
                #         family="monospace",
                #     )
                # def plot_reco_not_gen_polar_events(
                #     events_j,
                #     allsel_reco,
                #     allsel_gen,
                #     reco_jet=None,
                #     gen_jet=None,
                #     z_reco=None,
                #     z_gen=None,
                #     npv_mask=None,
                #     MET_mask=None,
                #     kinsel_reco_mask=None,
                #     toposel_reco_mask=None,
                #     kinsel_gen_mask=None,
                #     toposel_gen_mask=None,
                #     is_matched_gen_mask=None,
                #     outdir="outputs/plots/eventImage",
                #     max_events=50,
                # ):
                #     os.makedirs(outdir, exist_ok=True)
                
                #     mask = ak.to_numpy(~allsel_reco & allsel_gen)
                #     idxs = np.flatnonzero(mask)[:max_events]
                
                #     print(f"Found {mask.sum()} events with allsel_reco=True and allsel_gen=False")
                #     print(f"Saving first {len(idxs)} event displays to {outdir}")
                
                #     for i, iev in enumerate(idxs):
                #         ev = events_j[iev]
                
                #         # estimate a sensible radial max from all visible objects
                #         pts_all = []
                
                #         for name in ["FatJet", "GenJetAK8", "Muon", "Electron", "GenDressedLepton"]:
                #             if name in ev.fields and len(getattr(ev, name)) > 0:
                #                 pts_all.extend(list(_to_1d_numpy(getattr(ev, name).pt)))
                
                #         if reco_jet is not None:
                #             try:
                #                 pts_all.append(float(ak.to_numpy(ak.atleast_1d(reco_jet[iev].pt))[0]))
                #             except Exception:
                #                 pass
                
                #         if gen_jet is not None:
                #             try:
                #                 pts_all.append(float(ak.to_numpy(ak.atleast_1d(gen_jet[iev].pt))[0]))
                #             except Exception:
                #                 pass
                
                #         rmax = max(50, 1.15 * max(pts_all)) if len(pts_all) else 100
                
                #         fig = plt.figure(figsize=(14, 8))
                #         #gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.9], wspace=0.15)
                #         gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.45, 0.9], wspace=0.1)
                #         ax      = fig.add_subplot(gs[0, 0], projection="polar")
                #         ax_leg  = fig.add_subplot(gs[0, 1])
                #         ax_info = fig.add_subplot(gs[0, 2])
                #         ax_leg.axis("off")
                #         ax_info.axis("off")
                        

                
                #         # --- Plot full collections ---
                #         _plot_collection_polar(
                #             ax, _event_collection(ev, "GenDressedLepton"),
                #             label="Gen Dressed Lepton", color="royalblue",
                #             marker="o", open_marker=True, annotate_pt=False, max_r=rmax
                #         )
                
                #         _plot_collection_polar(
                #             ax, _event_collection(ev, "Electron"),
                #             label="Reco Electron", color="tab:green",
                #             marker="s", open_marker=True, annotate_pt=False, max_r=rmax
                #         )
                
                #         _plot_collection_polar(
                #             ax, _event_collection(ev, "Muon"),
                #             label="Reco Muon", color="tab:red",
                #             marker="^", open_marker=True, annotate_pt=False, max_r=rmax
                #         )
                
                #         _plot_collection_polar(
                #             ax, _event_collection(ev, "GenJetAK8"),
                #             label="Gen Jet", color="brown",
                #             marker="D", open_marker=False, annotate_pt=True, max_r=rmax
                #         )
                
                #         _plot_collection_polar(
                #             ax, _event_collection(ev, "FatJet"),
                #             label="Reco Jet", color="coral",
                #             marker="D", open_marker=True, annotate_pt=True, max_r=rmax
                #         )
                
                #         # --- Overlay selected reco/gen jet if provided ---
                #         # if reco_jet is not None:
                #         #     try:
                #         #         _plot_single_object_polar(
                #         #             ax, reco_jet[iev],
                #         #             label="Selected Reco Jet", color="darkorange",
                #         #             marker="D", open_marker=True, lw=2.6
                #         #         )
                #         #     except Exception:
                #         #         pass
                
                #         # if gen_jet is not None:
                #         #     try:
                #         #         _plot_single_object_polar(
                #         #             ax, gen_jet[iev],
                #         #             label="Selected Gen Jet", color="darkred",
                #         #             marker="D", open_marker=False, lw=2.6
                #         #         )
                #         #     except Exception:
                #         #         pass
                #         if reco_jet is not None:
                #             try:
                #                 _plot_single_object_polar(
                #                     ax, reco_jet[iev],
                #                     label="Selected Reco Jet",
                #                     color="gold",
                #                     marker="X",
                #                     open_marker=False,
                #                     lw=3.5,
                #                     alpha=1.0,
                #                 )
                #             except Exception:
                #                 pass
                        
                #         if gen_jet is not None:
                #             try:
                #                 _plot_single_object_polar(
                #                     ax, gen_jet[iev],
                #                     label="Selected Gen Jet",
                #                     color="black",
                #                     marker="*",
                #                     open_marker=False,
                #                     lw=3.5,
                #                     alpha=1.0,
                #                 )
                #             except Exception:
                #                 pass
                
                #         # --- Approximate Z from reco leptons ---
                #         reco_z = None
                #         if "Muon" in ev.fields and len(ev.Muon) >= 2:
                #             reco_z = _compute_event_z_from_leptons(ev.Muon)
                #         elif "Electron" in ev.fields and len(ev.Electron) >= 2:
                #             reco_z = _compute_event_z_from_leptons(ev.Electron)
                
                #         gen_z = _compute_event_z_from_leptons(_event_collection(ev, "GenDressedLepton"))
                
                #         # _plot_z_candidate(ax, reco_z,  label="Gen Z",  color="magenta", ls="-.", marker="p")
                #         # _plot_z_candidate(ax, gen_z, label="Reco Z", color="hotpink", ls="--", marker="p")

                #         reco_z = None
                #         gen_z  = None
                        
                #         if z_reco is not None:
                #             try:
                #                 reco_z = z_reco[iev]
                #             except:
                #                 pass
                        
                #         if z_gen is not None:
                #             try:
                #                 gen_z = z_gen[iev]
                #             except:
                #                 pass
                        
                #         _plot_single_object_polar(
                #             ax, reco_z,
                #             label="Reco Z",
                #             color="hotpink",
                #             marker="p",
                #             open_marker=True,
                #             lw=2.5,
                #         )
                        
                #         _plot_single_object_polar(
                #             ax, gen_z,
                #             label="Gen Z",
                #             color="magenta",
                #             marker="p",
                #             open_marker=False,
                #             lw=2.5,
                #         )
                
                #         # --- Optional Δφ between selected reco/gen jet ---
                #         title = f"Event {iev}"
                #         if reco_jet is not None and gen_jet is not None:
                #             try:
                #                 dphi = float(abs(reco_jet[iev].delta_phi(gen_jet[iev])))
                #                 title += f"   Δ(φ) = {dphi:.4f}"
                #             except Exception:
                #                 pass
                
                #         ax.set_title(title, pad=28, fontsize=16)
                #         ax.set_rmax(rmax)
                #         ax.set_rlabel_position(22.5)
                #         ax.set_xlabel(r"$\phi$ [rad]", labelpad=18)
                #         ax.set_ylabel(r"$\rho = p_T$ [GeV]", labelpad=25)
                #         ax.grid(True, alpha=0.65)
                
                #         handles, labels = ax.get_legend_handles_labels()
                #         ax_leg.legend(
                #             handles, labels,
                #             loc="upper left",
                #             fontsize=10,
                #             frameon=True,
                #         )

                #         add_event_debug_panel(
                #             ax_info,
                #             iev,
                #             allsel_reco,
                #             allsel_gen,
                #             npv_mask,
                #             MET_mask,
                #             kinsel_reco_mask,
                #             toposel_reco_mask,
                #             kinsel_gen_mask,
                #             toposel_gen_mask,
                #             is_matched_gen,
                #             reco_jet,
                #             gen_jet,
                #             z_reco,
                #             z_gen,
                #         )
                
                #         fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.08)
                #         fout = os.path.join(outdir, f"event_{i:03d}_entry_{iev}.png")
                #         plt.savefig(fout, dpi=160)
                #         plt.close(fig)
                
                #     return idxs
                # npv_mask          = sel.require(npv=True)
                # MET_mask          = sel.require(MET=True)
                # kinsel_reco_mask  = sel.require(kinsel_reco=True)
                # toposel_reco_mask = sel.require(toposel_reco=True)
                # kinsel_gen_mask   = sel.require(kinsel_gen=True)
                # toposel_gen_mask  = sel.require(toposel_gen=True)
                # is_matched_gen_mask = sel.require(is_matched_gen=True)

                # bad_idx = plot_reco_not_gen_polar_events(
                #     events_j=events_j,
                #     allsel_reco=allsel_reco,
                #     allsel_gen=allsel_gen,
                #     reco_jet=reco_jet,
                #     gen_jet=gen_jet,
                #     z_reco=z_reco,
                #     z_gen=z_gen,
                #     npv_mask=npv_mask,
                #     MET_mask=MET_mask,
                #     kinsel_reco_mask=kinsel_reco_mask,
                #     toposel_reco_mask=toposel_reco_mask,
                #     kinsel_gen_mask=kinsel_gen_mask,
                #     toposel_gen_mask=toposel_gen_mask,
                #     is_matched_gen_mask=is_matched_gen_mask,
                #     outdir="outputs/plots/eventImage",
                #     max_events=50,
                # )

                



                ######-----------_#############
                
                # Fill histograms
                # GEN level histograms
                self.logging.debug(f"Total reco events passing all selection: {sel.require(allsel_reco = True).sum()}",  )
                if self._do_gen:
                    self.logging.debug(f"Total gen events passing all selection: {sel.require(allsel_gen = True).sum()}")
                    self.logging.debug(f"Total events passing both reco and gen selections: {sel.require(allsel_reco = True, allsel_gen = True).sum()}")
                self.logging.debug(f"Total reco events (ee channel) passing all selection: {sel.require(allsel_reco = True, twoReco_ee = True).sum()}",  )
                self.logging.debug(f"Total reco events (mm channel) passing all selection: {sel.require(allsel_reco = True, twoReco_mm = True).sum()}",  )
                self.logging.debug(f"Weights sample: {weights.weight()[sel_reco][:10]}" )
                channels = ['mm', 'ee']
                for channel in channels:
                    
                    if channel == 'ee':
                        print("Now doing ee")
                        if self._do_gen:
                            sel_both = sel.require(allsel_reco = True, allsel_gen = True, twoGen_ee = True, twoReco_ee = True)
                            sel_gen = sel.require(allsel_gen = True, twoGen_ee = True)
                        sel_reco = sel.require(allsel_reco = True, twoReco_ee = True)
                    else:
                        print("Now doing mm")
                        if self._do_gen:
                            sel_both = sel.require(allsel_reco = True, allsel_gen = True, twoGen_mm = True, twoReco_mm = True)
                            sel_gen = sel.require(allsel_gen = True, twoGen_mm = True)
                        sel_reco = sel.require(allsel_reco = True, twoReco_mm = True)
                    jetR = 0.8
                    print(f"JET SYST {jet_syst}")
                    if jet_syst == "nominal":
                        self.logging.debug(f"list of systematics {self.systematics}")
                        # Some weight variations (e.g. ISR/FSR from PSWeight, q2, pdf)
                        # are absent in certain samples -- Herwig has no PSWeight branch,
                        # so "isr"/"fsr" never get registered above. Skip any weight
                        # systematic whose modifier isn't actually present rather than
                        # letting weights.weight(modifier=...) raise a KeyError.
                        available_variations = set(weights.variations)
                        for syst in self.systematics:
                            if syst != "nominal" and syst not in available_variations:
                                self.logging.debug(
                                    f"Skipping systematic {syst}: not available for this sample"
                                )
                                continue
                            self.logging.debug(f"Processing systematic {syst}")
                            if syst == "nominal":
                                if self._do_gen:
                                    weights_both = weights.weight()[sel_both]
                                weights_reco = weights.weight()[sel_reco]
                                
                                
                            else:
                                weights_reco = weights.weight(modifier=syst)[sel_reco]
                                if self._do_gen:
                                    weights_both = weights.weight(modifier=syst)[sel_both]
                            if hem_weight is not None:
                                weights_reco = weights_reco * hem_weight[sel_reco]
                                if self._do_gen:
                                    weights_both = weights_both * hem_weight[sel_both]
                            mass_jk_fill = {"jk": jk_index} if self._mode == "mass_jk" else {}
                            if self._do_gen:
                                gen_jet_truth = gen_jet[sel_gen]
                                groomed_gen_jet_truth = groomed_gen_jet[sel_gen]
                                #### GEN-truth distributions carry the generator weight
                                #### plus *theory* reweightings (ISR/FSR/q2/PDF), but NOT
                                #### detector effects (pu, L1 prefiring, lepton SFs, HEM).
                                #### Detector SFs belong on the response matrix / reco
                                #### hists (weights_both / weights_reco) so that the
                                #### unfolding efficiency = matched(full) / all_gen carries
                                #### the detector-correction SFs and 1/eff properly corrects
                                #### data back to the true gen-level spectrum.
                                ####
                                #### Theory weights, by contrast, are per-event reweightings
                                #### of the physics itself: under an ISR/FSR/q2/PDF variation
                                #### the *gen-level* prediction genuinely changes, so the gen
                                #### truth, the matched response and the misses must all move
                                #### consistently. Their nominal weight is 1.0 (see
                                #### GetPSweights / GetQ2weights / GetPDFweights), so including
                                #### them leaves the nominal gen spectrum unchanged -- only the
                                #### Up/Down modifier shifts it. Detector systematics pass
                                #### modifier=None here and so the gen truth stays nominal.
                                _theory_gen_systs = {
                                    "isrUp", "isrDown", "fsrUp", "fsrDown",
                                    "q2Up", "q2Down", "pdfUp", "pdfDown",
                                }
                                _gen_modifier = syst if syst in _theory_gen_systs else None
                                # Only include theory weights actually registered for this
                                # sample (Herwig lacks ISR/FSR PSWeights, so "isr"/"fsr"
                                # may be absent); partial_weight raises on unknown names.
                                _gen_include = [
                                    n for n in ['genWeight', 'isr', 'fsr', 'q2', 'pdf']
                                    if n in weights._weights
                                ]
                                weights_gen = weights.partial_weight(
                                    include=_gen_include,
                                    modifier=_gen_modifier,
                                )[sel_gen]

                                # Joint gen groomed-vs-ungroomed mass (same events) for the
                                # groomed<->ungroomed covariance. Built here while gen_jet_truth,
                                # groomed_gen_jet_truth and weights_gen are all aligned to sel_gen,
                                # with a SINGLE common valid mask so the two masses stay paired.
                                # No-ops unless the hist is registered ("mass_cov" mode).
                                if self._mode == "mass_cov":
                                    _mu_cov = gen_jet_truth.mass
                                    _mg_cov = groomed_gen_jet_truth.mass
                                    _pt_cov = gen_jet_truth.pt
                                    _ok_cov = ~(ak.is_none(_pt_cov) | ak.is_none(_mu_cov) | ak.is_none(_mg_cov))
                                    _pt_c = _pt_cov[_ok_cov]
                                    _mu_c = _mu_cov[_ok_cov]
                                    _mg_c = _mg_cov[_ok_cov]
                                    _w_c = weights_gen[_ok_cov]
                                    fill_hist(
                                        self.hists,
                                        "ptjet_mjet_g_vs_u_gen",
                                        dataset=dataset,
                                        channel=channel,
                                        ptgen=_pt_c,
                                        m_g_gen=_mg_c,
                                        m_u_gen=_mu_c,
                                        weight=_w_c,
                                        systematic=syst,
                                    )
                                    fill_hist(
                                        self.hists,
                                        "ptjet_rhojet_g_vs_u_gen",
                                        dataset=dataset,
                                        channel=channel,
                                        ptgen=_pt_c,
                                        mpt_g_gen=2*np.log10(_mg_c/(_pt_c*jetR)),
                                        mpt_u_gen=2*np.log10(_mu_c/(_pt_c*jetR)),
                                        weight=_w_c,
                                        systematic=syst,
                                    )
                                    # reco-level counterpart (Scope#3 asks gen AND reco
                                    # separately): same-event reco groomed×ungroomed, nominal
                                    # weight only (one fill/event/channel; jet_syst is nominal
                                    # in this branch). reco_jet is the JEC-corrected selected jet.
                                    if syst == "nominal":
                                        _rj = reco_jet[sel_reco]
                                        _ptr = _rj.pt
                                        _mur = _rj.mass
                                        _mgr = _rj.msoftdrop
                                        _okr = ~(ak.is_none(_ptr) | ak.is_none(_mur) | ak.is_none(_mgr))
                                        _ptr_c = _ptr[_okr]; _mur_c = _mur[_okr]; _mgr_c = _mgr[_okr]
                                        _wr_c = weights_reco[_okr]
                                        fill_hist(
                                            self.hists, "ptjet_mjet_g_vs_u_reco",
                                            dataset=dataset, channel=channel, ptreco=_ptr_c,
                                            m_g_reco=_mgr_c, m_u_reco=_mur_c,
                                            weight=_wr_c, systematic=syst,
                                        )
                                        fill_hist(
                                            self.hists, "ptjet_rhojet_g_vs_u_reco",
                                            dataset=dataset, channel=channel, ptreco=_ptr_c,
                                            mpt_g_reco=2*np.log10(_mgr_c/(_ptr_c*jetR)),
                                            mpt_u_reco=2*np.log10(_mur_c/(_ptr_c*jetR)),
                                            weight=_wr_c, systematic=syst,
                                        )


                                ptgen = gen_jet_truth.pt
                                ptgen = ptgen[~ak.is_none(ptgen)]
                                mgen = gen_jet_truth.mass
                                mgen = mgen[~ak.is_none(mgen)]
                                mgen_g = groomed_gen_jet_truth.mass
                                mgen_g = mgen_g[~ak.is_none(mgen_g)]
                                weights_gen = weights_gen[~ak.is_none(mgen)]

                                mpt_gen = 2*np.log10(mgen/(ptgen*jetR))
                                mpt_gen_g = 2*np.log10(mgen_g/(ptgen*jetR))
                                weights_gen_u = weights_gen
                                weights_gen_g = weights_gen

                                if self._do_reweight:
                                    herwig_weight_u, herwig_weight_g = self._get_herwig_reweights(
                                        ptgen,
                                        mgen,
                                        mgen_g,
                                        jetR,
                                    )
                                    weights_gen_u = weights_gen * herwig_weight_u
                                    weights_gen_g = weights_gen * herwig_weight_g

                                fill_hist(
                                    self.hists,
                                    "ptjet_mjet_u_gen",
                                    dataset=dataset,
                                    channel=channel,
                                    ptgen=ptgen,
                                    mgen=mgen,
                                    weight=weights_gen_u if self._is_mass_reweight() else weights_gen,
                                    systematic=syst,
                                    **mass_jk_fill,
                                )
                                fill_hist(
                                    self.hists,
                                    "ptjet_mjet_g_gen",
                                    dataset=dataset,
                                    channel=channel,
                                    ptgen=ptgen,
                                    mgen=mgen_g,
                                    weight=weights_gen_g if self._is_mass_reweight() else weights_gen,
                                    systematic=syst,
                                    **mass_jk_fill,
                                )

                                if self._do_reweight and self._is_rho_reweight():
                                    self.logging.debug("We are doing reweight right?")
                                    #self.logging.debug(f"And the weights are not false? {weights_gen_g}")

                                    fill_hist(self.hists, "ptjet_rhojet_u_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = mpt_gen, weight = weights_gen_u, systematic = syst)
                                    
                                    fill_hist(self.hists, "ptjet_rhojet_g_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = mpt_gen_g, weight = weights_gen_g, systematic = syst)
                                elif self._do_jk:
                                    fill_hist(self.hists, "ptjet_rhojet_u_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = 2*np.log10(mgen/(ptgen*jetR)), weight = weights_gen, jk = jk_index, systematic = syst)
                                    
                                    fill_hist(self.hists, "ptjet_rhojet_g_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = 2*np.log10(mgen_g/(ptgen*jetR)), weight = weights_gen, jk = jk_index, systematic = syst)
        



                                else:
                                    #self.logging.debug(f"No of GEN JET {len(mgen)}")  
                                    partonFlavour = gen_jet_truth.partonFlavour
                                    abs_pdg = abs(partonFlavour)
    
                                    is_quark = (
                                        (abs_pdg == 1) |
                                        (abs_pdg == 2) |
                                        (abs_pdg == 3) |
                                        (abs_pdg == 4) |
                                        (abs_pdg == 5)
                                    )
                                    
                                    is_gluon = (gen_jet_truth.partonFlavour == 21)
    
                                    parton_flavor = ak.where(is_quark, 1,
                                      ak.where(is_gluon, 2, 0))

                                    fill_hist(self.hists, "ptjet_rhojet_u_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = 2*np.log10(mgen/(ptgen*jetR)), weight = weights_gen, systematic = syst)
                                    
                                    fill_hist(self.hists, "ptjet_rhojet_g_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = 2*np.log10(mgen_g/(ptgen*jetR)), weight = weights_gen, systematic = syst)

                                    fill_hist(self.hists, "pt_flavor_jet0_gen", dataset = dataset, pt = ptgen, n = parton_flavor, systematic = syst, weight = weights_gen )
                                    fill_hist(self.hists, "y_flavor_jet0_gen", dataset = dataset, y = gen_jet_truth.rapidity, n = parton_flavor, systematic = syst, weight = weights_gen )

                                    valid_validation_jet = (
                                        ~ak.is_none(gen_jet_truth.pt)
                                        & ~ak.is_none(gen_jet_truth.mass)
                                        & ~ak.is_none(gen_jet_truth.partonFlavour)
                                    )
                                    validation_gen_jet = gen_jet_truth[valid_validation_jet]
                                    validation_abs_pdg = abs(validation_gen_jet.partonFlavour)
                                    validation_mpt = 2 * np.log10(
                                        validation_gen_jet.mass / (validation_gen_jet.pt * jetR)
                                    )

                                    validation_flavor_masks = {
                                        "quark": (
                                            (validation_abs_pdg >= 1)
                                            & (validation_abs_pdg <= 5)
                                        ),
                                        "gluon": validation_abs_pdg == 21,
                                        "other": (
                                            ~(
                                                (validation_abs_pdg >= 1)
                                                & (validation_abs_pdg <= 5)
                                            )
                                            & (validation_abs_pdg != 21)
                                        ),
                                    }
                                    # Carry the theory-varied gen weight (weights_gen picks
                                    # up ISR/FSR/q2/PDF via the syst modifier) onto the
                                    # systematic axis so the flavor split can be compared
                                    # Up/Down. Detector systematics leave the gen weight
                                    # nominal, matching the other gen-truth histograms.
                                    validation_weights = weights_gen[valid_validation_jet]
                                    for flavor_name, flavor_mask in validation_flavor_masks.items():
                                        fill_hist(
                                            self.hists,
                                            "mass_flavor_jet0_gen",
                                            dataset=dataset,
                                            mass=validation_gen_jet.mass[flavor_mask],
                                            parton_flavor=flavor_name,
                                            systematic=syst,
                                            weight=validation_weights[flavor_mask],
                                        )
                                        fill_hist(
                                            self.hists,
                                            "mpt_flavor_jet0_gen",
                                            dataset=dataset,
                                            mpt_gen=validation_mpt[flavor_mask],
                                            parton_flavor=flavor_name,
                                            systematic=syst,
                                            weight=validation_weights[flavor_mask],
                                        )

                                    


                                        
                                # if (not self._do_jk) or (not self._do_reweight):
                                #     fill_hist(self.hists, "ptjet_rhojet_u_gen", dataset = dataset, ptgen = ptgen,
                                #               mpt_gen = 2*np.log10(mgen/(ptgen*jetR)), weight = weights_gen, systematic = syst)
                                    
                                #     fill_hist(self.hists, "ptjet_rhojet_g_gen", dataset = dataset, ptgen = ptgen,
                                #               mpt_gen = 2*np.log10(mgen_g/(ptgen*jetR)), weight = weights_gen, systematic = syst)

                                # else:## Rho jk
                                    

                                
                                


                                
                                
                                gen_jet_both = gen_jet[sel_both]
                                reco_jet_both = reco_jet[sel_both]
                                groomed_gen_jet_both = groomed_gen_jet[sel_both]
        
                                ptgen_both = gen_jet_both.pt
                                ptgen_both = ptgen_both[~ak.is_none(ptgen_both)]
                                mgen_both = gen_jet_both.mass
                                mgen_both = mgen_both[~ak.is_none(mgen_both)]
                                mgen_both_g = groomed_gen_jet_both.mass

                                mpt_gen_both = 2*np.log10(mgen_both/(ptgen_both*jetR))
                                mpt_gen_both_g = 2*np.log10(mgen_both_g/(ptgen_both*jetR))
        
                                #self.logging.debug(f"No of GEN JET also passing RECO {len(mgen_both)}")
                                
                                ptreco_both = reco_jet_both.pt
                                ptreco_both = ptreco_both[~ak.is_none(ptreco_both)]
                                mreco_both = reco_jet_both.mass
                                mreco_both = mreco_both[~ak.is_none(mreco_both)]

                                ptreco_both_g = reco_jet_both.pt
                                ptreco_both_g = ptreco_both_g[~ak.is_none(ptreco_both_g)]
                                mreco_both_g = reco_jet_both.msoftdrop
                                mreco_both_g = mreco_both_g[~ak.is_none(mreco_both_g)]
                                weights_both_g = weights_both[~ak.is_none(mreco_both_g)]

                                mpt_reco_both = 2*np.log10(np.abs(mreco_both/(ptreco_both*jetR)))
                                mpt_reco_both_g = 2*np.log10(np.abs(mreco_both_g/(ptreco_both_g*jetR)))


                                
                                jetR = 0.8
                                weights_both_u = weights_both
                                weights_both_reweighted_g = weights_both_g

                                if self._do_reweight and self._is_data_prior_rho_reweight():
                                    herwig_weight_both_u, herwig_weight_both_g = self._get_herwig_reweights(
                                        ptgen_both,
                                        mgen_both,
                                        mgen_both_g,
                                        jetR,
                                    )
                                    weights_both_u = weights_both * herwig_weight_both_u
                                    weights_both_reweighted_g = weights_both_g * herwig_weight_both_g
                                elif self._do_reweight:
                                    herwig_weight_both_u, herwig_weight_both_g = self._get_herwig_reweights(
                                        ptreco_both,
                                        mreco_both,
                                        mreco_both_g,
                                        jetR,
                                    )
                                    herwig_weight_both_g = ak.nan_to_num(herwig_weight_both_g, 0)
                                    self.logging.debug(f"herwig WEights both g? {herwig_weight_both_g}" )
                                    weights_both_u = weights_both * herwig_weight_both_u
                                    weights_both_reweighted_g = weights_both_g * herwig_weight_both_g

                                fill_hist(
                                    self.hists,
                                    "response_matrix_u",
                                    dataset=dataset,
                                    channel=channel,
                                    ptreco=ptreco_both,
                                    mreco=mreco_both,
                                    ptgen=ptgen_both,
                                    mgen=mgen_both,
                                    systematic=syst,
                                    weight=weights_both_u if self._is_mass_reweight() else weights_both,
                                    **mass_jk_fill,
                                )

                                fill_hist(
                                    self.hists,
                                    "response_matrix_g",
                                    dataset=dataset,
                                    channel=channel,
                                    ptreco=ptreco_both_g,
                                    mreco=mreco_both_g,
                                    ptgen=ptgen_both,
                                    mgen=mgen_both_g,
                                    systematic=syst,
                                    weight=weights_both_reweighted_g if self._is_mass_reweight() else weights_both,
                                    **mass_jk_fill,
                                )

                                # fill_hist(self.hists, 'm_u_jet_reco_over_gen', dataset = dataset, ptgen = ptgen_both, mgen = mgen_both, 
                                #           frac = (mreco_both-mgen_both) /mgen_both, weight = weights_both)

                                # fill_hist(self.hists, 'm_g_jet_reco_over_gen', dataset = dataset, ptgen = ptgen_both, mgen = mgen_both_g, 
                                #           frac = (mreco_both_g-mgen_both_g) /mgen_both_g, weight = weights_both)
                                

                                # import matplotlib.pyplot as plt
                                # plt.hist(2*np.log10(mreco_both/(ptreco_both*jetR)))
                                # plt.show()
                                


                                
                                if self._do_jk:### Rho jk
                                    fill_hist(self.hists, "response_matrix_rho_u", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = 2*np.log10(mreco_both/(ptreco_both*jetR)), ptgen = ptgen_both,
                                              mpt_gen = 2*np.log10(mgen_both/(ptgen_both*jetR)), 
                                              weight = weights_both, jk = jk_index, systematic = syst)
                                        
                                    fill_hist(self.hists, "response_matrix_rho_g", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = 2*np.log10(mreco_both_g/(ptreco_both*jetR)), ptgen = ptgen_both,
                                              mpt_gen = 2*np.log10(mgen_both_g/(ptgen_both*jetR)), 
                                              weight = weights_both, jk = jk_index, systematic = syst)
                                elif self._do_reweight and self._is_rho_reweight():
                                    # self.logging.debug(f"WEights both g? {weights_both_g}" )
                                    # self.logging.debug(f"How many nan in weights_both_g? {ak.sum(ak.is_none(weights_both_g))}")
                                    # self.logging.debug(f"How many nan in mpt_reco? {ak.sum(ak.is_none(mpt_reco_both))}")
                                    # self.logging.debug(f"How many nan in mpt_gen? {ak.sum(ak.is_none(mpt_gen_both))}")
                                    # self.logging.debug(f"how many nan in ptreco_both? {}")
                                    # self.logging.debug(f"how many nan in ptgen_both? {}")

                                    #self.logging.debug(f"WEights both g? {weights_both_g}")
                                    # self.logging.debug(
                                    #     f"How many NaN in weights_both_g? {ak.sum(np.isnan(herwig_weight_both_g))}"
                                    # )
                                    # self.logging.debug(
                                    #     f"How many NaN in weights_both_g? {ak.sum(np.isnan(weights_both_g))}"
                                    # )
                                    
                                    # self.logging.debug(
                                    #     f"How many NaN in mpt_reco_both_g? {ak.sum(np.isnan(mpt_reco_both_g))}"
                                    # )
                                    
                                    # self.logging.debug(
                                    #     f"How many NaN in mpt_gen_both? {ak.sum(np.isnan(mpt_gen_both_g))}"
                                    # )
                                    
                                    # self.logging.debug(
                                    #     f"How many NaN in ptreco_both? {ak.sum(np.isnan(ptreco_both))}"
                                    # )
                                    
                                    # self.logging.debug(
                                    #     f"How many NaN in ptgen_both? {ak.sum(np.isnan(ptgen_both))}"
                                    # )
                                    fill_hist(self.hists, "response_matrix_rho_u", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = mpt_reco_both, ptgen = ptgen_both,
                                              mpt_gen = mpt_gen_both, systematic = syst, weight = weights_both_u)
                                    
                                    fill_hist(self.hists, "response_matrix_rho_g", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = mpt_reco_both_g, ptgen = ptgen_both,
                                              mpt_gen = mpt_gen_both_g, systematic = syst, weight = weights_both_reweighted_g)
                                
                                else:
                                    fill_hist(self.hists, "response_matrix_rho_u", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = 2*np.log10(mreco_both/(ptreco_both*jetR)), ptgen = ptgen_both,
                                              mpt_gen = 2*np.log10(mgen_both/(ptgen_both*jetR)), systematic = syst, weight = weights_both)
                                    
                                    fill_hist(self.hists, "response_matrix_rho_g", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = 2*np.log10(mreco_both_g/(ptreco_both*jetR)), ptgen = ptgen_both,
                                              mpt_gen = 2*np.log10(mgen_both_g/(ptgen_both*jetR)), systematic = syst, weight = weights_both)
                                
                                
                                fill_hist(self.hists, "jk_response_matrix_u",dataset = dataset, ptreco = ptreco_both, mreco = mreco_both, ptgen = ptgen_both, mgen = mgen_both, jk = jk_index, weight = weights_both)
                                fill_hist(self.hists, "jk_response_matrix_g", dataset = dataset, ptreco = ptreco_both_g, mreco = mreco_both_g, ptgen = ptgen_both, mgen = mgen_both_g, jk = jk_index, weight = weights_both)

                
                            reco_jet_meas = reco_jet[sel_reco]
                            ptreco = reco_jet_meas.pt
                            ptreco_g = reco_jet_meas.pt
                            mreco = reco_jet_meas.mass
                            mreco_g = reco_jet_meas.msoftdrop
                            #mreco_g2 = reco_jet_meas.msoftdrop_orig
                            reco_data_prior_weight_u = None
                            reco_data_prior_weight_g = None
                            if self._do_gen and self._is_data_prior_rho_reweight():
                                gen_jet_for_reco = gen_jet[sel_reco]
                                groomed_gen_jet_for_reco = groomed_gen_jet[sel_reco]
                                matched_to_selected_gen = ak.to_numpy(
                                    ak.fill_none(allsel_gen[sel_reco], False)
                                )
                                reco_data_prior_weight_u = np.ones(len(matched_to_selected_gen))
                                reco_data_prior_weight_g = np.ones(len(matched_to_selected_gen))
                                if np.any(matched_to_selected_gen):
                                    prior_u, prior_g = self._get_herwig_reweights(
                                        gen_jet_for_reco[matched_to_selected_gen].pt,
                                        gen_jet_for_reco[matched_to_selected_gen].mass,
                                        groomed_gen_jet_for_reco[matched_to_selected_gen].mass,
                                        jetR,
                                    )
                                    reco_data_prior_weight_u[matched_to_selected_gen] = prior_u
                                    reco_data_prior_weight_g[matched_to_selected_gen] = prior_g

                            self._fill_groomed_over_ungroomed_reco(
                                dataset=dataset,
                                channel=channel,
                                ptreco=ptreco,
                                ungroomed_mass=mreco,
                                groomed_mass=mreco_g,
                                weight=weights_reco,
                                systematic=syst,
                                **mass_jk_fill,
                            )
                            if syst == "nominal":
                                self._fill_groomed_over_ungroomed_reco(
                                    dataset=dataset,
                                    channel=channel,
                                    ptreco=ptreco,
                                    ungroomed_mass=reco_jet_meas.mass_raw_diagnostic,
                                    groomed_mass=reco_jet_meas.msoftdrop_raw_diagnostic,
                                    weight=weights_reco,
                                    systematic=syst,
                                    mass_ratio_name="m_g_over_m_u_raw_reco",
                                    mass_2d_name="m_g_vs_m_u_raw_reco",
                                    **mass_jk_fill,
                                )
                                self._fill_reco_jet_ntuple(
                                    dataset=dataset,
                                    channel=channel,
                                    systematic=syst,
                                    events=events_j[sel_reco],
                                    reco_jet=reco_jet_meas,
                                    weight=weights_reco,
                                    gen_jet=gen_jet[sel_reco] if self._do_gen else None,
                                    groomed_gen_jet=groomed_gen_jet[sel_reco] if self._do_gen else None,
                                    has_gen_match=is_matched_reco[sel_reco] if self._do_gen else None,
                                    passes_gen_selection=allsel_gen[sel_reco] if self._do_gen else None,
                                    passes_both=(sel_reco & allsel_gen)[sel_reco] if self._do_gen else None,
                                )

                            valid_reco_u = ~ak.is_none(mreco)
                            valid_reco_g = ~ak.is_none(mreco_g)
                            if reco_data_prior_weight_u is not None:
                                reco_data_prior_weight_u = reco_data_prior_weight_u[
                                    ak.to_numpy(valid_reco_u)
                                ]
                                reco_data_prior_weight_g = reco_data_prior_weight_g[
                                    ak.to_numpy(valid_reco_g)
                                ]

                            ptreco = ptreco[valid_reco_u]
                            mreco = mreco[valid_reco_u]
                            mreco_g = mreco_g[valid_reco_g]
                            #mreco_g2 = mreco_g2[~ak.is_none(mreco_g2)]
                            ptreco_g = ptreco_g[valid_reco_g]
                            weights_reco_g = weights_reco[valid_reco_g]
                            weights_reco = weights_reco[valid_reco_u]

                            mpt_reco = 2*np.log10(mreco/(ptreco*jetR))
                            mpt_reco_g = 2*np.log10(mreco_g/(ptreco*jetR))

                            events_j_meas = events_j[sel_reco]
                            z_reco_meas = z_reco[sel_reco]
                            print("Fatjet y ", reco_jet_meas.rapidity)
                            print("Fatjet eta ", reco_jet_meas.eta)
                            # print(
                            #     "energy:", reco_jet_meas.p4.energy,
                            #     "\npz:", reco_jet_meas.p4.pz,
                            #     "\np4.energy:", reco_jet_meas.p4.energy,
                            #     "\np4.pz:", reco_jet_meas.p4.pz
                            # )
                            if channel == "mm":
                                pt_mu0  = events_j_meas.Muon[:, 0].pt
                                eta_mu0 = events_j_meas.Muon[:, 0].eta
                                phi_mu0 = events_j_meas.Muon[:, 0].phi
                            
                                pt_mu1  = events_j_meas.Muon[:, 1].pt
                                eta_mu1 = events_j_meas.Muon[:, 1].eta
                                phi_mu1 = events_j_meas.Muon[:, 1].phi
                                
                                y_mu0 = events_j_meas.Muon[:,0].rapidity
                                y_mu1 = events_j_meas.Muon[:,1].rapidity
                                
                                                         
                                q0 = events_j_meas.Muon[:, 0].charge
                                q1 = events_j_meas.Muon[:, 1].charge
                            
                                pt_mupos  = ak.where(q0 > 0, pt_mu0, pt_mu1)
                                pt_muneg  = ak.where(q0 < 0, pt_mu0, pt_mu1)
                            
                                eta_mupos = ak.where(q0 > 0, eta_mu0, eta_mu1)
                                eta_muneg = ak.where(q0 < 0, eta_mu0, eta_mu1)
                            
                                phi_mupos = ak.where(q0 > 0, phi_mu0, phi_mu1)
                                phi_muneg = ak.where(q0 < 0, phi_mu0, phi_mu1)

                                y_mupos = ak.where(q0 > 0, y_mu0, y_mu1)
                                y_muneg = ak.where(q0 < 0, y_mu0, y_mu1)   
                            
                            
                            if channel == "ee":
                                pt_el0  = events_j_meas.Electron[:, 0].pt
                                eta_el0 = events_j_meas.Electron[:, 0].eta
                                phi_el0 = events_j_meas.Electron[:, 0].phi
                            
                                pt_el1  = events_j_meas.Electron[:, 1].pt
                                eta_el1 = events_j_meas.Electron[:, 1].eta
                                phi_el1 = events_j_meas.Electron[:, 1].phi
                                y_el0 = events_j_meas.Electron[:,0].rapidity
                                y_el1 = events_j_meas.Electron[:,1].rapidity
                                
                                q0 = events_j_meas.Electron[:, 0].charge
                                q1 = events_j_meas.Electron[:, 1].charge

                                
                                y_elpos = ak.where(q0 > 0, y_el0, y_el1)
                                y_elneg = ak.where(q0 < 0, y_el0, y_el1)
                                pt_elpos  = ak.where(q0 > 0, pt_el0, pt_el1)
                                pt_elneg  = ak.where(q0 < 0, pt_el0, pt_el1)
                            
                                eta_elpos = ak.where(q0 > 0, eta_el0, eta_el1)
                                eta_elneg = ak.where(q0 < 0, eta_el0, eta_el1)
                            
                                phi_elpos = ak.where(q0 > 0, phi_el0, phi_el1)
                                phi_elneg = ak.where(q0 < 0, phi_el0, phi_el1)

                                y_elpos = ak.where(q0 > 0, y_el0, y_el1)
                                y_elneg = ak.where(q0 < 0, y_el0, y_el1)


                            
                            pt_Z = z_reco_meas.pt
                            eta_Z = z_reco_meas.eta
                            phi_Z = z_reco_meas.phi
                            mass_Z = z_reco_meas.mass
                            nJet = ak.num(events_j_meas.FatJet, axis = 1)
                            
                            dr = z_jet_dr_reco[sel_reco]
                            dphi = z_jet_dphi_reco[sel_reco]
                            ptasym = z_pt_asym_reco[sel_reco]
                            
                            


                            
                            
                    
                            #self.logging.debug(f"No of RECO JET {len(mreco)}")
                            if syst == "nominal":
                                self.logging.debug(f"Len of ptreco {len(ptreco)} mreco {len(mreco)} syst {syst} channel {channel} dataset {dataset}")
                                self.logging.debug(f"ptreco sample {ptreco[:10]}")
                                self.logging.debug(f"mreco sample {mreco[:10]}")
                                self.logging.debug(f"mreco_g sample {mreco_g[:10]}")
                            weights_reco_u = weights_reco
                            weights_reco_reweighted_g = weights_reco_g

                            if self._do_reweight and self._is_data_prior_rho_reweight():
                                weights_reco_u = weights_reco * reco_data_prior_weight_u
                                weights_reco_reweighted_g = weights_reco_g * reco_data_prior_weight_g
                            elif self._do_reweight:
                                herwig_weight_reco_u, herwig_weight_reco_g = self._get_herwig_reweights(
                                    ptreco,
                                    mreco,
                                    mreco_g,
                                    jetR,
                                )
                                weights_reco_u = weights_reco * herwig_weight_reco_u
                                weights_reco_reweighted_g = weights_reco_g * herwig_weight_reco_g

                            fill_hist(
                                self.hists,
                                "ptjet_mjet_u_reco",
                                dataset=dataset,
                                channel=channel,
                                ptreco=ptreco,
                                mreco=mreco,
                                systematic=syst,
                                weight=weights_reco_u if self._is_mass_reweight() else weights_reco,
                                **mass_jk_fill,
                            )
                            fill_hist(
                                self.hists,
                                "ptjet_mjet_g_reco",
                                dataset=dataset,
                                channel=channel,
                                ptreco=ptreco_g,
                                mreco=mreco_g,
                                systematic=syst,
                                weight=weights_reco_reweighted_g if self._is_mass_reweight() else weights_reco_g,
                                **mass_jk_fill,
                            )


                                


                            if self._do_jk:## Rho jk
                                fill_hist(self.hists, "ptjet_rhojet_u_reco", dataset = dataset, ptreco = ptreco,
                                      mpt_reco = 2*np.log10(mreco/(ptreco*jetR)), jk = jk_index, weight = weights_reco, systematic = syst)
                            
                                fill_hist(self.hists, "ptjet_rhojet_g_reco", dataset = dataset, ptreco = ptreco_g,
                                          mpt_reco = 2*np.log10(mreco_g/(ptreco*jetR)), jk = jk_index, weight = weights_reco_g, systematic = syst)
                            elif self._do_reweight and self._is_rho_reweight():
                                fill_hist(self.hists, "ptjet_rhojet_u_reco", dataset = dataset, ptreco = ptreco,
                                      mpt_reco = 2*np.log10(mreco/(ptreco*jetR)), systematic = syst, weight = weights_reco_u)
                            
                                fill_hist(self.hists, "ptjet_rhojet_g_reco", dataset = dataset, ptreco = ptreco_g,
                                          mpt_reco = 2*np.log10(mreco_g/(ptreco*jetR)), systematic = syst, weight = weights_reco_reweighted_g)
                            else: # regular cases
                                
                                ### Required input distributions
                                fill_hist(self.hists, "ptjet_rhojet_u_reco", dataset = dataset, ptreco = ptreco,
                                      mpt_reco = 2*np.log10(mreco/(ptreco*jetR)), systematic = syst, weight = weights_reco)
                            
                                fill_hist(self.hists, "ptjet_rhojet_g_reco", dataset = dataset, ptreco = ptreco_g,
                                          mpt_reco = 2*np.log10(mreco_g/(ptreco*jetR)), systematic = syst, weight = weights_reco_g)

                                ### Required for data/MC validation
                                # weights_gen_reco = weights.partial_weight('genWeight')[sel_reco]
                                # print(weights_gen_reco)
                                ## Filling validation plots ##
                                if channel == "mm":
                                    fill_hist(self.hists, "pt_mupos",  dataset=dataset, pt=pt_mupos,   systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "eta_mupos", dataset=dataset, eta=eta_mupos, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "phi_mupos", dataset=dataset, phi=phi_mupos, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "y_mupos",   dataset=dataset, y=y_mupos,     systematic=syst, weight=weights_reco)
                                
                                    fill_hist(self.hists, "pt_muneg",  dataset=dataset, pt=pt_muneg,   systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "eta_muneg", dataset=dataset, eta=eta_muneg, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "phi_muneg", dataset=dataset, phi=phi_muneg, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "y_muneg",   dataset=dataset, y=y_muneg,     systematic=syst, weight=weights_reco)

                                    del (
                                        pt_mu0, pt_mu1,
                                        eta_mu0, eta_mu1,
                                        phi_mu0, phi_mu1,
                                        y_mu0, y_mu1,
                                        q0, q1,
                                        pt_mupos, pt_muneg,
                                        eta_mupos, eta_muneg,
                                        phi_mupos, phi_muneg,
                                        y_mupos, y_muneg
                                    )
                                
                                
                                # electrons (pos/neg)
                                if channel == "ee":
                                    fill_hist(self.hists, "pt_elpos",  dataset=dataset, pt=pt_elpos,   systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "eta_elpos", dataset=dataset, eta=eta_elpos, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "phi_elpos", dataset=dataset, phi=phi_elpos, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "y_elpos",   dataset=dataset, y=y_elpos,     systematic=syst, weight=weights_reco)
                                
                                    fill_hist(self.hists, "pt_elneg",  dataset=dataset, pt=pt_elneg,   systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "eta_elneg", dataset=dataset, eta=eta_elneg, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "phi_elneg", dataset=dataset, phi=phi_elneg, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "y_elneg",   dataset=dataset, y=y_elneg,     systematic=syst, weight=weights_reco)

                                    del (
                                        pt_el0, pt_el1,
                                        eta_el0, eta_el1,
                                        phi_el0, phi_el1,
                                        y_el0, y_el1,
                                        q0, q1,
                                        pt_elpos, pt_elneg,
                                        eta_elpos, eta_elneg,
                                        phi_elpos, phi_elneg,
                                        y_elpos, y_elneg
                                    )
                                
                                ## nJets
                                fill_hist(self.hists, "nJets", dataset = dataset, n = nJet, systematic = syst, weight = weights_reco )
                                
                                ## Z object
                                fill_hist(self.hists, "pt_Z", dataset = dataset, pt = pt_Z, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "eta_Z", dataset = dataset, eta = eta_Z, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "phi_Z", dataset = dataset, phi = phi_Z, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "mass_Z", dataset = dataset, mass = mass_Z, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "y_Z", dataset = dataset, y = getRapidity(z_reco_meas), systematic = syst, weight = weights_reco)

                                ## MET (PF Type-1) -- HEM implementation cross-check (2018)
                                fill_hist(self.hists, "met_pt",  dataset = dataset, pt  = events_j_meas.MET.pt,  systematic = syst, weight = weights_reco)
                                fill_hist(self.hists, "met_phi", dataset = dataset, phi = events_j_meas.MET.phi, systematic = syst, weight = weights_reco)
                                ## joint MET x jet-pt and MET x jet-mass (per-event correlation)
                                fill_hist(self.hists, "met_ptjet_reco", dataset = dataset, pt = events_j_meas.MET.pt, ptreco = reco_jet_meas.pt,   systematic = syst, weight = weights_reco)
                                fill_hist(self.hists, "mass_met_reco",  dataset = dataset, mass = reco_jet_meas.mass, pt = events_j_meas.MET.pt,   systematic = syst, weight = weights_reco)
                                met_pt_xy, met_phi_xy = METXYCorr(
                                    events_j_meas.MET.pt, events_j_meas.MET.phi, events_j_meas.PV.npvs,
                                    IOV, isMC = self._do_gen, run = events_j_meas.run)
                                fill_hist(self.hists, "met_pt_xy",  dataset = dataset, pt  = met_pt_xy,  systematic = syst, weight = weights_reco)
                                fill_hist(self.hists, "met_phi_xy", dataset = dataset, phi = met_phi_xy, systematic = syst, weight = weights_reco)

                                ## AK4-jet HEM occupancy cross-check (per jet; nominal only,
                                ## hists carry no systematic axis so fill once per event)
                                if syst == "nominal":
                                    ak4 = events_j_meas.Jet
                                    ak4 = ak4[(ak4.pt > 30) & (ak4.jetId >= 2)]
                                    ak4_w = ak.broadcast_arrays(weights_reco, ak4.pt)[0]
                                    fill_hist(self.hists, "ak4_eta_phi_reco", dataset = dataset,
                                              eta = ak.flatten(ak4.eta), phi = ak.flatten(ak4.phi),
                                              weight = ak.flatten(ak4_w))
                                    ak4_hem = ak4[(ak4.eta > -3.0) & (ak4.eta < -1.3)]
                                    ak4_hem_w = ak.broadcast_arrays(weights_reco, ak4_hem.pt)[0]
                                    fill_hist(self.hists, "ak4_phi_hemeta", dataset = dataset,
                                              phi = ak.flatten(ak4_hem.phi), weight = ak.flatten(ak4_hem_w))

                                    ## AK4-in-HEM event veto: reuse HEMVeto on the AK4 collection
                                    ## (data -> pass/fail mask; 2018 MC -> flat lumi weight). Only
                                    ## meaningful for 2018; elsewhere it's a no-op (weight 1).
                                    if IOV == "2018":
                                        ak4_veto = HEMVeto(events_j_meas.Jet, events_j_meas.run, isMC = self._do_gen)
                                        ak4_veto_w = ak.to_numpy(ak4_veto).astype(float)
                                    else:
                                        ak4_veto_w = np.ones(len(weights_reco))
                                    w_ak4veto = weights_reco * ak4_veto_w
                                    fill_hist(self.hists, "met_pt_ak4veto",    dataset = dataset, pt   = events_j_meas.MET.pt,  systematic = syst, weight = w_ak4veto)
                                    fill_hist(self.hists, "met_phi_ak4veto",   dataset = dataset, phi  = events_j_meas.MET.phi, systematic = syst, weight = w_ak4veto)
                                    fill_hist(self.hists, "mass_jet0_ak4veto", dataset = dataset, mass = reco_jet_meas.mass,     systematic = syst, weight = w_ak4veto)


                                ## Jet
                                fill_hist(self.hists, "pt_jet0", dataset = dataset, pt = ptreco, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "eta_jet0", dataset = dataset, eta = reco_jet_meas.eta, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "phi_jet0", dataset = dataset, phi = reco_jet_meas.phi, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "mass_jet0", dataset = dataset, mass = reco_jet_meas.mass, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "y_jet0", dataset = dataset, y = reco_jet_meas.rapidity, systematic = syst, weight = weights_reco)
                                fill_hist(self.hists, "eta_phi_jet_reco", dataset = dataset, eta = reco_jet_meas.eta, phi = reco_jet_meas.phi, weight = weights_reco)

                                ## Combo
                                fill_hist(self.hists, "ptasym", dataset = dataset, frac = ptasym, systematic = syst, weight = weights_reco)
                                fill_hist(self.hists, "dr", dataset = dataset, dr = dr, systematic = syst, weight = weights_reco)
                                fill_hist(self.hists, "dphi", dataset = dataset, dphi = dphi, systematic = syst, weight = weights_reco)
                                
                            

                            # fill_hist(self.hists, "ptjet_rhojet_g_reco2", dataset = dataset, ptreco = ptreco_g,
                            #           mpt_reco = 2*np.log10(mreco_g2/(ptreco*jetR)), systematic = syst, weight = weights_reco_g)
                            
                            fill_hist( self.hists, "jk_ptjet_mjet_u_reco", dataset = dataset,ptreco = ptreco, mreco = mreco, systematic = syst, weight = weights_reco, jk = jk_index)
                            fill_hist( self.hists, "jk_ptjet_mjet_g_reco", dataset = dataset,ptreco = ptreco_g, mreco = mreco_g, systematic = syst, weight = weights_reco_g, jk = jk_index)
                            
                            if not self._do_gen:
                                break # Break on nominal when running over data




                    else: # jet syst is not nominal
                        print("ARE WE ENTERING THIS SECTION OR NO?")
                        if self._do_gen:
                            weights_gen =  weights.partial_weight(include=['genWeight'])[sel_gen]
                            weights_both = weights.weight()[sel_both]
                        weights_reco = weights.weight()[sel_reco]
                        mass_jk_fill = {"jk": jk_index} if self._mode == "mass_jk" else {}
                        

                        reco_jet_meas = reco_jet[sel_reco]
                        ptreco = reco_jet_meas.pt
                        ptreco_g = reco_jet_meas.pt
                        mreco = reco_jet_meas.mass
                        mreco_g = reco_jet_meas.msoftdrop

                        self._fill_groomed_over_ungroomed_reco(
                            dataset=dataset,
                            channel=channel,
                            ptreco=ptreco,
                            ungroomed_mass=mreco,
                            groomed_mass=mreco_g,
                            weight=weights_reco,
                            systematic=jet_syst,
                            **mass_jk_fill,
                        )

                        ptreco = ptreco[~ak.is_none(mreco)]
                        mreco = mreco[~ak.is_none(mreco)]
                        mreco_g = mreco_g[~ak.is_none(mreco_g)]
                        ptreco_g = ptreco_g[~ak.is_none(mreco_g)]
                        weights_reco_g = weights_reco[~ak.is_none(mreco_g)]
                        weights_reco = weights_reco[~ak.is_none(mreco)]


                        events_j_meas = events_j[sel_reco]
                        z_reco_meas = z_reco[sel_reco]
                        if channel == "mm":
                            pt_mu0  = events_j_meas.Muon[:, 0].pt
                            eta_mu0 = events_j_meas.Muon[:, 0].eta
                            phi_mu0 = events_j_meas.Muon[:, 0].phi
                        
                            pt_mu1  = events_j_meas.Muon[:, 1].pt
                            eta_mu1 = events_j_meas.Muon[:, 1].eta
                            phi_mu1 = events_j_meas.Muon[:, 1].phi
                            
                            y_mu0 = events_j_meas.Muon[:,0].rapidity
                            y_mu1 = events_j_meas.Muon[:,1].rapidity
                            
                                                     
                            q0 = events_j_meas.Muon[:, 0].charge
                            q1 = events_j_meas.Muon[:, 1].charge
                        
                            pt_mupos  = ak.where(q0 > 0, pt_mu0, pt_mu1)
                            pt_muneg  = ak.where(q0 < 0, pt_mu0, pt_mu1)
                        
                            eta_mupos = ak.where(q0 > 0, eta_mu0, eta_mu1)
                            eta_muneg = ak.where(q0 < 0, eta_mu0, eta_mu1)
                        
                            phi_mupos = ak.where(q0 > 0, phi_mu0, phi_mu1)
                            phi_muneg = ak.where(q0 < 0, phi_mu0, phi_mu1)

                            y_mupos = ak.where(q0 > 0, y_mu0, y_mu1)
                            y_muneg = ak.where(q0 < 0, y_mu0, y_mu1)   
                        
                        
                        if channel == "ee":
                            pt_el0  = events_j_meas.Electron[:, 0].pt
                            eta_el0 = events_j_meas.Electron[:, 0].eta
                            phi_el0 = events_j_meas.Electron[:, 0].phi
                        
                            pt_el1  = events_j_meas.Electron[:, 1].pt
                            eta_el1 = events_j_meas.Electron[:, 1].eta
                            phi_el1 = events_j_meas.Electron[:, 1].phi
                            y_el0 = events_j_meas.Electron[:,0].rapidity
                            y_el1 = events_j_meas.Electron[:,1].rapidity
                            q0 = events_j_meas.Electron[:, 0].charge
                            q1 = events_j_meas.Electron[:, 1].charge
                            
                            y_elpos = ak.where(q0 > 0, y_el0, y_el1)
                            y_elneg = ak.where(q0 < 0, y_el0, y_el1)
                        

                        
                            pt_elpos  = ak.where(q0 > 0, pt_el0, pt_el1)
                            pt_elneg  = ak.where(q0 < 0, pt_el0, pt_el1)
                        
                            eta_elpos = ak.where(q0 > 0, eta_el0, eta_el1)
                            eta_elneg = ak.where(q0 < 0, eta_el0, eta_el1)
                        
                            phi_elpos = ak.where(q0 > 0, phi_el0, phi_el1)
                            phi_elneg = ak.where(q0 < 0, phi_el0, phi_el1)

                            y_elpos = ak.where(q0 > 0, y_el0, y_el1)
                            y_elneg = ak.where(q0 < 0, y_el0, y_el1)


                        
                        pt_Z = z_reco_meas.pt
                        eta_Z = z_reco_meas.eta
                        phi_Z = z_reco_meas.phi
                        mass_Z = z_reco_meas.mass
                        nJet = ak.num(events_j_meas.FatJet, axis = 1)
                        
                        dr = z_jet_dr_reco[sel_reco]
                        dphi = z_jet_dphi_reco[sel_reco]
                        ptasym = z_pt_asym_reco[sel_reco]

                        #self.logging.debug(f"No of RECO JET {len(mreco)}")
                            
                        fill_hist(self.hists, "ptjet_rhojet_u_reco", dataset = dataset, ptreco = ptreco,
                                  mpt_reco = 2*np.log10(mreco/(ptreco*jetR)), systematic = jet_syst, weight = weights_reco)
                        
                        fill_hist(self.hists, "ptjet_rhojet_g_reco", dataset = dataset, ptreco = ptreco_g,
                                  mpt_reco = 2*np.log10(mreco_g/(ptreco*jetR)), systematic = jet_syst, weight = weights_reco_g)
                        print("HOW IS DOJK TRUE?")
                        print(self._do_jk)
                        ## Filling Rho
                        if not self._do_jk:
                            print("ARE WE FILLING THIS OR NO?")
                            fill_hist(self.hists, "ptjet_mjet_u_reco", dataset = dataset, channel = channel, ptreco = ptreco, mreco = mreco, systematic = jet_syst, weight = weights_reco)
                            fill_hist(self.hists, "ptjet_mjet_g_reco", dataset = dataset, channel = channel, ptreco = ptreco_g, mreco = mreco_g, systematic = jet_syst, weight = weights_reco_g)

                            ### Required for data/MC validation
                            if channel == "mm":
                                fill_hist(self.hists, "pt_mupos",  dataset=dataset, pt=pt_mupos,   systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "eta_mupos", dataset=dataset, eta=eta_mupos, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "phi_mupos", dataset=dataset, phi=phi_mupos, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "y_mupos",   dataset=dataset, y=y_mupos,     systematic=jet_syst, weight=weights_reco)
                            
                                fill_hist(self.hists, "pt_muneg",  dataset=dataset, pt=pt_muneg,   systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "eta_muneg", dataset=dataset, eta=eta_muneg, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "phi_muneg", dataset=dataset, phi=phi_muneg, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "y_muneg",   dataset=dataset, y=y_muneg,     systematic=jet_syst, weight=weights_reco)

                                del (
                                    pt_mu0, pt_mu1,
                                    eta_mu0, eta_mu1,
                                    phi_mu0, phi_mu1,
                                    y_mu0, y_mu1,
                                    q0, q1,
                                    pt_mupos, pt_muneg,
                                    eta_mupos, eta_muneg,
                                    phi_mupos, phi_muneg,
                                    y_mupos, y_muneg
                                )
                            
                            
                            # electrons (pos/neg)
                            if channel == "ee":
                                fill_hist(self.hists, "pt_elpos",  dataset=dataset, pt=pt_elpos,   systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "eta_elpos", dataset=dataset, eta=eta_elpos, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "phi_elpos", dataset=dataset, phi=phi_elpos, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "y_elpos",   dataset=dataset, y=y_elpos,     systematic=jet_syst, weight=weights_reco)
                            
                                fill_hist(self.hists, "pt_elneg",  dataset=dataset, pt=pt_elneg,   systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "eta_elneg", dataset=dataset, eta=eta_elneg, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "phi_elneg", dataset=dataset, phi=phi_elneg, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "y_elneg",   dataset=dataset, y=y_elneg,     systematic=jet_syst, weight=weights_reco)

                                del (
                                    pt_el0, pt_el1,
                                    eta_el0, eta_el1,
                                    phi_el0, phi_el1,
                                    y_el0, y_el1,
                                    q0, q1,
                                    pt_elpos, pt_elneg,
                                    eta_elpos, eta_elneg,
                                    phi_elpos, phi_elneg,
                                    y_elpos, y_elneg
                                )
                            
                            ## nJets
                            fill_hist(self.hists, "nJets", dataset = dataset, n = nJet, systematic = jet_syst, weight = weights_reco )
                            
                            ## Z object
                            fill_hist(self.hists, "pt_Z", dataset = dataset, pt = pt_Z, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "eta_Z", dataset = dataset, eta = eta_Z, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "phi_Z", dataset = dataset, phi = phi_Z, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "mass_Z", dataset = dataset, mass = mass_Z, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "y_Z", dataset = dataset, y = getRapidity(z_reco_meas), systematic = jet_syst, weight = weights_reco)
                            ## MET (PF Type-1) -- HEM implementation cross-check (2018)
                            fill_hist(self.hists, "met_pt",  dataset = dataset, pt  = events_j_meas.MET.pt,  systematic = jet_syst, weight = weights_reco)
                            fill_hist(self.hists, "met_phi", dataset = dataset, phi = events_j_meas.MET.phi, systematic = jet_syst, weight = weights_reco)
                            ## joint MET x jet-pt and MET x jet-mass (per-event correlation)
                            fill_hist(self.hists, "met_ptjet_reco", dataset = dataset, pt = events_j_meas.MET.pt, ptreco = reco_jet_meas.pt,   systematic = jet_syst, weight = weights_reco)
                            fill_hist(self.hists, "mass_met_reco",  dataset = dataset, mass = reco_jet_meas.mass, pt = events_j_meas.MET.pt,   systematic = jet_syst, weight = weights_reco)
                            met_pt_xy, met_phi_xy = METXYCorr(
                                events_j_meas.MET.pt, events_j_meas.MET.phi, events_j_meas.PV.npvs,
                                IOV, isMC = self._do_gen, run = events_j_meas.run)
                            fill_hist(self.hists, "met_pt_xy",  dataset = dataset, pt  = met_pt_xy,  systematic = jet_syst, weight = weights_reco)
                            fill_hist(self.hists, "met_phi_xy", dataset = dataset, phi = met_phi_xy, systematic = jet_syst, weight = weights_reco)
                            ## Jet
                            fill_hist(self.hists, "pt_jet0", dataset = dataset, pt = ptreco, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "eta_jet0", dataset = dataset, eta = reco_jet_meas.eta, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "phi_jet0", dataset = dataset, phi = reco_jet_meas.phi, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "mass_jet0", dataset = dataset, mass = reco_jet_meas.mass, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "y_jet0", dataset = dataset, y = reco_jet_meas.rapidity, systematic = jet_syst, weight = weights_reco)

                            ## Combo
                            fill_hist(self.hists, "ptasym", dataset = dataset, frac = ptasym, systematic = jet_syst, weight = weights_reco)
                            fill_hist(self.hists, "dr", dataset = dataset, dr = dr, systematic = jet_syst, weight = weights_reco)
                            fill_hist(self.hists, "dphi", dataset = dataset, dphi = dphi, systematic = jet_syst, weight = weights_reco)

                        else:
                            fill_hist(self.hists, "ptjet_mjet_u_reco", dataset = dataset, channel = channel, ptreco = ptreco, mreco = mreco, systematic = jet_syst, weight = weights_reco, **mass_jk_fill)
                            fill_hist(self.hists, "ptjet_mjet_g_reco", dataset = dataset, channel = channel, ptreco = ptreco_g, mreco = mreco_g, systematic = jet_syst, weight = weights_reco_g, **mass_jk_fill)
                            

                        if self._do_gen:
                            gen_jet_both = gen_jet[sel_both]
                            reco_jet_both = reco_jet[sel_both]
                            groomed_gen_jet_both = groomed_gen_jet[sel_both]

                            ptgen_both = gen_jet_both.pt
                            ptgen_both = ptgen_both[~ak.is_none(ptgen_both)]
                            mgen_both = gen_jet_both.mass
                            mgen_both = mgen_both[~ak.is_none(mgen_both)]
                            mgen_both_g = groomed_gen_jet_both.mass

                            #self.logging.debug(f"No of GEN JET also passing RECO {len(mgen_both)}")
                            
                            ptreco_both = reco_jet_both.pt
                            ptreco_both = ptreco_both[~ak.is_none(ptreco_both)]
                            mreco_both = reco_jet_both.mass
                            mreco_both = mreco_both[~ak.is_none(mreco_both)]
                            weights_both = weights_both[~ak.is_none(mreco_both)]
                            
                            fill_hist(self.hists, "response_matrix_u", dataset = dataset, channel = channel, ptreco = ptreco_both, mreco = mreco_both, ptgen = ptgen_both, mgen = mgen_both, systematic = jet_syst, weight = weights_both, **mass_jk_fill)
                            ptreco_both_g = reco_jet_both.pt
                            ptreco_both_g = ptreco_both_g[~ak.is_none(ptreco_both_g)]
                            mreco_both_g = reco_jet_both.msoftdrop
                            mreco_both_g = mreco_both_g[~ak.is_none(mreco_both_g)]
                            weights_both_g = weights_both[~ak.is_none(mreco_both_g)]
                            fill_hist(self.hists, "response_matrix_g", dataset = dataset, channel = channel, ptreco = ptreco_both_g, mreco = mreco_both_g, ptgen = ptgen_both, mgen = mgen_both_g, systematic = jet_syst, weight = weights_both_g, **mass_jk_fill)

                            ### Rho filling

                            fill_hist(self.hists, "response_matrix_rho_u", dataset = dataset,  ptreco = ptreco_both,
                                      mpt_reco = 2*np.log10(mreco_both/(ptreco_both*jetR)), ptgen = ptgen_both,
                                      mpt_gen = 2*np.log10(mgen_both/(ptgen_both*jetR)), systematic = jet_syst,
                                      weight = weights_both)
                                
                            fill_hist(self.hists, "response_matrix_rho_g", dataset = dataset,  ptreco = ptreco_both,
                                      mpt_reco = 2*np.log10(mreco_both_g/(ptreco_both*jetR)), ptgen = ptgen_both,
                                      mpt_gen = 2*np.log10(mgen_both_g/(ptgen_both*jetR)), systematic = jet_syst,
                                      weight = weights_both)


                            
                            
                            #register_hist(self.hists, "jk_response_matrix_u", [ ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, binning.jackknife_axis])
                            fill_hist(self.hists, "jk_response_matrix_u",dataset = dataset, ptreco = ptreco_both, mreco = mreco_both, ptgen = ptgen_both, mgen = mgen_both, jk = jk_index, weight = weights_both)
                            fill_hist(self.hists, "jk_response_matrix_g",dataset = dataset, ptreco = ptreco_both_g, mreco = mreco_both_g, ptgen = ptgen_both, mgen = mgen_both_g, jk = jk_index, weight = weights_both_g)

                            
                            # End of channels loop
                
                if not self._do_gen:
                    self.logging.debug("This break condition is true")
                    break  # Exit the jet_syst loop if not doing GEN analysis

            if not self._do_jk:
                break  # Exit the JK loop if not doing jackknife resampling

        for name in sel.names:
            self.hists["cutflow"][f"{dataset}_{name}"] += sel.all(name).sum()
        

        return self.hists

    def postprocess(self, accumulator):
        hname_list = [key for key in accumulator.keys() if key not in ("cutflow", "nev", "sumw")]
        lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
        #hname_list = ["ptjet_mjet_u_reco", 'ptjet_mjet_g_reco', "ptjet_mjet_u_gen", "ptjet_mjet_g_gen", "response_matrix_u", "response_matrix_g"]
        sumw = accumulator["sumw"]
        
        for hname in hname_list:
            
            h = accumulator[hname]
            if hname == "reco_jet_ntuple":
                self._scale_reco_jet_ntuple_weights(h, sumw)
                continue
            if not hasattr(h, "axes") or "dataset" not in h.axes.name:
                continue
            for i,ds in enumerate(h.axes['dataset']):
                if ds.startswith("SingleMuon") or ds.startswith("EGamma") or ds.startswith("SingleElectron"):
                    continue
                elif 'pythia' in ds:
                    xsdb = {
                        'HT-70to100': 140.0	,
                        'HT-100to200': 139.2,
                        'HT-200to400': 38.4,
                        'HT-400to600': 5.174,
                        'HT-600to800': 1.258,
                        'HT-800to1200': 0.5598,
                        'HT-1200to2500': 0.1305,
                        'HT-2500toInf': 0.002997,  
                        
                    }
                    
                    lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    ht_bin = ds.split('_')[-1]
                    iov = ds.split('_')[-2]
                    xs = xsdb[ht_bin]
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    
                    

                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    
                    scale = (xs * lumi_fb * 1000) / sw
                    k_factor = 1.1297638966
                    scale = scale*k_factor
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")


                

                    
                elif 'herwig' in ds:
                    xs = 5.036
                    lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-2]
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif 'powheg' in ds:
                    xs = 1976.0
                    lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-2]
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ds.startswith('inclusive_'):
                    # NLO amcatnloFXFX DY M-50; xsdb GenXSecAnalyzer (UL ~6404 pb), no k-factor.
                    xs = 6404.0
                    lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{ds}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ds.startswith('ptz_'):
                    # PtZ-binned NLO amcatnloFXFX DY (LHEFilterPtZ, MatchEWPDG20).
                    # Each bin normalized to its own XSDB xs (pb); no k-factor (NLO).
                    ptz_xsdb = {
                        '0To50': 1485.0,
                        '50To100': 397.4,
                        '100To250': 97.2,
                        '250To400': 3.701,
                        '400To650': 0.5086,
                        '650ToInf': 0.04728,
                    }
                    lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    ptz_bin = ds.split('_')[1]
                    iov = ds.split('_')[-1]
                    xs = ptz_xsdb.get(ptz_bin)
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{ds}'. Skipping normalization.")
                        continue
                    lumi_fb = lumi_db[iov]
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{ds}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ('st' in ds) or ('ST' in ds):
                    xsdb = {'st_tW_antitop': 34.97,
                                'st_tW_top': 34.91	,
                                'ST_t-channel_antitop': 67.93,
                                'ST_t-channel_top':   69.09}
                    if ds.startswith('st_tW_antitop'):
                        xs = xsdb['st_tW_antitop']
                    elif ds.startswith('st_tW_top'):
                        xs = xsdb['st_tW_top']
                    elif ds.startswith('ST_t-channel_antitop'):
                        xs = xsdb['ST_t-channel_antitop']
                    else:
                        xs = xsdb['ST_t-channel_top']
                    lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    # self.logging.debug(f"{iov}")
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ('ww' in ds):
                    xs = 75.95
                    lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    # self.logging.debug(f"{iov}")
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ('wz' in ds):
                    xs = 27.6
                    lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    # self.logging.debug(f"{iov}")
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ('zz' in ds):
                    xs = 12.17
                    lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    # self.logging.debug(f"{iov}")
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ('ttjets' in ds):
                    xs = 471.7
                    lumi_db = {'UL16NanoAODv9':16.81 , 'UL16NanoAODAPVv9': 19.52 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    # self.logging.debug(f"{iov}")
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                else:
                    self.logging.info(f"Hist {hname} is not listed, no scaling done")
                    
            grouping = defaultdict(list)
            
            for ds in h.axes["dataset"]:
                if ds.startswith("SingleMuon") or ds.startswith("EGamma") or ds.startswith("SingleElectron"):
                    grouping[ds].append(ds)
                    continue
    
                
                if "pythia" in ds:
                    iov = ds.split("_")[-2]
                    new_key = f"pythia_{iov}"
                elif "herwig" in ds:
                    iov = ds.split("_")[-2]
                    new_key = f"herwig_{iov}"
                elif ("st" in ds) or ("ST" in ds):
                    iov = ds.split("_")[-1]
                    new_key = f"ST_{iov}"
                elif "ww" in ds:
                    iov = ds.split("_")[-1]
                    new_key = f"ww_{iov}"
                elif "wz" in ds:
                    iov = ds.split("_")[-1]
                    new_key = f"wz_{iov}"
                elif "zz" in ds:
                    iov = ds.split("_")[-1]
                    new_key = f"zz_{iov}"
                elif "ttjets" in ds:
                    iov = ds.split("_")[-1]
                    new_key = f"ttjets_{iov}"
                else:
                    iov = ds.split("_")[-1]
                    new_key = f"MC_{iov}"
                grouping[new_key].append(ds)
        
                # 3) Merge with the no-growth workaround (preserves axis order)
            h = group(h, oldname="dataset", newname="dataset", grouping=dict(grouping))
            accumulator[hname] = h

        return accumulator

                    
            
