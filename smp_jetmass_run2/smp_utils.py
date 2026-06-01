import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict


class util_constants: 
    def __init__(self):
        self.mclabels = [ "UL16NanoAODv9", "UL17NanoAODv9", "UL18NanoAODv9"]
        lumi = [35920,41530,59740]
        self.z_xs = 6077.22
        self.lumi = dict( zip( self.mclabels, lumi ) )





    

def get_z_gen_selection( events, selection, ptcut_e, ptcut_m, ptcut_e2=None, ptcut_m2=None):
    '''
    Function to get Z candidates from ee and mumu pairs in the "dressed" lepton gen collection
    '''
    isGenElectron = (np.abs(events.GenDressedLepton.pdgId) == 11)  & (events.GenDressedLepton.hasTauAnc == False)
    isGenMuon = (np.abs(events.GenDressedLepton.pdgId) == 13) & (events.GenDressedLepton.hasTauAnc == False)
    gen_charge = ak.where( events.GenDressedLepton.pdgId > 0, +1, -1)

    if ptcut_e2 == None:
        ptcut_e2 = ptcut_e
    if ptcut_m2 == None:
        ptcut_m2 = ptcut_m
    # print("Sum of pdg", ak.sum(events.GenDressedLepton[:, :2].pdgId , axis = 1) ==0)
    # print("two gen dressed ", ak.sum(isGenElectron, axis=1) >= 2)
    selection.add("twoGen_ee", 
                  (ak.sum(isGenElectron, axis=1) == 2) & 
                  (ak.all(events.GenDressedLepton.pt > ptcut_e2, axis=1)) & 
                  (ak.max(events.GenDressedLepton.pt, axis=1) > ptcut_e) &
                  (ak.all( np.abs(events.GenDressedLepton.eta) < 2.4, axis=1)) & 
                  (ak.sum(events.GenDressedLepton[:, :2].pdgId , axis = 1) == 0)
                  #(ak.sum(gen_charge, axis=1) == 0)
                 )
    selection.add("twoGen_mm", 
                  (ak.sum(isGenMuon, axis=1) == 2) & 
                  (ak.all(events.GenDressedLepton.pt > ptcut_m2, axis=1)) & 
                  (ak.max(events.GenDressedLepton.pt, axis=1) > ptcut_m) &
                  (ak.all( np.abs(events.GenDressedLepton.eta) < 2.4, axis=1)) & 
                  (ak.sum(events.GenDressedLepton[:, :2].pdgId , axis = 1) == 0)
                  #(ak.sum(gen_charge, axis=1) == 0)
                 )
    selection.add("twoGen_leptons",
                  selection.all("twoGen_ee") | selection.all("twoGen_mm")
                 )
    # selection.add("twoGen_leptons",selection.all("twoGen_mm")
    #              )
    sel = selection.all("twoGen_leptons")
    z_gen = events.GenDressedLepton[:, :2].sum(axis=1)
    #z_gen = ak.where( sel, ak.sum( events.GenDressedLepton, axis=1), None )
    return z_gen

def get_z_reco_selection( events, selection, ptcut_e, ptcut_m, ptcut_e2=20, ptcut_m2=20):
    '''
    Function to get Z candidates from ee and mumu pairs from reconstructed leptons. 
    If ptcut_e2 or ptcut_m2 are not None, then the cuts on the pt are asymmetric
    '''

    if ptcut_e2 == None:
        ptcut_e2 = ptcut_e
    if ptcut_m2 == None:
        ptcut_m2 = ptcut_m
    #print("Hello from function electron pt", events.Electron.pt)
    
    selection.add("twoReco_ee", 
                  (ak.num(events.Electron) == 2) & 
                  (ak.all(events.Electron.pt > ptcut_e2, axis=1)) & ## Subleading electron pt cut
                  (ak.max(events.Electron.pt, axis=1) > ptcut_e) &## leading electron pt cut
                  (ak.all( np.abs(events.Electron.eta) < 2.5, axis=1)) & ## leading electron eta cut
                  (ak.sum(events.Electron[:, :2].charge, axis=1) == 0) &  ### 
                  (ak.all(events.Electron.pfRelIso03_all < 0.2, axis=1)) &
                  (ak.all(events.Electron.cutBased > 3, axis=1) ) ## Tight ID
    )
    
    selection.add("number of electron is 2", (ak.num(events.Electron) >= 2))
    selection.add("ptcut_e2", (ak.all(events.Electron.pt > ptcut_e2, axis=1)))
    selection.add("ptcut_e", (ak.max(events.Electron.pt, axis=1) > ptcut_e))
    # selection.add("eta_cut_e", (ak.all( np.abs(events.Electron.eta) < 2.5, axis=1)))
    # selection.add("opposite_signed_ee",(ak.sum(events.Electron[:, :2].charge, axis=1) == 0))
    # selection.add("pfRelIso_cut_e",  (ak.all(events.Electron.pfRelIso03_all < 0.2, axis=1)))
    # selection.add("cutBased_e", (ak.all(events.Electron.cutBased > 3, axis=1) ))

    #print("muon pt: ", events.Muon.pt)
    #print("Max muon pt: ", ak.max(events.Muon.pt, axis=1))
    #print("Number of muons", ak.num(events.Muon))

    
    
    selection.add("twoReco_mm", 
                  (ak.num(events.Muon) == 2) & 
                  (ak.all(events.Muon.pt > ptcut_m2, axis=1)) & 
                  (ak.max(events.Muon.pt, axis=1) > ptcut_m) &
                  (ak.all( np.abs(events.Muon.eta) < 2.5, axis=1)) & 
                  (ak.sum(events.Muon[:, :2].charge, axis=1) == 0) &
                  #(ak.all(events.Muon.pfRelIso04_all < 0.25, axis=1)) &
                  (ak.all(events.Muon.tightId == True, axis=1))
    )
    
    selection.add("number of muon is 2", (ak.num(events.Muon) >= 2))
    selection.add("ptcut_m2", (ak.all(events.Muon.pt > ptcut_m2, axis=1)))
    selection.add("ptcut_m", (ak.max(events.Muon.pt, axis=1) > ptcut_m))
    # selection.add("eta_cut_m", (ak.all( np.abs(events.Muon.eta) < 2.5, axis=1)))
    # selection.add("opposite_signed_mm", (ak.sum(events.Muon[:, :2].charge, axis=1) == 0))
    # selection.add("pfRelIso_cut_m",  (ak.all(events.Muon.pfRelIso03_all < 0.2, axis=1)))
    # selection.add("looseId_m", (ak.all(events.Muon.mediumId == True, axis=1)) )

     
    
    selection.add("twoReco_leptons",
                  selection.all("twoReco_ee") | selection.all("twoReco_mm")
                 )

    #selection.add(" "  )
    ## remove this part and uncomment above section
    # selection.add("twoReco_leptons",selection.all("twoReco_mm")
    #              )
    

    #print("Two leptons cut ", ak.sum(selection.require(twoReco_leptons = True)))
    z_reco = ak.where( selection.all("twoReco_ee"), events.Electron[:,:2].sum(axis=1), events.Muon[:,:2].sum(axis=1) )
    return z_reco

def n_obj_selection( events, selection, coll, nmin=1, ptmin=120, etamax=2.5):
    '''
    Function to require at least nmin objects from events.coll that satisfy pt > ptmin and |eta| < etamax
    '''
    selection.add("oneGenJet", 
                  ak.sum( (getattr(events, coll).pt > ptmin) & (np.abs(getattr(events, coll).eta) < etamax), axis=1 ) >= nmin
                 )


def find_closest_dr( a, coll , verbose = False):
    '''
    Find the objects within coll that are closest to a. 
    Return it and the delta R between it and a.
    '''
    combs = ak.cartesian( (a, coll), axis=1 )
    dr = combs['0'].delta_r(combs['1'])
    dr_min = ak.singletons( ak.argmin( dr, axis=1 ) )
    sel = combs[dr_min]['1']
    return ak.firsts(sel),ak.firsts(dr[dr_min])

    

def get_groomed_jet( jet, subjets , verbose = False):
    '''
    Find the subjets that correspond to the given jet using delta R matching. 
    This is suboptimal, but it's hard to fix upstream. 
    '''
    combs = ak.cartesian( (jet, subjets), axis=1 )
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    sel = dr_jet_subjets < 0.8
    total = combs[sel]['1'].sum(axis=1)
    return total, sel

# def apply_lepton_separation(jets, muons, electrons):
#     '''
#     Input -- jet collection and muon collection.
#     Finds which jets are within DeltaR 0.4 of a muon and discards them. Returns cleaned jets.
    
#     '''
#     combs_muons = ak.cartesian( (jets, muons), axis = 1 )
#     dr_jet_muons = combs_muons['0'].delta_r(combs_muons['1'])

#     sel = (dr_jet_muons > 0.4)

#     jets = combs_muons[sel]['0']
#     ele_sep = False
#     if ele_sep:
#         del sel
#         combs_electrons = ak.cartesian( (jets, electrons), axis = 1 )
#         dr_jet_electrons = combs_electrons['0'].delta_phi(combs_electrons['1'])    
#         sel =  (dr_jet_electrons > 0.4)
#         good_jets = combs_electrons[sel]['0']
#         return good_jets
#     else:
#         return jets
def apply_lepton_separation(jets, muons, electrons, dr_cut=0.4):
    """
    Remove jets within dR < dr_cut of any selected muon or electron.
    Returns the cleaned jet collection.
    """
    # jets:    shape (events, n_jets)
    # muons:   shape (events, n_muons)
    # For each jet, broadcast muon coordinates and find minimum dR to any muon

    # Expand dims to get (events, n_jets, 1) vs (events, 1, n_muons) broadcasting
    jet_eta  = jets.eta[:, :, np.newaxis]
    jet_phi  = jets.phi[:, :, np.newaxis]
    mu_eta   = muons.eta[:, np.newaxis, :]
    mu_phi   = muons.phi[:, np.newaxis, :]
    el_eta   = electrons.eta[:, np.newaxis, :]
    el_phi   = electrons.phi[:, np.newaxis, :]

    deta_mu  = jet_eta - mu_eta
    dphi_mu  = (jet_phi - mu_phi + np.pi) % (2 * np.pi) - np.pi
    dR_mu    = np.sqrt(deta_mu**2 + dphi_mu**2)   # (events, n_jets, n_muons)

    deta_el  = jet_eta - el_eta
    dphi_el  = (jet_phi - el_phi + np.pi) % (2 * np.pi) - np.pi
    dR_el    = np.sqrt(deta_el**2 + dphi_el**2)   # (events, n_jets, n_electrons)

    # A jet is clean if it is far from ALL muons AND ALL electrons
    clean_of_mu = ak.all(dR_mu > dr_cut, axis=2)  # (events, n_jets)
    clean_of_el = ak.all(dR_el > dr_cut, axis=2)  # (events, n_jets)

    return jets[clean_of_mu & clean_of_el]

def apply_lepton_separation_gen(jets, leptons, dr_cut=0.4):
    """
    Remove jets within dR < dr_cut of any lepton in `leptons`.
    jets:    (events, n_jets)
    leptons: (events, n_leptons)
    """
    jet_eta = jets.eta[:, :, np.newaxis]
    jet_phi = jets.phi[:, :, np.newaxis]

    lep_eta = leptons.eta[:, np.newaxis, :]
    lep_phi = leptons.phi[:, np.newaxis, :]

    deta = jet_eta - lep_eta
    dphi = (jet_phi - lep_phi + np.pi) % (2 * np.pi) - np.pi
    dR = np.sqrt(deta**2 + dphi**2)

    clean_of_lep = ak.all(dR > dr_cut, axis=2)
    return jets[clean_of_lep]
# def get_dphi( a, coll, verbose=False ):
#     '''
#     Find the highest-pt object in coll and return the highest pt,
#     as well as the delta phi to a. 
#     '''
#     combs = ak.cartesian( (a, coll), axis=1 )
#     dphi = np.abs(combs['0'].delta_phi(combs['1']))
    
#     return ak.firsts( combs['1'] ), ak.firsts(dphi)

def get_dphi(a, coll, verbose=False):
    index = ak.argsort(coll.pt, ascending=False)
    sorted_coll = coll[index]
    lead_jet = ak.firsts(sorted_coll)
    dphi = np.abs(a.delta_phi(lead_jet))
    return lead_jet, dphi
    

def get_dphi_reco(a, coll, verbose=False):
    # Sort coll by pt descending so ak.firsts gives the highest-pt object
    coll_sorted = coll[ak.argsort(coll.pt, axis=1, ascending=False)]
    combs = ak.cartesian((a, coll_sorted), axis=1)
    dphi = np.abs(combs['0'].delta_phi(combs['1']))
    sel = dphi > 0.4

    best_obj = ak.firsts(combs[sel]['1'])
    best_dphi = ak.firsts(dphi[sel])

    if verbose:
        print(f"Best pt: {best_obj.pt}, Best dphi: {best_dphi}")

    return best_obj, best_dphi  # or return best_obj.pt, best_dphi
# def get_dphi_reco( a, coll, verbose=False ):
#     '''
#     Find the highest-pt object in coll and return the highest pt,
#     as well as the delta phi to a. 
#     '''
#     combs = ak.cartesian( (a, coll), axis=1 )
#     dphi = np.abs(combs['0'].delta_phi(combs['1']))
#     sel = dphi > 0.4
    
#     return ak.firsts( combs[sel]['1'] ), ak.firsts(dphi[sel])

def plot_pt_threshold(events0):
    import matplotlib.pyplot as plt
    
    # Extract the pt values for leading electrons in events with >= 2 electrons
    lead_electron_pts = events0[ak.num(events0.Electron) >= 2].Electron[:, 0].pt
    
    # Set up bins with an edge at 40
    bins = [i for i in range(0, 401, 4)]  # Bin width 4, covers 0–400, includes 40
    
    # Calculate histogram
    counts, edges, _ = plt.hist(lead_electron_pts, bins=bins, range=(0, 400), color = 'orange')
    
    # Calculate percentages
    total = sum(counts)
    left = sum([c for e, c in zip(edges, counts) if e < 40])
    right = total - left
    
    left_pct = (left / total) * 100
    right_pct = (right / total) * 100
    
    # Draw vertical line at 40
    plt.axvline(x=40, color='red', linestyle='--', linewidth=1)
    
    # Add percentage text
    plt.text(38, max(counts)*0.5, f"{left_pct:.1f}% <= 40", color='darkred', fontsize=12, ha='center', rotation = 'vertical')
    plt.text(42, max(counts)*0.5, f"{right_pct:.1f}% > 40", color='darkred', fontsize=12, ha='center', rotation = 'vertical')
    
    # Final plot settings
    plt.xlim(0, 100)
    plt.xlabel("Leading Electron $p_T$ [GeV]")
    plt.ylabel("Events")
    plt.title("Leading Electron $p_T$ Distribution")
    plt.savefig("lead_electron_pt_threshold40.png")
    plt.show()

    
    # Extract the pt values for leading electrons in events with >= 2 electrons
    lead_electron_pts = events0[ak.num(events0.Electron) >= 2].Electron[:, 1].pt
    
    # Set up bins with an edge at 40
    bins = [i for i in range(0, 401, 4)]  # Bin width 4, covers 0–400, includes 40
    
    # Calculate histogram
    counts, edges, _ = plt.hist(lead_electron_pts, bins=bins, range=(0, 400), color = 'orange')
    
    # Calculate percentages
    total = sum(counts)
    left = sum([c for e, c in zip(edges, counts) if e < 40])
    right = total - left
    
    left_pct = (left / total) * 100
    right_pct = (right / total) * 100
    
    # Draw vertical line at 40
    plt.axvline(x=40, color='red', linestyle='--', linewidth=1)
    
    # Add percentage text
    plt.text(38, max(counts)*0.5, f"{left_pct:.1f}% <= 40", color='darkred', fontsize=12, ha='center', rotation = 'vertical')
    plt.text(42, max(counts)*0.5, f"{right_pct:.1f}% > 40", color='darkred', fontsize=12, ha='center', rotation = 'vertical')
    
    # Final plot settings
    plt.xlim(0, 100)
    plt.xlabel("Sub-leading Electron $p_T$ [GeV]")
    plt.ylabel("Events")
    plt.title("Sub-leading Electron $p_T$ Distribution")
    plt.savefig("sublead_electron_pt_threshold40.png")
    plt.show()



    ### Muon
    
    # Extract the pt values for leading electrons in events with >= 2 electrons
    lead_muon_pts = events0[ak.num(events0.Muon) >= 2].Muon[:, 0].pt
    
    # Set up bins with an edge at 40
    bins = [i for i in range(0, 401, 4)]  # Bin width 4, covers 0–400, includes 40
    
    # Calculate histogram
    counts, edges, _ = plt.hist(lead_muon_pts, bins=bins, range=(0, 400))
    
    # Calculate percentages
    total = sum(counts)
    left = sum([c for e, c in zip(edges, counts) if e < 29])
    right = total - left
    
    left_pct = (left / total) * 100
    right_pct = (right / total) * 100
    
    # Draw vertical line at 40
    plt.axvline(x=29, color='red', linestyle='--', linewidth=1)
    
    # Add percentage text
    plt.text(27, max(counts)*0.5, f"{left_pct:.1f}% <= 40 GeV", color='darkred', fontsize=12, ha='center', rotation = 'vertical')
    plt.text(31, max(counts)*0.5, f"{right_pct:.1f}% > 40 GeV", color='darkred', fontsize=12, ha='center', rotation = 'vertical')
    
    # Final plot settings
    plt.xlim(0, 100)
    plt.xlabel("Leading Muon $p_T$ [GeV]")
    plt.ylabel("Events")
    plt.title("Leading Muon $p_T$ Distribution")
    plt.savefig("lead_muon_pt_threshold40.png")
    plt.show()

    
    # Extract the pt values for leading electrons in events with >= 2 electrons
    lead_muon_pts = events0[ak.num(events0.Muon) >= 2].Muon[:, 1].pt
    
    # Set up bins with an edge at 40
    bins = [i for i in range(0, 401, 4)]  # Bin width 4, covers 0–400, includes 40
    
    # Calculate histogram
    counts, edges, _ = plt.hist(lead_muon_pts, bins=bins, range=(0, 400))
    
    # Calculate percentages
    total = sum(counts)
    left = sum([c for e, c in zip(edges, counts) if e < 40])
    right = total - left
    
    left_pct = (left / total) * 100
    right_pct = (right / total) * 100
    
    # Draw vertical line at 40
    plt.axvline(x=29, color='red', linestyle='--', linewidth=1)
    
    # Add percentage text
    plt.text(27, max(counts)*0.2, f"{left_pct:.1f}% <= 29 GeV", color='darkred', fontsize=12, ha='center', rotation = 'vertical')
    plt.text(31, max(counts)*0.2, f"{right_pct:.1f}% > 29 GeV", color='darkred', fontsize=12, ha='center', rotation = 'vertical')
    
    # Final plot settings
    plt.xlim(0, 100)
    plt.xlabel("Sub-leading Muon $p_T$ [GeV]")
    plt.ylabel("Events")
    plt.title("Sub-leading Muon $p_T$ Distribution")
    plt.savefig("sublead_muon_pt_threshold40.png")
    plt.show()


def plot_nlepton(events0):
    import matplotlib.pyplot as plt
    import hist
    h_nleptons = hist.Hist(hist.axis.StrCategory(['Electron', 'Muon'], name = "lepton"), hist.axis.Variable([-0.5,0.5,1.5, 2.5, 3.5, 4.5, 5.5], name = "number", label = "# Leptons"))

    h_nleptons.fill(lepton = "Muon",number =  ak.num(events0.Muon))
    h_nleptons.fill(lepton = "Electron", number =  ak.num(events0.Electron) )
    
    #plt.hist(ak.num(events.Muon), bins = [-0.5,0.5,1.5, 2.5, 3.5], label = "Muons")
    #plt.hist(ak.num(events.Electron), bins = [-0.5,0.5,1.5, 2.5, 3.5], label = "Electrons")
    fig, ax = plt.subplots()
    
    h_total = h_nleptons[0,:].project("number")
    h_total.plot(stack=True, histtype='fill', ax=ax, label = "Muon")
    # Add percentage labels
    total = h_total.values().sum()
    bin_centers = h_total.axes[0].centers
    
    for i, center in enumerate(bin_centers):
        count = h_total.values()[i]
        if count > 0:
            percent = 100 * count / total
            ax.text(
                center, count + 50,       # adjust 50 as needed for spacing
                f"{percent:.1f}%", 
                ha="center", 
                va="bottom", 
                fontsize=9
            )
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    
    h_total = h_nleptons[1,:].project("number")
    h_total.plot(stack=True, histtype='fill', ax=ax, label = "Electron", color = 'orange')
    # Add percentage labels
    total = h_total.values().sum()
    bin_centers = h_total.axes[0].centers
    
    for i, center in enumerate(bin_centers):
        count = h_total.values()[i]
        if count > 0:
            percent = 100 * count / total
            ax.text(
                center, count + 50,       # adjust 50 as needed for spacing
                f"{percent:.1f}%", 
                ha="center", 
                va="bottom", 
                fontsize=9
            )

    plt.legend()
    plt.tight_layout()
    plt.show()

# class tunfold_binning:
#     def __init__(self, mbins, ptbins, uflow1,oflow1, uflow2,  oflow2):
#         self.mbins = mbins
#         self.ptbins = ptbins
#         self.nmbins = len(mbins) - 1
#         self.nptbins = len(ptbins) - 1
#         self.uflow1 = uflow1
#         self.uflow2 = uflow2

#         self.oflow1 = oflow1
#         self.oflow2 = oflow2
        
#         if (uflow1 & oflow1) :
#             self.nmbins += 2
#         elif (uflow1 | oflow1):
#             self.nmbins += 1

#         if (uflow2 & oflow2):
#             self.nptbins += 2 
#         elif (uflow2 | oflow2):
#             self.nptbins += 1
#         self.total_nbin = self.nmbins * self.nptbins
        
#     def getGlobalBinNumber(self, mass, pt):
#         '''
#         Function to derive Global Bin Number. Inspired by GetGlobalBinNumber in TUnfoldDensity. Currently not used in this code.
#         '''
#         mbin_number = np.digitize(mass, self.mbins )
#         ptbin_number = np.digitize(pt, self.ptbins )
        
        
#         # print("mbin number: ", mbin_number)
#         # print("ptbin number: ", ptbin_number)
        
#         # print("total bin number: ", self.total_nbin)
        
#         if self.uflow1 == False and self.oflow1 == False and self.uflow2 == True and self.oflow2 == False:               
            
#             globB =  globB = ak.where( (mbin_number == 0), -1, ak.where( (mbin_number == self.nmbins+1) | (ptbin_number == self.nptbins+1), self.total_nbin, (ptbin_number )*self.nmbins + mbin_number ))
        
#         if self.uflow1 == True and self.oflow1 == False and self.uflow2 == True and self.oflow2 == False:               
            
#             globB = ak.where( (mbin_number == self.nmbins+1) | (ptbin_number == self.nptbins+1), self.total_nbin, (ptbin_number )*self.nmbins + mbin_number + 1)
        
#         if self.uflow1 == False and self.oflow1 == False and self.uflow2 == False and self.oflow2 == False:               
#             globB = ak.where( (mbin_number == 0) | (ptbin_number == 0) , 0, ak.where((mbin_number == self.nmbins+1) | (ptbin_number == self.nptbins+1), self.total_nbin, (ptbin_number -1)*self.nmbins + mbin_number) )

                
#         if self.uflow1 == True and self.oflow1 == True and self.uflow2 == False and self.oflow2 == False:
#             globB = ak.where( ptbin_number == 0, 0, ak.where(ptbin_number == self.nptbins+1, self.total_nbin,(ptbin_number -1)*self.nmbins + mbin_number + 1 ) )
                
#         if self.uflow1 == True and self.uflow2 == True and self.oflow1 == True and self.oflow2 == True:
#             globB = (ptbin_number )*self.nmbins + mbin_number + 1
            
#         return globB

# class util_binning :
#     '''
#     Class to implement the binning schema for jet mass and pt 2d unfolding. The gen-level mass is twice as fine. 
#     '''
#     def __init__(self):
#         #self.ptreco_axis = hist.axis.Variable([200,260,350,460,550,650,760,13000], name="ptreco", label=r"p_{T,RECO} (GeV)")   
        
#         #self.mgen_axis = hist.axis.Variable([0, 10, 20, 40, 60, 80, 100, 13000], name="mgen", label=r"Mass (GeV)")

#         ### Original version
#         #self.mgen_axis = hist.axis.Variable([0, 10, 20, 40, 60, 80, 100, 150, 200, 13000], name="mgen", label=r"Mass (GeV)")
        
#         #self.mreco_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 6200, 13000], name="mreco", label=r"$m_{RECO}$ (GeV)")
#         ############

#         #self.mgen_axis = hist.axis.Variable([0, 10, 20, 40, 60, 80, 100, 120, 140, 160, 200, 13000], name="mgen", label=r"Mass (GeV)")
#         #self.mreco_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130 , 140, 150, 160, 180, 200, 250, 300, 350, 400, 500, 1000, 13000], name="mreco", label=r"$m_{RECO}$ (GeV)")

#         #self.mgen_axis = hist.axis.Regular(100,0,200, name="mgen", label=r"Mass (GeV)")
#         #self.mreco_axis = hist.axis.Regular(100,0,200, name="mreco", label=r"$m_{RECO}$")

#         ### Only for making the response matrix for Herwig TOY MC
#         #self.mgen_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 13000], name="mgen", label=r"Mass (GeV)")

        
#         #self.mreco_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 13000] , name="mreco", label=r"$m_{RECO}$ (GeV)")
#         # Original bins
#         # self.mgen_axis = hist.axis.Variable([0, 5, 10, 20, 40, 60, 80, 100, 120, 140,160, 180, 200, 13000], name="mgen", label=r"Mass (GeV)")
#         # self.mreco_axis = hist.axis.Variable([0, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 500, 13000] , name="mreco", label=r"$m_{RECO}$ (GeV)")

#         self.mgen_axis = hist.axis.Variable([0, 5, 10, 20, 40, 60, 80, 100, 150, 200, 13000], name="mgen", label=r"Mass (GeV)")
#         self.mreco_axis = hist.axis.Variable([0, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 500, 13000] , name="mreco", label=r"$m_{RECO}$ (GeV)")

#         self.mgen_axis = hist.axis.Variable([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
#                                                 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
#                                                 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 
#                                                 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 
#                                                 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 
#                                                 94, 96, 98, 100, 102, 104, 106, 108, 110, 
#                                                 112, 114, 116, 118, 120, 122, 124, 126, 
#                                                 128, 130, 132, 134, 136, 138, 140, 142, 
#                                                 144, 146, 148, 150, 152, 154, 156, 158, 
#                                                 160, 162, 164, 166, 168, 170, 172, 174, 
#                                                 176, 178, 180, 182, 184, 186, 188, 190, 
#                                                 192, 194, 196, 198, 200, 1000], name="mgen", label=r"Mass (GeV)")

#         self.mreco_axis = hist.axis.Variable([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 
#                                                     5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 
#                                                     10, 11.0, 12, 13.0, 14, 15.0, 16, 17.0, 18, 19.0, 
#                                                     20, 21.0, 22, 23.0, 24, 25.0, 26, 27.0, 28, 29.0, 
#                                                     30, 31.0, 32, 33.0, 34, 35.0, 36, 37.0, 38, 39.0, 
#                                                     40, 41.0, 42, 43.0, 44, 45.0, 46, 47.0, 48, 49.0, 
#                                                     50, 51.0, 52, 53.0, 54, 55.0, 56, 57.0, 58, 59.0, 
#                                                     60, 61.0, 62, 63.0, 64, 65.0, 66, 67.0, 68, 69.0, 
#                                                     70, 71.0, 72, 73.0, 74, 75.0, 76, 77.0, 78.0, 79.0, 80, 81.0, 
#                                                     82, 83.0, 84, 85.0, 86, 87.0, 88, 89.0, 90, 91.0, 
#                                                     92, 93.0, 94, 95.0, 96, 97.0, 98, 99.0, 100, 101.0, 
#                                                     102, 103.0, 104, 105.0, 106, 107.0, 108, 109.0, 110, 111.0, 
#                                                     112, 113.0, 114, 115.0, 116, 117.0, 118, 119.0, 120, 121.0, 
#                                                     122, 123.0, 124, 125.0, 126, 127.0, 128, 129.0, 130, 131.0, 
#                                                     132, 133.0, 134, 135.0, 136, 137.0, 138, 139.0, 140, 141.0, 
#                                                     142, 143.0, 144, 145.0, 146, 147.0, 148, 149.0, 150, 151.0, 
#                                                     152, 153.0, 154, 155.0, 156, 157.0, 158, 159.0, 160, 161.0, 
#                                                     162, 163.0, 164, 165.0, 166, 167.0, 168, 169.0, 170, 171.0, 
#                                                     172, 173.0, 174, 175.0, 176, 177.0, 178, 179.0, 180, 181.0, 
#                                                     182, 183.0, 184, 185.0, 186, 187.0, 188, 189.0, 190, 191.0, 
#                                                     192, 193.0, 194, 195.0, 196, 197.0, 198, 199.0, 200, 500, 1000], name="mreco", label=r"$m_{RECO}$ (GeV)")

#         # self.mgen_over_pt_axis = hist.axis.Variable([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10], name = 'mpt_gen', label = r'$\rho$ ')
#         # self.mreco_over_pt_axis = hist.axis.Variable([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5 , 7.5, 10], name = 'mpt_reco', label = r'$\rho$ (Detector)')


#         # Negative counterpart of mgen_over_pt_axis
#         self.mgen_over_pt_axis = hist.axis.Variable(
#             [-10, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0],
#             name='mpt_gen', label=r'$-\rho$'
#         )
        
#         # Negative counterpart of mreco_over_pt_axis
#         self.mreco_over_pt_axis = hist.axis.Variable(
#             [-10, -7.5, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3,
#              -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0],
#             name='mpt_reco', label=r'$-\rho$ (Detector)'
#         )
#         #self.ptgen_axis = hist.axis.Variable([200,260,350,460,550,650,760,13000], name="ptgen", label=r"p_{T,RECO} (GeV)")   

#         #self.ptgen_axis = hist.axis.Variable([140, 200, 260, 330, 408, 13000], name="ptgen", label=r"p_{T,GEN} (GeV)")  
#         #self.ptreco_axis = hist.axis.Variable([140, 200, 260, 330, 408, 13000], name="ptreco", label=r"p_{T,RECO} (GeV)")

        
#         # self.ptgen_axis = hist.axis.Variable([  200.,   260.,   350.,   460., 13000.], name="ptgen", label=r"$p_{T,GEN}$ (GeV)")  
#         # self.ptreco_axis = hist.axis.Variable([ 200.,   260.,   350.,   460., 13000.], name="ptreco", label=r"$p_{T,RECO}$ (GeV)")
#         self.ptgen_axis = hist.axis.Variable([  200.,   290.,   400.,    13000.], name="ptgen", label=r"$p_{T,GEN}$ (GeV)")  
#         self.ptreco_axis = hist.axis.Variable([ 200.,   290.,   400.,    13000.], name="ptreco", label=r"$p_{T,RECO}$ (GeV)")

#         self.mcut_reco_u_axis = hist.axis.Variable([ 0, 20, 10000], name="mreco", label=r"Mass (GeV)" )
#         self.mcut_reco_g_axis = hist.axis.Variable([ 0, 10, 10000], name="mreco", label=r"Mass (GeV)" )

#         self.mcut_gen_u_axis = hist.axis.Variable([ 0, 20, 10000], name="mgen", label=r"Mass (GeV)" )
#         self.mcut_gen_g_axis = hist.axis.Variable([ 0, 10, 10000], name="mgen", label=r"Mass (GeV)" )

#         self.ptgen_axis_fine = hist.axis.Variable([  200., 210,  220, 230, 240, 260., 280, 300, 320, 340,  350., 370, 390, 410, 430, 450, 500, 550, 600, 650, 700, 800, 900, 1000, 13000.], name="ptgen_fine", label=r"$p_{T,GEN}$ (GeV)")  
        
        
#         #self.mgen_axis = hist.axis.Variable( [0,2.5,5,7.5,10,15,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,325,350,1000], name="mgen", label=r"Mass [GeV]")
        
#         self.gen_binning = tunfold_binning( self.mgen_axis.edges, self.ptgen_axis.edges, False, False, False, False )
#         self.reco_binning = tunfold_binning( self.mreco_axis.edges, self.ptreco_axis.edges, False, False, False, False )
        
#         self.gen_axis = hist.axis.Regular(self.gen_binning.total_nbin, 0, self.gen_binning.total_nbin, name = "bin_gen", label = "Generator")
#         self.reco_axis =  hist.axis.Regular(self.reco_binning.total_nbin, 0, self.reco_binning.total_nbin, name = "bin_reco", label = "Detector")
        
#         self.dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
#         self.dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
#         self.lep_axis = hist.axis.StrCategory(["ee", "mm"], name="lep")
#         self.n_axis = hist.axis.Regular(10, 0, 10, name="n", label=r"Number")
#         self.mass_axis = hist.axis.Regular(100, 0, 1000, name="mass", label=r"$m$ [GeV]")
#         self.diff_axis = hist.axis.Regular(100, -20, 20, name="diff", label=r"$\Delta$ [GeV]")
#         self.diff_axis_large = hist.axis.Regular(100, -50, 50, name="diff", label=r"$\Delta$ [GeV]")
#         self.zmass_axis = hist.axis.Regular(100, 80, 100, name="mass", label=r"$m$ [GeV]")
#         self.pt_axis = hist.axis.Regular(150, 0, 1500, name="pt", label=r"$p_{T}$ [GeV]")                
#         self.frac_axis = hist.axis.Regular(200, -1.5, 1.5, name="frac", label=r"Fraction")                
#         self.dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
#         self.dr_fine_axis = hist.axis.Regular(150, 0, 1.5, name="dr", label=r"$\Delta R$")
#         self.dphi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")
#         self.eta_axis = hist.axis.Regular(40, -5, 5, name="eta", label=r"$ \eta$")
#         self.phi_axis = hist.axis.Regular(40, -5, 5, name="phi", label=r"$ \phi$")
#         self.ptfine_axis = hist.axis.Regular(20, 200, 500, name="pt", label=r"p_{T,RECO} (GeV)")
#         self.jackknife_axis = hist.axis.IntCategory([], growth = True, name = 'jk', label = "Jackknife categories" )
        
#         self.syst_axis=hist.axis.StrCategory([],growth = True, name = "systematic", label = "Systematic Uncertainty")

# def fill_tunfold_hist_1d(hist, mass, pt, weight,  recogen, dataset , systematic):
#     binning = util_binning()
#     if recogen == 'gen':
#         globB = binning.gen_binning.getGlobalBinNumber(mass, pt)
#         hist.fill(dataset = dataset, bin_gen = globB-0.5, systematic = systematic, weight = weight)
#     if recogen == 'reco':
#         globB = binning.reco_binning.getGlobalBinNumber(mass, pt)
#         hist.fill(dataset = dataset, bin_reco= globB-0.5, systematic = systematic, weight = weight)
    
    


# def fill_tunfold_hist_2d(hist, mass_gen, pt_gen, mass_reco, pt_reco, weight, dataset, systematic, jk_index = None):
#     binning = util_binning()
#     globBgen = binning.gen_binning.getGlobalBinNumber(mass_gen, pt_gen)
#     globBreco = binning.reco_binning.getGlobalBinNumber(mass_reco, pt_reco)
#     if jk_index == None:
#         hist.fill(dataset = dataset, bin_reco= globBreco-0.5, bin_gen = globBgen-0.5, systematic = systematic,  weight = weight)
#     else:
#         hist.fill(dataset = dataset, jk = jk_index, bin_reco= globBreco-0.5, bin_gen = globBgen-0.5, systematic = systematic,  weight = weight)