// -*- C++ -*-
//
// Rivet routine for the CMS Z+jet groomed/ungroomed jet-mass measurement
// (Run 2, smp_jetmass_run2 / zjet channel).
//
// This reproduces the PARTICLE-LEVEL (fiducial) selection of the coffea
// QJetMassProcessor gen path, byte-for-byte where the generator level allows:
//
//   Dressed leptons (dR = 0.1 dressing, no tau ancestors):
//       electrons pT > 40 GeV, muons pT > 29 GeV, |eta| < 2.4
//   Z boson: exactly two same-flavour, opposite-sign dressed leptons;
//       pT(Z) > 90 GeV and 71 < m(ll) < 111 GeV
//   AK8 jets (anti-kT, R = 0.8): |y| < 2.4, cleaned of any dressed lepton
//       within dR < 0.4; require >= 1 such jet
//   Candidate jet: the highest-pT cleaned jet
//   Topology (Z vs candidate jet):
//       dphi(Z, jet) > 1.57  and  pT-asymmetry |pTZ - pTjet|/(pTZ + pTjet) < 0.3
//
//   Observables, measured in jet-pT bins [200, 290, 400, inf]:
//       ungroomed AK8 jet mass m_u
//       soft-drop jet mass m_g  (beta = 0, z_cut = 0.1)
//       rho = 2 * log10(m / (pT * R)) with R = 0.8, for both m_u and m_g
//
// NOTE: the official Rivet analysis ID (CMS_<year>_I<inspire>) should replace
// the placeholder name once the paper/HepData entry exists.

#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/VisibleFinalState.hh"
#include "Rivet/Projections/LeptonFinder.hh"
#include "Rivet/Projections/FastJets.hh"

#include "fastjet/contrib/SoftDrop.hh"

namespace Rivet {

  /// Z(ll)+jet groomed and ungroomed jet-mass cross sections
  class CMS_ZJET_JETMASS : public Analysis {
  public:

    RIVET_DEFAULT_ANALYSIS_CTOR(CMS_ZJET_JETMASS);

    void init() {

      // --- Dressed leptons (dR = 0.1 cone dressing) ---------------------
      // LeptonOrigin::NODECAY selects prompt leptons not coming from any decay
      // (hadron or tau), matching GenDressedLepton.hasTauAnc == False.
      // The Cut is applied to the *dressed* lepton, so abspid restricts flavour.
      // electrons: pT > 40, |eta| < 2.4 ; muons: pT > 29, |eta| < 2.4
      LeptonFinder dressed_el(0.1, Cuts::abspid == PID::ELECTRON &&
                              Cuts::pT > 40*GeV && Cuts::abseta < 2.4,
                              LeptonOrigin::NODECAY);
      LeptonFinder dressed_mu(0.1, Cuts::abspid == PID::MUON &&
                              Cuts::pT > 29*GeV && Cuts::abseta < 2.4,
                              LeptonOrigin::NODECAY);
      declare(dressed_el, "DressedElectrons");
      declare(dressed_mu, "DressedMuons");

      // AK8 gen jets are clustered from all visible final-state particles
      // (everything except neutrinos), exactly as CMS slimmedGenJetsAK8.
      // Leptons that end up inside jets are removed afterwards by the dR < 0.4
      // jet-lepton cleaning, so they are intentionally kept in the input here.
      VisibleFinalState vfs(Cuts::abseta < 5.0);
      FastJets jets(vfs, fastjet::JetDefinition(fastjet::antikt_algorithm, 0.8));
      declare(jets, "JetsAK8");

      // --- Binning (matches util_binning gen axes) ----------------------
      // Jet-pT bins: [200, 290, 400, inf]
      _ptedges = {200., 290., 400., 13000.};
      _ptlabel = {"200_290", "290_400", "400_Inf"};

      // Ungroomed/groomed jet-mass bins [GeV]
      const vector<double> medges =
        {0., 10., 20., 30., 50., 70., 90., 110., 130., 150., 170., 200.};
      // rho = 2*log10(m/(pT*R)) bins
      const vector<double> rhoedges =
        {-10., -6., -5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0.};

      // --- Histograms ---------------------------------------------------
      for (size_t i = 0; i < _ptlabel.size(); ++i) {
        book(_h_mass_u[i], "mass_u_pt" + _ptlabel[i], medges);
        book(_h_mass_g[i], "mass_g_pt" + _ptlabel[i], medges);
        book(_h_rho_u[i],  "rho_u_pt"  + _ptlabel[i], rhoedges);
        book(_h_rho_g[i],  "rho_g_pt"  + _ptlabel[i], rhoedges);
      }
      // pT-inclusive (pT > 200) spectra
      book(_h_mass_u_incl, "mass_u_incl", medges);
      book(_h_mass_g_incl, "mass_g_incl", medges);

      // Z-boson control distributions (after full event selection)
      book(_h_z_pt,   "z_pt",   25, 90., 590.);
      book(_h_z_mass, "z_mass", 40, 71., 111.);
      book(_h_jet_pt, "jet_pt", 30, 200., 800.);
    }

    void analyze(const Event& event) {

      // --- Dressed-lepton Z reconstruction ------------------------------
      const DressedLeptons els =
        apply<LeptonFinder>(event, "DressedElectrons").dressedLeptons();
      const DressedLeptons mus =
        apply<LeptonFinder>(event, "DressedMuons").dressedLeptons();

      // exactly two same-flavour, opposite-sign dressed leptons
      DressedLeptons leps;
      if (els.size() == 2 && mus.empty() &&
          els[0].charge() + els[1].charge() == 0) {
        leps = els;
      } else if (mus.size() == 2 && els.empty() &&
                 mus[0].charge() + mus[1].charge() == 0) {
        leps = mus;
      } else {
        vetoEvent;
      }

      const FourMomentum zmom = leps[0].mom() + leps[1].mom();
      if (zmom.pT() <= 90*GeV) vetoEvent;
      if (zmom.mass() <= 71*GeV || zmom.mass() >= 111*GeV) vetoEvent;

      // --- AK8 jets: |y| < 2.4, cleaned of dressed leptons (dR < 0.4) ----
      const Jets alljets =
        apply<FastJets>(event, "JetsAK8").jetsByPt(Cuts::absrap < 2.4);
      Jets cleanjets;
      for (const Jet& j : alljets) {
        bool overlap = false;
        for (const DressedLepton& l : leps) {
          if (deltaR(j, l) < 0.4) { overlap = true; break; }
        }
        if (!overlap) cleanjets.push_back(j);
      }
      if (cleanjets.empty()) vetoEvent;

      // candidate = highest-pT cleaned jet
      const Jet& jet = cleanjets[0];

      // --- Z vs jet topology --------------------------------------------
      const double dphi = deltaPhi(zmom, jet.mom());
      if (dphi <= 1.57) vetoEvent;
      const double ptasym =
        std::abs(zmom.pT() - jet.pT()) / (zmom.pT() + jet.pT());
      if (ptasym >= 0.3) vetoEvent;

      // measurement starts at jet pT = 200 GeV
      if (jet.pT() < 200*GeV) vetoEvent;

      // --- Observables ---------------------------------------------------
      const double m_u = jet.mass();

      // Soft-drop mass: recluster the jet constituents with Cambridge/Aachen
      // (R = 0.8) and apply SoftDrop(beta = 0, z_cut = 0.1), as in CMS AK8.
      double m_g = 0.0;
      const vector<fastjet::PseudoJet> constits = jet.pseudojet().constituents();
      if (!constits.empty()) {
        fastjet::JetDefinition ca_def(fastjet::cambridge_algorithm, 0.8);
        fastjet::ClusterSequence cs(constits, ca_def);
        const vector<fastjet::PseudoJet> reclustered =
          fastjet::sorted_by_pt(cs.inclusive_jets());
        if (!reclustered.empty()) {
          fastjet::contrib::SoftDrop sd(0.0, 0.1);  // beta, z_cut
          const fastjet::PseudoJet groomed = sd(reclustered[0]);
          if (groomed != 0) m_g = groomed.m();
        }
      }

      const double R = 0.8;
      const double rho_u =
        (m_u > 0) ? 2.0 * std::log10(m_u / (jet.pT() * R)) : -99.0;
      const double rho_g =
        (m_g > 0) ? 2.0 * std::log10(m_g / (jet.pT() * R)) : -99.0;

      // --- Fill ----------------------------------------------------------
      _h_z_pt->fill(zmom.pT());
      _h_z_mass->fill(zmom.mass());
      _h_jet_pt->fill(jet.pT());

      _h_mass_u_incl->fill(m_u);
      _h_mass_g_incl->fill(m_g);

      const int ipt = ptBin(jet.pT());
      if (ipt >= 0) {
        _h_mass_u[ipt]->fill(m_u);
        _h_mass_g[ipt]->fill(m_g);
        if (rho_u > -99.0) _h_rho_u[ipt]->fill(rho_u);
        if (rho_g > -99.0) _h_rho_g[ipt]->fill(rho_g);
      }
    }

    void finalize() {
      // Normalise to fiducial cross section: sigma = sum(w) / sum(w_gen) * xs.
      // crossSection()/sumOfWeights() yields the per-event xs weight [pb].
      const double norm = crossSection() / sumOfWeights();
      for (size_t i = 0; i < _ptlabel.size(); ++i) {
        scale(_h_mass_u[i], norm);
        scale(_h_mass_g[i], norm);
        scale(_h_rho_u[i],  norm);
        scale(_h_rho_g[i],  norm);
      }
      scale(_h_mass_u_incl, norm);
      scale(_h_mass_g_incl, norm);
      scale(_h_z_pt,   norm);
      scale(_h_z_mass, norm);
      scale(_h_jet_pt, norm);
    }

  private:

    /// Return the jet-pT bin index, or -1 if out of range.
    int ptBin(double pt) const {
      for (size_t i = 0; i + 1 < _ptedges.size(); ++i) {
        if (pt >= _ptedges[i] && pt < _ptedges[i + 1]) return int(i);
      }
      return -1;
    }

    vector<double> _ptedges;
    vector<string> _ptlabel;

    Histo1DPtr _h_mass_u[3], _h_mass_g[3], _h_rho_u[3], _h_rho_g[3];
    Histo1DPtr _h_mass_u_incl, _h_mass_g_incl;
    Histo1DPtr _h_z_pt, _h_z_mass, _h_jet_pt;
  };

  RIVET_DECLARE_PLUGIN(CMS_ZJET_JETMASS);

}
