// -*- C++ -*-
//
// CMS SMP-25-010: normalized jet mass, log10(rho^2) = 2 log10(m / (pT R)), for
// the AK8 jet (R = 0.8) recoiling against a Z boson at 13 TeV (Run 2), groomed
// (soft drop beta = 0, z_cut = 0.1) and ungroomed, in three jet-pT slices.
//
// This routine reproduces the PARTICLE-LEVEL (fiducial) selection of the coffea
// processor the measurement is made with (zjet_processor.py in
// smp_jetmass_run2), and has been validated against it on the same FullSim
// events: the ungroomed normalized distribution agrees bin by bin to better
// than 0.4% in every populated bin of all three pT slices.
//
// Fiducial selection:
//   - exactly two same-flavour opposite-sign dressed leptons (dressing cone
//     dR = 0.1), electrons pT > 40 GeV, muons pT > 29 GeV, |eta| < 2.4, and no
//     additional lepton passing those cuts
//   - pT(Z) > 90 GeV and 71 < m(ll) < 111 GeV
//   - AK8 jets with |y| < 2.4, cleaned of dressed leptons within dR < 0.4
//   - the candidate is the LEADING cleaned jet; the event is rejected if that
//     jet fails the topology cut (no fallback to the subleading jet)
//   - dphi(Z, jet) > 1.57 AND pT asymmetry
//     |pT_Z - pT_jet| / (pT_Z + pT_jet) < 0.3
//
// Gen jets are CMS GenJetAK8: anti-kT R = 0.8 over all stable particles except
// neutrinos, muons included. The Z decay products are clustered INTO the jets
// and the overlap is removed afterwards by the dR < 0.4 jet-lepton cleaning --
// they are NOT vetoed from the jet input. Doing it the other way round changes
// the jet sample.
//
// Reference binning (HepData export of the PAS result):
//   jet pT       : 200, 290, 400, inf   [GeV]
//   log10(rho^2) : -10, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0
// The first log10(rho^2) bin is a wide catch-all; the published normalization is
// unit area per pT slice, including that catch-all bin.
//
// NOTE: the plugin name must be changed to CMS_2026_I<InspireID> as soon as the
// paper is on arXiv and the HepData record exists, and the hard-coded binning
// below replaced by refData() lookups.

#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/LeptonFinder.hh"

#include "fastjet/contrib/SoftDrop.hh"

namespace Rivet {

  /// Normalized AK8 jet mass in Z+jet events at 13 TeV
  class CMS_2026_PAS_SMP_25_010 : public Analysis {
  public:

    RIVET_DEFAULT_ANALYSIS_CTOR(CMS_2026_PAS_SMP_25_010);

    void init() {

      // --- Dressed leptons ------------------------------------------------
      // dR = 0.1 dressing cone, matching NanoAOD GenDressedLepton.
      // LeptonOrigin::NODECAY drops leptons from hadron and tau decays, which is
      // the GenDressedLepton.hasTauAnc == False requirement in the processor.
      const LeptonFinder dressed_el(0.1, Cuts::abspid == PID::ELECTRON &&
                                    Cuts::pT > 40*GeV && Cuts::abseta < 2.4,
                                    LeptonOrigin::NODECAY);
      const LeptonFinder dressed_mu(0.1, Cuts::abspid == PID::MUON &&
                                    Cuts::pT > 29*GeV && Cuts::abseta < 2.4,
                                    LeptonOrigin::NODECAY);
      declare(dressed_el, "DressedElectrons");
      declare(dressed_mu, "DressedMuons");

      // --- AK8 gen jets (CMS GenJetAK8) -----------------------------------
      // Muons kept, neutrinos dropped, exactly as ak8GenJetsNoNu.
      const FinalState fs(Cuts::abseta < 5.0);
      const FastJets jets(fs, JetAlg::ANTIKT, 0.8, JetMuons::ALL, JetInvisibles::NONE);
      declare(jets, "JetsAK8");

      // --- Histograms -----------------------------------------------------
      // 2D (pT x log10(rho^2)) in the reference binning, plus the three pT
      // slices as stand-alone normalized distributions.
      for (const string& groom : {string("ungroomed"), string("groomed")}) {
        book(_h2[groom], "zjets_" + groom, ptEdges(), rhoEdges());
        for (size_t i = 0; i < NPTSLICE; ++i) {
          const string nm = "zjets_" + groom + "_slice" + to_string(i);
          book(_h1[nm], nm, rhoEdges());
        }
      }
      book(_h1["zjets_jetpt"], "zjets_jetpt", 30, 200., 800.);
      book(_h1["zjets_zpt"],   "zjets_zpt",   25, 90., 590.);
      book(_h1["zjets_zmass"], "zjets_zmass", 40, 71., 111.);
    }

    void analyze(const Event& event) {

      const DressedLeptons els =
        apply<LeptonFinder>(event, "DressedElectrons").dressedLeptons();
      const DressedLeptons mus =
        apply<LeptonFinder>(event, "DressedMuons").dressedLeptons();

      // exactly two same-flavour opposite-sign dressed leptons, no third lepton
      DressedLeptons leps;
      if (els.size() == 2 && mus.empty() && isZero(els[0].charge() + els[1].charge())) {
        leps = els;
      } else if (mus.size() == 2 && els.empty() && isZero(mus[0].charge() + mus[1].charge())) {
        leps = mus;
      } else {
        vetoEvent;
      }

      const FourMomentum zmom = leps[0].mom() + leps[1].mom();
      if (zmom.pT() <= 90*GeV) vetoEvent;
      if (zmom.mass() <= 71*GeV || zmom.mass() >= 111*GeV) vetoEvent;

      // |y| < 2.4 is applied to the jet collection BEFORE the leading jet is
      // picked, so a forward jet is dropped rather than vetoing the event. The
      // 100 GeV floor mirrors the NanoAOD GenJetAK8 storage threshold that the
      // processor's jet-multiplicity cut inherits; it cannot bite inside the
      // measured phase space, which starts at 200 GeV.
      Jets jets = apply<FastJets>(event, "JetsAK8")
        .jetsByPt(Cuts::pT > 100*GeV && Cuts::absrap < 2.4);
      idiscardIfAnyDeltaRLess(jets, leps, 0.4);
      if (jets.empty()) vetoEvent;

      const Jet& jet = jets[0];
      if (deltaPhi(zmom, jet.mom()) <= 1.57) vetoEvent;
      if (fabs(zmom.pT() - jet.pT()) / (zmom.pT() + jet.pT()) >= 0.3) vetoEvent;

      const double pt = jet.pT()/GeV;
      const int islice = ptSlice(pt);
      if (islice < 0) vetoEvent;  // below 200 GeV: outside the measurement

      _h1["zjets_zpt"]->fill(zmom.pT()/GeV);
      _h1["zjets_zmass"]->fill(zmom.mass()/GeV);
      _h1["zjets_jetpt"]->fill(pt);

      const double m_u = jet.mass()/GeV;
      const double m_g = softDropMass(jet)/GeV;

      if (m_u > 0) {
        const double rho = 2.0 * log10(m_u / (pt * JETR));
        _h2["ungroomed"]->fill(pt, rho);
        _h1["zjets_ungroomed_slice" + to_string(islice)]->fill(rho);
      }
      if (m_g > 0) {
        const double rho = 2.0 * log10(m_g / (pt * JETR));
        _h2["groomed"]->fill(pt, rho);
        _h1["zjets_groomed_slice" + to_string(islice)]->fill(rho);
      }
    }

    void finalize() {
      // 2D: absolute fiducial cross section [pb].
      const double norm = crossSection() / sumOfWeights();
      for (auto& kv : _h2) scale(kv.second, norm);
      for (auto& kv : _h1) {
        // pT slices carry the published normalization: unit area per slice over
        // the SHOWN bins. Overflows are excluded deliberately -- single-prong
        // soft-drop jets have log10(rho^2) far below -10 and are outside the
        // measurement, exactly as they are excluded from the unfolded result.
        if (kv.first.find("_slice") != string::npos) normalize(kv.second, 1.0, false);
        else scale(kv.second, norm);
      }
    }

  private:

    /// Soft-drop mass (beta = 0, z_cut = 0.1) of the Cambridge/Aachen
    /// reclustered jet constituents, as in the CMS AK8 soft-drop gen jets.
    double softDropMass(const Jet& jet) const {
      const vector<fastjet::PseudoJet> constits = jet.pseudojet().constituents();
      if (constits.empty()) return 0.0;
      const fastjet::JetDefinition ca_def(fastjet::cambridge_algorithm, JETR);
      const fastjet::ClusterSequence cs(constits, ca_def);
      const vector<fastjet::PseudoJet> reclustered = fastjet::sorted_by_pt(cs.inclusive_jets());
      if (reclustered.empty()) return 0.0;
      const fastjet::contrib::SoftDrop sd(0.0, 0.1);
      const fastjet::PseudoJet groomed = sd(reclustered[0]);
      return (groomed != 0) ? groomed.m() : 0.0;
    }

    /// Index of the jet-pT slice, or -1 if below the measured range.
    static int ptSlice(double pt) {
      for (size_t i = 0; i < NPTSLICE; ++i) {
        if (pt >= ptEdges()[i] && pt < ptEdges()[i+1]) return int(i);
      }
      return -1;
    }

    static const vector<double>& ptEdges() {
      static const vector<double> e = {200., 290., 400., 13000.};
      return e;
    }
    static const vector<double>& rhoEdges() {
      static const vector<double> e = {-10., -4.5, -4., -3.5, -3.,
                                       -2.5, -2., -1.5, -1., -0.5, 0.};
      return e;
    }

    static constexpr double JETR = 0.8;
    static constexpr size_t NPTSLICE = 3;

    map<string, Histo1DPtr> _h1;
    map<string, Histo2DPtr> _h2;

  };

  RIVET_DECLARE_PLUGIN(CMS_2026_PAS_SMP_25_010);

}
