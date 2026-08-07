// -*- C++ -*-
//
// Rivet routine for the CMS dijet + trijet groomed/ungroomed jet-mass
// measurement (Run 2, smp_jetmass_run2 hadronic channels).
//
// This reproduces the PARTICLE-LEVEL (fiducial) gen selections of the coffea
// DijetProcessor / TrijetProcessor (smp_jetmass_run2/{dijet,trijet}_processor.py,
// genTot_seq), byte-for-byte where the generator level allows:
//
//   AK8 jets: anti-kT R = 0.8 clustered from all visible final-state
//   particles (neutrinos excluded), as CMS slimmedGenJetsAK8.
//
//   DIJET (fills BOTH leading jets):
//     >= 2 AK8 jets; |y| < 2.5 for both leading jets;
//     |dphi(j1,j2)| > 2.0;  pT asymmetry |pT1-pT2|/(pT1+pT2) < 0.3;
//     trijet-priority veto: event is DROPPED if it lies in the trijet
//     measurement phase space (>= 3 jets, |y(j3)| < 2.5, min pairwise dphi
//     of the three leading > 1.0, pT3 > 185 GeV) -- mirrors
//     HadronicProcessorBase._in_trijet_phase_space with the same 185 floor.
//
//   TRIJET (fills the 3rd-leading jet only):
//     >= 3 AK8 jets; |y(j3)| < 2.5;
//     min pairwise dphi of the three leading jets > 1.0.
//     (No pT-asymmetry cut and no veto: trijet has priority.)
//
//   No explicit gen jet-pT cut (as in the processors); the pT axis
//   [185, 200, 290, 400, 480, 570, 680, 760, 820, inf] handles it, with
//   [185, 200) acting as the unreported sink bin.
//
//   Observables: ungroomed mass m_u, soft-drop mass m_g (beta = 0,
//   z_cut = 0.1), rho = 2*log10(m/(pT*R)) with R = 0.8.
//
// NOTE: m_g is the C/A-reclustered SoftDrop mass; the coffea gen path uses
// the SubGenJetAK8 subjet sum (same algorithm/parameters) -- expect the same
// few-% grooming-definition offset documented for CMS_ZJET_JETMASS.
//
// NOTE: the processors count NanoAOD-stored GenJetAK8 (implicit storage pT
// threshold); here the jet list uses pT > 30 GeV. All *measured*
// configurations put the relevant jets well above both thresholds (asym<0.3
// forces pT2 > ~0.54*pT1; the veto and trijet fills carry explicit floors),
// so this only matters below the 185 GeV sink edge.

#include "Rivet/Analysis.hh"
#include "Rivet/Projections/VisibleFinalState.hh"
#include "Rivet/Projections/FastJets.hh"

#include "fastjet/contrib/SoftDrop.hh"

#include <fstream>
#include <cstdlib>

namespace Rivet {

  /// Dijet + trijet groomed and ungroomed jet-mass cross sections
  class CMS_HADRONIC_JETMASS : public Analysis {
  public:

    RIVET_DEFAULT_ANALYSIS_CTOR(CMS_HADRONIC_JETMASS);

    static constexpr double JET_R = 0.8;
    static constexpr double YCUT  = 2.5;
    static constexpr double TRIJET_PT3_FLOOR = 185.;  // veto phase-space floor

    void init() {

      // AK8 gen jets from all visible final-state particles (no neutrinos),
      // exactly as CMS slimmedGenJetsAK8. No lepton cleaning: the hadronic
      // gen selections involve no leptons.
      VisibleFinalState vfs(Cuts::abseta < 5.0);
      FastJets jets(vfs, fastjet::JetDefinition(fastjet::antikt_algorithm, JET_R));
      declare(jets, "JetsAK8");

      // --- Binning (matches hist_utils hadronic gen axes) ----------------
      // Measurement pT bins (the [185,200) sink is not booked; the ntuple
      // keeps every selected jet for offline rebinned/reweight use).
      _ptedges = {200., 290., 400., 480., 570., 680., 760., 820., 13000.};
      _ptlabel = {"200_290", "290_400", "400_480", "480_570",
                  "570_680", "680_760", "760_820", "820_Inf"};

      // Ungroomed rho (mgen_over_pt_axis) and groomed rho (hadronic
      // binning-study gen edges), rho = 2*log10(m/(pT*R)).
      const vector<double> rho_u_edges =
        {-10., -8., -7., -6., -5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0.};
      const vector<double> rho_g_edges =
        {-10., -5., -4., -3.4, -2.85, -2.25, -1.8, -1.5,
         -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.};
      // Gen mass edges (mgen_axis)
      const vector<double> medges =
        {0., 10., 20., 30., 50., 70., 90., 110., 130., 150., 170., 200., 300., 500., 13000.};

      for (const string ch : {"dijet", "trijet"}) {
        for (size_t i = 0; i < _ptlabel.size(); ++i) {
          book(_h[ch + "_mass_u_pt" + _ptlabel[i]], ch + "_mass_u_pt" + _ptlabel[i], medges);
          book(_h[ch + "_mass_g_pt" + _ptlabel[i]], ch + "_mass_g_pt" + _ptlabel[i], medges);
          book(_h[ch + "_rho_u_pt"  + _ptlabel[i]], ch + "_rho_u_pt"  + _ptlabel[i], rho_u_edges);
          book(_h[ch + "_rho_g_pt"  + _ptlabel[i]], ch + "_rho_g_pt"  + _ptlabel[i], rho_g_edges);
        }
        book(_h[ch + "_jet_pt"], ch + "_jet_pt", 60, 185., 1500.);
      }

      // --- Optional per-event gen ntuples --------------------------------
      // If $HAD_NTUPLE is set, dump one row per FILLED JET per channel to
      // $HAD_NTUPLE_dijet.txt / $HAD_NTUPLE_trijet.txt so the shapes can be
      // rebinned / reweighted at ANY granularity offline (the varied/nominal
      // ratio for the model reweight). Off by default.
      const char* ntpath = std::getenv("HAD_NTUPLE");
      _do_ntuple = (ntpath != nullptr && ntpath[0] != '\0');
      if (_do_ntuple) {
        _nt_dijet.open(string(ntpath) + "_dijet.txt");
        _nt_trijet.open(string(ntpath) + "_trijet.txt");
        for (auto* nt : {&_nt_dijet, &_nt_trijet}) {
          (*nt) << "# jet_pt m_u m_g rho_u rho_g weight\n";
          nt->precision(8);
        }
      }
    }

    void analyze(const Event& event) {

      // pT-ordered jet list; modest floor only (see header note).
      const Jets jets = apply<FastJets>(event, "JetsAK8")
                          .jetsByPt(Cuts::pT > 30*GeV);
      if (jets.size() < 2) vetoEvent;

      const double w = (event.weights().size() > 0) ? event.weights()[0] : 1.0;

      // --- Trijet phase space (shared by the trijet selection and the
      // --- dijet veto): >= 3 jets, |y(j3)| < 2.5, min pairwise dphi of the
      // --- three leading > 1.0; the dijet veto additionally floors pT3.
      bool trijet_topo = false;
      double pt3 = -1.;
      if (jets.size() >= 3) {
        const Jet& j1 = jets[0];
        const Jet& j2 = jets[1];
        const Jet& j3 = jets[2];
        const double dphimin = std::min({deltaPhi(j1, j2), deltaPhi(j1, j3),
                                         deltaPhi(j2, j3)});
        trijet_topo = (std::abs(j3.rap()) < YCUT) && (dphimin > 1.0);
        pt3 = j3.pT()/GeV;
      }

      // ----------------------------------------------------------------
      // DIJET
      // ----------------------------------------------------------------
      {
        const Jet& j1 = jets[0];
        const Jet& j2 = jets[1];
        const bool rap_ok  = (std::abs(j1.rap()) < YCUT) && (std::abs(j2.rap()) < YCUT);
        const bool dphi_ok = deltaPhi(j1, j2) > 2.0;
        const bool asym_ok = std::abs(j1.pT() - j2.pT()) / (j1.pT() + j2.pT()) < 0.3;
        const bool veto    = trijet_topo && (pt3 > TRIJET_PT3_FLOOR);
        if (rap_ok && dphi_ok && asym_ok && !veto) {
          for (const Jet* jj : {&j1, &j2}) fillJet("dijet", *jj, w, _nt_dijet);
        }
      }

      // ----------------------------------------------------------------
      // TRIJET (no veto, no asymmetry cut: trijet has priority)
      // ----------------------------------------------------------------
      if (trijet_topo) fillJet("trijet", jets[2], w, _nt_trijet);
    }

    void finalize() {
      const double norm = crossSection() / sumOfWeights();
      for (auto& kv : _h) scale(kv.second, norm);
      if (_do_ntuple) {
        for (auto* nt : {&_nt_dijet, &_nt_trijet}) {
          (*nt) << "# xsec_pb " << crossSection() << '\n';
          (*nt) << "# sumw "    << sumOfWeights()  << '\n';
          nt->close();
        }
      }
    }

  private:

    void fillJet(const string& ch, const Jet& jet, double w, std::ofstream& nt) {
      const double pt  = jet.pT()/GeV;
      const double m_u = jet.mass()/GeV;

      // Soft-drop mass: recluster constituents with Cambridge/Aachen (R=0.8)
      // and apply SoftDrop(beta=0, z_cut=0.1), as in CMS AK8.
      double m_g = 0.0;
      const vector<fastjet::PseudoJet> constits = jet.pseudojet().constituents();
      if (!constits.empty()) {
        fastjet::JetDefinition ca_def(fastjet::cambridge_algorithm, JET_R);
        fastjet::ClusterSequence cs(constits, ca_def);
        const vector<fastjet::PseudoJet> reclustered =
          fastjet::sorted_by_pt(cs.inclusive_jets());
        if (!reclustered.empty()) {
          fastjet::contrib::SoftDrop sd(0.0, 0.1);  // beta, z_cut
          const fastjet::PseudoJet groomed = sd(reclustered[0]);
          if (groomed != 0) m_g = groomed.m();
        }
      }

      const double rho_u = (m_u > 0) ? 2.0 * std::log10(m_u / (pt * JET_R)) : -99.0;
      const double rho_g = (m_g > 0) ? 2.0 * std::log10(m_g / (pt * JET_R)) : -99.0;

      _h[ch + "_jet_pt"]->fill(pt);
      const int ipt = ptBin(pt);
      if (ipt >= 0) {
        const string& lbl = _ptlabel[ipt];
        _h[ch + "_mass_u_pt" + lbl]->fill(m_u);
        _h[ch + "_mass_g_pt" + lbl]->fill(m_g);
        if (rho_u > -99.0) _h[ch + "_rho_u_pt" + lbl]->fill(rho_u);
        if (rho_g > -99.0) _h[ch + "_rho_g_pt" + lbl]->fill(rho_g);
      }

      // every selected jet (no pT floor) so offline rebinning stays free
      if (_do_ntuple)
        nt << pt << ' ' << m_u << ' ' << m_g << ' '
           << rho_u << ' ' << rho_g << ' ' << w << '\n';
    }

    /// Return the jet-pT bin index, or -1 if out of range.
    int ptBin(double pt) const {
      for (size_t i = 0; i + 1 < _ptedges.size(); ++i) {
        if (pt >= _ptedges[i] && pt < _ptedges[i + 1]) return int(i);
      }
      return -1;
    }

    vector<double> _ptedges;
    vector<string> _ptlabel;
    map<string, Histo1DPtr> _h;

    bool _do_ntuple = false;
    std::ofstream _nt_dijet, _nt_trijet;
  };

  RIVET_DECLARE_PLUGIN(CMS_HADRONIC_JETMASS);

}
