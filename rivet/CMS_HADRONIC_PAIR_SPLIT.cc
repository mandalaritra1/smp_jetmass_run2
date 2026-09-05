// -*- C++ -*-
//
// Particle-level implementation of the adopted hadronic pair_split fiducial.
// This analysis is intentionally separate from CMS_HADRONIC_JETMASS: the
// latter is the frozen legacy selection used by the 2026-08-02 campaigns.
//
// AK8 jets are anti-kT R=0.8 jets clustered from visible final-state
// particles. The two channels are defined on the same pT-ordered jet list:
//
//   dijet:  pT(j1,j2)>200 GeV, |y(j1,j2)|<2.4, and
//           dphi12 > pi - 200 GeV / ((pT1+pT2)/2). Both jets are measured.
//
//   trijet: pT(j1,j2)>200 GeV, pT(j3)>185 GeV, |y(j1,j2,j3)|<2.4,
//           min(dphi12,dphi13,dphi23)>1, and the exact pair complement
//           dphi12 < pi - 200 GeV / ((pT1+pT2)/2). Jet 3 is measured.
//
// There is no pT-asymmetry cut and no legacy trijet-priority veto. Soft-drop
// uses C/A reclustering with beta=0 and zcut=0.1. No groomed-mass floor or
// event reweighting is applied.

#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/VisibleFinalState.hh"

#include "fastjet/contrib/SoftDrop.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>

namespace Rivet {

  class CMS_HADRONIC_PAIR_SPLIT : public Analysis {
  public:

    RIVET_DEFAULT_ANALYSIS_CTOR(CMS_HADRONIC_PAIR_SPLIT);

    static constexpr double JET_R = 0.8;
    static constexpr double YCUT = 2.4;
    static constexpr double LEAD_PT_FLOOR_GEV = 200.0;
    static constexpr double THIRD_PT_FLOOR_GEV = 185.0;
    static constexpr double RECOIL_SCALE_GEV = 200.0;

    void init() {
      VisibleFinalState visible(Cuts::abseta < 5.0);
      FastJets jets(
        visible, fastjet::JetDefinition(fastjet::antikt_algorithm, JET_R));
      declare(jets, "JetsAK8");

      _ptedges = {200., 290., 400., 480., 570., 680., 760., 820., 13000.};
      _ptlabel = {"200_290", "290_400", "400_480", "480_570",
                  "570_680", "680_760", "760_820", "820_Inf"};

      const vector<double> rho_u_edges =
        {-10., -8., -7., -6., -5., -4.5, -4., -3.5, -3., -2.5,
         -2., -1.5, -1., -0.5, 0.};
      const vector<double> rho_g_edges =
        {-10., -5., -4., -3.4, -2.85, -2.25, -1.8, -1.5,
         -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.};
      const vector<double> mass_edges =
        {0., 10., 20., 30., 50., 70., 90., 110., 130., 150.,
         170., 200., 300., 500., 13000.};

      for (const string channel : {"dijet", "trijet"}) {
        for (size_t index = 0; index < _ptlabel.size(); ++index) {
          const string suffix = "_pt" + _ptlabel[index];
          book(_h[channel + "_mass_u" + suffix],
               channel + "_mass_u" + suffix, mass_edges);
          book(_h[channel + "_mass_g" + suffix],
               channel + "_mass_g" + suffix, mass_edges);
          book(_h[channel + "_rho_u" + suffix],
               channel + "_rho_u" + suffix, rho_u_edges);
          book(_h[channel + "_rho_g" + suffix],
               channel + "_rho_g" + suffix, rho_g_edges);
        }
        book(_h[channel + "_jet_pt"], channel + "_jet_pt", 60, 185., 1500.);
      }

      const char* ntuple_path = std::getenv("HAD_NTUPLE");
      _do_ntuple = ntuple_path != nullptr && ntuple_path[0] != '\0';
      if (_do_ntuple) {
        _nt_dijet.open(string(ntuple_path) + "_dijet.txt");
        _nt_trijet.open(string(ntuple_path) + "_trijet.txt");
        for (auto* ntuple : {&_nt_dijet, &_nt_trijet}) {
          (*ntuple) << "# jet_pt m_u m_g rho_u rho_g weight\n";
          ntuple->precision(8);
        }
      }

      const char* metrics_path = std::getenv("HAD_SELECTION_METRICS");
      if (metrics_path != nullptr) _selection_metrics_path = metrics_path;
    }

    void analyze(const Event& event) {
      ++_events_seen;
      const double weight = event.weights().size() > 0 ? event.weights()[0] : 1.0;

      const Jets jets = apply<FastJets>(event, "JetsAK8")
                          .jetsByPt(Cuts::pT > 30*GeV);
      if (jets.size() < 2) {
        ++_neither_events;
        return;
      }

      const Jet& jet1 = jets[0];
      const Jet& jet2 = jets[1];
      const double pt1 = jet1.pT()/GeV;
      const double pt2 = jet2.pT()/GeV;
      const double pt_average = 0.5 * (pt1 + pt2);
      const double dphi12 = deltaPhi(jet1, jet2);
      const double pair_boundary = M_PI - RECOIL_SCALE_GEV / pt_average;

      const bool leading_pt =
        pt1 > LEAD_PT_FLOOR_GEV && pt2 > LEAD_PT_FLOOR_GEV;
      const bool leading_rapidity =
        std::abs(jet1.rap()) < YCUT && std::abs(jet2.rap()) < YCUT;
      const bool dijet_selected =
        leading_pt && leading_rapidity && dphi12 > pair_boundary;

      bool trijet_selected = false;
      if (jets.size() >= 3) {
        const Jet& jet3 = jets[2];
        const double min_dphi = std::min(
          {dphi12, deltaPhi(jet1, jet3), deltaPhi(jet2, jet3)});
        const bool trijet_rapidity =
          leading_rapidity && std::abs(jet3.rap()) < YCUT;
        trijet_selected =
          leading_pt && jet3.pT()/GeV > THIRD_PT_FLOOR_GEV &&
          trijet_rapidity && min_dphi > 1.0 && dphi12 < pair_boundary;
      }

      if (dijet_selected) {
        ++_dijet_events;
        _dijet_sumw += weight;
        _dijet_sumw2 += weight * weight;
        fillJet("dijet", jet1, weight, _nt_dijet);
        fillJet("dijet", jet2, weight, _nt_dijet);
      }
      if (trijet_selected) {
        ++_trijet_events;
        _trijet_sumw += weight;
        _trijet_sumw2 += weight * weight;
        fillJet("trijet", jets[2], weight, _nt_trijet);
      }
      if (dijet_selected && trijet_selected) ++_overlap_events;
      if (!dijet_selected && !trijet_selected) ++_neither_events;
    }

    void finalize() {
      const double normalization = crossSection() / sumOfWeights();
      for (auto& item : _h) scale(item.second, normalization);

      if (_do_ntuple) {
        for (auto* ntuple : {&_nt_dijet, &_nt_trijet}) {
          (*ntuple) << "# xsec_pb " << crossSection() << '\n';
          (*ntuple) << "# sumw " << sumOfWeights() << '\n';
          ntuple->close();
        }
      }

      if (!_selection_metrics_path.empty()) {
        std::ofstream metrics(_selection_metrics_path);
        metrics.precision(17);
        metrics << "events_seen=" << _events_seen << '\n'
                << "dijet_selected_events=" << _dijet_events << '\n'
                << "trijet_selected_events=" << _trijet_events << '\n'
                << "overlap_events=" << _overlap_events << '\n'
                << "neither_selected_events=" << _neither_events << '\n'
                << "dijet_selected_sumw=" << _dijet_sumw << '\n'
                << "dijet_selected_sumw2=" << _dijet_sumw2 << '\n'
                << "trijet_selected_sumw=" << _trijet_sumw << '\n'
                << "trijet_selected_sumw2=" << _trijet_sumw2 << '\n';
      }
    }

  private:

    void fillJet(
      const string& channel, const Jet& jet, double weight, std::ofstream& ntuple) {
      const double pt = jet.pT()/GeV;
      const double ungroomed_mass = jet.mass()/GeV;

      double groomed_mass = 0.0;
      const vector<fastjet::PseudoJet> constituents =
        jet.pseudojet().constituents();
      if (!constituents.empty()) {
        fastjet::JetDefinition ca_definition(
          fastjet::cambridge_algorithm, JET_R);
        fastjet::ClusterSequence sequence(constituents, ca_definition);
        const vector<fastjet::PseudoJet> reclustered =
          fastjet::sorted_by_pt(sequence.inclusive_jets());
        if (!reclustered.empty()) {
          fastjet::contrib::SoftDrop soft_drop(0.0, 0.1);
          const fastjet::PseudoJet groomed = soft_drop(reclustered[0]);
          if (groomed != 0) groomed_mass = groomed.m();
        }
      }

      const double rho_u = ungroomed_mass > 0.0
        ? 2.0 * std::log10(ungroomed_mass / (pt * JET_R)) : -99.0;
      const double rho_g = groomed_mass > 0.0
        ? 2.0 * std::log10(groomed_mass / (pt * JET_R)) : -99.0;

      _h[channel + "_jet_pt"]->fill(pt);
      const int pt_bin = ptBin(pt);
      if (pt_bin >= 0) {
        const string suffix = "_pt" + _ptlabel[pt_bin];
        _h[channel + "_mass_u" + suffix]->fill(ungroomed_mass);
        _h[channel + "_mass_g" + suffix]->fill(groomed_mass);
        if (rho_u > -99.0) _h[channel + "_rho_u" + suffix]->fill(rho_u);
        if (rho_g > -99.0) _h[channel + "_rho_g" + suffix]->fill(rho_g);
      }

      if (_do_ntuple) {
        ntuple << pt << ' ' << ungroomed_mass << ' ' << groomed_mass << ' '
               << rho_u << ' ' << rho_g << ' ' << weight << '\n';
      }
    }

    int ptBin(double pt) const {
      for (size_t index = 0; index + 1 < _ptedges.size(); ++index) {
        if (pt >= _ptedges[index] && pt < _ptedges[index + 1])
          return int(index);
      }
      return -1;
    }

    vector<double> _ptedges;
    vector<string> _ptlabel;
    map<string, Histo1DPtr> _h;

    bool _do_ntuple = false;
    string _selection_metrics_path;
    std::ofstream _nt_dijet;
    std::ofstream _nt_trijet;

    long _events_seen = 0;
    long _dijet_events = 0;
    long _trijet_events = 0;
    long _overlap_events = 0;
    long _neither_events = 0;
    double _dijet_sumw = 0.0;
    double _dijet_sumw2 = 0.0;
    double _trijet_sumw = 0.0;
    double _trijet_sumw2 = 0.0;
  };

  RIVET_DECLARE_PLUGIN(CMS_HADRONIC_PAIR_SPLIT);

}
