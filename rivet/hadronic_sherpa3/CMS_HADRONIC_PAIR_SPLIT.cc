// -*- C++ -*-
// Particle-level implementation of the adopted hadronic pair_split fiducial.
//
// This analysis intentionally contains no legacy selection path.  In
// particular, there is no fixed dphi > 2 cut, pT-asymmetry cut, or
// trijet-priority veto.

#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/VisibleFinalState.hh"

#include "fastjet/contrib/SoftDrop.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace Rivet {

class CMS_HADRONIC_PAIR_SPLIT : public Analysis {
public:
  RIVET_DEFAULT_ANALYSIS_CTOR(CMS_HADRONIC_PAIR_SPLIT);

  static constexpr double JET_R = 0.8;
  static constexpr double YCUT = 2.4;
  static constexpr double LEAD_PT_MIN = 200.0;
  static constexpr double THIRD_PT_MIN = 185.0;
  static constexpr double PAIR_RECOIL_SCALE = 200.0;
  static constexpr double PI = 3.14159265358979323846;

  void init() {
    VisibleFinalState visible(Cuts::abseta < 5.0);
    FastJets jets(visible,
                  fastjet::JetDefinition(fastjet::antikt_algorithm, JET_R));
    declare(jets, "JetsAK8");

    _ptEdges = {185., 200., 290., 400., 480., 570., 680., 760., 820.,
                1500.};
    _ptLabels = {"185_200", "200_290", "290_400", "400_480",
                 "480_570", "570_680", "680_760", "760_820",
                 "820_1500"};
    const std::vector<double> rhoGEdges = {
        -10., -5., -4., -3.4, -2.85, -2.25, -1.8, -1.5,
        -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.};

    for (const std::string channel : {"dijet", "trijet"}) {
      for (const std::string& ptLabel : _ptLabels) {
        const std::string name = channel + "_rho_g_pt" + ptLabel;
        book(_histograms[name], name, rhoGEdges);
      }
    }

    const char* jetNtuplePath = std::getenv("HAD_PAIR_SPLIT_NTUPLE");
    _writeJetNtuple = jetNtuplePath != nullptr && jetNtuplePath[0] != '\0';
    if (_writeJetNtuple) {
      _dijetNtuple.open(std::string(jetNtuplePath) + "_dijet.txt");
      _trijetNtuple.open(std::string(jetNtuplePath) + "_trijet.txt");
      for (std::ofstream* stream : {&_dijetNtuple, &_trijetNtuple}) {
        *stream << "# event_index jet_index jet_pt jet_y jet_phi m_g rho_g weight\n";
        stream->precision(12);
      }
    }

    const char* eventNtuplePath = std::getenv("HAD_PAIR_SPLIT_EVENT_NTUPLE");
    _writeEventNtuple =
        eventNtuplePath != nullptr && eventNtuplePath[0] != '\0';
    if (_writeEventNtuple) {
      _eventNtuple.open(eventNtuplePath);
      _eventNtuple
          << "# event_index njet pt1 pt2 pt3 y1 y2 y3 phi1 phi2 phi3 "
             "dphi12 dphi13 dphi23 pair_boundary pass_dijet pass_trijet weight\n";
      _eventNtuple.precision(12);
    }
  }

  void analyze(const Event& event) {
    ++_eventIndex;
    const double eventWeight =
        event.weights().size() == 0 ? 1.0 : event.weights()[0];
    _rawSumWeights += eventWeight;
    _rawSumWeightsSquared += eventWeight * eventWeight;
    const Jets jets = apply<FastJets>(event, "JetsAK8")
                          .jetsByPt(Cuts::pT > 30.0 * GeV);
    if (jets.size() < 2) return;

    const Jet& jet1 = jets[0];
    const Jet& jet2 = jets[1];
    const double pt1 = jet1.pT() / GeV;
    const double pt2 = jet2.pT() / GeV;
    const double ptAverage = 0.5 * (pt1 + pt2);
    const double dphi12 = deltaPhi(jet1, jet2);
    const double pairBoundary = PI - PAIR_RECOIL_SCALE / ptAverage;
    const bool dijetPass =
        pt1 > LEAD_PT_MIN && pt2 > LEAD_PT_MIN &&
        std::abs(jet1.rap()) < YCUT && std::abs(jet2.rap()) < YCUT &&
        dphi12 > pairBoundary;

    bool trijetPass = false;
    double pt3 = -1.0;
    double y3 = 99.0;
    double phi3 = 99.0;
    double dphi13 = -1.0;
    double dphi23 = -1.0;
    if (jets.size() >= 3) {
      const Jet& jet3 = jets[2];
      pt3 = jet3.pT() / GeV;
      y3 = jet3.rap();
      phi3 = jet3.phi();
      dphi13 = deltaPhi(jet1, jet3);
      dphi23 = deltaPhi(jet2, jet3);
      const double minPairwiseDphi = std::min({dphi12, dphi13, dphi23});
      trijetPass =
          pt1 > LEAD_PT_MIN && pt2 > LEAD_PT_MIN && pt3 > THIRD_PT_MIN &&
          std::abs(jet1.rap()) < YCUT && std::abs(jet2.rap()) < YCUT &&
          std::abs(jet3.rap()) < YCUT && minPairwiseDphi > 1.0 &&
          dphi12 < pairBoundary;
    }

    if (_writeEventNtuple) {
      _eventNtuple << _eventIndex << ' ' << jets.size() << ' ' << pt1 << ' '
                   << pt2 << ' ' << pt3 << ' ' << jet1.rap() << ' '
                   << jet2.rap() << ' ' << y3 << ' ' << jet1.phi() << ' '
                   << jet2.phi() << ' ' << phi3 << ' ' << dphi12 << ' '
                   << dphi13 << ' ' << dphi23 << ' ' << pairBoundary << ' '
                   << int(dijetPass) << ' ' << int(trijetPass) << ' '
                   << eventWeight << '\n';
    }

    if (dijetPass) {
      fillJet("dijet", jet1, 1, eventWeight, _dijetNtuple);
      fillJet("dijet", jet2, 2, eventWeight, _dijetNtuple);
    }
    if (trijetPass) {
      fillJet("trijet", jets[2], 3, eventWeight, _trijetNtuple);
    }
  }

  void finalize() {
    const double sumWeights = sumOfWeights();
    const double crossSectionPb = crossSection() / picobarn;
    if (sumWeights != 0.0) {
      for (auto& entry : _histograms) {
        scale(entry.second, crossSectionPb / sumWeights);
      }
    }

    if (_writeJetNtuple) {
      for (std::ofstream* stream : {&_dijetNtuple, &_trijetNtuple}) {
        *stream << "# generated_events " << _eventIndex << '\n';
        *stream << "# cross_section_pb " << crossSectionPb << '\n';
        *stream << "# sumw " << sumWeights << '\n';
        *stream << "# raw_sumw " << _rawSumWeights << '\n';
        *stream << "# sumw2 " << _rawSumWeightsSquared << '\n';
        stream->close();
      }
    }
    if (_writeEventNtuple) {
      _eventNtuple << "# generated_events " << _eventIndex << '\n';
      _eventNtuple << "# cross_section_pb " << crossSectionPb << '\n';
      _eventNtuple << "# sumw " << sumWeights << '\n';
      _eventNtuple << "# raw_sumw " << _rawSumWeights << '\n';
      _eventNtuple << "# sumw2 " << _rawSumWeightsSquared << '\n';
      _eventNtuple.close();
    }
  }

private:
  void fillJet(const std::string& channel, const Jet& jet, int jetIndex,
               double eventWeight, std::ofstream& ntuple) {
    const double pt = jet.pT() / GeV;
    const std::vector<fastjet::PseudoJet> constituents =
        jet.pseudojet().constituents();
    if (constituents.empty()) return;

    fastjet::JetDefinition caDefinition(fastjet::cambridge_algorithm, JET_R);
    fastjet::ClusterSequence clusterSequence(constituents, caDefinition);
    const std::vector<fastjet::PseudoJet> caJets =
        fastjet::sorted_by_pt(clusterSequence.inclusive_jets());
    if (caJets.empty()) return;

    fastjet::contrib::SoftDrop softDrop(0.0, 0.1);
    const fastjet::PseudoJet groomedJet = softDrop(caJets.front());
    if (groomedJet == 0 || groomedJet.m() <= 0.0) return;

    const double groomedMass = groomedJet.m();
    const double rhoG = 2.0 * std::log10(groomedMass / (pt * JET_R));
    const int ptIndex = ptBin(pt);
    if (ptIndex >= 0) {
      _histograms[channel + "_rho_g_pt" + _ptLabels[ptIndex]]->fill(
          rhoG, eventWeight);
    }

    if (_writeJetNtuple) {
      ntuple << _eventIndex << ' ' << jetIndex << ' ' << pt << ' '
             << jet.rap() << ' ' << jet.phi() << ' ' << groomedMass << ' '
             << rhoG << ' ' << eventWeight << '\n';
    }
  }

  int ptBin(double pt) const {
    for (size_t index = 0; index + 1 < _ptEdges.size(); ++index) {
      if (pt >= _ptEdges[index] && pt < _ptEdges[index + 1]) {
        return static_cast<int>(index);
      }
    }
    return -1;
  }

  std::vector<double> _ptEdges;
  std::vector<std::string> _ptLabels;
  std::map<std::string, Histo1DPtr> _histograms;
  long long _eventIndex = 0;
  double _rawSumWeights = 0.0;
  double _rawSumWeightsSquared = 0.0;

  bool _writeJetNtuple = false;
  bool _writeEventNtuple = false;
  std::ofstream _dijetNtuple;
  std::ofstream _trijetNtuple;
  std::ofstream _eventNtuple;
};

RIVET_DECLARE_PLUGIN(CMS_HADRONIC_PAIR_SPLIT);

}  // namespace Rivet
