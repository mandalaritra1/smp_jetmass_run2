// Pythia8 -> Rivet driver via HepMC3 + a directly-owned Rivet AnalysisHandler.
//
//   pythia_rivet <cmnd-file> <out.yoda> <nevents> [analysis]
//
// Why not Pythia8Plugins/Pythia8Rivet.h: that wrapper (a) is co-versioned with a
// specific Rivet and mismatches in several LCG views, and (b) forwards ALL LHE
// scale/PDF multiweights to Rivet -> ~50k weighted objects/event from a madgraph
// gridpack LHE (huge yoda, big CPU/RAM). Driving our own AnalysisHandler lets us
// call skipMultiWeights(true) -> nominal weight only.  MLM jet matching is applied
// via CombineMatchingInput when the cmnd sets JetMatching:merge.
//
// build (inside a sourced LCG view, -std last so it wins over pythia8-config):
//   g++ pythia_rivet.cc -o pythia_rivet \
//       $(pythia8-config --cxxflags --libs) $(rivet-config --cppflags --ldflags --libs) -std=c++20

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include "Pythia8Plugins/CombineMatchingInput.h"
#include "Rivet/AnalysisHandler.hh"
#include "HepMC3/GenEvent.h"
#include <string>

using namespace Pythia8;

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: pythia_rivet <cmnd> <out.yoda> <nevents> [analysis]\n";
    return 1;
  }
  const std::string cmnd = argv[1];
  const std::string yoda = argv[2];
  const int nEvents = std::atoi(argv[3]);
  const std::string analysis = (argc > 4) ? argv[4] : "CMS_ZJET_JETMASS";

  Pythia pythia;
  pythia.readFile(cmnd);
  pythia.readString("Main:numberOfEvents = " + std::to_string(nEvents));

  // MLM jet matching: register the matching UserHook (scheme=1 -> MadGraph MLM)
  // before init, only when the cmnd enabled JetMatching:merge.
  CombineMatchingInput combined;
  if (pythia.flag("JetMatching:merge")) combined.setHook(pythia);

  if (!pythia.init()) return 1;

  HepMC3::Pythia8ToHepMC3 toHepMC;
  Rivet::AnalysisHandler ah;
  ah.addAnalysis(analysis);
  ah.skipMultiWeights(true);   // nominal weight only

  bool inited = false;
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    if (!pythia.next()) continue;
    HepMC3::GenEvent ge(HepMC3::Units::GEV, HepMC3::Units::MM);
    toHepMC.fill_next_event(pythia, &ge);
    if (!inited) { ah.init(ge); inited = true; }
    ah.analyze(ge);
  }

  // cross section [pb] (mb->pb = 1e9) so the analysis finalize scaling is sane;
  // the offline stitch uses the ntuple sumw + fixed LO xsec, not this value.
  ah.setCrossSection(pythia.info.sigmaGen() * 1.0e9, pythia.info.sigmaErr() * 1.0e9);
  ah.finalize();
  ah.writeData(yoda);
  pythia.stat();
  return 0;
}
