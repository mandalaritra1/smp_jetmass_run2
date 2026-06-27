// Minimal Pythia8 -> Rivet driver (for environments without a prebuilt
// pythia8-main144, e.g. the cvmfs LCG views on lxplus/LPC).
//
//   pythia_rivet <cmnd-file> <out.yoda> <nevents> [analysis]
//
// Runs Pythia configured by <cmnd-file>, feeds each event to Rivet via the
// Pythia8Rivet interface, and writes <out.yoda>. The analysis .so must be on
// RIVET_ANALYSIS_PATH. Works for internal hard processes and for showering an
// LHE file (set Beams:frameType=4, Beams:LHEF=... in the cmnd file).
//
// build (inside a sourced LCG view):
//   g++ pythia_rivet.cc -o pythia_rivet \
//       $(pythia8-config --cxxflags --libs) $(rivet-config --cppflags --ldflags --libs)

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/Pythia8Rivet.h"
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
  if (!pythia.init()) return 1;

  Pythia8Rivet rivet(pythia, yoda);
  rivet.addAnalysis(analysis);

  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    if (!pythia.next()) continue;
    rivet(pythia.event);
  }

  rivet.done();
  pythia.stat();
  return 0;
}
