// Dedicated Pythia 8.313 MESS + Vincia -> Rivet driver.
//
// MESS returns some successfully generated events with exactly zero nominal
// weight.  Those are merging trial/veto events, not physical events.  They
// must be removed before HepMC/Rivet; otherwise the text ntuples contain many
// zero-weight rows and raw event-count diagnostics are misleading.

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include "Pythia8Plugins/aMCatNLOHooks.h"
#include "Rivet/AnalysisHandler.hh"
#include "HepMC3/GenEvent.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>

using namespace Pythia8;

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "usage: pythia_rivet_mess <cmnd> <out.yoda> <nevents> "
                 "<metrics.txt> [analysis]\n";
    return 1;
  }

  const std::string cmnd = argv[1];
  const std::string yoda = argv[2];
  const int nEvents = std::atoi(argv[3]);
  const std::string metricsPath = argv[4];
  const std::string analysis = (argc > 5) ? argv[5] : "CMS_HADRONIC_JETMASS";

  Pythia pythia;
  pythia.readFile(cmnd);
  pythia.readString("Main:numberOfEvents = " + std::to_string(nEvents));

  // Follow Pythia 8.313 main164 for internal merging (scheme 0 = LO).  MESS
  // is internal CKKW-L merging and must not be combined with the MLM hook.
  if (pythia.word("Merging:Process") == "void") {
    std::cerr << "FATAL: Merging:Process is not configured\n";
    return 2;
  }
  auto mergingHooks = std::make_shared<amcnlo_unitarised_interface>(0);
  pythia.setUserHooksPtr(mergingHooks);

  if (!pythia.init()) return 3;

  // main164 normalization for an external LHE sample: the input cross
  // section is in pb, while Pythia's accumulated sample cross section is mb.
  double inputCrossSectionPb = 0.0;
  for (int iProcess = 0; iProcess < pythia.info.nProcessesLHEF(); ++iProcess)
    inputCrossSectionPb += pythia.info.sigmaLHEF(iProcess);

  HepMC3::Pythia8ToHepMC3 toHepMC;
  Rivet::AnalysisHandler analysisHandler;
  analysisHandler.addAnalysis(analysis);
  analysisHandler.skipMultiWeights(true);

  long nextOk = 0;
  long nextFailed = 0;
  long zeroWeightSkipped = 0;
  long brokenSkipped = 0;
  long analyzed = 0;
  double analyzedSumw = 0.0;
  double analyzedSumw2 = 0.0;
  bool inited = false;

  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    if (!pythia.next()) {
      ++nextFailed;
      if (pythia.info.atEndOfFile()) break;
      continue;
    }
    ++nextOk;

    const double weight = pythia.info.weightValueByIndex();
    if (weight == 0.0) {
      ++zeroWeightSkipped;
      continue;
    }
    if (pythia.event.size() < 3) {
      ++brokenSkipped;
      continue;
    }

    double normalizationMb = inputCrossSectionPb / static_cast<double>(nEvents);
    if (std::abs(pythia.info.lhaStrategy()) == 4)
      normalizationMb = 1.0 / static_cast<double>(nEvents);
    normalizationMb /= MB2PB;
    pythia.info.weightContainerPtr->accumulateXsec(normalizationMb);

    HepMC3::GenEvent genEvent(HepMC3::Units::GEV, HepMC3::Units::MM);
    toHepMC.fill_next_event(pythia, &genEvent);
    if (!inited) {
      analysisHandler.init(genEvent);
      inited = true;
    }
    analysisHandler.analyze(genEvent);
    ++analyzed;
    analyzedSumw += weight;
    analyzedSumw2 += weight * weight;
  }

  if (!inited) {
    std::cerr << "FATAL: no nonzero-weight event reached Rivet\n";
    return 4;
  }

  const double sampleCrossSectionMb =
      pythia.info.weightContainerPtr->getSampleXsec()[0];
  const double sampleCrossSectionErrMb =
      pythia.info.weightContainerPtr->getSampleXsecErr()[0];
  analysisHandler.setCrossSection(
      sampleCrossSectionMb * MB2PB, sampleCrossSectionErrMb * MB2PB);
  analysisHandler.finalize();
  analysisHandler.writeData(yoda);
  pythia.stat();

  std::ofstream metrics(metricsPath);
  metrics << std::setprecision(17)
          << "requested=" << nEvents << '\n'
          << "next_ok=" << nextOk << '\n'
          << "next_failed=" << nextFailed << '\n'
          << "zero_weight_skipped=" << zeroWeightSkipped << '\n'
          << "broken_skipped=" << brokenSkipped << '\n'
          << "analyzed=" << analyzed << '\n'
          << "analyzed_sumw=" << analyzedSumw << '\n'
          << "analyzed_sumw2=" << analyzedSumw2 << '\n'
          << "input_xsec_pb=" << inputCrossSectionPb << '\n'
          << "sample_xsec_pb=" << sampleCrossSectionMb * MB2PB << '\n'
          << "sample_xsec_err_pb=" << sampleCrossSectionErrMb * MB2PB << '\n';
  return metrics.good() ? 0 : 5;
}
