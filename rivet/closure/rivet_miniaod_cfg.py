"""CMSSW config: run the SMP-25-010 Rivet routine on a FullSim MiniAOD file.

This is the Rivet half of the gen-level closure test. It reads the packed gen
particles out of MiniAOD, converts them to HepMC, and runs the analysis plugin.
The coffea half runs zjet_processor.py over the CHILD NanoAOD of the same
MiniAOD file, so both sides see the identical event stream.

Usage (LPC or lxplus, inside a CMSSW area whose Rivet the plugin was built
against -- see closure/README.md):

    cmsRun rivet_miniaod_cfg.py \
        inputFiles=/store/mc/RunIISummer20UL18MiniAODv2/DYJetsToLL.../file.root \
        outputFile=closure_rivet.yoda maxEvents=-1

The plugin .so must be on RIVET_ANALYSIS_PATH.
"""

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

opts = VarParsing("analysis")
opts.register("analysisName", "CMS_2026_PAS_SMP_25_010",
              VarParsing.multiplicity.singleton, VarParsing.varType.string,
              "Rivet analysis to run")
opts.register("crossSection", 1.0,
              VarParsing.multiplicity.singleton, VarParsing.varType.float,
              "Cross section [pb]; only rescales the absolute histograms, the "
              "normalized per-slice distributions used for the closure are "
              "insensitive to it")
# NOT VarParsing's own outputFile: in "analysis" mode it rewrites the name
# (appends _numEventN and forces a .root extension), which breaks YODA's
# format-from-extension detection and kills the job in endRun.
opts.register("yodaFile", "closure_rivet.yoda",
              VarParsing.multiplicity.singleton, VarParsing.varType.string,
              "Output YODA file")
opts.setDefault("maxEvents", -1)
opts.parseArguments()

process = cms.Process("runRivetAnalysis")

process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(opts.maxEvents))
process.source = cms.Source("PoolSource",
                            fileNames=cms.untracked.vstring(opts.inputFiles))

# MiniAOD keeps prunedGenParticles + packedGenParticles; mergedGenParticles
# stitches them back into one collection with the full final state, which is
# what the jet clustering in the routine needs.
process.load("GeneratorInterface.RivetInterface.mergedGenParticles_cfi")
process.load("GeneratorInterface.RivetInterface.genParticles2HepMC_cfi")
process.genParticles2HepMC.genParticles = cms.InputTag("mergedGenParticles")

process.load("GeneratorInterface.RivetInterface.rivetAnalyzer_cfi")
process.rivetAnalyzer.AnalysisNames = cms.vstring(opts.analysisName)
process.rivetAnalyzer.HepMCCollection = cms.InputTag("genParticles2HepMC:unsmeared")
process.rivetAnalyzer.OutputFile = cms.string(opts.yodaFile)
process.rivetAnalyzer.CrossSection = cms.double(opts.crossSection)
# The DY sample carries LHE and GEN weights; the closure compares shapes filled
# with the same weight the processor uses (genWeight), so keep GEN weights only.
process.rivetAnalyzer.useLHEweights = cms.bool(False)
process.rivetAnalyzer.useGENweights = cms.bool(True)

process.p = cms.Path(process.mergedGenParticles *
                     process.genParticles2HepMC *
                     process.rivetAnalyzer)
