import FWCore.ParameterSet.Config as cms

# DYJetsToLL_M-50 HT-binned, madgraphMLM + Herwig7-CH3 (MLM/TreeMG5 merging).
# Generator block copied VERBATIM from CMSSW GeneratorInterface/Herwig7Interface/test/
# DYToLL01234Jets_5FS_TuneCH3_13TeV_madgraphMLM_herwig7_cfg.py (CMSSW_10_6_28) -- the
# self-contained merging block that creates LesHouchesHandler+FxFxLHReader+FxFxHandler.
# externalLHEProducer prepended to run the per-HT-bin UL gridpack -> cmsgrid_final.lhe.

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.5/DYJets_HT/DYJets_HT-2500toInf_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)

generator = cms.EDFilter("Herwig7GeneratorFilter",
    configFiles = cms.vstring(),
    crossSection = cms.untracked.double(-1),
    dataLocation = cms.string('${HERWIGPATH:-6}'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    filterEfficiency = cms.untracked.double(1.0),
    generatorModule = cms.string('/Herwig/Generators/EventGenerator'),
    herwig7CH3AlphaS = cms.vstring(
        'cd /Herwig/Shower', 
        'set AlphaQCD:AlphaIn 0.118', 
        'cd /'
    ),
    herwig7CH3MPISettings = cms.vstring(
        'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.4712', 
        'set /Herwig/UnderlyingEvent/MPIHandler:pTmin0 3.04', 
        'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 1.284', 
        'set /Herwig/UnderlyingEvent/MPIHandler:Power 0.1362'
    ),
    herwig7CH3PDF = cms.vstring(
        'cd /Herwig/Partons', 
        'create ThePEG::LHAPDF PDFSet_nnlo ThePEGLHAPDF.so', 
        'set PDFSet_nnlo:PDFName NNPDF31_nnlo_as_0118.LHgrid', 
        'set PDFSet_nnlo:RemnantHandler HadronRemnants', 
        'set /Herwig/Particles/p+:PDF PDFSet_nnlo', 
        'set /Herwig/Particles/pbar-:PDF PDFSet_nnlo', 
        'set /Herwig/Partons/PPExtractor:FirstPDF  PDFSet_nnlo', 
        'set /Herwig/Partons/PPExtractor:SecondPDF PDFSet_nnlo', 
        'set /Herwig/Shower/ShowerHandler:PDFA PDFSet_nnlo', 
        'set /Herwig/Shower/ShowerHandler:PDFB PDFSet_nnlo', 
        'create ThePEG::LHAPDF PDFSet_lo ThePEGLHAPDF.so', 
        'set PDFSet_lo:PDFName NNPDF31_lo_as_0130.LHgrid', 
        'set PDFSet_lo:RemnantHandler HadronRemnants', 
        'set /Herwig/Shower/ShowerHandler:PDFARemnant PDFSet_lo', 
        'set /Herwig/Shower/ShowerHandler:PDFBRemnant PDFSet_lo', 
        'set /Herwig/Partons/MPIExtractor:FirstPDF PDFSet_lo', 
        'set /Herwig/Partons/MPIExtractor:SecondPDF PDFSet_lo', 
        'cd /'
    ),
    herwig7StableParticlesForDetector = cms.vstring(
        'set /Herwig/Decays/DecayHandler:MaxLifeTime 10*mm', 
        'set /Herwig/Decays/DecayHandler:LifeTimeOption Average'
    ),
    hw_mg_merging_settings = cms.vstring(
        'cd /Herwig/EventHandlers', 
        'library HwFxFx.so', 
        'create Herwig::FxFxEventHandler LesHouchesHandler', 
        'set LesHouchesHandler:PartonExtractor /Herwig/Partons/PPExtractor', 
        'set LesHouchesHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler', 
        'set LesHouchesHandler:DecayHandler /Herwig/Decays/DecayHandler', 
        'set LesHouchesHandler:WeightOption VarNegWeight', 
        'set /Herwig/Generators/EventGenerator:EventHandler  /Herwig/EventHandlers/LesHouchesHandler', 
        'create ThePEG::Cuts /Herwig/Cuts/NoCuts', 
        'cd /Herwig/EventHandlers', 
        'create Herwig::FxFxFileReader FxFxLHReader', 
        'insert LesHouchesHandler:FxFxReaders[0] FxFxLHReader', 
        'cd /Herwig/Shower', 
        'library HwFxFxHandler.so', 
        'create Herwig::FxFxHandler FxFxHandler', 
        'set /Herwig/Shower/FxFxHandler:SplittingGenerator /Herwig/Shower/SplittingGenerator', 
        'set /Herwig/Shower/FxFxHandler:KinematicsReconstructor /Herwig/Shower/KinematicsReconstructor', 
        'set /Herwig/Shower/FxFxHandler:PartnerFinder /Herwig/Shower/PartnerFinder', 
        'set /Herwig/EventHandlers/LesHouchesHandler:CascadeHandler /Herwig/Shower/FxFxHandler', 
        'set /Herwig/Partons/PDFSet_nnlo:PDFName NNPDF31_nnlo_as_0118', 
        'set /Herwig/Partons/RemnantDecayer:AllowTop Yes', 
        'set /Herwig/Partons/PDFSet_nnlo:RemnantHandler /Herwig/Partons/HadronRemnants', 
        'set /Herwig/Particles/p+:PDF /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Particles/pbar-:PDF /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Partons/PPExtractor:FirstPDF  /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Partons/PPExtractor:SecondPDF /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Shower/ShowerHandler:PDFA /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Shower/ShowerHandler:PDFB /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/EventHandlers/FxFxLHReader:FileName cmsgrid_final.lhe', 
        'set /Herwig/EventHandlers/FxFxLHReader:WeightWarnings false', 
        'set /Herwig/EventHandlers/FxFxLHReader:AllowedToReOpen No', 
        'set /Herwig/EventHandlers/FxFxLHReader:InitPDFs 0', 
        'set /Herwig/EventHandlers/FxFxLHReader:Cuts /Herwig/Cuts/NoCuts', 
        'set /Herwig/EventHandlers/FxFxLHReader:MomentumTreatment RescaleEnergy', 
        'set /Herwig/EventHandlers/FxFxLHReader:PDFA /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/EventHandlers/FxFxLHReader:PDFB /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Shower/ShowerHandler:MaxPtIsMuF Yes', 
        'set /Herwig/Shower/ShowerHandler:RestrictPhasespace Yes', 
        'set /Herwig/Shower/PartnerFinder:PartnerMethod Random', 
        'set /Herwig/Shower/PartnerFinder:ScaleChoice Partner', 
        'set /Herwig/Shower/KinematicsReconstructor:InitialInitialBoostOption LongTransBoost', 
        'set /Herwig/Shower/KinematicsReconstructor:ReconstructionOption General', 
        'set /Herwig/Shower/KinematicsReconstructor:InitialStateReconOption Rapidity', 
        'set /Herwig/Shower/ShowerHandler:SpinCorrelations Yes', 
        'cd /Herwig/Shower', 
        'set /Herwig/Shower/FxFxHandler:MPIHandler  /Herwig/UnderlyingEvent/MPIHandler', 
        'set /Herwig/Shower/FxFxHandler:RemDecayer  /Herwig/Partons/RemnantDecayer', 
        'set /Herwig/Shower/FxFxHandler:ShowerAlpha  AlphaQCD', 
        'set FxFxHandler:HeavyQVeto Yes', 
        'set FxFxHandler:HardProcessDetection Automatic', 
        'set FxFxHandler:drjmin 0', 
        'cd /Herwig/Shower', 
        'set FxFxHandler:VetoIsTurnedOff VetoingIsOn', 
        'set FxFxHandler:ETClus 20*GeV', 
        'set FxFxHandler:RClus 1.0', 
        'set FxFxHandler:EtaClusMax 10', 
        'set FxFxHandler:RClusFactor 1.5'
    ),
    hw_user_settings = cms.vstring(
        'set FxFxHandler:MergeMode TreeMG5', 
        'set FxFxHandler:njetsmax 4'
    ),
    parameterSets = cms.vstring(
        'herwig7CH3PDF', 
        'herwig7CH3AlphaS', 
        'herwig7CH3MPISettings', 
        'herwig7StableParticlesForDetector', 
        'hw_mg_merging_settings', 
        'hw_user_settings'
    ),
    repository = cms.string('${HERWIGPATH}/HerwigDefaults.rpo'),
    run = cms.string('InterfaceMatchboxTest'),
    runModeList = cms.untracked.string('read,run'),
    seed = cms.untracked.int32(12345)
)
ProductionFilterSequence = cms.Sequence(generator)
