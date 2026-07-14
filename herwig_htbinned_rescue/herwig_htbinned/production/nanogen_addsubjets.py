import FWCore.ParameterSet.Config as cms

def addSubGenJetAK8(process):
    """Add SubGenJetAK8 (gen soft-drop subjets: beta=0, zcut=0.1, R=0.8) to NanoGEN.
    Replicates the MiniAOD applySubstructure gen chain
    (genParticlesForJetsNoNu -> ak8GenJetsNoNu -> Constituents -> ak8GenJetsNoNuSoftDrop:SubJets)
    so the coffea processor's groomed-mass path (get_groomed_jet on events.SubGenJetAK8)
    works on NanoGEN. Apply AFTER the NANOGEN step (needs process.nanogenSequence)."""
    from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJets, genParticlesForJetsNoNu
    from RecoJets.JetProducers.ak8GenJets_cfi import ak8GenJets, ak8GenJetsSoftDrop, ak8GenJetsConstituents
    from PhysicsTools.NanoAOD.jets_cff import genSubJetAK8Table

    process.genParticlesForJets      = genParticlesForJets.clone()          # reads 'genParticles'
    process.genParticlesForJetsNoNu  = genParticlesForJetsNoNu.clone()      # drop neutrinos (visible FS)
    process.ak8GenJetsNoNu             = ak8GenJets.clone(src = 'genParticlesForJetsNoNu')
    process.ak8GenJetsNoNuConstituents = ak8GenJetsConstituents.clone(src = 'ak8GenJetsNoNu')
    process.ak8GenJetsNoNuSoftDrop     = ak8GenJetsSoftDrop.clone(
        src = cms.InputTag('ak8GenJetsNoNuConstituents', 'constituents'))  # beta=0,zcut=0.1,R=0.8 defaults

    # the SubGenJetAK8 flat table (from jets_cff), pointed at OUR gen soft-drop subjets
    process.genSubJetAK8Table = genSubJetAK8Table.clone(
        src = cms.InputTag('ak8GenJetsNoNuSoftDrop', 'SubJets'))

    # producers run unscheduled via a Task; the table goes in the sequence
    process.subGenJetAK8Task = cms.Task(
        process.genParticlesForJets, process.genParticlesForJetsNoNu,
        process.ak8GenJetsNoNu, process.ak8GenJetsNoNuConstituents, process.ak8GenJetsNoNuSoftDrop)
    process.nanogenSequence.associate(process.subGenJetAK8Task)
    process.nanogenSequence += process.genSubJetAK8Table

    # --- slim: drop ONLY GenPart (genParticleTable) -- ~11.6 kB/ev (~80% of the file),
    # and it has no extension table so this is safe. (Dropping genJetTable etc. is NOT
    # safe: their flavour EXTENSION tables would then have no main -> output module error.)
    for om in process.outputModules_().values():
        om.outputCommands.append("drop nanoaodFlatTable_genParticleTable_*_*")
    return process
