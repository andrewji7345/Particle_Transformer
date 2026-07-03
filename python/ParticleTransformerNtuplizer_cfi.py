import FWCore.ParameterSet.Config as cms

particleTransformerNtuplizer = cms.EDAnalyzer(
    "ParticleTransformerNtuplizer",

    packedPFCandidates = cms.InputTag("packedPFCandidates"),

    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),

    ak4Jets = cms.InputTag("slimmedJets"),

    ak8Jets = cms.InputTag("slimmedJetsAK8"),

    met = cms.InputTag("slimmedMETs"),

    rho = cms.InputTag("fixedGridRhoFastjetAll"),

    genParticles = cms.InputTag("prunedGenParticles"),

    genJets = cms.InputTag("slimmedGenJets"),

    genAK8Jets = cms.InputTag("slimmedGenJetsAK8"),

    isMC = cms.bool(True)
)