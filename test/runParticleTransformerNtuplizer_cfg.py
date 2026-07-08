import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')

options.register(
    'inputRootFiles',
    'SuuAnalysis/ParticleTransformer/test/signalMCFiles/WbWb_4000_1000.txt', # to test ntuplizer
    #'SuuAnalysis/ParticleTransformer/test/signalMCFiles/WbWb_all.txt', # for realsies
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    'Text file containing list of ROOT files'
)

options.register(
    'outputRootFile',
    'rootfiles_particleTransformer/WbWb_4000_1000.root', # to test ntuplizer
    #'rootfiles_particleTransformer/WbWb_all.root', # for realsies
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    'Output ROOT file'
)

options.parseArguments()

def readFileList(fname):
    with open(fname) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]

process = cms.Process("NTUPLE")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        *readFileList(options.inputRootFiles)
    )
)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputRootFile)
)

process.load("SuuAnalysis.ParticleTransformer.ParticleTransformerNtuplizer_cfi")

process.p = cms.Path(
    process.particleTransformerNtuplizer
)