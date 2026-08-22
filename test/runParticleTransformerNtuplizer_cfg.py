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
    '',
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    'Output ROOT file'
)

options.register(
    'algorithmMode',
    'slimmed',
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Reconstruction teacher: 'slimmed' (fixed reference) or 'packed' (configurable)"
)

options.register(
    'jetPtCut',
    300.0,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.float,
    'Jet pT threshold'
)

options.register(
    'akRadius',
    0.8,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.float,
    'AK jet clustering radius'
)

options.register(
    'caRadius',
    0.8,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.float,
    'CA jet clustering radius'
)

options.register(
    'cosThrust',
    0.85,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.float,
    'Cosine similarity to assign CA jet to SJ'
)

options.parseArguments()

if options.algorithmMode not in ('slimmed', 'packed'):
    raise ValueError("algorithmMode must be 'slimmed' or 'packed'")

if options.algorithmMode == 'slimmed':
    fixed = (options.jetPtCut, options.akRadius, options.caRadius, options.cosThrust)
    reference = (300.0, 0.8, 0.8, 0.85)
    if fixed != reference:
        raise ValueError(
            "The slimmed reference is fixed to pt300/AK8/CA8/th85. "
            "Use algorithmMode=packed to change reconstruction hyperparameters."
        )

if not options.outputRootFile:
    if options.algorithmMode == 'slimmed':
        filename = 'WbWb_4000_1000_slimmed.root'
    else:
        filename = (
            f'WbWb_4000_1000_packed_pt{int(options.jetPtCut)}'
            f'_ak{int(round(10 * options.akRadius))}'
            f'_ca{int(round(10 * options.caRadius))}'
            f'_th{int(round(100 * options.cosThrust))}.root'
        )
    options.outputRootFile = f'rootfiles_particleTransformer/{filename}'

def readFileList(fname):
    with open(fname) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]

process = cms.Process("NTUPLE")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1) # for realsies
    #input = cms.untracked.int32(1000) # for ntuplizer evaluation
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

process.particleTransformerNtuplizer.algorithmMode = cms.string(options.algorithmMode)
process.particleTransformerNtuplizer.jetPtCut = cms.double(options.jetPtCut)

process.particleTransformerNtuplizer.akRadius = cms.double(options.akRadius)

process.particleTransformerNtuplizer.caRadius = cms.double(options.caRadius)

process.particleTransformerNtuplizer.cosThrust = cms.double(options.cosThrust)

process.p = cms.Path(
    process.particleTransformerNtuplizer
)
