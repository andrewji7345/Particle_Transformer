// Record all relevant event information to a root file.

#include <memory>
#include <vector>
#include <unordered_map>
#include <cmath>

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// Data formats
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

// ROOT
#include "TTree.h"

// TFileService
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//--------------------------------------------------
// Helper function: find last copy of a particle
//--------------------------------------------------

const reco::Candidate* findLastCopy(const reco::Candidate* p)
{
  const reco::Candidate* current = p;

  bool foundSameParticle = true;

  while (foundSameParticle) {

    foundSameParticle = false;

    for (unsigned int i=0; i<current->numberOfDaughters(); i++) {

      const reco::Candidate* d = current->daughter(i);

      if (d->pdgId() == current->pdgId()) {

        current = d;
        foundSameParticle = true;
        break;
      }
    }
  }

  return current;
}

//--------------------------------------------------
// Helper struct: Store 4-vec and boost vec of a particle
//--------------------------------------------------

struct ParticleKinematics {
    float px = 0;
    float py = 0;
    float pz = 0;
    float E = 0;
    float pt = 0;
    float eta = 0;
    float phi = 0;
    float mass = 0;
};

//--------------------------------------------------
// Helper function: fill 4-vec and boost vec of a particle
//--------------------------------------------------

void fillParticle(const reco::Candidate* src,
                  ParticleKinematics& dst)
{
    if (!src) return;

    dst.px   = src->px();
    dst.py   = src->py();
    dst.pz   = src->pz();
    dst.E    = src->energy();

    dst.pt   = src->pt();
    dst.eta  = src->eta();
    dst.phi  = src->phi();
    dst.mass = src->mass();
}

//--------------------------------------------------
// Helper function: branch particle 4-vec and boost vec in ttree
//--------------------------------------------------

void branchParticle(TTree* tree,
                    const std::string& name,
                    ParticleKinematics& p)
{
    tree->Branch((name+"_px").c_str(),   &p.px);
    tree->Branch((name+"_py").c_str(),   &p.py);
    tree->Branch((name+"_pz").c_str(),   &p.pz);
    tree->Branch((name+"_E").c_str(),    &p.E);

    tree->Branch((name+"_pt").c_str(),   &p.pt);
    tree->Branch((name+"_eta").c_str(),  &p.eta);
    tree->Branch((name+"_phi").c_str(),  &p.phi);
    tree->Branch((name+"_mass").c_str(), &p.mass);
}

//--------------------------------------------------
// Helper function: recursively find the truth label for a genParticle's origin
//--------------------------------------------------

int getTruthLabel(const reco::Candidate* p,
                  const reco::Candidate* gen_chi_ptr[2],

                  const reco::Candidate* gen_W_q0_ptr[2],
                  const reco::Candidate* gen_W_q1_ptr[2],
                  const reco::Candidate* gen_b_ptr[2],
                  const reco::Candidate* gen_Z_q0_ptr[2],
                  const reco::Candidate* gen_Z_q1_ptr[2],
                  const reco::Candidate* gen_h_q0_ptr[2],
                  const reco::Candidate* gen_h_q1_ptr[2],
                  const reco::Candidate* gen_t_W_q0_ptr[2],
                  const reco::Candidate* gen_t_W_q1_ptr[2],
                  const reco::Candidate* gen_t_b_ptr[2])
{
    if (!p)
        return -1;

    // Check both chi decay chains
    for (int i = 0; i < 2; ++i) {

        int offset = 11 * i;

        if (p == gen_W_q0_ptr[i])   return offset + 1;
        if (p == gen_W_q1_ptr[i])   return offset + 2;
        if (p == gen_b_ptr[i])      return offset + 3;
        if (p == gen_Z_q0_ptr[i])   return offset + 4;
        if (p == gen_Z_q1_ptr[i])   return offset + 5;
        if (p == gen_h_q0_ptr[i])   return offset + 6;
        if (p == gen_h_q1_ptr[i])   return offset + 7;
        if (p == gen_t_W_q0_ptr[i]) return offset + 8;
        if (p == gen_t_W_q1_ptr[i]) return offset + 9;
        if (p == gen_t_b_ptr[i])    return offset + 10;
        if (p == gen_chi_ptr[i])    return offset + 11; // FSR

    }

    // Reached beam / no more ancestry
    if (p->numberOfMothers() == 0)
        return 0;

    // Recursively search all mothers
    for (size_t i = 0; i < p->numberOfMothers(); ++i) {

        int label = getTruthLabel(
            p->mother(i),

            gen_chi_ptr,

            gen_W_q0_ptr,
            gen_W_q1_ptr,
            gen_b_ptr,
            gen_Z_q0_ptr,
            gen_Z_q1_ptr,
            gen_h_q0_ptr,
            gen_h_q1_ptr,
            gen_t_W_q0_ptr,
            gen_t_W_q1_ptr,
            gen_t_b_ptr
        );

        // Found a valid ancestry match
        if (label != 0)
            return label;
    }

    // No ancestry path matched any target particle
    return 0;
}

//--------------------------------------------------
// Helper function: calculate delta phi
//--------------------------------------------------

float delta_Phi(float phi1, float phi2) {
  float dphi = phi1 - phi2;
  while (dphi > M_PI) dphi -= 2*M_PI;
  while (dphi <= -M_PI) dphi += 2*M_PI;
  return dphi;
}

//--------------------------------------------------
// Helper function: wrap phi bins
//--------------------------------------------------

int wrapPhiBin(int bin, int nphi) {
    bin %= nphi;
    if (bin < 0)
      bin += nphi;
    return bin;
}

//--------------------------------------------------
// Helper struct: describe cell, with custom operator '=='
//--------------------------------------------------

struct Cell {
  int ieta;
  int iphi;

  bool operator==(const Cell &other) const {
    return ieta == other.ieta && iphi == other.iphi;
  }
};

//--------------------------------------------------
// Helper struct: hash for unordered map of cells
//--------------------------------------------------

struct CellHash {
  std::size_t operator()(const Cell &c) const {
    return std::hash<int>()(c.ieta) ^ (std::hash<int>()(c.iphi) << 1);
  }
};

//--------------------------------------------------
// Ntuplizer class declaration
//--------------------------------------------------

class ParticleTransformerNtuplizer: public edm::one::EDAnalyzer<edm::one::SharedResources>{
  public:
    explicit ParticleTransformerNtuplizer(const edm::ParameterSet&);
    ~ParticleTransformerNtuplizer() override {}

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void beginJob() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  //------------------------------------
  // Tokens
  //------------------------------------

  edm::EDGetTokenT<pat::PackedCandidateCollection> packedPFToken_;
  edm::EDGetTokenT<pat::JetCollection> ak4Token_;
  edm::EDGetTokenT<pat::JetCollection> ak8Token_;
  edm::EDGetTokenT<pat::METCollection> metToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  edm::EDGetTokenT<double> rhoToken_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> genToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genAK4JetToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genAK8JetToken_;

  //------------------------------------
  // TTree
  //------------------------------------

  TTree* tree_;
  std::vector<float> particle_pt;
  std::vector<float> particle_eta;
  std::vector<float> particle_phi;
  std::vector<float> particle_energy;
  std::vector<float> particle_mass;
  
  std::vector<float> particle_px;
  std::vector<float> particle_py;
  std::vector<float> particle_pz;
  
  std::vector<int> particle_charge;
  std::vector<int> particle_pdgId;
  
  std::vector<float> particle_dz;
  std::vector<float> particle_dxy;
  
  std::vector<float> particle_puppiWeight;
  std::vector<int> particle_fromPV;
  std::vector<int> particle_pvAssociationQuality;

  std::vector<float> particle_vx;
  std::vector<float> particle_vy;
  std::vector<float> particle_vz;

  std::vector<int> particle_ak4Index;
  std::vector<int> particle_ak8Index;

  std::vector<float> ak8_pt;
  std::vector<float> ak8_eta;
  std::vector<float> ak8_phi;
  std::vector<float> ak8_mass;

  std::vector<float> ak8_softdropMass;

  std::vector<float> ak8_tau1;
  std::vector<float> ak8_tau2;
  std::vector<float> ak8_tau3;
  std::vector<float> ak8_tau4;

  std::vector<float> ak4_pt;
  std::vector<float> ak4_eta;
  std::vector<float> ak4_phi;
  std::vector<float> ak4_mass;

  std::vector<float> ak4_jetArea;

  ParticleKinematics gen_Suu;
  ParticleKinematics gen_chi[2];

  ParticleKinematics gen_W[2];
  ParticleKinematics gen_W_q0[2];
  ParticleKinematics gen_W_q1[2];
  ParticleKinematics gen_b[2];

  ParticleKinematics gen_Z[2];
  ParticleKinematics gen_Z_q0[2];
  ParticleKinematics gen_Z_q1[2];

  ParticleKinematics gen_h[2];
  ParticleKinematics gen_h_q0[2];
  ParticleKinematics gen_h_q1[2];

  ParticleKinematics gen_t[2];
  ParticleKinematics gen_t_W[2];
  ParticleKinematics gen_t_W_q0[2];
  ParticleKinematics gen_t_W_q1[2];
  ParticleKinematics gen_t_b[2];

  UInt_t run;
  UInt_t lumi;
  ULong64_t event;
  float HT;
  float MET_pt;
  float MET_phi;
  float rho;
  int nPV;
  unsigned int nParticles;
  float PV_x;
  float PV_y;
  float PV_z;
  bool isMC;

  std::vector<int> particle_match_pdgid;
  std::vector<float> particle_match_dr2;
  std::vector<int> particle_truthLabel;
};

//------------------------------------
// Constructor
//------------------------------------

ParticleTransformerNtuplizer::ParticleTransformerNtuplizer(
    const edm::ParameterSet& iConfig)
{
    packedPFToken_ = consumes<pat::PackedCandidateCollection>(
        iConfig.getParameter<edm::InputTag>("packedPFCandidates"));

    ak4Token_ = consumes<pat::JetCollection>(
        iConfig.getParameter<edm::InputTag>("ak4Jets"));

    ak8Token_ = consumes<pat::JetCollection>(
        iConfig.getParameter<edm::InputTag>("ak8Jets"));

    metToken_ = consumes<pat::METCollection>(
        iConfig.getParameter<edm::InputTag>("met"));

    vertexToken_ = consumes<reco::VertexCollection>(
        iConfig.getParameter<edm::InputTag>("vertices"));

    rhoToken_ = consumes<double>(
        iConfig.getParameter<edm::InputTag>("rho"));

    genToken_ = consumes<reco::GenParticleCollection>(
        iConfig.getParameter<edm::InputTag>("genParticles"));

    genAK4JetToken_ = consumes<reco::GenJetCollection>(
        iConfig.getParameter<edm::InputTag>("genJets"));

    genAK8JetToken_ = consumes<reco::GenJetCollection>(
        iConfig.getParameter<edm::InputTag>("genAK8Jets"));

    isMC = iConfig.getParameter<bool>("isMC");
}

//------------------------------------
// beginJob
//------------------------------------

void ParticleTransformerNtuplizer::beginJob(){
  usesResource("TFileService");

  edm::Service<TFileService> fs;

  tree_ = fs->make<TTree>("Events", "Events");

  // Event information
  tree_->Branch("run",   &run);
  tree_->Branch("lumi",  &lumi);
  tree_->Branch("event", &event);

  tree_->Branch("HT",      &HT);
  tree_->Branch("MET_pt",  &MET_pt);
  tree_->Branch("MET_phi", &MET_phi);
  tree_->Branch("rho",     &rho);
  tree_->Branch("nPV",     &nPV);
  tree_->Branch("PV_x",    &PV_x);
  tree_->Branch("PV_y",    &PV_y);
  tree_->Branch("PV_z",    &PV_z);
  tree_->Branch("nParticles", &nParticles);
  tree_->Branch("isMC",    &isMC);

  // Particle information
  tree_->Branch("particle_pt",     &particle_pt);
  tree_->Branch("particle_eta",    &particle_eta);
  tree_->Branch("particle_phi",    &particle_phi);
  tree_->Branch("particle_energy", &particle_energy);
  tree_->Branch("particle_mass", &particle_mass);
  tree_->Branch("particle_px", &particle_px);
  tree_->Branch("particle_py", &particle_py);
  tree_->Branch("particle_pz", &particle_pz);

  tree_->Branch("particle_charge", &particle_charge);
  tree_->Branch("particle_pdgId",  &particle_pdgId);

  tree_->Branch("particle_dz",  &particle_dz);
  tree_->Branch("particle_dxy", &particle_dxy);

  tree_->Branch("particle_puppiWeight",          &particle_puppiWeight);
  tree_->Branch("particle_fromPV",               &particle_fromPV);
  tree_->Branch("particle_pvAssociationQuality", &particle_pvAssociationQuality);

  tree_->Branch("particle_vx", &particle_vx);
  tree_->Branch("particle_vy", &particle_vy);
  tree_->Branch("particle_vz", &particle_vz);

  tree_->Branch("particle_match_pdgid", &particle_match_pdgid);
  tree_->Branch("particle_match_dr2", &particle_match_dr2);
  tree_->Branch("particle_truthLabel", &particle_truthLabel);

  tree_->Branch("particle_ak4Index", &particle_ak4Index);
  tree_->Branch("particle_ak8Index", &particle_ak8Index);

  // AK8 jets
  tree_->Branch("ak8_pt",   &ak8_pt);
  tree_->Branch("ak8_eta",  &ak8_eta);
  tree_->Branch("ak8_phi",  &ak8_phi);
  tree_->Branch("ak8_mass", &ak8_mass);

  tree_->Branch("ak8_softdropMass", &ak8_softdropMass);

  tree_->Branch("ak8_tau1", &ak8_tau1);
  tree_->Branch("ak8_tau2", &ak8_tau2);
  tree_->Branch("ak8_tau3", &ak8_tau3);
  tree_->Branch("ak8_tau4", &ak8_tau4);

  // AK4 jets
  tree_->Branch("ak4_pt",   &ak4_pt);
  tree_->Branch("ak4_eta",  &ak4_eta);
  tree_->Branch("ak4_phi",  &ak4_phi);
  tree_->Branch("ak4_mass", &ak4_mass);

  tree_->Branch("ak4_jetArea", &ak4_jetArea);

  // Target info
  branchParticle(tree_, "gen_Suu", gen_Suu);

  for (int i = 0; i < 2; ++i) {
    branchParticle(tree_, "gen_chi" + std::to_string(i), gen_chi[i]);

    branchParticle(tree_, "gen_W" + std::to_string(i), gen_W[i]);
    branchParticle(tree_, "gen_W" + std::to_string(i) + "_q0", gen_W_q0[i]);
    branchParticle(tree_, "gen_W" + std::to_string(i) + "_q1", gen_W_q1[i]);
    branchParticle(tree_, "gen_b" + std::to_string(i), gen_b[i]);
    branchParticle(tree_, "gen_Z" + std::to_string(i), gen_Z[i]);
    branchParticle(tree_, "gen_Z" + std::to_string(i) + "_q0", gen_Z_q0[i]);
    branchParticle(tree_, "gen_Z" + std::to_string(i) + "_q1", gen_Z_q1[i]);
    branchParticle(tree_, "gen_h" + std::to_string(i), gen_h[i]);
    branchParticle(tree_, "gen_h" + std::to_string(i) + "_q0", gen_h_q0[i]);
    branchParticle(tree_, "gen_h" + std::to_string(i) + "_q1", gen_h_q1[i]);
    branchParticle(tree_, "gen_t" + std::to_string(i), gen_t[i]);
    branchParticle(tree_, "gen_t" + std::to_string(i) + "_W", gen_t_W[i]);
    branchParticle(tree_, "gen_t" + std::to_string(i) + "_W_q0", gen_t_W_q0[i]);
    branchParticle(tree_, "gen_t" + std::to_string(i) + "_W_q1", gen_t_W_q1[i]);
    branchParticle(tree_, "gen_t" + std::to_string(i) + "_b", gen_t_b[i]);
  }
}

//------------------------------------
// Event loop
//------------------------------------

void ParticleTransformerNtuplizer::analyze(const edm::Event& iEvent,
                          const edm::EventSetup&) {

  // Clear vectors
  particle_pt.clear();
  particle_eta.clear();
  particle_phi.clear();
  particle_energy.clear();
  particle_mass.clear();
  particle_px.clear();
  particle_py.clear();
  particle_pz.clear();
  
  particle_charge.clear();
  particle_pdgId.clear();
  
  particle_dz.clear();
  particle_dxy.clear();
  
  particle_puppiWeight.clear();
  particle_fromPV.clear();
  particle_pvAssociationQuality.clear();

  particle_vx.clear();
  particle_vy.clear();
  particle_vz.clear();

  particle_match_pdgid.clear();
  particle_match_dr2.clear();
  particle_truthLabel.clear();

  particle_ak4Index.clear();
  particle_ak8Index.clear();

  ak8_pt.clear();
  ak8_eta.clear();
  ak8_phi.clear();
  ak8_mass.clear();

  ak8_softdropMass.clear();

  ak8_tau1.clear();
  ak8_tau2.clear();
  ak8_tau3.clear();
  ak8_tau4.clear();

  ak4_pt.clear();
  ak4_eta.clear();
  ak4_phi.clear();
  ak4_mass.clear();

  ak4_jetArea.clear();

  HT = 0;
  rho = 0;
  PV_x = PV_y = PV_z = 0;

  gen_Suu = {};

  for (int i = 0; i < 2; ++i) {
    gen_chi[i] = {};

    gen_W[i] = {};
    gen_W_q0[i] = {};
    gen_W_q1[i] = {};
    gen_b[i] = {};
    gen_Z[i] = {};
    gen_Z_q0[i] = {};
    gen_Z_q1[i] = {};
    gen_h[i] = {};
    gen_h_q0[i] = {};
    gen_h_q1[i] = {};
    gen_t[i] = {};
    gen_t_W[i] = {};
    gen_t_W_q0[i] = {};
    gen_t_W_q1[i] = {};
    gen_t_b[i] = {};
  }
  
  // Retrieve collections

  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vertexToken_, vertices);

  edm::Handle<pat::PackedCandidateCollection> packedPFCands;
  iEvent.getByToken(packedPFToken_, packedPFCands);

  edm::Handle<pat::JetCollection> ak4Jets;
  iEvent.getByToken(ak4Token_, ak4Jets);

  edm::Handle<pat::JetCollection> ak8Jets;
  iEvent.getByToken(ak8Token_, ak8Jets);

  edm::Handle<pat::METCollection> mets;
  iEvent.getByToken(metToken_, mets);

  edm::Handle<double> rhoHandle;
  iEvent.getByToken(rhoToken_, rhoHandle);

  edm::Handle<std::vector<reco::GenParticle>> genParticles;
  iEvent.getByToken(genToken_, genParticles);

  edm::Handle<reco::GenJetCollection> genAK4Jets;
  iEvent.getByToken(genAK4JetToken_, genAK4Jets);

  edm::Handle<reco::GenJetCollection> genAK8Jets;
  iEvent.getByToken(genAK8JetToken_, genAK8Jets);

  // Reserve vector space

  particle_pt.reserve(packedPFCands->size());
  particle_eta.reserve(packedPFCands->size());
  particle_phi.reserve(packedPFCands->size());
  particle_energy.reserve(packedPFCands->size());
  particle_mass.reserve(packedPFCands->size());
  particle_px.reserve(packedPFCands->size());
  particle_py.reserve(packedPFCands->size());
  particle_pz.reserve(packedPFCands->size());
  
  particle_charge.reserve(packedPFCands->size());
  particle_pdgId.reserve(packedPFCands->size());
  
  particle_dz.reserve(packedPFCands->size());
  particle_dxy.reserve(packedPFCands->size());
  
  particle_puppiWeight.reserve(packedPFCands->size());
  particle_fromPV.reserve(packedPFCands->size());
  particle_pvAssociationQuality.reserve(packedPFCands->size());

  particle_vx.reserve(packedPFCands->size());
  particle_vy.reserve(packedPFCands->size());
  particle_vz.reserve(packedPFCands->size());

  particle_ak4Index.reserve(packedPFCands->size());
  particle_ak8Index.reserve(packedPFCands->size());

  ak8_pt.reserve(ak8Jets->size());
  ak8_eta.reserve(ak8Jets->size());
  ak8_phi.reserve(ak8Jets->size());
  ak8_mass.reserve(ak8Jets->size());

  ak8_softdropMass.reserve(ak8Jets->size());

  ak8_tau1.reserve(ak8Jets->size());
  ak8_tau2.reserve(ak8Jets->size());
  ak8_tau3.reserve(ak8Jets->size());
  ak8_tau4.reserve(ak8Jets->size());

  ak4_pt.reserve(ak4Jets->size());
  ak4_eta.reserve(ak4Jets->size());
  ak4_phi.reserve(ak4Jets->size());
  ak4_mass.reserve(ak4Jets->size());

  ak4_jetArea.reserve(ak4Jets->size());

  // Fill event information

  run = iEvent.id().run();
  lumi = iEvent.luminosityBlock();
  event = iEvent.id().event();
  nPV = vertices->size();

  // Fill ak8 jet information
  for (const auto& jet : *ak8Jets) {

    ak8_pt.push_back(jet.pt());
    ak8_eta.push_back(jet.eta());
    ak8_phi.push_back(jet.phi());
    ak8_mass.push_back(jet.mass());

    if (jet.hasUserFloat("ak8PFJetsPuppiSoftDropMass")) {
      ak8_softdropMass.push_back(jet.userFloat("ak8PFJetsPuppiSoftDropMass"));
    }
    else {
      ak8_softdropMass.push_back(-1.);
    }

    ak8_tau1.push_back(jet.userFloat("NjettinessAK8Puppi:tau1"));
    ak8_tau2.push_back(jet.userFloat("NjettinessAK8Puppi:tau2"));
    ak8_tau3.push_back(jet.userFloat("NjettinessAK8Puppi:tau3"));
    ak8_tau4.push_back(jet.userFloat("NjettinessAK8Puppi:tau4"));

    // Only run if want to see all userFloats
    //for (const auto& name : jet.userFloatNames()) std::cout << name << std::endl;
    //for (const auto& pair : jet.getPairDiscri()) std::cout << pair.first << " : " << pair.second << std::endl;
    
  }

  // Determine AK8 jet membership for each PF candidate

  std::vector<int> ak4Membership(packedPFCands->size(), -1);

  for (size_t jetIdx = 0; jetIdx < ak4Jets->size(); ++jetIdx) {

    const auto& jet = (*ak4Jets)[jetIdx];

    for (const auto& daughter : jet.daughterPtrVector()) {

      if (daughter.key() < ak4Membership.size())
          ak4Membership[daughter.key()] = jetIdx;
    }
  }

  // Fill ak4 jet information
  HT = 0;

  for (const auto& jet : *ak4Jets){

    ak4_pt.push_back(jet.pt());
    ak4_eta.push_back(jet.eta());
    ak4_phi.push_back(jet.phi());
    ak4_mass.push_back(jet.mass());

    ak4_jetArea.push_back(jet.jetArea());

    HT += jet.pt();

    // Only run if want to see all userFloats
    //for (const auto& name : jet.userFloatNames()) std::cout << name << std::endl;
    //for (const auto& pair : jet.getPairDiscri()) std::cout << pair.first << std::endl;

  }

  // Determine AK4 jet membership for each PF candidate

  std::vector<int> ak8Membership(packedPFCands->size(), -1);

  for (size_t jetIdx = 0; jetIdx < ak8Jets->size(); ++jetIdx) {

    const auto& jet = (*ak8Jets)[jetIdx];

    for (const auto& daughter : jet.daughterPtrVector()) {

      if (daughter.key() < ak8Membership.size())
          ak8Membership[daughter.key()] = jetIdx;
    }
  }

  // Fill particle information
  for (const auto& p : *packedPFCands) {

    particle_pt.push_back(p.pt());
    particle_eta.push_back(p.eta());
    particle_phi.push_back(p.phi());
    particle_energy.push_back(p.energy());
    particle_mass.push_back(p.mass());
    particle_px.push_back(p.px());
    particle_py.push_back(p.py());
    particle_pz.push_back(p.pz());
  
    particle_charge.push_back(p.charge());
    particle_pdgId.push_back(p.pdgId());
  
    particle_dz.push_back(p.dz());
    particle_dxy.push_back(p.dxy());
  
    particle_puppiWeight.push_back(p.puppiWeight());
    particle_fromPV.push_back(p.fromPV());
    particle_pvAssociationQuality.push_back(p.pvAssociationQuality());

    particle_vx.push_back(p.vx());
    particle_vy.push_back(p.vy());
    particle_vz.push_back(p.vz());
  }

  for (long unsigned int i = 0; i < packedPFCands->size(); ++i) {
    particle_ak4Index.push_back(ak4Membership[i]);
    particle_ak8Index.push_back(ak8Membership[i]);
  }

  nParticles = particle_pt.size();

  // Fill MET information
  if (!mets->empty()) {
    MET_pt = mets->front().pt();
    MET_phi = mets->front().phi();
  } else {
    MET_pt = 0.f;
    MET_phi = 0.f;
  }

  // Fill rho
  rho = *rhoHandle;

  // Fill PV location
  if (!vertices->empty()) {
    PV_x = vertices->front().x();
    PV_y = vertices->front().y();
    PV_z = vertices->front().z();
  } else {
    PV_x = PV_y = PV_z = 0.f;
  }

  // Find and fill target info
  // Loop over genParticles to find Suu
  // Record pointers to 1) record kinematic vars and 2) to match with PF candidates
  const reco::Candidate* gen_Suu_ptr = nullptr;
  const reco::Candidate* gen_chi_ptr[2] = {nullptr, nullptr};

  for (const auto &p : *genParticles) {

    if (std::abs(p.pdgId()) == 9936661) { // The Suu pdgID

      gen_Suu_ptr = findLastCopy(&p);
      fillParticle(gen_Suu_ptr, gen_Suu);

      // Find the correct ordering (by pt) of chis
      if (gen_Suu_ptr->numberOfDaughters() != 2) {
        continue;
      }

      if (std::abs(gen_Suu_ptr->daughter(0)->pdgId()) != 9936661 || std::abs(gen_Suu_ptr->daughter(1)->pdgId()) != 9936661) {
        continue;
      }

      const reco::Candidate* tmp_gen_chi_ptr_0 = findLastCopy(gen_Suu_ptr->daughter(0));
      const reco::Candidate* tmp_gen_chi_ptr_1 = findLastCopy(gen_Suu_ptr->daughter(1));

      if (tmp_gen_chi_ptr_0->pt() > tmp_gen_chi_ptr_1->pt()) {
        gen_chi_ptr[0] = tmp_gen_chi_ptr_0;
        gen_chi_ptr[1] = tmp_gen_chi_ptr_1;
      }

      else {
        gen_chi_ptr[0] = tmp_gen_chi_ptr_0;
        gen_chi_ptr[1] = tmp_gen_chi_ptr_1;
      }  
      
      fillParticle(gen_chi_ptr[0], gen_chi[0]);
      fillParticle(gen_chi_ptr[1], gen_chi[1]);

      break;
    }
  }

  // Loop over genParticles to find decays
  // Record pointers to 1) record kinematic vars and 2) to match with PF candidates
  const reco::Candidate* gen_W_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_W_q0_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_W_q1_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_b_ptr[2] = {nullptr, nullptr};

  const reco::Candidate* gen_Z_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_Z_q0_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_Z_q1_ptr[2] = {nullptr, nullptr};

  const reco::Candidate* gen_h_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_h_q0_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_h_q1_ptr[2] = {nullptr, nullptr};

  const reco::Candidate* gen_t_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_t_W_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_t_W_q0_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_t_W_q1_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_t_b_ptr[2] = {nullptr, nullptr};

  for (int idx = 0; idx < 2; ++idx) { // for each chi

    const reco::Candidate* gen_chi_ptr_temp = gen_chi_ptr[idx];

    if (!gen_chi_ptr_temp) continue;
    
    for (unsigned int i=0; i<gen_chi_ptr_temp->numberOfDaughters(); i++) { // for each chi daughter

      const reco::Candidate* daughter = findLastCopy(gen_chi_ptr_temp->daughter(i));

      // determine if decay is Wb, ht, or Zt
      if (std::abs(daughter->pdgId()) == 24) { // if daughter is W

        gen_W_ptr[idx] = daughter;
        fillParticle(gen_W_ptr[idx], gen_W[idx]);

        if (gen_W_ptr[idx]->numberOfDaughters() != 2) continue;

        if (std::abs(gen_W_ptr[idx]->daughter(0)->pdgId()) > 5 ||
            std::abs(gen_W_ptr[idx]->daughter(1)->pdgId()) > 5) continue;

        gen_W_q0_ptr[idx] = findLastCopy(gen_W_ptr[idx]->daughter(0));
        gen_W_q1_ptr[idx] = findLastCopy(gen_W_ptr[idx]->daughter(1));

        fillParticle(gen_W_q0_ptr[idx], gen_W_q0[idx]);
        fillParticle(gen_W_q1_ptr[idx], gen_W_q1[idx]);

      }

      else if (std::abs(daughter->pdgId()) == 5) { // if daughter is b

        gen_b_ptr[idx] = daughter;
        fillParticle(gen_b_ptr[idx], gen_b[idx]);

      }

      else if (std::abs(daughter->pdgId()) == 23) { // if daughter is Z

        gen_Z_ptr[idx] = daughter;
        fillParticle(gen_Z_ptr[idx], gen_Z[idx]);

        if (gen_Z_ptr[idx]->numberOfDaughters() != 2) continue;

        if (std::abs(gen_Z_ptr[idx]->daughter(0)->pdgId()) > 5 ||
            std::abs(gen_Z_ptr[idx]->daughter(1)->pdgId()) > 5) continue;

        gen_Z_q0_ptr[idx] = findLastCopy(gen_Z_ptr[idx]->daughter(0));
        gen_Z_q1_ptr[idx] = findLastCopy(gen_Z_ptr[idx]->daughter(1));

        fillParticle(gen_Z_q0_ptr[idx], gen_Z_q0[idx]);
        fillParticle(gen_Z_q1_ptr[idx], gen_Z_q1[idx]);

      }

      else if (std::abs(daughter->pdgId()) == 25) { // if daughter is h

        gen_h_ptr[idx] = daughter;
        fillParticle(gen_h_ptr[idx], gen_h[idx]);

        if (gen_h_ptr[idx]->numberOfDaughters() != 2) continue;

        if (std::abs(gen_h_ptr[idx]->daughter(0)->pdgId()) > 5 ||
            std::abs(gen_h_ptr[idx]->daughter(1)->pdgId()) > 5) continue;

        gen_h_q0_ptr[idx] = findLastCopy(gen_h_ptr[idx]->daughter(0));
        gen_h_q1_ptr[idx] = findLastCopy(gen_h_ptr[idx]->daughter(1));

        fillParticle(gen_h_q0_ptr[idx], gen_h_q0[idx]);
        fillParticle(gen_h_q1_ptr[idx], gen_h_q1[idx]);

      }

      else if (std::abs(daughter->pdgId()) == 6) { // if daughter is t

        gen_t_ptr[idx] = daughter;
        fillParticle(gen_t_ptr[idx], gen_t[idx]);

        if (gen_t_ptr[idx]->numberOfDaughters() != 2) continue;

        if (std::abs(gen_t_ptr[idx]->daughter(0)->pdgId()) == 24 && std::abs(gen_t_ptr[idx]->daughter(1)->pdgId()) == 5) {
          gen_t_W_ptr[idx] = findLastCopy(gen_t_ptr[idx]->daughter(0));
          gen_t_b_ptr[idx] = findLastCopy(gen_t_ptr[idx]->daughter(1));
        }

        else if (std::abs(gen_t_ptr[idx]->daughter(0)->pdgId()) == 5 && std::abs(gen_t_ptr[idx]->daughter(1)->pdgId()) == 24) {
          gen_t_W_ptr[idx] = findLastCopy(gen_t_ptr[idx]->daughter(1));
          gen_t_b_ptr[idx] = findLastCopy(gen_t_ptr[idx]->daughter(0));
        }

        else continue;

        fillParticle(gen_t_W_ptr[idx], gen_t_W[idx]);
        fillParticle(gen_t_b_ptr[idx], gen_t_b[idx]);

        if (gen_t_W_ptr[idx]->numberOfDaughters() != 2) continue;

        if (std::abs(gen_t_W_ptr[idx]->daughter(0)->pdgId()) > 5 ||
            std::abs(gen_t_W_ptr[idx]->daughter(1)->pdgId()) > 5) continue;

        gen_t_W_q0_ptr[idx] = findLastCopy(gen_t_W_ptr[idx]->daughter(0));
        gen_t_W_q1_ptr[idx] = findLastCopy(gen_t_W_ptr[idx]->daughter(1));

        fillParticle(gen_t_W_q0_ptr[idx], gen_t_W_q0[idx]);
        fillParticle(gen_t_W_q1_ptr[idx], gen_t_W_q1[idx]);

      }
    } // end loop over chi daughters
  } // end loop over chis

  // Find and fill truth information on each PF particle's ancestry, using delta R matching between PF cands and generator particles.
  // Build spatial hash map taking in eta, phi of PF candidate and outputting list of gen particles within 3x3 cell grid of that region.
  
  // Cell size constants, delta R^2 limit
  constexpr float ETA_BIN = 0.05;
  constexpr float PHI_BIN = 0.05;
  constexpr float DR2_MAX = ETA_BIN * PHI_BIN;
  int nphi = std::ceil(2 * M_PI / PHI_BIN);

  // Build eta-phi map for status=1 gen particles
  std::unordered_map<Cell, std::vector<const reco::GenParticle*>, CellHash> genGrid;

  for (const auto &gen : *genParticles) {

    if (gen.status() != 1) continue;

    float eta = gen.eta();
    float phi = gen.phi();

    Cell c;
    c.ieta = std::floor(eta / ETA_BIN);
    c.iphi = wrapPhiBin(std::floor(phi / PHI_BIN), nphi);

    genGrid[c].push_back(&gen);
  }

  // Loop over PF cands, search nearby bins
  for (size_t ipf = 0; ipf < packedPFCands->size(); ++ipf) {

    const pat::PackedCandidate &pf = (*packedPFCands)[ipf];

    float pf_eta = pf.eta();
    float pf_phi = pf.phi();

    Cell base;
    base.ieta = std::floor(pf_eta / ETA_BIN);
    base.iphi = std::floor(pf_phi / PHI_BIN);

    const reco::GenParticle* bestGen = nullptr;
    float bestDR2 = DR2_MAX;

    // search 3x3 neighborhood
    for (int di = -1; di <= 1; ++di) {
      for (int dj = -1; dj <= 1; ++dj) {

        Cell c;
        c.ieta = base.ieta + di;
        c.iphi = wrapPhiBin(base.iphi + dj, nphi);

        auto it = genGrid.find(c);
        if (it == genGrid.end()) continue;

        for (const auto *gen : it->second) {

          float dEta = pf_eta - gen->eta();
          float dPhi = delta_Phi(pf_phi, gen->phi());
          float dR2  = dEta*dEta + dPhi*dPhi;

          if (dR2 < bestDR2) {
            bestDR2 = dR2;
            bestGen = gen;
          }
        }
      }
    }

    // store result
    if (bestGen) {
      particle_match_pdgid.push_back(bestGen->pdgId());
      particle_match_dr2.push_back(bestDR2);
    } else {
      particle_match_pdgid.push_back(0); // unmatched
      particle_match_dr2.push_back(-1.);
    }

    // recursively find ancestry
    if (bestGen) {
      particle_truthLabel.push_back(getTruthLabel(bestGen,
                                                  gen_chi_ptr,

                                                  gen_W_q0_ptr, 
                                                  gen_W_q1_ptr, 
                                                  gen_b_ptr,
                                                  gen_Z_q0_ptr, 
                                                  gen_Z_q1_ptr,
                                                  gen_h_q0_ptr, 
                                                  gen_h_q1_ptr,
                                                  gen_t_W_q0_ptr, 
                                                  gen_t_W_q1_ptr, 
                                                  gen_t_b_ptr));
    }
    else {
      particle_truthLabel.push_back(-1);
    }

  } // end loop over pf cands
  
  // Fill tree
  assert(particle_pt.size() == particle_truthLabel.size());
  assert(particle_pt.size() == particle_eta.size());
  assert(particle_pt.size() == particle_phi.size());
  assert(particle_pt.size() == particle_energy.size());
  tree_->Fill();
  
} // end analyze()

//------------------------------------
// Parameter descriptions
//------------------------------------

void ParticleTransformerNtuplizer::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>(
    "packedPFCandidates",
    edm::InputTag("packedPFCandidates"));

  desc.add<edm::InputTag>(
    "vertices",
    edm::InputTag("offlineSlimmedPrimaryVertices"));

  desc.add<edm::InputTag>(
    "ak4Jets",
    edm::InputTag("slimmedJets"));

  desc.add<edm::InputTag>(
    "ak8Jets",
    edm::InputTag("slimmedJetsAK8"));

  desc.add<edm::InputTag>(
    "met",
    edm::InputTag("slimmedMETs"));

  desc.add<edm::InputTag>(
    "rho",
    edm::InputTag("fixedGridRhoFastjetAll"));

  desc.add<edm::InputTag>(
    "genParticles",
    edm::InputTag("prunedGenParticles"));

  desc.add<edm::InputTag>(
    "genJets",
    edm::InputTag("slimmedGenJets"));

  desc.add<edm::InputTag>(
    "genAK8Jets",
    edm::InputTag("slimmedGenJetsAK8"));

  desc.add<bool>("isMC", true);

  descriptions.add(
    "ParticleTransformerNtuplizer",
    desc);
}

//------------------------------------
// Define plugin
//------------------------------------

DEFINE_FWK_MODULE(ParticleTransformerNtuplizer);
