// copyright ############################### #
// This file is part of the Xcoll package.   #
// Copyright (c) CERN, 2026.                 #
// ######################################### #
//
// Thin-gas-target benchmark of the Xcoll proton beam-gas models against
// Geant4 (FTFP_BERT).
//
// A proton beam traverses a slab of a pure-element gas whose column density
// is chosen so that the probability of an inelastic interaction is a few
// percent. For each event the following is written (little-endian float64,
// 9 columns per event) to the output file:
//
//   0  first hadronic process in the target: 0 none, 1 hadElastic,
//      2 protonInelastic
//   1  hadElastic: |t| of the primary [GeV^2]; otherwise NaN
//   2  protonInelastic: momentum of the leading (most energetic) proton
//      among the secondaries [GeV/c], 0 if none; otherwise NaN
//   3  protonInelastic: polar angle of the leading proton with respect to
//      the direction of the primary [rad]; otherwise NaN
//   4  no hadronic interaction: polar angle of the primary at the exit of
//      the target [rad]; otherwise NaN
//   5  no hadronic interaction: energy lost by the primary in the target
//      [GeV]; otherwise NaN
//   6  largest energy lost by the primary in a single step [GeV]
//   7  process of that step: 0 other, 1 hIoni, 2 hBrems, 3 hPairProd,
//      4 CoulombScat, 5 msc
//   8  largest polar deflection of the primary in a single step [rad]
//
// After the first hadronic interaction the primary is stopped, so that
// every event has at most one hadronic interaction. All secondaries are
// killed (their properties are read before).
//
// Usage:
//   beamgas_benchmark Z p_GeV column_density_cm2 n_events seed output.bin
//
// Build (Geant4 11.4.2):
//   mkdir build && cd build && cmake .. && make
// (or, on a system where the conda linker clashes with the system glibc,
//  compile directly with /usr/bin/g++ -B/usr/bin/ $(geant4-config --cflags)
//  beamgas_benchmark.cc $(geant4-config --libs))

#include "G4RunManager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserEventAction.hh"
#include "G4UserStackingAction.hh"
#include "G4VUserActionInitialization.hh"
#include "G4ParticleGun.hh"
#include "G4Proton.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4ProductionCuts.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "FTFP_BERT.hh"
#include "Randomize.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

double gZ = 7;
double gP = 450.*GeV;
double gColumn = 1e23/cm2;
const double gLength = 1.*m;

struct EventRecord {
  double first_had;
  double t;
  double p_lead;
  double theta_lead;
  double theta_final;
  double de_final;
  double de_max;
  double de_max_proc;
  double dtheta_max;
  bool done;
  G4ThreeVector p0;
  double e0;
};

EventRecord gRec;
FILE* gOut = nullptr;

int ProcessCode(const G4String& name)
{
  if (name == "hIoni") return 1;
  if (name == "hBrems") return 2;
  if (name == "hPairProd") return 3;
  if (name == "CoulombScat") return 4;
  if (name == "msc") return 5;
  return 0;
}

class DetectorConstruction : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override
  {
    auto* nist = G4NistManager::Instance();
    G4Material* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    G4Element* element = nist->FindOrBuildElement((G4int)gZ);
    // Pure-element gas with the requested column density over gLength
    double atoms_per_volume = gColumn/gLength;
    double density = atoms_per_volume*element->GetAtomicMassAmu()*amu;
    auto* gas = new G4Material("TargetGas", density, 1, kStateGas,
                               293.*kelvin, 1.*atmosphere);
    gas->AddElement(element, 1);

    auto* world_box = new G4Box("World", 2.*m, 2.*m, 2.*m);
    auto* world_lv = new G4LogicalVolume(world_box, vacuum, "World");
    auto* world_pv = new G4PVPlacement(nullptr, G4ThreeVector(), world_lv,
                                       "World", nullptr, false, 0);
    auto* target_box = new G4Box("Target", 1.*m, 1.*m, 0.5*gLength);
    auto* target_lv = new G4LogicalVolume(target_box, gas, "Target");
    new G4PVPlacement(nullptr, G4ThreeVector(), target_lv, "Target",
                      world_lv, false, 0);

    // Only the delta rays above a few MeV are produced explicitly (the
    // benchmark looks at transfers far above that): a range cut of
    // ~2.5 g/cm^2 corresponds to ~5 MeV electrons
    auto* region = new G4Region("TargetRegion");
    region->AddRootLogicalVolume(target_lv);
    auto* cuts = new G4ProductionCuts();
    cuts->SetProductionCut(2.5*g/cm2/density);
    region->SetProductionCuts(cuts);
    return world_pv;
  }
};

class PrimaryGenerator : public G4VUserPrimaryGeneratorAction {
 public:
  PrimaryGenerator() : fGun(new G4ParticleGun(1))
  {
    const double mp = G4Proton::Proton()->GetPDGMass();
    fGun->SetParticleDefinition(G4Proton::Proton());
    fGun->SetParticleEnergy(std::sqrt(gP*gP + mp*mp) - mp);
    fGun->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    fGun->SetParticlePosition(G4ThreeVector(0, 0, -0.5*gLength - 1.*cm));
  }
  ~PrimaryGenerator() override { delete fGun; }
  void GeneratePrimaries(G4Event* event) override
  {
    fGun->GeneratePrimaryVertex(event);
  }
 private:
  G4ParticleGun* fGun;
};

class EventAction : public G4UserEventAction {
 public:
  void BeginOfEventAction(const G4Event*) override
  {
    const double nan = std::nan("");
    gRec = {0, nan, nan, nan, nan, nan, 0, 0, 0, false,
            G4ThreeVector(0, 0, gP), 0};
    gRec.e0 = std::sqrt(gP*gP + std::pow(G4Proton::Proton()->GetPDGMass(), 2));
  }
  void EndOfEventAction(const G4Event*) override
  {
    double row[9] = {gRec.first_had, gRec.t, gRec.p_lead, gRec.theta_lead,
                     gRec.theta_final, gRec.de_final, gRec.de_max,
                     gRec.de_max_proc, gRec.dtheta_max};
    fwrite(row, sizeof(double), 9, gOut);
  }
};

class SteppingAction : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override
  {
    G4Track* track = step->GetTrack();
    if (track->GetParentID() != 0 || gRec.done) return;
    const G4StepPoint* pre = step->GetPreStepPoint();
    const G4StepPoint* post = step->GetPostStepPoint();
    const bool in_target = pre->GetPhysicalVolume() &&
                           pre->GetPhysicalVolume()->GetName() == "Target";
    if (!in_target) return;

    const G4VProcess* proc = post->GetProcessDefinedStep();
    const G4String name = proc ? proc->GetProcessName() : "";

    if (name == "hadElastic") {
      G4LorentzVector p1(pre->GetMomentum(), pre->GetTotalEnergy());
      G4LorentzVector p2(post->GetMomentum(), post->GetTotalEnergy());
      G4LorentzVector q = p1 - p2;
      gRec.first_had = 1;
      gRec.t = -q.m2()/(GeV*GeV);
      gRec.done = true;
      track->SetTrackStatus(fStopAndKill);
      return;
    }
    if (name == "protonInelastic") {
      gRec.first_had = 2;
      const G4ThreeVector dir = pre->GetMomentumDirection();
      double best_e = -1;
      gRec.p_lead = 0;
      gRec.theta_lead = std::nan("");
      for (const G4Track* sec : *step->GetSecondaryInCurrentStep()) {
        if (sec->GetDefinition() != G4Proton::Proton()) continue;
        if (sec->GetTotalEnergy() > best_e) {
          best_e = sec->GetTotalEnergy();
          gRec.p_lead = sec->GetMomentum().mag()/GeV;
          gRec.theta_lead = sec->GetMomentumDirection().angle(dir);
        }
      }
      gRec.done = true;
      track->SetTrackStatus(fStopAndKill);
      return;
    }

    // Electromagnetic steps
    const double de = pre->GetTotalEnergy() - post->GetTotalEnergy();
    if (de > gRec.de_max*GeV) {
      gRec.de_max = de/GeV;
      gRec.de_max_proc = ProcessCode(name);
    }
    const double dtheta = post->GetMomentumDirection().angle(
        pre->GetMomentumDirection());
    if (dtheta > gRec.dtheta_max) gRec.dtheta_max = dtheta;

    // Leaving the target without hadronic interaction
    if (post->GetStepStatus() == fGeomBoundary) {
      gRec.theta_final = post->GetMomentumDirection().angle(
          G4ThreeVector(0, 0, 1));
      gRec.de_final = (gRec.e0 - post->GetTotalEnergy())/GeV;
      gRec.done = true;
      track->SetTrackStatus(fStopAndKill);
    }
  }
};

class StackingAction : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* track) override
  {
    return track->GetParentID() == 0 ? fUrgent : fKill;
  }
};

class ActionInitialization : public G4VUserActionInitialization {
 public:
  void Build() const override
  {
    SetUserAction(new PrimaryGenerator());
    SetUserAction(new EventAction());
    SetUserAction(new SteppingAction());
    SetUserAction(new StackingAction());
  }
};

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 7) {
    fprintf(stderr, "Usage: %s Z p_GeV column_density_cm2 n_events seed "
                    "output.bin\n", argv[0]);
    return 1;
  }
  gZ = atof(argv[1]);
  gP = atof(argv[2])*GeV;
  gColumn = atof(argv[3])/cm2;
  const long n_events = atol(argv[4]);
  G4Random::setTheSeed(atol(argv[5]));
  gOut = fopen(argv[6], "wb");

  auto* run_manager = new G4RunManager();
  run_manager->SetUserInitialization(new DetectorConstruction());
  auto* physics = new FTFP_BERT(0);
  run_manager->SetUserInitialization(physics);
  run_manager->SetUserInitialization(new ActionInitialization());
  run_manager->Initialize();
  run_manager->BeamOn(n_events);

  fclose(gOut);
  delete run_manager;
  return 0;
}
