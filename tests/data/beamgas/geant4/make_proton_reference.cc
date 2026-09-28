// copyright ############################### #
// This file is part of the Xcoll package.   #
// Copyright (c) CERN, 2026.                 #
// ######################################### #
//
// Generator of the Geant4 reference values used by tests/test_beamgas_proton.py.
//
// It calls the Geant4 cross-section and final-state classes that FTFP_BERT
// uses for protons directly (no simulation is run) and writes the results to
// JSON on stdout. The reference file proton_reference.json was produced with
// Geant4 11.4.2 (geant4-11-04-patch-02):
//
//   g++ -O2 -std=c++17 $(geant4-config --cflags) make_proton_reference.cc \
//       $(geant4-config --libs) -o make_proton_reference
//   source <geant4-install>/bin/geant4.sh
//   ./make_proton_reference > proton_reference.json
//
// (On a system where the conda linker clashes with the system glibc, compile
// with /usr/bin/g++ -B/usr/bin/ instead.)

#include "G4NistManager.hh"
#include "G4Proton.hh"
#include "G4Neutron.hh"
#include "G4Deuteron.hh"
#include "G4Triton.hh"
#include "G4He3.hh"
#include "G4Alpha.hh"
#include "G4GenericIon.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4PionPlus.hh"
#include "G4PionMinus.hh"
#include "G4PionZero.hh"
#include "G4Lambda.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4ProcessManager.hh"
#include "G4DynamicParticle.hh"
#include "G4BGGNucleonInelasticXS.hh"
#include "G4BGGNucleonElasticXS.hh"
#include "G4ComponentGGHadronNucleusXsc.hh"
#include "G4HadronNucleonXsc.hh"
#include "G4ChipsProtonElasticXS.hh"
#include "G4WentzelOKandVIxSection.hh"
#include "G4BetheBlochModel.hh"
#include "G4EmParameters.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4Version.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

void print_array(const char* key, const std::vector<double>& v, bool last=false)
{
  printf("  \"%s\": [", key);
  for (size_t i = 0; i < v.size(); ++i) {
    printf("%.17g%s", v[i], (i + 1 < v.size()) ? ", " : "");
  }
  printf("]%s\n", last ? "" : ",");
}

std::vector<double> quantiles(std::vector<double> x, const std::vector<double>& probs)
{
  std::sort(x.begin(), x.end());
  std::vector<double> out;
  for (double pr : probs) {
    double pos = pr*(x.size() - 1);
    size_t i = static_cast<size_t>(pos);
    double f = pos - i;
    out.push_back((i + 1 < x.size()) ? x[i]*(1 - f) + x[i + 1]*f : x.back());
  }
  return out;
}

}  // namespace

int main()
{
  G4Proton::ProtonDefinition();
  G4Neutron::NeutronDefinition();
  G4Deuteron::DeuteronDefinition();
  G4Triton::TritonDefinition();
  G4He3::He3Definition();
  G4Alpha::AlphaDefinition();
  G4GenericIon::GenericIonDefinition();
  G4Electron::ElectronDefinition();
  G4Positron::PositronDefinition();
  G4Gamma::GammaDefinition();
  G4PionPlus::PionPlusDefinition();
  G4PionMinus::PionMinusDefinition();
  G4PionZero::PionZeroDefinition();
  G4Lambda::LambdaDefinition();
  // The CHIPS elastic model needs the ion table to get the nuclear masses
  G4GenericIon::GenericIon()->SetProcessManager(
      new G4ProcessManager(G4GenericIon::GenericIon()));
  G4ParticleTable::GetParticleTable()->SetReadiness();
  G4IonTable::GetIonTable()->CreateAllIon();
  G4Random::setTheSeed(20260928);

  const G4ParticleDefinition* proton = G4Proton::Proton();
  const G4ParticleDefinition* neutron = G4Neutron::Neutron();
  const double mp = proton->GetPDGMass();
  auto* nist = G4NistManager::Instance();

  auto* bggIn = new G4BGGNucleonInelasticXS(proton);
  auto* bggEl = new G4BGGNucleonElasticXS(proton);
  bggIn->BuildPhysicsTable(*proton);
  bggEl->BuildPhysicsTable(*proton);
  auto* gg = new G4ComponentGGHadronNucleusXsc();
  auto* hn = new G4HadronNucleonXsc();
  auto* chips = new G4ChipsProtonElasticXS();

  const int zmax = 92;
  std::vector<double> zs;
  for (int z = 1; z <= zmax; ++z) zs.push_back(z);
  const std::vector<double> plist = {100., 177.827941, 316.227766, 450., 562.341325,
                                     1000., 1778.27941, 3162.27766, 5623.41325,
                                     6800., 7000., 10000.};

  printf("{\n");
  printf("  \"geant4_version\": \"%s\",\n", G4Version.c_str());
  printf("  \"proton_mass_MeV\": %.17g,\n", mp);
  printf("  \"neutron_mass_MeV\": %.17g,\n", neutron->GetPDGMass());
  printf("  \"amu_c2_MeV\": %.17g,\n", amu_c2/MeV);
  print_array("Z", zs);

  // ---- Per-element constants ----------------------------------------------
  std::vector<double> A_bgg, mass_amu, a27, fac_in, fac_el;
  {
    G4DynamicParticle dp(proton, G4ThreeVector(0, 0, 1), 450.*GeV);
    for (int z = 1; z <= zmax; ++z) {
      int A = (z == 1) ? 1 : G4lrint(nist->GetAtomicMassAmu(z));
      A_bgg.push_back(A);
      mass_amu.push_back(nist->GetAtomicMassAmu(z));
      a27.push_back(nist->GetA27(z));
      if (z == 1) {
        fac_in.push_back(1.0115);
        fac_el.push_back(1.0115);
      } else {
        gg->ComputeCrossSections(proton, dp.GetKineticEnergy(), z, A);
        fac_in.push_back(bggIn->GetElementCrossSection(&dp, z, nullptr)
                         /gg->GetInelasticGlauberGribovXsc());
        fac_el.push_back(bggEl->GetElementCrossSection(&dp, z, nullptr)
                         /gg->GetElasticGlauberGribovXsc());
      }
    }
  }
  print_array("A_bgg", A_bgg);
  print_array("nist_mass_amu", mass_amu);
  print_array("nist_a27", a27);
  print_array("glauber_factor_inelastic", fac_in);
  print_array("glauber_factor_elastic", fac_el);

  // ---- Hadronic cross sections on a momentum grid [mb] ---------------------
  print_array("p_GeV", plist);
  std::vector<double> v_in, v_el, v_gtot, v_gin, v_gprod, v_gel;
  std::vector<double> pp_tot, pp_el, pn_tot, pn_el;
  for (double pg : plist) {
    double p = pg*GeV;
    double ekin = std::sqrt(p*p + mp*mp) - mp;
    G4DynamicParticle dp(proton, G4ThreeVector(0, 0, 1), ekin);
    for (int z = 1; z <= zmax; ++z) {
      v_in.push_back(bggIn->GetElementCrossSection(&dp, z, nullptr)/millibarn);
      v_el.push_back(bggEl->GetElementCrossSection(&dp, z, nullptr)/millibarn);
      if (z > 1) {
        gg->ComputeCrossSections(proton, ekin, z, (int)A_bgg[z - 1]);
        v_gtot.push_back(gg->GetTotalGlauberGribovXsc()/millibarn);
        v_gin.push_back(gg->GetInelasticGlauberGribovXsc()/millibarn);
        v_gprod.push_back(gg->GetProductionGlauberGribovXsc()/millibarn);
        v_gel.push_back(gg->GetElasticGlauberGribovXsc()/millibarn);
      } else {
        v_gtot.push_back(0.); v_gin.push_back(0.); v_gprod.push_back(0.); v_gel.push_back(0.);
      }
    }
    hn->HadronNucleonXscNS(proton, proton, ekin);
    pp_tot.push_back(hn->GetTotalHadronNucleonXsc()/millibarn);
    pp_el.push_back(hn->GetElasticHadronNucleonXsc()/millibarn);
    hn->HadronNucleonXscNS(proton, neutron, ekin);
    pn_tot.push_back(hn->GetTotalHadronNucleonXsc()/millibarn);
    pn_el.push_back(hn->GetElasticHadronNucleonXsc()/millibarn);
  }
  print_array("bgg_inelastic_mb", v_in);
  print_array("bgg_elastic_mb", v_el);
  print_array("gg_total_mb", v_gtot);
  print_array("gg_inelastic_mb", v_gin);
  print_array("gg_production_mb", v_gprod);
  print_array("gg_elastic_mb", v_gel);
  print_array("ns_pp_total_mb", pp_tot);
  print_array("ns_pp_elastic_mb", pp_el);
  print_array("ns_pn_total_mb", pn_tot);
  print_array("ns_pn_elastic_mb", pn_el);

  // ---- CHIPS elastic: cross section and |t| quantiles ----------------------
  const std::vector<double> chips_z = {1, 2, 6, 7, 8, 10, 18, 36, 54};
  const std::vector<double> chips_p = {100., 450., 3000., 6800.};
  std::vector<double> probs;
  for (int i = 1; i < 200; ++i) probs.push_back(i*0.005);
  probs.push_back(0.999);
  probs.push_back(0.9999);
  const int nsample = 200000;
  print_array("chips_Z", chips_z);
  print_array("chips_p_GeV", chips_p);
  print_array("chips_probabilities", probs);
  std::vector<double> ch_xs, ch_tmax, ch_q;
  for (double pg : chips_p) {
    for (double zd : chips_z) {
      int z = (int)zd;
      int N = (int)A_bgg[z - 1] - z;
      double p = pg*GeV;
      ch_xs.push_back(chips->GetChipsCrossSection(p, z, N, 2212)/millibarn);
      ch_tmax.push_back(2.*chips->GetHMaxT()/(GeV*GeV));
      std::vector<double> ts;
      ts.reserve(nsample);
      for (int i = 0; i < nsample; ++i) {
        ts.push_back(chips->GetExchangeT(z, N, 2212)/(GeV*GeV));
      }
      for (double q : quantiles(ts, probs)) ch_q.push_back(q);
    }
  }
  print_array("chips_elastic_mb", ch_xs);
  print_array("chips_tmax_GeV2", ch_tmax);
  print_array("chips_t_quantiles_GeV2", ch_q);

  // ---- Single Coulomb scattering off the nucleus (WentzelOKandVI) ----------
  // Majorant cross section between two angles, and the fraction of the
  // majorant accepted by the form-factor and spin rejection of
  // SampleSingleScattering: their product is the effective cross section.
  G4EmParameters::Instance();
  const std::vector<double> ws_z = {1, 2, 7, 18, 54};
  const std::vector<double> ws_p = {450., 6800.};
  const std::vector<double> ws_theta = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2};
  print_array("coulomb_Z", ws_z);
  print_array("coulomb_p_GeV", ws_p);
  print_array("coulomb_theta_edges", ws_theta);
  std::vector<double> ws_major, ws_accept;
  const int nws = 1000000;
  for (double pg : ws_p) {
    for (double zd : ws_z) {
      int z = (int)zd;
      auto* wokvi = new G4WentzelOKandVIxSection(false);
      wokvi->Initialise(proton, -1.0);
      double p = pg*GeV;
      double ekin = std::sqrt(p*p + mp*mp) - mp;
      wokvi->SetupKinematic(ekin, nullptr);
      wokvi->SetupTarget(z, DBL_MAX);
      for (size_t i = 0; i + 1 < ws_theta.size(); ++i) {
        double c1 = std::cos(ws_theta[i]);
        double c2 = std::cos(ws_theta[i + 1]);
        ws_major.push_back(wokvi->ComputeNuclearCrossSection(c1, c2)/millibarn);
        int nacc = 0;
        for (int k = 0; k < nws; ++k) {
          G4ThreeVector& d = wokvi->SampleSingleScattering(c1, c2, 0.0);
          if (d.z() < 1.0) ++nacc;
        }
        ws_accept.push_back(double(nacc)/nws);
      }
      delete wokvi;
    }
  }
  print_array("coulomb_majorant_mb", ws_major);
  print_array("coulomb_acceptance", ws_accept);

  // ---- Knock-on electrons (BetheBloch) ------------------------------------
  auto* bb = new G4BetheBlochModel(proton);
  const std::vector<double> ko_z = {1, 7, 18};
  const std::vector<double> ko_p = {450., 6800.};
  const std::vector<double> ko_v = {1e-6, 1e-4, 1e-3, 1e-2};
  print_array("knock_on_Z", ko_z);
  print_array("knock_on_p_GeV", ko_p);
  print_array("knock_on_energy_fraction_cut", ko_v);
  std::vector<double> ko_xs;
  for (double pg : ko_p) {
    double p = pg*GeV;
    double ekin = std::sqrt(p*p + mp*mp) - mp;
    for (double zd : ko_z) {
      int z = (int)zd;
      double A = nist->GetAtomicMassAmu(z);
      for (double v : ko_v) {
        double cut = v*(ekin + mp);
        ko_xs.push_back(bb->ComputeCrossSectionPerAtom(proton, ekin, z, A, cut, ekin)/millibarn);
      }
    }
  }
  print_array("knock_on_mb", ko_xs, true);
  printf("}\n");
  return 0;
}
