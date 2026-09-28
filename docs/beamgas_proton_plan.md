# Beam-gas Monte Carlo for proton beams: plan

Status: **plan for review, nothing implemented yet.** Branch `beamgas_proton`,
created from `beamgas` at `0832d895`.

Scope: extend the beam-gas Monte Carlo of PR #230 (e±) to LHC protons from
450 GeV to 6.8 TeV, for beam-gas lifetime and loss maps. The e± physics stays
untouched and keeps producing bit-identical results.

All Geant4 references are to the local clone, **Geant4 11.4.2**
(`geant4-11-04-patch-02`, clone at `v11.4.2-19-gcb836de6e6`). All Geant4
numbers quoted below come from small scratch programs linked against the local
install (`/home/gbroggi/software/geant4-install`). They call the Geant4
classes directly; no simulation is run.

---

## 1. Summary of the recommendation

A hybrid approach. Cross sections come from Geant4, as used by FTFP_BERT. The
elastic final state comes from Geant4 CHIPS. Single diffraction comes from the
Everest/K2 formulas, with its normalisation still to be settled. Everything is
written in Python/NumPy next to the e± models, with the same weighted,
forced-interaction scheme.

| # | Process | Cross section | Final state | Main reason |
|---|---------|---------------|-------------|-------------|
| 1 | Absorption (inelastic) | `G4BGGNucleonInelasticXS`, minus QE and SD | proton removed; event written to a source table | Geant4 standard above 91 GeV; BGG σ_inel+σ_el reproduces the LHC DR totals within 3.5% (except He) |
| 2a | Coherent elastic pA / pp elastic on H | `G4BGGNucleonElasticXS` | `G4ChipsElasticModel` (CHIPS dσ/dt), exact two-body kinematics | this is what FTFP_BERT uses; forward slopes agree with Everest within 7% |
| 2b | Quasi-elastic pN | Glauber-Gribov σ_in − σ_prod (same code as 1) — **your call, §7 Q1** | pN elastic (CHIPS pp) on a bound nucleon | consistent with σ_inel; the energy dependence is physical |
| 3 | Single diffraction (target dissociation) | Everest/K2 `N_eff·σ_SD,pp(s)`, H special-cased — **normalisation: your call, §7 Q2** | Everest/K2: dN/dM² ∝ 1/M², slope b(M²), Δp/p = −M²/s | Geant4 has no standalone SD; K2 is the collimation-community standard |
| 4 | Single Coulomb on nucleus | analytic, from the dσ/dt below | screened Rutherford × exponential form factor × spin factor, as in `G4WentzelOKandVIxSection` | reuses the e± Coulomb port; importance sampled |
| 2+4 | Coulomb–nuclear interference | sampled jointly with 2a as \|F_C + F_N\|² — **your call, §7 Q3** | – | a 2–5% effect on the elastic loss rate; cheap within weighted sampling |
| 5 | MCS, mean ionisation | not simulated; analytic diagnostics only | – | emittance growth only (see §4.6) |
| 5' | Knock-on electrons (hard tail of ionisation) | `G4BetheBlochModel` per-electron dσ/dT | exact p–e kinematics (gives both Δp/p and θ) | 9–29% of SD for Δp/p > 1e-3 at injection — **your call, §7 Q4** |
| 5'' | e⁺e⁻ pair production | `G4hPairProductionModel` | Δp/p only | up to 15% of SD (Ar, 6.8 TeV); lowest priority — **§7 Q4** |

---

## 2. What the e± implementation actually is

One correction to the brief: the e± physics is **not in C kernels**. It lives
in Python/NumPy:

- `xcoll/beam_elements/elements_src/beamgas.h` is a no-op. `BeamGasScattering`
  is a passive marker during tracking.
- `xcoll/beamgas/cross_sections.py` holds `ElementData`,
  `BremsstrahlungCalculator` (port of `G4eBremsstrahlungRelModel`,
  `G4ModifiedTsai`) and `CoulombScatteringCalculator` (port of
  `G4ScreeningMottCrossSection`, `G4WentzelVIModel`). Each calculator exposes
  `compute_xsec()` and `sample_deflections(px, py, delta, rng) -> ScatteringSample`.
- `xcoll/beam_elements/beamgas.py`: `BeamGasScattering.scatter()` draws n
  macro-particles from the local matched Gaussian, including the closed orbit
  and dispersion. It picks a gas species with probability ∝ n_i σ_i and calls
  the species calculator. It returns an `xt.Particles` with
  `weight = interaction_rate/n × importance_weight`, `s` and `at_element` set
  to the element, and fills `scatter_log`.
- `xcoll/beamgas/study.py`: `BeamGasStudy` validates the PDG id (±11 only),
  builds one calculator per species for **one process per study**
  (`'brems'` or `'coulomb'`), and integrates the atomic density over the
  section each element represents. `run()` tracks each element's sample from
  the element back to itself for `n_turns`. It counts `state <= 0` as lost,
  and returns `BeamGasResult`: rates, lifetime, MC error, `cutoff_scan`,
  `rate_above_theta_max`, and `interaction_log`.
- Gas input is an `xt.Table` with `s` and one column per **chemical element**,
  holding **atomic** density. Molecules are therefore already handled per
  constituent atom (e.g. `H = 2 n_H2 + 4 n_CH4 + 2 n_H2O`). The e± code
  supports Z = 1–54.
- Tests: `tests/test_beamgas.py`, 90 tests passing in 16 s on `beamgas`.
  Example: `examples/beamgas.py` (toy ring).

Doing the proton sampling in Python is fine too. It happens once per event,
before tracking, and vectorised NumPy handles ~10⁶ events/s. C kernels would
only pay off if interactions happened *during* tracking, which the
forced-interaction scheme avoids. **I propose to stay in Python** (§7 Q9).

---

## 3. Options assessed

### (a) Reusing Everest (K2/SixTrack heritage)

What Everest has (`xcoll/scattering_routines/everest/`):

- `properties.h::calculate_scattering` energy-scales reference cross sections:
  `σ_tot(E) = σ_tot,ref + N_eff·(σ_pp,tot(E) − 40 mb)`, with σ_inel and the
  nuclear slope b_N scaled by σ_tot(E)/σ_tot,ref. It has pp fits (COMPETE
  σ_tot, TOTEM σ_el, SD `4.3 + 0.3 ln s` mb, and
  `b_pp = 7.156 + 1.439 ln √s`). Here `N_eff = 1.618·A^{1/3}`.
- `nuclear_interaction.h` selects the channel (absorption, pN elastic
  e^{−b_N t}, pp elastic e^{−b_pp t}, SD, Rutherford) and applies the
  kinematics.

The pp fits agree well with Geant4. σ_pp,tot is 39.7 vs 40.1 mb at 450 GeV
and 47.2 vs 47.0 mb at 6.8 TeV. σ_pp,el is 7.07 vs 7.19 mb and 8.65 vs
9.12 mb. b_pp is 12.0 vs 12.2 GeV⁻² (CHIPS) and 14.0 vs 14.3 GeV⁻².

What stops direct reuse for gas:

1. **Code coupling.** The functions take `MaterialData`, `EverestData` and
   `LocalParticle` inside the collimator jaw loop (MCS stepping, ionisation).
   They can't be called from the Python beam-gas sampler without a dedicated
   kernel. Only the *formulas* are reusable.
2. **No gas data.** The material database has reference σ_tot, σ_inel, b_N
   and Rutherford values only for Be, Al, Si, Cu, Ge, Mo, W, Pb, C
   (graphite/CFC) and four mixtures. H, He, N, O, Ne and Ar are all missing.
3. **Hydrogen breaks the model.** With `N_eff(A=1) = 1.62` instead of 1, QE
   and SD are overcounted by 62%. The "coherent pN elastic" channel is
   meaningless for H: it should be pp elastic.
4. **Coulomb normalisation looks wrong for gas.** Everest's Rutherford cross
   section is the tabulated `cross_section[5]`, e.g. 7.6·10⁻⁵ b = 0.076 mb for
   C, between |t| = 0.998·10⁻³ and 0.02 GeV². The analytic single-Coulomb
   cross section 4πα²Z²(ħc)²(1/t₁ − 1/t₂) over the same window is **8.9 mb**,
   about 117× larger. Be (×113) and Pb (×174) show the same pattern. I have
   not traced this back to K2. In a thick jaw it barely matters because MCS
   dominates, but in gas it would make the Coulomb channel essentially
   disappear. The fixed |t| > 10⁻³ GeV² cut is also θ > 70 µrad at 450 GeV,
   above the injection loss threshold (~40 µrad, §5).
5. **MCS and ionisation are thick-target models.** Highland with the
   `1 + 0.038 ln(x/X0)` term goes negative for x/X0 ~ 10⁻⁵, and the
   Bethe–Bloch straggling is tuned for thick absorbers. Neither applies to
   single scattering in gas.
6. **SD normalisation.** `σ_SD,pp = 4.3 + 0.3 ln s` mb gives 6.3 mb at
   √s = 29 GeV and 7.1 mb at 113 GeV. That matches the *two-arm* SD values
   (e.g. CDF's 7.9/9.5 mb at 546/1800 GeV follow the same curve). Yet K2 uses
   it for target dissociation only, where the beam proton survives. If so, it
   is ~2× too high. This is to be settled; see §7 Q2.

**Conclusion.** Reuse the Everest SD *sampling* (ported to Python), and use
the Everest pp fits and `14.1·A^0.65` slopes as cross-checks. Don't reuse
Everest cross sections, Coulomb, MCS or ionisation for gas.

### (b) Porting Geant4

What FTFP_BERT actually uses for protons at 450 GeV–7 TeV, checked in the
source:

| Piece | Geant4 11.4.2 | Where |
|---|---|---|
| Inelastic XS | `G4BGGNucleonInelasticXS` | `physics_lists/builders/src/G4FTFPProtonBuilder.cc:84` |
| Inelastic final state | `G4TheoFSGenerator("FTFP")`: `G4FTFModel` + `G4ExcitedStringDecay` + `G4GeneratorPrecompoundInterface`. **No** CHIPS quasi-elastic channel: `G4HadronPhysicsFTFP_BERT` passes `quasiElastic=false`. QE happens inside FTF as elastic hN collisions with probability σ_el/σ_tot, switched off for A < 2 (`G4FTFModel.cc:290`, `G4FTFParameters.cc:378`) | `G4HadronPhysicsFTFP_BERT.cc:84,183` |
| Elastic XS | `G4BGGNucleonElasticXS` | `constructors/hadron_elastic/src/G4HadronElasticPhysics.cc:137` |
| Elastic final state | `G4ChipsElasticModel` → `G4ChipsProtonElasticXS::GetExchangeT` | `G4HadronElasticPhysics.cc:138` |
| Single Coulomb | `G4CoulombScattering` with **`G4eCoulombScatteringModel`** (combined with WentzelVI msc), cross section `G4WentzelOKandVIxSection`. `G4hCoulombScatteringModel` is only used in the `_SS`/`_WVI` EM lists, not in FTFP_BERT | `G4EmBuilder.cc:175-203`, `G4CoulombScattering.cc:147` |
| MSC | `G4hMultipleScattering` + `G4WentzelVIModel` | `G4EmBuilder.cc:175` |
| δ-rays / pairs / brems | `G4hIonisation` (`G4BetheBlochModel`), `G4hPairProduction`, `G4hBremsstrahlung` | `G4EmBuilder.cc:179-187` |

How BGG works above 91 GeV:

- **Z > 1:** `σ = f_Z · σ_GG(p)`. Here σ_GG comes from
  `G4ComponentGGHadronNucleusXsc::ComputeCrossSections`: Glauber–Gribov with
  σ_tot = 2πR² ln(1+x), σ_in = (2πR²/2.4) ln(1+2.4x),
  x = (Zσ_pp + Nσ_pn)/(2πR²), `R = G4NuclearRadii::RadiusHNGG(A)`, times the
  per-Z "baryon correction" tables. `f_Z` is a per-Z constant that joins
  Barashenkov at 91 GeV.
- **Z = 1:** `1.0115 × G4HadronNucleonXsc::HadronNucleonXscNS`. σ_tot comes
  from the PDG-2017 fit and σ_el from the NS/TOTEM fit, with σ_in = σ_tot − σ_el.

This is ~100 lines of formulas plus three per-Z tables (93 entries each). The
f_Z factors need the Barashenkov data at 91 GeV, so I would tabulate them from
the local Geant4 build rather than port Barashenkov. The GG component also
gives σ_prod, hence σ_QE = σ_in − σ_prod, at no extra cost.

The CHIPS elastic model (`G4ChipsProtonElasticXS`: `GetPTables`,
`GetTabValues`, `GetExchangeT`, `GetQ2max`) is a closed-form parametrisation:

- pp: 3 components.
- pA: 4 components, with separate A < 6.5 and A > 6.5 branches.

All components are generalised exponentials in t, so they can be sampled by
direct inversion. Porting is ~250 lines; the Geant4 in-memory tables (AMDB)
aren't needed. Above p = e⁸ = 2981 GeV/c Geant4 evaluates the formula directly
instead of interpolating a table. Below that, the tables are linear in ln p,
which differs from direct evaluation by < 1% (to be checked in tests).

Things to know about CHIPS:

- Its own integrated σ_el is **not** used by Geant4, and it differs from BGG
  by up to +34% (N at 6.8 TeV: 148.6 vs 111.2 mb). Like Geant4, I would use
  BGG for σ and CHIPS only for the shape.
- Forward-peak slopes (fit to |t| < 0.01 GeV²), nearly energy-independent:

  | | C | N | O | Ar | pp |
  |---|---|---|---|---|---|
  | CHIPS [GeV⁻²] | 67 | 73 | 80 | 150 | 12.2 → 14.3 |
  | Everest `14.1·A^0.65` [GeV⁻²] | 71 | 78 | 86 | 155 | 12.0 → 14.0 |

- CHIPS adds a large-|t| tail (0.6–0.9% of events above 0.5 GeV² for C/N/O).
  That tail sits far beyond any aperture, so it is irrelevant for losses.

`G4WentzelOKandVIxSection` for protons:

- dσ/dz ∝ Z²/(z + screenZ)² with z = 1 − cos θ, times an exponential nuclear
  form factor `(1 + formfactA·z)⁻²`, R = 1.27 fm·A^0.27, times a spin/recoil
  factor.
- This is the same screening and form factor as the e± port
  (`CoulombScatteringCalculator`). Only the prefactor (m_e r_e/(pβ))², the
  kinematics and the Mott→spin-½ factor change.
- Atomic electrons add Z (so Z² → Z(Z+1)), but only for
  θ < θ_e,max = m_e/m_p ≈ 0.54 mrad. For H this **doubles** the Coulomb rate
  in the loss-relevant range. I propose to treat the electrons as a separate,
  kinematically exact knock-on process (§4.6) and keep Z² for the nucleus.

### (c) Existing Xcoll Geant4 and FLUKA couplings

- The Geant4 coupling runs BDSIM through `g4interface` (collimasim) over RPyC
  and models collimator jaws. `g4interface` is **not installed** here. BDSIM
  is (`bdsim-install`).
- The FLUKA coupling needs a FLUKA installation, which is absent.
- Both are built around jaws hit by tracked particles. A thin gas slab with a
  controlled geometry, particle and physics list is simpler and more
  transparent as a **standalone Geant4 application** against the local
  11.4.2 install.

**Conclusion.** Use the couplings neither for production nor for the
benchmark. The benchmark will be a standalone FTFP_BERT thin-target program
(§8.4).

---

## 4. Physics per process

Notation:

- p: beam momentum
- s: pN centre-of-mass energy squared
- t: four-momentum transfer, |t| ≈ p²θ²
- ξ = M_X²/s

All cross sections are per **atom**, evaluated at the reference momentum p0c,
as in the e± code. The event-by-event δ ~ 10⁻⁴ changes σ negligibly.

### 4.1 Absorption (inelastic)

- σ_abs = σ_inel^BGG − σ_QE − σ_SD. Geant4's σ_inel contains quasi-elastic
  and all diffractive channels; the two survivor channels are generated
  separately (§4.3, §4.4), so they are subtracted here.
- Absorption covers non-diffractive events, projectile and double
  diffraction, and target diffraction beyond ξ_max. In all of these the beam
  proton is removed at the interaction point, as requested.
- Leading baryons with x_F ~ 0.3–0.9 exist, but they are lost within tens of
  metres downstream. Treating them as lost at the IP is the standard
  approximation, and the FLUKA source is the IP anyway.
- No final state is generated. Each event is written to a source table with:
  `s`, element, x, px, y, py, zeta, delta, total momentum, gas Z and A, and
  weight [1/s]. §6.4 covers the bookkeeping.
- Source: `G4BGGNucleonInelasticXS::GetElementCrossSection`,
  `G4ComponentGGHadronNucleusXsc::ComputeCrossSections`,
  `G4HadronNucleonXsc::HadronNucleonXscNS/PDG` (proton branch, p ≥ 100 GeV/c),
  `G4NuclearRadii::RadiusHNGG`, `fProtonBarCorrectionTot/In`.

### 4.2 Coherent elastic (pA) and pp elastic (H), with Coulomb

- **Hadronic σ_el:** `G4BGGNucleonElasticXS`, the same structure as §4.1.
- **Hadronic shape:** CHIPS dσ_N/dt, the port of `G4ChipsProtonElasticXS`.
  pp for H; pA for Z ≥ 2, with the same `N = A − Z` conventions as
  `G4ChipsElasticModel::SampleInvariantT`.
- **Kinematics:** exact two-body elastic on a target of mass M_A. This gives
  θ from t and the recoil Δp/p = −|t|/(2 M_A p): 1.2·10⁻⁵ for pp at
  |t| = 0.01 GeV² and 450 GeV. It is negligible, but exact at no cost.
- **Coulomb on the nucleus:** dσ_C/dt = 4πα²Z²(ħc)² G²(t)/t², with Moliere
  screening and the exponential form factor G, as in
  `G4WentzelOKandVIxSection::SetupTarget` / `SampleSingleScattering`. The
  spin-½ factor (`1 − z·factB + factB1·Z·√(z·factB)(2−z)`) and the recoil
  factor `1/(1 + z·factD)` are kept for fidelity, though they are 10⁻⁵
  corrections at loss-relevant angles.
- **CNI (proposed default, §7 Q3):** sample the channel from
  dσ/dt = |F_C + F_N|². Here:
  - F_N = (ρ + i)/√(1+ρ²) · √(dσ_N/dt) (constant phase, valid before the
    first diffraction minimum, which is where interference matters);
  - F_C = −2αZ√π ħc G(t)/|t| · e^{iαΦ(t)};
  - Φ = −[γ_E + ln(B|t|/2)] (Bethe/West–Yennie), with B the CHIPS forward
    slope;
  - ρ from the PDG/COMPETE pp parametrisation, with ρ_pA ≈ ρ_pp.

  Order of magnitude, N₂ with loss threshold t₀ (§5): the interference term
  is −αZσ_tot(ρ + αΦ)E₁(Bt₀/2). That is ≈ −3.8 mb out of ~82 mb above t₀ at
  6.8 TeV (−5%), and ≈ −3 mb out of ~131 mb at 450 GeV (−2%; ρ ≈ 0 at
  √s = 29 GeV).
- **Sampling:** a mixture proposal, log-uniform in t (the Coulomb part,
  exactly as the e± `sample_theta`) plus the CHIPS sampler (the hadronic
  part). Weight = (dσ/dt)/(σ_norm·Σ_k c_k g_k(t)), the balance heuristic, so
  the weights stay bounded. Only the Coulomb part needs a lower cut, reported
  via `cutoff_scan` and `rate_above_theta_max` as in e±.
- Switches `coulomb=True/False`, `nuclear=True/False` and
  `interference=True/False` allow each piece in isolation (tests, and
  comparison with Geant4, where the two are separate processes).

### 4.3 Quasi-elastic pN (nuclei only)

- Proton scatters elastically off one bound nucleon; the nucleus breaks up.
  This is incoherent with Coulomb, so there is no CNI.
- **Cross section:** three candidates, all ±15% of each other at 450 GeV but
  with different energy dependence (N, mb):

  | model | 450 GeV | 6.8 TeV | Δ |
  |---|---|---|---|
  | GG σ_in − σ_prod (× f_Z) | 31.2 | 35.6 | +14% |
  | Everest N_eff·σ_el,pp | 27.6 | 33.7 | +22% |
  | CHIPS `G4QuasiElRatios` fraction × σ_inel (QGSP lists, not FTFP_BERT) | 26.6 | 26.9 | +1% |

  Since σ_el,pp grows by 27% over this range, a flat QE (CHIPS) looks
  unphysical. **Recommendation:** GG, since it is the same Glauber
  calculation as σ_inel and needs no extra code. §7 Q1.
- **Final state:** t from the CHIPS **pp** dσ/dt, with M = m_p in the
  kinematics, giving Δp/p ≈ −|t|/(2m_p p). Fermi motion and binding (~10 MeV)
  are neglected (Δp/p < 3·10⁻⁵ at 450 GeV). I'd leave them out unless you
  want them.

### 4.4 Single diffraction, target dissociation (pp → pX, pA → pX)

- Geant4 has no standalone model; SD lives inside FTF
  (`G4FTFParameters.cc:420`, target-diffraction probability 6 mb/σ_in,hN for
  baryons). **Proposal:** port the Everest/K2 channel
  (`nuclear_interaction.h`, `ichoix == 4`):
  - M² = exp(u·ln(0.15 s)), i.e. dN/dM² ∝ 1/M² on [1, 0.15 s] GeV²;
  - slope b_SD = 2b_pp for M² < 2, (106 − 17M²)b_pp/36 for 2 ≤ M² ≤ 5,
    7b_pp/12 above;
  - p′ = p(1 − M²/s), θ from |t|/√(pp′).
- Changes with respect to K2:
  - **H uses N_eff = 1**, not 1.62.
  - M²_min = (m_p + m_π)² = 1.162 GeV² instead of 1 GeV². Open for your
    view; it is a small effect.
  - Exact kinematics for p′ and θ.
- Resulting ξ ranges:
  - 450 GeV: ξ ∈ [1.4·10⁻³, 0.15], so all SD events have Δp/p ≥ 1.4·10⁻³.
  - 6.8 TeV: ξ ∈ [9·10⁻⁵, 0.15], and 66% of events have ξ > 10⁻³.
- **Normalisation is the main open point** (§7 Q2). K2's `4.3 + 0.3 ln s` mb
  tracks the two-arm SD. The Geant4 benchmark (§8.4) will measure FTF's
  leading-proton spectrum on H, N and Ar. That, plus data from ISR and FNAL
  fixed-target experiments (Albrow et al., Armitage et al., Schamberger et
  al.), should fix a per-arm σ_SD(s). I'd add a `sd_scale` parameter until
  then.

### 4.5 Single Coulomb

Merged into §4.2. It remains available alone as `process='coulomb'`: the
elastic channel with the nuclear amplitude off, mirroring the e± name.

### 4.6 MCS and ionisation (your item 5)

**MCS: agreed, it only contributes emittance growth.** It is the small-angle
part of the same single scattering. The Monte Carlo covers θ > θ_min, and the
diffusion below θ_min only grows the core emittance.

Magnitude at n = 10¹⁵ molecules/m³, β = 100 m, ε_n = 2.5 µm, using
Highland's 13.6 MeV without the log term (the full second moment with
E_s ≈ 21 MeV is 2.4× larger):

| gas | 450 GeV: ε doubling time | 6.8 TeV |
|---|---|---|
| H₂ | ~20 h | ~300 h |
| N₂ | ~0.9 h | ~13 h |

So it is irrelevant for losses. At injection with heavy gas, though, it is
comparable to the beam-gas lifetime (σ_abs(N₂) gives τ ≈ 19 h at the same n).
**Proposal:** no stochastic MCS. Report dε/dt = ½⟨β⟩ n c ∫_{θ<θ_min} θ² dσ
per element as a diagnostic, computed from the same Coulomb dσ/dθ, so the MC
and the diffusion estimate tile the angle range exactly. This is cheap.

**Mean ionisation: agreed, negligible.** dE/dx_min gives 0.04 eV/turn (H₂)
and 0.23 eV/turn (N₂) at 10¹⁵ m⁻³. RF absorbs this as a synchronous-phase
shift. For scale, synchrotron radiation is 0.1 eV/turn at 450 GeV.

**But the hard tails of ionisation and of pair production are single-event
momentum kicks.** Per atom, for Δp/p > 10⁻³, from the Geant4 11.4.2 models:

| mb/atom | 450 GeV: knock-on e⁻ | e⁺e⁻ pair | brems | SD (K2, ξ > 10⁻³) | 6.8 TeV: knock-on | pair | brems | SD |
|---|---|---|---|---|---|---|---|---|
| H | 0.56 | 0.016 | 4·10⁻⁵ | 6.3 | 0.037 | 0.026 | 5·10⁻⁵ | 4.7 |
| N | 3.9 | 0.46 | 2·10⁻³ | 24.6 | 0.26 | 0.66 | 2·10⁻³ | 18.4 |
| Ar | 10.0 | 2.8 | 0.011 | 35.0 | 0.67 | 3.8 | 0.014 | 26.1 |

- **Knock-on at injection** is 9–29% of SD, above the same Δp/p. The table
  divides by the K2 SD; with a one-arm SD (Q2) the ratio doubles.
- **Pair production at 6.8 TeV** is 15% of SD for Ar, 4% for N.
- **Proton bremsstrahlung** is negligible everywhere.

Proposals:

- **Knock-on:** include it. It is analytic, from
  `G4BetheBlochModel::ComputeCrossSectionPerElectron`:
  dσ/dT = 2πr_e²m_ec²Z/(β²T²)·(1 − β²T/T_max + T²/2E²). Exact p–e kinematics
  give both Δp/p = −T/p and θ ≤ m_e/m_p. This also accounts for the Z(Z+1)
  electron term of Coulomb scattering in a kinematically correct way.
- **Pair production:** include only if you want off-momentum studies at top
  energy with heavy gas. The final state is just Δp/p. The cross section
  would come as a dσ/dv table computed with `G4hPairProductionModel`, since
  the Kelner–Kokoulin–Petrukhin double integral is not worth porting.
- **Brems:** no.

---

## 5. Cross sections

Loss-relevant reference angle for the Coulomb columns. A 5σ amplitude at a
TCP from an arc location with β ≈ 100 m (ε_n = 3.5 µm) needs:

- θ_ref ≈ 40 µrad at 450 GeV (|t| = 3.2·10⁻⁴ GeV²);
- θ_ref ≈ 10 µrad at 6.8 TeV (|t| = 4.6·10⁻³ GeV²).

Coulomb scales as 1/θ_ref². "e⁻ (knock-on)" counts the atomic electrons for
the same θ cut.

Columns: inel = σ_inel^BGG; el = σ_el^BGG (for H: pp elastic); QE = CHIPS
fraction (see §4.3 for alternatives); SD = K2 form with N_eff (H: 1; see Q2);
abs = inel − QE − SD.

### 5.1 Per atom [mb], Geant4 11.4.2

**450 GeV/c (√s_pN = 29.1 GeV):**

| | A | inel | el | QE | SD | abs | Coulomb, nucleus (θ > 40 µrad) | e⁻ (knock-on) |
|---|---|---|---|---|---|---|---|---|
| H | 1 | 33.3 | 7.2 | – | 6.3 | 27.0 | 0.80 | 0.80 |
| He | 4 | 104.3 | 26.5 | 13.4 | 16.2 | 74.7 | 3.2 | 1.6 |
| C | 12 | 254.0 | 80.0 | 24.4 | 23.4 | 206.2 | 29.0 | 4.8 |
| N | 14 | 288.7 | 92.0 | 26.6 | 24.6 | 237.5 | 39.4 | 5.6 |
| O | 16 | 322.3 | 114.4 | 28.6 | 25.8 | 268.0 | 51.5 | 6.4 |
| Ne | 20 | 375.4 | 149.8 | 31.1 | 27.8 | 316.5 | 80.4 | 8.0 |
| Ar | 40 | 589.2 | 288.9 | 39.3 | 35.0 | 514.9 | 260.6 | 14.5 |
| Kr | 84 | 985.3 | 566.6 | 50.6 | 44.8 | 889.9 | 1042 | 29.0 |
| Xe | 131 | 1349.8 | 822.7 | 58.3 | 51.9 | 1239.5 | 2345 | 43.4 |

**6.8 TeV/c (√s_pN = 113.0 GeV):**

| | inel | el | QE | SD | abs | Coulomb, nucleus (θ > 10 µrad) | e⁻ (knock-on) |
|---|---|---|---|---|---|---|---|
| H | 38.5 | 9.1 | – | 7.1 | 31.3 | 0.056 | 0.056 |
| He | 115.8 | 32.9 | 14.2 | 18.3 | 83.3 | 0.23 | 0.11 |
| C | 278.3 | 97.0 | 24.8 | 26.4 | 227.0 | 2.0 | 0.34 |
| N | 315.7 | 111.2 | 26.9 | 27.8 | 260.9 | 2.8 | 0.39 |
| O | 351.8 | 136.9 | 28.8 | 29.1 | 294.0 | 3.6 | 0.45 |
| Ne | 408.6 | 177.0 | 31.2 | 31.3 | 346.0 | 5.6 | 0.56 |
| Ar | 631.3 | 329.1 | 38.5 | 39.5 | 553.3 | 18.3 | 1.0 |
| Kr | 1047.6 | 632.4 | 49.3 | 50.6 | 947.7 | 73.0 | 2.0 |
| Xe | 1430.9 | 911.8 | 56.9 | 58.6 | 1315.3 | 164.3 | 3.0 |

**Energy dependence**, 450 GeV → 6.8 TeV:

| | H | He | C | N | O | Ar |
|---|---|---|---|---|---|---|
| σ_inel | ×1.153 | ×1.110 | ×1.096 | ×1.093 | ×1.092 | ×1.071 |
| σ_el | ×1.27 | – | – | ×1.21 | – | ×1.14 |

BGG inelastic scan [mb] from 100 GeV to 10 TeV:

| p [GeV/c] | H | He | C | N | O | Ne | Ar | Kr | Xe |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 31.9 | 101.6 | 248.1 | 282.1 | 315.2 | 367.3 | 578.8 | 970.0 | 1329.7 |
| 316 | 32.9 | 103.4 | 252.1 | 286.5 | 320.0 | 372.7 | 585.8 | 980.2 | 1343.1 |
| 1000 | 34.6 | 107.0 | 259.6 | 294.9 | 329.1 | 383.0 | 598.9 | 999.8 | 1368.6 |
| 3162 | 36.7 | 111.9 | 270.0 | 306.5 | 341.8 | 397.3 | 617.1 | 1026.6 | 1403.6 |
| 10000 | 39.4 | 117.9 | 282.8 | 320.6 | 357.3 | 414.7 | 638.9 | 1058.9 | 1445.6 |

It is smooth and monotonic, driven by σ_pp(s) inside Glauber–Gribov.

### 5.2 Per molecule vs the LHC Design Report (7 TeV)

| | σ_inel | σ_el | **σ_inel + σ_el** | DR | (inel+el)/DR | inel/DR | σ_abs |
|---|---|---|---|---|---|---|---|
| H₂ | 77.0 | 18.3 | **95.3** | 94 | 1.014 | 0.82 | 62.8 |
| He | 116.0 | 33.0 | **148.9** | 126 | 1.182 | 0.92 | 83.4 |
| CH₄ | 432.7 | 133.9 | **566.6** | 566 | 1.001 | 0.76 | 352.8 |
| H₂O | 429.3 | 155.5 | **584.7** | 565 | 1.035 | 0.76 | 357.1 |
| CO | 630.8 | 234.4 | **865.3** | 870 | 0.995 | 0.73 | 521.7 |
| CO₂ | 983.1 | 371.6 | **1354.7** | 1317 | 1.029 | 0.75 | 816.0 |

- **The DR values are total hadronic cross sections (inelastic + nuclear
  elastic), not inelastic.** With the additivity rule, BGG reproduces them
  within 0–3.5%. The exception is He, where BGG is 18% higher. The DR He
  value lies between our σ_inel (116) and σ_tot (149). I don't know its
  source.
- A DR-style lifetime therefore assumes every hadronic interaction is a loss.
  At 6.8 TeV that is nearly true, because most elastic kicks exceed the TCP
  threshold. The Monte Carlo decides it properly.

At 450 GeV / 6.8 TeV [mb]:

| | σ_inel | σ_el | σ_abs |
|---|---|---|---|
| H₂ | 66.7 / 76.9 | 14.4 / 18.2 | 54.0 / 62.6 |
| CH₄ | 387.4 / 432.1 | 108.8 / 133.5 | 314.3 / 352.3 |
| H₂O | 389.0 / 428.7 | 128.8 / 155.1 | 322.0 / 356.6 |
| CO | 576.3 / 630.1 | 194.4 / 233.9 | 474.2 / 521.0 |
| CO₂ | 898.7 / 981.9 | 308.8 / 370.7 | 742.2 / 815.0 |
| N₂ | 577.4 / 631.3 | 184.0 / 222.3 | 475.0 / 521.9 |
| Ar | 589.2 / 631.3 | 288.9 / 329.1 | 514.9 / 553.3 |

---

## 6. Architecture

### 6.1 Files

- **`xcoll/beamgas/hadronic.py`** (new): proton–gas physics.
  - `ProtonNucleusCrossSections(Z, p0c)`: BGG port; σ_inel, σ_el, σ_prod,
    σ_QE; H branch.
  - `ChipsElasticDistribution(Z, A, p0c)`: dσ/dt and sampler; pp and pA.
  - Calculators with the e± interface
    (`compute_xsec()`, `sample_deflections(px, py, delta, rng)`):
    `ProtonAbsorptionCalculator`, `ProtonElasticCalculator`
    (hadronic + Coulomb + CNI), `ProtonQuasiElasticCalculator`,
    `ProtonDiffractionCalculator`, `ProtonKnockOnCalculator`,
    optionally `ProtonPairProductionCalculator`.
  - `ProtonScatteringSample`: a new dataclass, so `ScatteringSample` stays as
    is. It adds `process`, `t`, `xi`, `energy_transfer` and `absorbed`.
  - Per-Z constant tables ported from Geant4 (baryon corrections, f_Z, A)
    with provenance comments.
  - The Geant4 licence notice (§6.7).
- **`xcoll/beamgas/study.py`**:
  - The PDG check accepts 2212 as well. Dispatch in `_build_calculators`,
    with the e± branch byte-for-byte unchanged.
  - Proton process names, stratified sampling, the absorption table and the
    per-process breakdown.
  - New optional `BeamGasResult` fields, defaulting to `None`.
- **`xcoll/beam_elements/beamgas.py`**:
  - `_resolve_process` accepts the proton names.
  - `scatter()` gains a proton path that loops over processes. The e± path
    keeps its exact RNG call sequence.
- **`xcoll/beamgas/__init__.py`**: exports.
- Optional helper **`molecular_to_atomic_density(table)`**: `'H2O'` →
  `{'H': 2, 'O': 1}`. The example needs it, it is additive, and the e± API is
  unchanged.

### 6.2 Process selection and sampling

- Proton processes: `'absorption'`, `'elastic'`, `'quasi_elastic'`,
  `'diffractive'`, `'coulomb'` (the elastic channel with the nuclear
  amplitude off), `'knock_on'`, optionally `'pair'`, and `'all'`.
- `process` accepts a string or a list for protons. For e± it stays a single
  string.
- **Stratified by (element, process).** `n_scattering_events` is an int
  (per element and process) or a dict `{process: n}`. Within a process, the
  species is drawn ∝ n_i σ_i, as in e±.
- Weight = rate(element, process)/n × importance weight. This beats a random
  process mixture: rare channels such as SD get their own statistics.
- Absorption events are not tracked, so they cost nothing.

### 6.3 What happens to each event

| process | Δ to the macro-particle | tracked? | logged columns |
|---|---|---|---|
| absorption | none; row in the absorption table | no | gas, Z, A, weight, coordinates, momentum |
| elastic / coulomb | θ, φ, δ recoil | yes | t, θ, Coulomb fraction at t |
| quasi_elastic | θ, φ, δ recoil | yes | t, θ |
| diffractive | θ, φ, δ = −ξ (exact) | yes | t, θ, ξ, M_X² |
| knock_on | θ, φ, δ = −T/p | yes | T, θ |

### 6.4 Absorption bookkeeping and loss-map output

**Proposal (§7 Q5):** absorption events are not turned into particles.

- They go into `BeamGasResult.absorption_events`, an `xt.Table` with one row
  per event: `name`, `s`, `x`, `px`, `y`, `py`, `zeta`, `delta`, `p [eV]`,
  `gas`, `Z`, `A` and `weight [1/s]`. It can be written to CSV as a FLUKA
  source.
- Their rate enters `rate_tracking` exactly, with no MC noise.
- This avoids a new particle-state constant and any change to `xc.LossMap`.
  The alternative (particles with a new `XC_LOST_IN_GAS` state) touches
  shared headers and the loss-map classification.

Loss map:

- For the tracked part, `xc.LossMap(line, part=result.particles,
  weights=result.particles.weight)` already works; it accepts weights.
- For the absorption part, a weighted histogram of `absorption_events.s`.
- `BeamGasResult.loss_map_table(...)` bins both into one table with the
  columns `absorption_in_gas`, `aperture` and `collimator` [1/s per bin].

### 6.5 Diagnostics

- The `cutoff_scan` and truncation warnings keep working for the channels
  with a lower cut: elastic/coulomb (θ) and knock-on (T).
- New:
  - `result.process_rates`: interaction rate and loss rate per process and
    per species;
  - the MCS emittance-growth diagnostic (§4.6);
  - a warning when tracked survivors sit outside the RF bucket (§7 Q6).

### 6.6 e± isolation

- A golden regression test comes first, on unmodified code: bitwise-equal
  e± outputs for both processes (rates, `cutoff_scan`, first particles,
  weights) with fixed seeds.
- All proton physics goes in the new module. The e± branches of `study.py`
  and `beamgas.py` keep their RNG call order. The existing 90 tests must pass
  unchanged.

### 6.7 Geant4 licence

The Geant4 Software License §1 requires reproducing the copyright notice and
licence conditions in redistributions "in whole or in part, with or without
modification". The e± port cites Geant4 class names but carries neither.

**Proposal:**

- Put a short Geant4 notice and acknowledgement in the header of
  `hadronic.py`: "This product includes software developed by Members of the
  Geant4 Collaboration ( http://cern.ch/geant4 )".
- Add the full licence text as `LICENSE-GEANT4` at the repo root.
- Every ported function cites the Geant4 version, file and class.
- Whether to add the same notice to the e± `cross_sections.py` is your call.
  It is a comment-only change (§7 Q8).

---

## 7. Open questions (need your decision)

1. **QE cross section.** GG σ_in − σ_prod (recommended), Everest
   N_eff·σ_el,pp, or the CHIPS ratio? They agree within ±15% at 450 GeV and
   differ in energy dependence (§4.3).
2. **SD.**
   - Normalisation: K2 `4.3 + 0.3 ln s` per N_eff, which looks two-arm (so
     ~2× high for target dissociation)? Half of it? Or tuned to the FTF
     benchmark and ISR/FNAL one-arm data?
   - Keep K2's M² shape (1/M², ξ ≤ 0.15, b(M²)), or do you prefer a
     triple-Pomeron form with a low-mass resonance enhancement (e.g.
     Schuler–Sjöstrand)?
   - Recommendation: K2 shape now, `sd_scale` parameter, normalisation fixed
     after the benchmark.
3. **CNI.** Sample elastic + Coulomb jointly from |F_C + F_N|² (recommended,
   a 2–5% effect), or keep them as separate incoherent processes? ρ_pA ≈ ρ_pp
   from the PDG/COMPETE parametrisation. Is that acceptable?
4. **EM tails.** Include knock-on electrons (recommended)? Include e⁺e⁻ pair
   production via a Geant4-derived table (optional, top energy with heavy
   gas)?
5. **Absorption bookkeeping.** A table without particles (recommended), or
   particles with a new lost state?
6. **Off-momentum survivors.**
   - At 6.8 TeV the bucket half-height is ≈ 3–4·10⁻⁴ (12–16 MV). SD,
     knock-on and pair events with larger |δ| leave the bucket and only
     reach IR3 after up to ~10⁶ turns of SR energy loss.
   - At 450 GeV they coast (abort-gap population) unless |δ| exceeds the
     momentum acceptance.
   - Options:
     (a) track long with RF and SR, too slow for the example;
     (b) after `n_turns`, classify out-of-bucket survivors as
     "eventually lost in momentum cleaning", assigned to IR3 in the loss map
     and counted in the lifetime;
     (c) report them separately and leave the decision to the user.
   - I'd implement (c) with an option for (b).
7. **Process API.** Are the process names and stratified per-process counts
   OK? `process='all'` as the default for protons?
8. **Licence notice.** Add it to the e± module as well (comment-only)?
9. **Python, not C kernels.** Confirm, per §2.
10. **Injection example.** The only lattice with apertures in the repos is
    Run 3 at 6.8 TeV (`examples/machines/lhc_run3_b1.json`).
    `xtrack/test_data/lhc_2024/{lhc.seq,injection_optics.madx}` has injection
    optics but no aperture model. Plan: the example runs at 6.8 TeV. A
    450 GeV variant would use the 2024 injection optics with collimators only
    and no aperture losses, unless you have a 450 GeV json with apertures.
11. **Pressure profile file.** I have no real LHC profile (e.g. from VASCO).
    I'd ship a clearly labelled *synthetic* profile: H₂-dominated cold arcs,
    and CO/CO₂/CH₄ in warm straights. It would be a small CSV in
    `examples/data/`. Or do you have one to use?

### Assumptions (stated explicitly)

- Cross sections are evaluated at the reference momentum. Additivity holds
  for molecules; molecular binding, shadowing between atoms and screening by
  molecular electrons are negligible at the relevant angles.
- A for each Z is the rounded standard atomic weight, as in e±. BGG uses the
  rounded NIST mean mass; these agree for all Z ≤ 54 except Tc (Z = 43:
  97 vs 98). The hadronic tables will follow Geant4.
- The absorbed proton is lost at the interaction point.
- The beam-gas interaction probability per turn is ≪ 1, so single
  interaction per macro-particle holds, as in e±.
- Gas densities are static. There is no beam-induced pressure feedback.

---

## 8. Validation plan

### 8.1 e± unchanged

- Golden bitwise test first (commit 1).
- The full existing suite passes. Pytest runs send the pytest-html report to
  a temporary directory so the repo stays clean.
- Before/after comparison of `examples/beamgas.py` output.

### 8.2 Unit tests per process

Against Geant4 reference values, committed as JSON with the generator source.

**Cross sections**

- σ_inel, σ_el (BGG), and the GG σ_tot/σ_in/σ_prod, for Z = 1–54 at
  p = 100, 450, 1000, 3162 and 6800 GeV. Target rtol 10⁻⁶, since it is an
  exact port.
- The DR comparison as a test: rtol 5% for H₂/CH₄/H₂O/CO/CO₂, and an xfail
  documented for He.

**Energy scaling**

- Monotonic in p.
- Matches Geant4 on the grid.
- σ_pp,tot/el vs the Everest fits within 6%.

**CHIPS elastic**

- The analytic dσ/dt integrates to the sampler's normalisation.
- Sampled |t| vs our own density: binned χ² and KS.
- Sampled |t| vs Geant4 `GetExchangeT` samples (2·10⁵ per case,
  two-sample KS) for H, He, C, N, O and Ar at 450 GeV and 6.8 TeV.

**Coulomb / CNI**

- σ in a window vs `scipy.quad`.
- θ histogram vs the analytic density.
- The small-angle precision test, as in e±.
- With nuclear=False, reduces to the Wentzel form.
- With ρ = 0 and Φ = 0, the interference term integrates to its analytic
  E₁ expression.
- Cross-check of the Coulomb σ against `G4WentzelOKandVIxSection` (scratch
  program).

**SD**

- ξ distribution ∝ 1/ξ on [ξ_min, 0.15].
- δ = −ξ.
- t-slope per M² band.
- σ normalisation and its s dependence.

**QE**

- σ vs GG.
- t-slope = pp slope.
- Recoil δ.

**Knock-on**

- dσ/dT vs the Bethe–Bloch formula.
- σ(T > T_cut) vs `G4BetheBlochModel` (the numbers in §4.6).
- θ ≤ m_e/m_p.
- Energy–momentum conservation.

**Kinematics**

- |p′| is as expected for each process.
- φ is uniform.
- Weights have mean 1 per process (MC tolerance).

### 8.3 Lifetime and bookkeeping

On the toy ring and on the LHC:

- Uniform n, `process='absorption'`: the loss rate equals N·n·σ_abs·β·c
  exactly (no MC noise).
- Absorption, QE and SD with all survivors forced lost, via a tiny aperture:
  the rate equals n·σ_inel·c within statistics. This is your τ = 1/(n σ_inel c)
  check.
- Huge aperture and no collimators: tracked losses → 0, rate =
  n·σ_abs·c.
- Tiny aperture, all hadronic processes: rate = n·(σ_inel + σ_el)·c, the
  "DR lifetime".
- Weights sum to the section rates, and the per-process breakdown sums to
  the total.

### 8.4 Geant4 thin-target benchmark

Geant4 11.4.2 is built here, and a scratch program already links against it.

- **Setup:** a standalone app with FTFP_BERT; a slab of G4_H, G4_N or G4_Ar
  gas (pure element, thickness for P_int ~ 10⁻³); primary protons at 450 GeV
  and 6.8 TeV, 10⁶–10⁷ events each.
- **Recorded per event:** first-step process (`hadElastic`,
  `protonInelastic`, `CoulombScat`, `msc`, `hIoni`, `hPairProd`), and the
  leading outgoing proton's θ and Δp/p.
- **Comparisons:**
  1. Per-process σ from interaction counts vs our σ.
  2. dN/d|t| for `hadElastic` vs our CHIPS port.
  3. For `protonInelastic`, the leading-proton spectrum in x_F and t. The
     x_F > 0.85 region measures FTF's effective QE + SD, which calibrates
     §7 Q1 and Q2.
  4. Single-Coulomb dN/dθ above the Geant4 msc/ss split angle vs ours.
- **Commit:** the app source (CMake), the analysis script and summary
  plots, under `examples/beamgas_geant4_benchmark/`. It is not part of CI.
  The resulting numbers go into this document.

### 8.5 Example

`examples/beamgas_proton_lhc.py`:

- **Lattice:** LHC Run 3 B1 at 6.8 TeV with apertures
  (`examples/machines/lhc_run3_b1.json`, 102k elements) and Everest
  collimators from `examples/colldbs/lhc_run3.yaml`.
- **Scattering elements:** ~100–200 `BeamGasScattering` elements.
- **Gas:** case 1 uniform (H₂ + CO); case 2 a pressure profile read from
  `examples/data/*.csv` (molecular densities → atomic densities).
- **Output:** a loss map with absorption in gas, aperture and collimator
  losses; the per-process rate table; the lifetime; and the absorption source
  table written to CSV.
- **Runtime budget:** measured here, LHC tracking costs ≈ 48 µs per
  particle-turn on 28 OpenMP threads. ~1.5·10⁵ tracked events × 20 turns is
  ≈ 2.5 min.
- **Note:** OpenMP kernel builds fail if launched from the xcoll repo root.
  Setuptools sees the tracked `dev/` folder as a second top-level package.
  The example runs from `examples/`, which is fine.

---

## 9. Implementation steps (one commit each)

1. Golden e± regression test (no code change).
2. `hadronic.py`: BGG/GG/NS port, per-Z tables, licence notice. Geant4
   reference JSON and generator source under `tests/data/beamgas/`. Unit tests.
3. CHIPS elastic port and tests vs Geant4 samples.
4. Proton calculators: absorption, elastic (+Coulomb, CNI), QE, SD,
   knock-on (optionally pair). Unit tests.
5. `study.py` / `beamgas.py` integration: dispatch, process list,
   stratification, absorption table, result fields, diagnostics. The e±
   golden test stays green.
6. Lifetime and bookkeeping tests (§8.3).
7. Geant4 thin-target benchmark app and analysis. Results go into §5, and
   Q1/Q2 are revisited with you if the numbers move.
8. LHC example and pressure-profile file.
9. Update this document to what was done, with results tables and
   limitations.

---

## Appendix: provenance of the numbers

- **Hadronic:** a scratch C++ program calling `G4BGGNucleonInelasticXS`,
  `G4BGGNucleonElasticXS`, `G4ComponentGGHadronNucleusXsc`,
  `G4HadronNucleonXsc`, `G4ChipsProtonElasticXS` (including 2·10⁵
  `GetExchangeT` samples per case) and `G4QuasiElRatios`.
- **EM:** `G4BetheBlochModel`, `G4hPairProductionModel` and
  `G4hBremsstrahlungModel::ComputeCrossSectionPerAtom`, with a cut at
  v = Δp/p.
- **Analytic:** the Coulomb and electron columns use 4πα²Z²(ħc)²/|t|,
  resp. ×Z/Z². The Everest columns use the formulas in `properties.h`.
- The sources will be committed with the reference data in step 2.
