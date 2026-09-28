# Beam-gas Monte Carlo for proton beams: design and implementation

**Status: implemented on branch `beamgas_proton`** (from `beamgas` at
`0832d895`). The e± physics is unchanged: a regression test freezes its
results.

**Scope.** The e± beam-gas Monte Carlo of PR #230 is extended to LHC protons
from 450 GeV to 6.8 TeV, for beam-gas lifetimes and loss maps.

**Geant4 references.** All refer to **Geant4 11.4.2**
(`geant4-11-04-patch-02`), from the local clone (`v11.4.2-19-gcb836de6e6`)
and install. Reference values were produced by calling the Geant4 classes
directly (`tests/data/beamgas/geant4/make_proton_reference.cc`). The
thin-target benchmark runs full FTFP_BERT simulations
(`examples/beamgas_geant4_benchmark/`).

This document started as the plan submitted for review. §1–§5 describe what
was built; §6 records the options assessment that led to it; §7 lists the
decisions, remaining open points and limitations.

---

## 1. What was built

The approach is hybrid. Cross sections come from Geant4 (as used by
FTFP_BERT) and the elastic shape from Geant4 CHIPS. Single diffraction uses
the Everest/K2 shape with exact kinematics and a benchmark-calibrated
normalisation. Everything is Python/NumPy, next to the e± models, with the
same weighted, forced-interaction scheme.

| Process (`process=` name) | Cross section | Final state | Source |
|---|---|---|---|
| `'absorption'` | σ_inel − σ_QE − σ_SD | proton created lost at the scattering element, state `LOST_ON_BEAMGAS` | `G4BGGNucleonInelasticXS` |
| `'elastic'` | \|F_C + F_N\|², Coulomb + coherent hadronic + interference, in a θ window | \|t\| sampled, exact two-body kinematics with recoil | BGG elastic × CHIPS shape (`G4BGGNucleonElasticXS`, `G4ChipsProtonElasticXS`); Coulomb from `G4WentzelOKandVIxSection`; West–Yennie interference, ρ from the PDG pp fit |
| `'nuclear_elastic'`, `'coulomb'` | hadronic only / Coulomb only variants of `'elastic'` | as above | same |
| `'quasi_elastic'` | Glauber–Gribov σ_in − σ_prod (0 for H) | CHIPS pp \|t\| on a free nucleon | `G4ComponentGGHadronNucleusXsc`, `G4ChipsProtonElasticXS` |
| `'diffractive'` (target dissociation) | `sd_scale · N_eff · (4.3 + 0.3 ln s)` mb, `N_eff = 1` for H, **`sd_scale = 0.65`** | dN/dM² ∝ 1/M² on [(m_p+m_π)², 0.15 s], Everest slope b(M²), \|t\| ≥ \|t\|_min, exact ν = (M² − m_p² + \|t\|)/2m_p | Everest/K2 (`nuclear_interaction.h`, `properties.h`) |
| `'knock_on'` | Bethe–Bloch per electron × Z, with the proton form factor and magnetic-moment correction, T ≥ `knock_on_cut`·E | exact p–e kinematics (Δp/p = −T/p, θ ≤ m_e/m_p) | `G4BetheBlochModel` |
| `'all'` (default) | absorption + elastic + quasi_elastic + diffractive + knock_on | | |

Not included, per your decisions and the item-5 analysis: e⁺e⁻ pair
production, proton bremsstrahlung, MCS, and the mean ionisation loss (§7.3).

---

## 2. Architecture as built

### 2.1 Files

- **`xcoll/beamgas/proton_data.py`** (new): per-Z Geant4 constants.
  - BGG mass numbers and NIST masses.
  - `fProtonBarCorrectionTot/In`, copied from the Geant4 source.
  - Glauber-to-Barashenkov factors, tabulated from Geant4 for Z = 1–92.
- **`xcoll/beamgas/proton_cross_sections.py`** (new): the physics.
  - `ProtonNucleusCrossSections` (BGG/GG port) and
    `proton_nucleon_cross_sections` (`G4HadronNucleonXsc` NS/PDG);
    `proton_proton_rho`.
  - `ChipsElasticDistribution`: CHIPS parameters, including Geant4's
    128-node table interpolation below p = e⁸ GeV/c; density, CDF, and
    windowed sampling by component inversion.
  - `WentzelCoulombCrossSection`.
  - Calculators with the e± interface (`compute_xsec`,
    `sample_deflections`): `ProtonAbsorptionCalculator`,
    `ProtonElasticCalculator`, `ProtonQuasiElasticCalculator`,
    `ProtonDiffractionCalculator`, `ProtonKnockOnCalculator`.
  - `ProtonScatteringSample`: the e± `ScatteringSample` is untouched.
- **`xcoll/beamgas/proton_study.py`** (new): `ProtonBeamGasStudy`, a
  subclass of `BeamGasStudy`.
- **`xcoll/beamgas/gas.py`** (new): `parse_molecule` and
  `molecular_to_atomic_density`, e.g. `H = 2 n_H2 + 4 n_CH4`.
- **`xcoll/beamgas/study.py`**: behaviour-preserving hooks only.
  - `__new__` dispatches proton lines to `ProtonBeamGasStudy`.
  - `_validate_beam`, `_resolve_study_process` and `_compute_xsecs` are
    methods.
  - The cutoff scan is a shared function, `_cutoff_scan_table`.
  - Optional `BeamGasResult` fields: `process_rates`, `rate_out_of_bucket`.
  - The "line end not represented" warning fires only when more than 0.1% of
    the gas lies beyond the last element.
- **`xcoll/beam_elements/beamgas.py`**: a proton path in `scatter()`
  (`_scatter_protons`, `_generate_primaries_in_bucket`). The e± path and its
  RNG call order are untouched, and the C kernel and xofields are
  unchanged.
- **`xcoll/headers/particle_states.py`**: new state
  `LOST_ON_BEAMGAS = -340`. It is not in `USE_IN_LOSSMAP`, so `xc.LossMap`
  ignores it.
- **`xcoll/line_tools.py`**: `beamgas_configure(process=None)` uses the
  study default (`'brems'` for e±, `'all'` for protons).

### 2.2 How a proton study runs

1. **Construction.** `xc.BeamGasStudy(line=...)` or
   `line.xcoll.beamgas_configure(...)` with a proton `particle_ref` returns a
   `ProtonBeamGasStudy`. It builds one calculator per (process, species),
   sharing one `ProtonNucleusCrossSections` per species.
2. **`initialise_beamgas()`.** Computes the rate of each process in the
   section represented by each element (bunch intensity × f_rev × ∫n_i ds ×
   σ_i,process). It stores these on the element, together with the number
   of events per process and the RF bucket from the line cavities.
3. **`scatter()`.** Stratified by process. For each process, `n_events`
   macro-particles are drawn from the matched Gaussian, **truncated at the
   RF separatrix**. A gas species is chosen ∝ n_i σ_i and the calculator
   applies the event. The weight is (process rate / n) × importance weight.
   Absorbed protons are created lost at the element, and the log records
   the process, θ, \|t\|, energy loss, M_X², Coulomb fraction, weight and
   the absorbed flag.
4. **`run(track=True, n_turns=...)`.** Tracks each element's sample for
   `n_turns`. Survivors outside the RF bucket are handled by `out_of_bucket`:
   - `'report'` (default): their rate is returned as `rate_out_of_bucket`,
     with a warning if above 1% of the loss rate;
   - `'drift'`: all of them complete their turn, are merged, and are
     tracked from the start of the line while their momentum is lowered by
     `drift_per_turn` per turn. This is an accelerated
     synchrotron-radiation energy loss, to find where they are lost at top
     energy without 10⁶ turns.
5. **Result.** The loss rate, lifetime and Monte Carlo error, with each
   (element, process) an independent stratum.
   - `process_rates`: interaction rate, loss rate, error and loss fraction
     per process.
   - `cutoff_scan`: for `'elastic'` in θ and `'knock_on'` in T.
   - `rate_above_theta_max`, `local_rates` (with
     `interaction_rate_<process>` columns), and `interaction_log`.
   - `study.absorption_events(result)` is the source table for FLUKA:
     s, coordinates, momentum, gas, Z, A and weight.

---

## 3. Cross sections (final models)

All values are per **atom**, in mb. QE is Glauber–Gribov; SD uses
`sd_scale = 0.65`; abs = inel − QE − SD. The Coulomb and knock-on columns
count scattering above the loss-relevant reference angle θ_ref: 40 µrad at
450 GeV, 10 µrad at 6.8 TeV (a 5σ amplitude at a TCP from an arc with
β ≈ 100 m). The Coulomb column includes the nuclear form factor.

### 3.1 Per atom

**450 GeV/c (√s_pN = 29.1 GeV)**

| | inel | el | QE | SD | abs | Coulomb, θ > 40 µrad | knock-on e⁻, θ > 40 µrad |
|---|---|---|---|---|---|---|---|
| H | 33.3 | 7.2 | – | 4.1 | 29.2 | 0.80 | 0.79 |
| He | 104.3 | 26.5 | 13.0 | 10.6 | 80.8 | 3.1 | 1.6 |
| C | 254.0 | 80.0 | 28.0 | 15.2 | 210.8 | 27.7 | 4.7 |
| N | 288.7 | 92.0 | 31.2 | 16.0 | 241.4 | 37.6 | 5.5 |
| O | 322.3 | 114.4 | 34.3 | 16.7 | 271.3 | 49.0 | 6.3 |
| Ne | 375.4 | 149.8 | 38.7 | 18.1 | 318.6 | 76.1 | 7.9 |
| Ar | 589.2 | 288.9 | 50.0 | 22.7 | 516.5 | 242 | 14.1 |
| Kr | 985.3 | 566.6 | 74.5 | 29.1 | 881.7 | 942 | 28.2 |
| Xe | 1349.8 | 822.7 | 97.4 | 33.8 | 1218.6 | 2070 | 42.4 |

**6.8 TeV/c (√s_pN = 113.0 GeV)**

| | inel | el | QE | SD | abs | Coulomb, θ > 10 µrad | knock-on e⁻, θ > 10 µrad |
|---|---|---|---|---|---|---|---|
| H | 38.5 | 9.1 | – | 4.6 | 33.8 | 0.053 | 0.054 |
| He | 115.8 | 32.9 | 15.1 | 11.9 | 88.8 | 0.18 | 0.11 |
| C | 278.3 | 97.0 | 32.0 | 17.2 | 229.1 | 1.44 | 0.33 |
| N | 315.7 | 111.2 | 35.6 | 18.1 | 262.0 | 1.91 | 0.38 |
| O | 351.8 | 136.9 | 38.9 | 18.9 | 294.0 | 2.45 | 0.44 |
| Ne | 408.6 | 177.0 | 43.8 | 20.4 | 344.3 | 3.69 | 0.54 |
| Ar | 631.3 | 329.1 | 55.7 | 25.6 | 549.9 | 10.5 | 0.98 |
| Kr | 1047.6 | 632.4 | 82.5 | 32.8 | 932.2 | 35.2 | 2.0 |
| Xe | 1430.9 | 911.8 | 107.6 | 38.1 | 1285.2 | 69.5 | 2.9 |

Energy scaling of σ_inel from 450 GeV to 6.8 TeV: H ×1.153, He ×1.110,
C ×1.096, N ×1.093, O ×1.092, Ar ×1.071. σ_el grows by ×1.27 (H), ×1.21 (N)
and ×1.14 (Ar). Everything is smooth and monotonic from 100 GeV to 10 TeV
(tested).

### 3.2 Per molecule vs the LHC Design Report (7 TeV)

| | σ_inel | σ_el | **σ_inel + σ_el** | DR | ratio | σ_abs | σ_QE | σ_SD |
|---|---|---|---|---|---|---|---|---|
| H₂ | 77.0 | 18.3 | **95.3** | 94 | 1.014 | 67.8 | – | 9.3 |
| He | 116.0 | 33.0 | **149.0** | 126 | 1.182 | 88.9 | 15.1 | 11.9 |
| CH₄ | 432.7 | 133.9 | **566.6** | 566 | 1.001 | 364.8 | 32.1 | 35.8 |
| H₂O | 429.3 | 155.5 | **584.7** | 565 | 1.035 | 362.0 | 39.0 | 28.2 |
| CO | 630.8 | 234.4 | **865.3** | 870 | 0.995 | 523.6 | 71.1 | 36.1 |
| CO₂ | 983.1 | 371.6 | **1354.7** | 1317 | 1.029 | 817.9 | 110.1 | 55.1 |

- **The DR values are total hadronic cross sections (inelastic + nuclear
  elastic).** With additivity, BGG reproduces them within 0–3.5%, except He
  (+18%: the DR value lies between our σ_inel and σ_tot).
- σ_inel alone is 73–92% of the DR values.

At 450 GeV the molecular totals are H₂ 81.1, CH₄ 496.1, H₂O 517.8, CO 770.7,
CO₂ 1207.5, N₂ 761.3 and Ar 878.1 mb.

---

## 4. Validation

### 4.1 Test suite

Run with `cd tests && python -m pytest test_beamgas*.py`.

| File | Content | Result |
|---|---|---|
| `test_beamgas.py` | existing e± tests, unmodified | 90 passed |
| `test_beamgas_regression.py` | e± results frozen before any change (brems, e⁻ and e⁺ Coulomb: xsecs, generated samples, tracked states, rates, cutoff scans; rtol 1e-12, states exact) | 3 passed |
| `test_beamgas_proton_physics.py` | ports vs Geant4 reference values, sampled distributions, kinematics | 51 passed |
| `test_beamgas_proton_study.py` | dispatch, validation, rates, stratified weights, lifetime, out-of-bucket, reproducibility | 27 passed |

**Proton physics unit tests: agreement with the Geant4 reference values.**

- **BGG and Glauber–Gribov:** σ_inel, σ_el and GG tot/in/prod/el for Z = 1–92
  at 12 momenta from 100 GeV to 10 TeV agree with Geant4 to better than
  10⁻⁸; the residual is the CODATA vs CLHEP masses.
- **pp/pn (NS/PDG):** agree to 10⁻⁸.
- **CHIPS:**
  - its elastic σ, which exercises the parameter port and the table
    interpolation, agrees exactly;
  - the \|t\| CDF at 199 Geant4 quantiles (2·10⁵ samples each, 9 elements ×
    4 momenta) deviates by at most 0.006, which is sampling noise;
  - the density integrates to 1, and the windowed sampler matches the
    density.
- **Wentzel Coulomb:** the majorant agrees to 10⁻⁸. Below 10 µrad, Geant4's
  cos θ₁ − cos θ₂ loses precision and our 2 sin²(θ/2) form is the accurate
  one. The form-factor/spin acceptance agrees with 10⁶ Geant4 samples per
  window within 5σ.
- **Knock-on:** the bare σ agrees with `G4BetheBlochModel` to 10⁻⁷. The θ
  limit m_e/m_p and the weighted tail are verified.
- **Elastic channel:**
  - σ agrees with `scipy.quad` to 10⁻⁸;
  - Coulomb + nuclear = no-interference total;
  - the interference is destructive and within 10%;
  - the weighted histogram reproduces dσ/d\|t\| in 12 bins within 5σ, and
    ⟨w⟩ = 1;
  - with nuclear only and no window, the weights are exactly 1.
- **Other processes:**
  - QE: σ = f_Z(σ_in − σ_prod), with the pp slope;
  - SD: the K2 formula with N_eff (H: 1), ln M² uniform, and
    ν = (M² − m_p² + \|t\|)/2m_p exact;
  - absorption: σ = inel − QE − SD;
  - all processes: momentum and angle consistency to 10⁻¹², and the
    expected energy dependence (e.g. σ_inel(6.8 TeV)/σ_inel(450 GeV) =
    1.153, 1.093, 1.071 for H, N, Ar).
- **LHC Design Report totals** (H₂, CH₄, H₂O, CO, CO₂ within 4%) and the
  He exception are both tests.

**Proton study integration tests (toy ring, N + H gas).**

- **Rates:** the per-process rates equal N·f_rev·L·Σ n_i σ_i. The weights
  sum to the process rates, exactly for the unweighted processes, and the
  species are drawn ∝ n σ.
- **Lifetime, as you required:**
  - absorption only: **τ = 1/(c Σ n_i σ_abs,i) to 10⁻¹⁰**, with no Monte
    Carlo noise;
  - absorption + QE + SD with an aperture that stops every scattered
    proton: **τ = 1/(c Σ n_i σ_inel,i) to 10⁻¹⁰**;
  - no aperture: only absorption is lost.
- **Bookkeeping:** the rate error equals the quadrature sum of the
  per-process errors, the process rates sum to the totals, and the
  absorption table matches the absorption rate.
- **RF bucket:** δ_max = 2Q_s/(h\|η\|) against 6D twiss (2%); generated
  protons start inside the bucket.
- **Drift mode:** it turns the out-of-bucket rate of the report mode into
  losses (rates consistent to 10⁻⁶).

### 4.2 Geant4 thin-target benchmark (FTFP_BERT)

`examples/beamgas_geant4_benchmark/`: `beamgas_benchmark.cc` (the
application), `run_benchmark.py` (driver and analysis), and the results in
`results_geant4_11.4.2.json`. The plots are in
`examples/plots/beamgas_geant4_benchmark/`.

Setup: 4·10⁶ protons per case on pure H, N and Ar gas slabs (P_inel = 5%),
at 450 GeV/c and 6.8 TeV/c; 44 s on 28 cores.

| | σ_inel G4 / Xcoll [mb] | σ_el G4 / Xcoll [mb] | elastic ⟨\|t\|⟩ G4 / Xcoll [GeV²] | FTF leading p with x_F > 0.99 (fraction of inel) | Xcoll QE + SD (0.65), same cut |
|---|---|---|---|---|---|
| H 450 GeV | 33.42 ± 0.08 / 33.34 | 7.17 ± 0.04 / 7.19 | 0.0897 / 0.0893 | 0.053 | 0.054 |
| N 450 GeV | 288.8 ± 0.7 / 288.7 | 91.7 ± 0.4 / 92.0 | 0.0245 / 0.0247 | 0.104 | 0.133 |
| Ar 450 GeV | 588.5 ± 1.3 / 589.2 | 289.5 ± 0.9 / 288.9 | 0.0087 / 0.0086 | 0.077 | 0.102 |
| H 6.8 TeV | 38.52 ± 0.09 / 38.45 | 9.04 ± 0.04 / 9.12 | 0.0769 / 0.0771 | 0.078 | 0.077 |
| N 6.8 TeV | 316.3 ± 0.7 / 315.7 | 111.1 ± 0.4 / 111.2 | 0.0425 / 0.0435 | 0.108 | 0.149 |
| Ar 6.8 TeV | 631.3 ± 1.4 / 631.3 | 329.3 ± 1.0 / 329.1 | 0.0092 / 0.0091 | 0.074 | 0.114 |

**Cross sections.** Agree within 0.1–1.8σ (±0.3–0.5%).

**Elastic \|t\|.** The histograms agree over four decades, including the
CHIPS tails and structure. KS distances are 0.003–0.005 (1.5–3σ of the 10⁵
Geant4 events).

**Single diffraction.**

- *Hydrogen.* FTF shows the same flat dN/d ln ξ plateau as the K2 shape.
  Its lower edge is at the kinematic threshold reproduced by our exact
  kinematics (ν_min = (M²_min − m_p²)/2m_p). Its level is 0.64–0.68 of
  K2's at both energies (Δ = 0.01–0.03), which fixes
  **`sd_scale = 0.65`**. Above x_F ≈ 0.9, FTF adds non-diffractive
  leading protons, which count as absorption here.
- *Nuclei.* FTF's picture differs.
  - Near-elastic peak: FTF's is ≈ 10% of σ_inel for N, the same size as our
    Glauber–Gribov QE, but it sits at an energy loss of ≈ 0.35 GeV (like a
    Δ(1232) excitation of the struck nucleon) instead of pure pN elastic
    kinematics. At injection that is Δp/p ≈ 10⁻³ instead of ~10⁻⁴.
  - High-mass diffraction: FTF has almost no surviving protons between
    Δp/p = 10⁻² and 10⁻¹, while K2 × 0.65 puts ≈ 4–6% of σ_inel there.

  This is the main remaining physics uncertainty (§7.2). `sd_scale` accepts
  a per-species dict, e.g. `{'H': 0.65, 'C': 0.1, ...}`, to follow FTF for
  nuclei.

**Knock-on.** P(single hIoni transfer > vE) agrees within 1–2σ at v = 10⁻³
and 10⁻² for all cases. At v = 10⁻⁴ and 450 GeV, Geant4 is 10–14% higher
because the recorded step also carries the ~10 MeV continuous restricted
loss.

**Coulomb and electrons.** P(θ > θ₀) for protons without hadronic
interaction:

- at θ₀ = 30 × θ_MCS it agrees within statistics;
- at 10 × θ_MCS Geant4 is 3–10% higher, which is plural-scattering tails
  not in a single-scattering model.

### 4.3 LHC example

`examples/beamgas_proton_lhc.py`: LHC Run 3 B1 at 6.8 TeV with apertures and
Everest collimators (`machines/lhc_run3_b1.json`, `colldbs/lhc_run3.yaml`).

- **Target section:** 100 m of arc 67 (s = 18000–18100 m, ≈ 1.8 km upstream
  of the IR7 TCPs), with 20 scattering elements and perfect vacuum
  elsewhere.
- **Gas cases:**
  - `'uniform'`: H₂ 1e15, CH₄ 1e14, CO 3e14, CO₂ 1e14 molecules/m³;
  - `'profile'`: read from `data/beamgas_pressure_profile_lhc_arc67.csv`, a
    synthetic H₂ background plus a CH₄/CO/CO₂ bump, clearly labelled as
    not real data.
- **Run:** all processes, 100 events per element and process, 20 turns,
  out-of-bucket drift at 5·10⁻⁶ per turn.
- **Outputs:** process-rate table, lifetime, loss map (absorption in gas,
  collimators, aperture), and the absorption source CSV.

**Results** (28-core CPU, **3.5 min** for both cases, including ~60 s of
machine setup and kernel compilation; reproducible with the seed). Rates are
for the 100 m section only (bunch of 1.4·10¹¹); lifetimes are ~5000 h.

| uniform gas | interaction rate [1/s] | loss rate [1/s] | lost fraction |
|---|---|---|---|
| absorption | 5395 | 5395 (exact) | 1 |
| elastic (θ ≥ 0.1 µrad) | 4.66·10⁵ | 1281 ± 113 | 0.3% |
| quasi-elastic | 558 | 509 ± 4 | 91% |
| diffractive | 459 | 449 ± 2 | 98% |
| knock-on (T ≥ 10⁻⁶ E) | 5548 | 29 ± 3 | 0.5% |
| **total** | 4.78·10⁵ | **7663 ± 110** | τ = 5075 ± 75 h |

Where the loss rate goes:

- 70% absorbed in the gas section;
- about 25% on the IR7 collimators (TCP.D/C/B6L7 10.1/7.4/2.4%, then the
  TCSGs);
- 3.0% on the aperture, mostly in the arc right downstream (large-ξ
  diffraction) and in the IR7 dispersion suppressor;
- ~10⁻³ in IR3 from the protons pushed out of the RF bucket and drifted to
  the momentum cleaning.

No out-of-bucket protons remain after the drift. The elastic rate above
50 mrad is negligible (10⁻¹⁹ 1/s).

The profile case (CO/CO₂ bump) gives 9160 ± 210 1/s and τ = 4245 ± 96 h,
with 69% absorbed in the gas, a loss peak at the bump, and the same
downstream pattern. The absorbed protons are written to
`beamgas_absorption_events_<case>.csv` (s, coordinates, momentum, gas, Z,
A, weight), ready to be turned into a FLUKA source. The plot is
`examples/plots/beamgas_proton_lhc/beamgas_proton_lhc_lossmap.png`.

---

## 5. Commits

1. `8ec34b12` Freeze the e± results in a regression test.
2. `43e401eb` Proton-gas cross sections and samplers ported from Geant4, the
   Geant4 reference data with its generator, and unit tests.
3. `d2606a01` Proton beam-gas studies dispatched on the beam species:
   absorption as lost particles, RF-bucket truncation, out-of-bucket
   report/drift, per-process results, molecular densities.
4. `5bf42cd0` Geant4 thin-target benchmark, and the SD calibration
   (`sd_scale = 0.65`, per-species override).
5. `8a4b9c63` LHC example (`examples/beamgas_proton_lhc.py`), its synthetic
   pressure profile, and this document updated to what was done.
6. `dfafee8e` Per-particle random generators (used by the Everest
   collimators) seeded from the study, for reproducibility with collimators.
7. Updated example results in this document.

---

## 6. Options assessment (from the plan)

### 6.1 Everest (K2 heritage): reused for the SD formulas only

**Useful.** The pp fits agree with Geant4:

- σ_tot 39.7/47.2 vs 40.1/47.0 mb;
- σ_el 7.07/8.65 vs 7.19/9.12 mb;
- b_pp 12.0/14.0 vs 12.2/14.3 GeV⁻² (CHIPS);
- nuclear slopes 14.1·A^0.65 within 7% of CHIPS.

**Not reusable for gas:**

- **Coupling.** The code is tied to `MaterialData`, `EverestData` and the
  jaw loop.
- **No gas data.** There are no reference cross sections for H, He, N, O,
  Ne or Ar.
- **Hydrogen.** `N_eff(A=1) = 1.62` instead of 1.
- **Coulomb normalisation.** `cross_section[5]` is ~113–174× below the
  analytic single-Coulomb cross section in the same \|t\| window (Be, C,
  Pb). Its fixed cut \|t\| > 10⁻³ GeV² is θ > 70 µrad at 450 GeV. **Worth
  checking for collimators too**, though MCS dominates there.
- **MCS and ionisation.** These are thick-target models.

### 6.2 Geant4: used for everything but SD

What FTFP_BERT uses for protons:

- **Inelastic:** `G4BGGNucleonInelasticXS`, with final state FTFP. There is
  no CHIPS quasi-elastic channel in FTFP_BERT (`quasiElastic=false`): QE
  happens inside FTF.
- **Elastic:** `G4BGGNucleonElasticXS` with `G4ChipsElasticModel`.
- **EM:**
  - single Coulomb scattering is `G4eCoulombScatteringModel` +
    `G4WentzelOKandVIxSection`, not `G4hCoulombScatteringModel`, which is
    only used in the `_SS`/`_WVI` lists;
  - msc is WentzelVI;
  - hIoni is BetheBloch;
  - hBrems and hPairProd are also present.

### 6.3 Xcoll Geant4/FLUKA couplings: not used

`g4interface` (collimasim) and FLUKA are not installed here, and both are
built around collimator jaws. A standalone FTFP_BERT thin-target
application (§4.2) proved simpler and more controllable for benchmarking.

---

## 7. Decisions, open points and limitations

### 7.1 Your decisions (review of the plan)

| Topic | Decision |
|---|---|
| QE | Glauber–Gribov σ_in − σ_prod |
| SD | K2 shape; normalisation fixed by the benchmark (`sd_scale = 0.65`) |
| CNI | elastic and Coulomb sampled together |
| EM tails | knock-on included; pair production left out |
| Absorbed protons | lost particles with a dedicated state (`LOST_ON_BEAMGAS`) |
| 10⁶-turn off-momentum losses | avoided with the drift continuation |
| API and language | process API free; Python |
| Licence | no notice; credit by class names |
| LHC example | 100 m section only |

### 7.2 Open physics points

1. **SD on nuclei (main uncertainty).** FTF predicts far fewer surviving
   protons than the K2 A-scaling, and places the near-elastic nuclear peak at
   ≈ 0.35 GeV energy loss (§4.2). Survivors vs local absorption on nuclei is
   uncertain by a factor of a few. Data on pA → pX at √s ~ 30–110 GeV would
   settle it. Meanwhile, use the per-species `sd_scale`.
2. **QE energy loss.** Fermi motion and nuclear excitation are neglected
   (Δp/p ≈ \|t\|/2m_p p). At injection FTF's near-elastic protons lose about
   10× more. This matters for momentum acceptance only near 10⁻³ at 450 GeV.
3. **CNI.** Uses ρ_pA ≈ ρ_pp and the simple West–Yennie phase with Zα, which
   is approximate for heavy nuclei (Zα Φ ~ 0.6 rad for Ar at small \|t\|).
   It is a 2–5% effect on the elastic loss rate.
4. **Coulomb form factor.** Exponential (Geant4's default). At 6.8 TeV it
   already suppresses Ar by ~20% at 10 µrad; a realistic charge form factor
   changes this at the ~10% level.

### 7.3 Limitations and choices of the implementation

- **Beyond the scope of single scattering:**
  - no MCS and no mean ionisation (item 5: emittance growth only; mean loss
    0.04–0.23 eV/turn at 10¹⁵ m⁻³);
  - no e⁺e⁻ pair production (up to 15% of SD for Ar at 6.8 TeV,
    Δp/p ≳ 10⁻³) and no proton bremsstrahlung (negligible).
- **Not done:** the MCS emittance-growth diagnostic (∫θ²dσ below θ_min)
  proposed in the plan.
- **Energy range:** p ≥ 91 GeV kinetic (the Barashenkov part of BGG is not
  ported), and Z ≤ 92.
- **Molecules:** additivity; the gas is given as atomic densities, with a
  helper for molecules.
- **Cross sections:** evaluated at the reference momentum; the per-particle
  δ only enters the kinematics.
- **RF bucket:** stationary, single harmonic, from the line cavities. The
  primaries are truncated at the separatrix (the Gaussian has tails outside
  it).
- **Drift mode:** mimics the SR energy loss at top energy, with a
  user-chosen rate. Larger rates give larger impact parameters at the
  momentum collimators. It is not meaningful at injection, where
  out-of-bucket protons coast; use `'report'` there.
- **Absorbed protons:** `LOST_ON_BEAMGAS` is not in `USE_IN_LOSSMAP`, so
  `xc.LossMap` shows collimator and aperture losses only. The example builds
  the full map (gas + collimator + aperture) itself.
- **Prebuilt Xsuite kernels:** these are matched by class name and version.
  The implementation deliberately leaves the `BeamGasScattering` kernel and
  xofields untouched, so stale prebuilt kernels cannot interfere.

### 7.4 Suggested next steps

1. Decide on SD for nuclear targets: follow FTF with a per-species default,
   keep K2, or confront pA → pX data. The same for the QE energy loss
   (Fermi motion and Δ excitation).
2. Injection: build a 450 GeV LHC lattice with apertures (e.g. from
   `xtrack/test_data/lhc_2024/injection_optics.madx` plus the aperture
   model) and run the example there. The out-of-bucket protons need a
   dedicated treatment (abort-gap population).
3. Optionally add e⁺e⁻ pair production (a `G4hPairProductionModel` table) for
   top-energy momentum losses with heavy gases.
4. Add the MCS emittance-growth diagnostic.
5. Loss maps: either teach `xc.LossMap` the `LOST_ON_BEAMGAS` state, or add
   a `BeamGasResult.loss_map()` helper, and export the absorption table in
   FLUKA's source format.
6. Benchmark losses end-to-end (IR7/IR3/DS pattern) against a FLUKA or
   Geant4 simulation of the same gas bump, and against measured
   beam-gas-induced losses if available.
