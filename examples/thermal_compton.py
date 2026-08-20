# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
#
# Validation of the thermal Compton event generator. No lattice is needed: a
# single ThermalComptonScattering element is configured "by hand" and the
# generated sample is compared with analytic expectations.
#
# The checks follow the validation suite of A. Natochii
# (BNL-229489-2026-TECH / https://github.com/eic/thermal-compton-mc), plus one
# additional check (E) on the differential (flux-weighted) rate:
#
#   A) blackbody photon density and mean photon energy,
#   B) total interaction rate: sum of the event weights vs N c n_gamma sigma,
#   C) energy loss rate vs the analytic inverse-Compton power
#      P = (4/3) sigma_T c gamma^2 beta^2 U_gamma,
#   D) maximum momentum deviation vs the kinematic limit 4 gamma^2 k / E,
#   E) mean energy transfer per scattering vs (4/3) gamma^2 <k>: this is the
#      check that fails if the incoming photon direction is sampled
#      isotropically instead of with the Moeller flux factor (1 - beta*mu),
#      in which case one obtains gamma^2 <k>, i.e. 25% too little, and a
#      loss-relevant tail underestimated by 25-45%.

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import xcoll as xc
from scipy.constants import c as C_LIGHT

E_BEAM = 18e9          # [eV]
TEMPERATURE = 300.0    # [K]
N_PARTICLES = 1e11
SECTION_LENGTH = 100.0
CIRCUMFERENCE = 3800.0
N_MACRO = 500
N_TRIALS = 500
DELTA_THRESHOLD = 0.0  # keep everything, so that the totals can be checked

particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=-1, energy0=E_BEAM)
gamma0 = float(particle_ref.gamma0[0])
beta0 = float(particle_ref.beta0[0])

n_gamma = xc.blackbody_photon_density(TEMPERATURE)
k_mean = xc.blackbody_mean_photon_energy(TEMPERATURE)

# ------------------------------------------------------------------ #
# A) photon gas
# ------------------------------------------------------------------ #
print('--- A) blackbody photon gas at T = %.0f K' % TEMPERATURE)
print(f'    n_gamma      = {n_gamma*1e-6:.4e} cm^-3   (expected ~5.48e8)')
print(f'    <k>          = {k_mean:.5f} eV          (expected ~0.0698)')

# ------------------------------------------------------------------ #
# Configure a single scattering element
# ------------------------------------------------------------------ #
section_rate = (N_PARTICLES*SECTION_LENGTH/CIRCUMFERENCE
                * C_LIGHT*n_gamma*xc.THOMSON_CROSS_SECTION)

elem = xc.ThermalComptonScattering(
    particle_ref=particle_ref,
    temperature=TEMPERATURE,
    section_length=SECTION_LENGTH,
    section_rate=section_rate,
    betx=10.0, bety=10.0, alfx=0.0, alfy=0.0,
    gemitt_x=24e-9, gemitt_y=2e-9,
    sigma_z=0.9e-2, sigma_delta=10.9e-4,
    n_macroparticles=N_MACRO,
    n_trials=N_TRIALS,
    delta_threshold=DELTA_THRESHOLD,
    max_events_per_macro=N_TRIALS,   # nothing is thinned away here
)

part = elem.scatter(seed=12345)
weight = part.weight[part.state > -1e5]
delta = part.delta[part.state > -1e5]
photon_energy = elem.photon_energy_log

# ------------------------------------------------------------------ #
# B) absolute rate
# ------------------------------------------------------------------ #
rate_mc = elem.rate_scattering
rate_thomson = section_rate
sigma_ratio = rate_mc/rate_thomson
print('--- B) absolute interaction rate')
print(f'    MC rate            = {rate_mc:.6e} Hz')
print(f'    Thomson rate       = {rate_thomson:.6e} Hz')
print(f'    sigma_KN/sigma_T   = {sigma_ratio:.5f} '
      f'(a few % below 1 at 18 GeV, as expected)')

# ------------------------------------------------------------------ #
# C) energy loss rate
# ------------------------------------------------------------------ #
u_gamma = n_gamma*k_mean
n_sec = N_PARTICLES*SECTION_LENGTH/CIRCUMFERENCE
p_analytic = (n_sec*4./3.*xc.THOMSON_CROSS_SECTION*C_LIGHT
              * gamma0**2*beta0**2*u_gamma)
print('--- C) energy loss rate (inverse Compton power)')
print(f'    MC        = {elem.energy_loss_rate:.6e} eV/s')
print(f'    analytic  = {p_analytic:.6e} eV/s')
print(f'    ratio     = {elem.energy_loss_rate/p_analytic:.5f} '
      f'(< 1 by the Klein-Nishina recoil correction)')

# ------------------------------------------------------------------ #
# D) kinematic limit and E) mean energy transfer
# ------------------------------------------------------------------ #
delta_shift = np.abs(delta)   # the incoming beam has <delta> = 0
mean_transfer = elem.energy_loss_rate/rate_mc
print('--- D) kinematics')
print(f'    max |delta|           = {np.max(delta_shift):.4e}')
print(f'    4 gamma^2 <k> / E     = {4*gamma0**2*k_mean/E_BEAM:.4e} '
      f'(mean-photon scale; the maximum can exceed it for hard photons)')
print('--- E) mean energy transfer per scattering')
print(f'    MC                    = {mean_transfer:.4e} eV')
print(f'    (4/3) gamma^2 <k>     = {4./3.*gamma0**2*k_mean:.4e} eV')
print(f'    ratio                 = '
      f'{mean_transfer/(4./3.*gamma0**2*k_mean):.4f} '
      f'(1 - O(k*/m); it would be 0.75 with isotropic photon sampling)')

# ------------------------------------------------------------------ #
# Plots
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].hist(photon_energy*1e-6, bins=100, weights=weight, histtype='step')
ax[0].set_xlabel('scattered photon energy [MeV]')
ax[0].set_ylabel('rate [Hz]')
ax[0].set_yscale('log')
ax[0].grid()

ax[1].hist(np.abs(delta)*100, bins=100, weights=weight, histtype='step')
ax[1].set_xlabel(r'$|\delta|$ of the scattered lepton [%]')
ax[1].set_ylabel('rate [Hz]')
ax[1].set_yscale('log')
ax[1].grid()

thresholds = np.logspace(-4, -1.5, 40)
tail = [np.sum(weight[np.abs(delta) > tt]) for tt in thresholds]
ax[2].loglog(thresholds*100, tail)
ax[2].set_xlabel(r'threshold on $|\delta|$ [%]')
ax[2].set_ylabel('rate above threshold [Hz]')
ax[2].grid()

plt.tight_layout()
plt.show()