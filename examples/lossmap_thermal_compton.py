# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
#
# Compton scattering of a stored electron beam on the thermal (blackbody)
# photons of the vacuum chamber: loss map and beam lifetime.
#
# The example follows the following workflow: 
#   1. insert scattering centres around the ring,
#   2. install apertures,
#   3. configure the study (beam parameters, temperature, thinning threshold),
#   4. generate the weighted scattered particles and track them,
#   5. build the loss map and the lifetime from the weights of the lost
#      particles.
#
import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import xcoll as xc

######################################################
# Beam parameters
######################################################
nemitt_x = 1e-5
nemitt_y = 1e-7

sigma_z = 4e-3
sigma_delta = 1e-3

beam_current = 0.3            # [A]
temperature = 300.0           # [K] warm vacuum chamber

######################################################
# Load a toy ring and make it an 18 GeV electron ring
######################################################
line = xt.load('./machines/four_cell_ring.json')
env = line.env

line.set_particle_ref('electron', p0c=10e9)

######################################################
# Insert thermal Compton scattering centers
######################################################
tt = line.get_ttle()
tt_bends_quads = tt.rows[
    (tt.element_type == 'Bend') | (tt.element_type == 'Quadrupole')
]

placements = []
for ii, nn in enumerate(tt_bends_quads.name):
    name = f'ThermalCompton.{ii}'
    env.elements[name] = xc.ThermalComptonScattering()
    placements.append(env.place(name, at=0.0, from_=nn))

name = f'ThermalCompton.{ii+1}'
env.elements[name] = xc.ThermalComptonScattering()
placements.append(env.place(name, at=tt.s[-1]))

line.insert(placements)

######################################################
# Install apertures
######################################################
tt = line.get_ttle()
needs_aperture = tt.rows.match_not(element_type='Drift.*|Marker|').name

aper_size = 0.040
env.new('aper', xt.LimitRect,
        min_x=-aper_size, max_x=aper_size,
        min_y=-aper_size, max_y=aper_size)

placements = []
for nn in needs_aperture:
    env.new(f'{nn}_aper_entry', 'aper')
    env.new(f'{nn}_aper_exit', 'aper')
    placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))
    placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))

line.insert(placements)

######################################################
# Thermal Compton study
######################################################
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

nturns = 200

compton = xc.ThermalComptonStudy(
    line=line,
    temperature=temperature,
    beam_current=beam_current,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    sigma_delta=sigma_delta,
    n_macroparticles=2000,     # local beam sample at each scattering centre
    n_trials=200,              # scattering trials per macro-particle
    delta_threshold=1e-3,      # thinning threshold, NOT the machine acceptance
    section_assignment='centered',
    seed=1997,
    method='4d',               # forwarded to line.twiss()
)

compton.initialise()

print(f'Photon density:            {compton.photon_density*1e-6:.3e} cm^-3')
print(f'Mean photon energy:        {compton.mean_photon_energy:.4f} eV')
print(f'Stored particles:          {compton.n_particles:.3e}')
print(f'Analytic scattering rate:  {compton.analytic_rate()*1e-3:.3f} kHz')
print(f'Minimum possible lifetime: {compton.minimum_lifetime()/3600:.2f} h')

result = compton.run(
    track=True,
    n_turns=nturns,
    keep_particles=True,
    with_progress=1,
)

print(result.local_rates)

print(f'Compton scattering rate:  {result.rate_scattering*1e-3:.3f} kHz '
      f'(analytic: {compton.analytic_rate()*1e-3:.3f} kHz)')
print(f'Rate above threshold:     {result.rate_tail*1e-3:.3f} kHz')
print(f'Energy loss rate:         {result.energy_loss_rate*1.602e-19:.3e} W '
      f'(analytic: '
      f'{compton.analytic_energy_loss_rate()*1.602e-19:.3e} W)')
print(f'Tracked loss rate:        {result.rate_tracking*1e-3:.3f} kHz')
print(f'Thermal Compton lifetime: {result.lifetime_tracking/3600:.2f} h')

######################################################
# Optional: refine loss locations
######################################################
loss_loc_refinement = xt.LossLocationRefinement(
    line,
    n_theta=360,
    r_max=0.5,
    dr=50e-6,
    ds=0.1,
)
loss_loc_refinement.refine_loss_location(result.particles)
lost_particles = result.particles.filter(result.particles.state == 0)

######################################################
# Plot: thermal Compton loss map
######################################################
circumference = line.get_length()
binwidth = 0.1

plt.figure()
plt.title(f'Toy ring thermal Compton loss map '
          f'(lifetime: {result.lifetime_tracking/3600:.2f} h)')
plt.hist(
    lost_particles.s,
    bins=np.arange(0, circumference + binwidth, binwidth),
    weights=lost_particles.weight*1e-3,
)
plt.xlabel('s [m]')
plt.ylabel('Loss rate [kHz]')
plt.grid()

######################################################
# Plot: momentum deviation of the generated events
######################################################
# Allocated slots only (unallocated ones carry state = -999999999)
generated = result.particles.filter(result.particles.state > -1e5)

plt.figure()
plt.hist(generated.delta*100, bins=100, weights=generated.weight*1e-3,
         histtype='step', label='generated')
plt.hist(lost_particles.delta*100, bins=100,
         weights=lost_particles.weight*1e-3, histtype='step', label='lost')
plt.yscale('log')
plt.xlabel(r'$\delta$ [%]')
plt.ylabel('Rate [kHz]')
plt.legend()
plt.grid()

plt.show()