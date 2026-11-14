# copyright ############################### #
# This file is part of the Xcoll Package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
from xcoll.thermal_photons import ThermalComptonCalculator

######################################################
# Beam parameters
######################################################
nemitt_x = 1e-5
nemitt_y = 1e-7

sigma_z = 4e-3
sigma_delta = 1e-3

bunch_intensity = 4e9

######################################################
# Vacuum chamber temperature
######################################################
temperature = 300.0  # K

######################################################
# Load a toy ring
######################################################
line = xt.load('machines/four_cell_ring.json')
env = line.env

# The beam scatters on the thermal photons of the vacuum chamber, which are
# boosted to energies of the order of 4 gamma^2 k_B T. The resulting energy
# loss only matters at high energy: the toy ring is given the energy of the
# Z pole (LEP, FCC-ee), where the mean relative energy loss is 1.5 %. At the
# 1 GeV of the toy ring it would be 0.03 %.
line.set_particle_ref('electron', p0c=45.6e9)
line.configure_bend_model(core='full', edge=None)

######################################################
# Install thermal-photon scattering markers
######################################################
# Each marker represents the section of the line from the previous marker to
# itself. They are installed in the middle of each magnet, to follow the
# variations of the optical functions and of the momentum acceptance. A marker
# is also added at the end of the line, so that the sections cover the whole
# ring.
tab = line.get_table()
tab_bends_quads = tab.rows[
    (tab.element_type == 'Bend') | (tab.element_type == 'Quadrupole')
]
line.xcoll.thermal_photons.install_markers(at=list(tab_bends_quads.name))

######################################################
# Install apertures
######################################################
tab = line.get_table()
needs_aperture = tab.rows.match_not(
    element_type='Drift.*|Marker|ThermalPhotonMarker|').name

aper_size = 0.040  # m

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
# Evaluate local momentum acceptance profile
######################################################
# The scattering lowers the energy of the particle: only the negative side of
# the momentum acceptance matters.
thermal_photon_markers = line.xcoll.thermal_photons.marker_names

lma = line.get_local_momentum_acceptance(
    elements=thermal_photon_markers,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    y_offset=1e-9,
    delta_negative_limit=-0.012,
    delta_positive_limit=0.012,
    delta_step_size=1e-4,
    n_turns=1000,
    method='4d',
)

plt.plot(lma.s, lma.delta_neg*100, c='r')
plt.plot(lma.s, lma.delta_pos*100, c='r')
plt.title('Toy ring: local momentum acceptance profile')
plt.xlabel('s [m]')
plt.ylabel(r'$\delta$ [%]')
plt.grid()
plt.show()

######################################################
# Configure the thermal-photon markers
######################################################
local_momentum_acceptance_scale = 0.85
n_scattering_events = 2000
nturns = 100

line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

# Integrates the scattering rate over the section of each marker, and stores
# on the marker the interaction rate together with the temperature, the
# energy-loss cut, the local optics, the beam parameters and the Monte Carlo
# settings. Only the scatterings with a relative energy loss above the cut are
# generated: here the cut of each marker is the smallest negative momentum
# acceptance of its section times `local_momentum_acceptance_scale`; a single
# value can be given instead with `energy_loss_cut`. The configured line can
# be saved here (e.g. with line.to_json): once loaded again, it generates the
# same particles, e.g. in separate jobs.
line.xcoll.thermal_photons.configure(
    # A number for a uniform temperature, or an xt.Table with columns `s` and
    # `temperature` for a profile, e.g. with cryogenic sections (see the end
    # of this example)
    temperature=temperature,
    local_momentum_acceptance=lma,
    local_momentum_acceptance_scale=local_momentum_acceptance_scale,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    sigma_delta=sigma_delta,
    bunch_intensity=bunch_intensity,
    n_scattering_events=n_scattering_events,
    seed=1997,
    method='4d',
)

print(line.xcoll.thermal_photons.analyse().local_rates)

######################################################
# Generate and track the scattered particles
######################################################
# For each marker:
#   1. Generate the particles scattered in its section, each with one Compton
#      scattering on a thermal photon, and the record of the interactions
#   2. Track them around the ring for `nturns` turns, starting and ending at
#      the marker
particles_by_marker = {}
logs = {}
for name in thermal_photon_markers:
    particles, log = line.xcoll.thermal_photons.generate(
        at=name, return_log=True)
    line.track(particles, ele_start=name, ele_stop=name, num_turns=nturns,
               with_progress=1)
    particles_by_marker[name] = particles
    logs[name] = log

######################################################
# Thermal-photon loss rate and lifetime
######################################################
# Marker by marker, with the records of the interactions: the result also
# contains the Monte Carlo error and the scan of the energy-loss cut. The
# merged particles alone give the loss rate and the lifetime.
result = line.xcoll.thermal_photons.analyse(particles_by_marker, logs=logs)

# All of the above in one call:
# result = line.xcoll.thermal_photons.run(n_turns=nturns)

print(f'Scattering rate (any energy loss):  {result.rate_total*1e-3:.3f} kHz')
print(f'Scattering rate (above the cuts):   '
      f'{result.rate_scattering*1e-3:.3f} kHz')
print(f'Loss rate:                          {result.rate_tracking*1e-3:.3f} kHz')
print(f'Lifetime if every scattering is lost: '
      f'{result.lifetime_total/3600:.2f} h')
print(f'Thermal-photon lifetime:            '
      f'{result.lifetime_tracking/3600:.2f} '
      f'+- {result.lifetime_tracking_error/3600:.2f} h')

# For comparison, the lifetime of a ring in which every energy loss above the
# average momentum acceptance, and none below, leads to a loss
mean_acceptance = np.mean(np.abs(lma.delta_neg))
calculator = ThermalComptonCalculator(
    line.particle_ref.p0c[0], temperature, energy_loss_cut=mean_acceptance)
print(f'Lifetime for a sharp acceptance of {100*mean_acceptance:.2f} %: '
      f'{1/calculator.compute_rate()/3600:.2f} h')

# Loss rate that cuts `cut_scale` times higher would have given: the events
# just above the cuts must not be lost (loss_probability = 0 in the first
# rows), otherwise local_momentum_acceptance_scale is too high
print(result.cutoff_scan)

# The weights are rates [1/s]: merged, the samples of all the markers
# represent the thermal-photon scattering of the whole ring
particles = xt.Particles.merge(list(particles_by_marker.values()))

######################################################
# Optional: refine loss locations
######################################################
loss_loc_refinement = xt.LossLocationRefinement(
    line,
    n_theta=360,  # Angular resolution of the polygonal aperture approximation
    r_max=0.5,    # Maximum transverse aperture in m
    dr=50e-6,     # Transverse loss refinement accuracy [m]
    ds=0.1,       # Longitudinal loss refinement accuracy [m]
)

loss_loc_refinement.refine_loss_location(particles)

lost_particles = particles.filter(particles.state == 0)  # lost on an aperture

######################################################
# Plot: Toy ring thermal-photon loss map
######################################################
circumference = line.get_length()
binwidth = 0.1

plt.title(
    f'Toy ring thermal-photon loss map '
    f'(lifetime: {result.lifetime_tracking/3600:.1f} h)'
)
plt.hist(
    lost_particles.s,
    bins=np.arange(0, circumference + binwidth, binwidth),
    weights=lost_particles.weight*1e-3,
)
plt.xlabel('s [m]')
plt.ylabel('Loss rate [kHz]')
plt.grid()
plt.show()

######################################################
# Optional: temperature profile
######################################################
# The temperature can vary along the ring. It is given as a table of
# positions and temperatures, interpolated linearly; two entries at the same
# position make a step. Here the second half of the ring is a cryogenic
# section at 4.5 K: the photon density there is (4.5/300)^3 of that of the
# warm half, and its photons are too soft to cause an energy loss above the
# cut. The markers of the cold half are skipped by generate() and run().
temperature_profile = xt.Table({
    'name': np.array(['start', 'warm_end', 'cold_start', 'end']),
    's': np.array([0.0, circumference/2, circumference/2, circumference]),
    'temperature': np.array([300.0, 300.0, 4.5, 4.5]),
})

line.xcoll.thermal_photons.configure(
    temperature=temperature_profile,
    local_momentum_acceptance=lma,
    local_momentum_acceptance_scale=local_momentum_acceptance_scale,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    sigma_delta=sigma_delta,
    bunch_intensity=bunch_intensity,
    n_scattering_events=n_scattering_events,
    seed=1997,
    method='4d',
    verbose=False,
)

result_profile = line.xcoll.thermal_photons.analyse()
print(result_profile.local_rates.cols[
    's', 'temperature', 'energy_loss_cut', 'interaction_rate', 'total_rate'])
print(f'Scattering rate above the cuts with the cold half: '
      f'{result_profile.rate_scattering*1e-3:.3f} kHz '
      f'(warm ring: {result.rate_scattering*1e-3:.3f} kHz)')
