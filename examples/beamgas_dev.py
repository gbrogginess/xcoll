# copyright ############################### #
# This file is part of the Xcoll Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import xcoll as xc

######################################################
# Constants
######################################################
import scipy.constants as sc

KB = sc.Boltzmann
T_ROOM = 293.15 # K

######################################################
# Beam parameters
######################################################
nemitt_x = 1e-5
nemitt_y = 1e-7

sigma_z = 4e-3
sigma_delta = 1e-3

bunch_population = 4e9

######################################################
# Build a toy ring
######################################################
lbend = 3
angle = np.pi / 2

lquad = 0.3
k1qf = 0.1
k1qd = 0.7

# Create environment
env = xt.Environment()

# Define the line (toy ring)
line = env.new_line(components=[
    env.new('mqf.1', xt.Quadrupole, length=lquad, k1=k1qf),
    env.new('d1.1',  xt.Drift, length=1),
    env.new('mb1.1', xt.Bend, length=lbend, angle=angle),
    env.new('d2.1',  xt.Drift, length=1),

    env.new('mqd.1', xt.Quadrupole, length=lquad, k1=-k1qd),
    env.new('d3.1',  xt.Drift, length=1),
    env.new('mb2.1', xt.Bend, length=lbend, angle=angle),
    env.new('d4.1',  xt.Drift, length=1),

    env.new('mqf.2', xt.Quadrupole, length=lquad, k1=k1qf),
    env.new('d1.2',  xt.Drift, length=1),
    env.new('mb1.2', xt.Bend, length=lbend, angle=angle),
    env.new('d2.2',  xt.Drift, length=1),

    env.new('mqd.2', xt.Quadrupole, length=lquad, k1=-k1qd),
    env.new('d3.2',  xt.Drift, length=1),
    env.new('mb2.2', xt.Bend, length=lbend, angle=angle),
    env.new('d4.2',  xt.Drift, length=1),
])

# Set the reference particle
line.set_particle_ref('electron', p0c=1e9)

# Configure the bend model
line.configure_bend_model(core='full', edge=None)

######################################################
# Insert beam-gas scattering centers
######################################################
# # We insert beam-gas scattering centers in the middle of each magnet
# tab = line.get_table()
# tab_bends_quads = tab.rows[(tab.element_type == 'Bend') | (tab.element_type == 'Quadrupole')]

# for ii, nn in enumerate(tab_bends_quads.name):
#     beamgas_name = f'BeamGasScattering.{ii}'
#     env.elements[beamgas_name] = xc.BeamGasScattering()
#     line.insert(beamgas_name, at=0.0, from_=nn)

# # The last BeamGasScattering element has to be placed at the end of the line
# beamgas_name = f'BeamGasScattering.{ii+1}'
# env.elements[beamgas_name] = xc.BeamGasScattering()
# line.insert(beamgas_name, at=tab.s[-1])

tab = line.get_table()

# The last BeamGasScattering element has to be placed at the end of the line
s_beamgas_to_insert = np.linspace(0, 21.2, 16)[1:]
for ii, ss in enumerate(s_beamgas_to_insert):
    beamgas_name = f'BeamGasScattering.{ii}'
    env.elements[beamgas_name] = xc.BeamGasScattering()
    line.insert(beamgas_name, at=ss)

######################################################
# Install apertures
######################################################
tab = line.get_table()
needs_aperture = np.unique(tab.element_type)[
    ~np.isin(np.unique(tab.element_type), ["", "Drift", "Marker"])
]

aper_size = 0.040 # m

placements = []
for nn, ee in zip(tab.name, tab.element_type):
    if ee not in needs_aperture:
        continue

    env.new(
        f'{nn}_aper_entry', xt.LimitRect,
        min_x=-aper_size, max_x=aper_size,
        min_y=-aper_size, max_y=aper_size
    )
    placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))

    env.new(
        f'{nn}_aper_exit', xt.LimitRect,
        min_x=-aper_size, max_x=aper_size,
        min_y=-aper_size, max_y=aper_size
    )
    placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))

line.insert(placements)

######################################################
# Define an example flat pressure profile
######################################################
tab = line.get_table()
tt_beamgas = tab.rows[tab.element_type == 'BeamGasScattering']

# Let's consider N_2 and pressure = 1e-5 mbar at room temperature
_mbar_to_pascal = 1e2
pressure_mbar = 1e-7
pressure_pascal = pressure_mbar * _mbar_to_pascal

atomic_density = 2 * pressure_pascal / (KB * T_ROOM)

xt.Table({'name': tt_beamgas.name,
          's': tt_beamgas.s})

gas_density = xt.Table(
    {
        'name': tt_beamgas.name,
        's': tt_beamgas.s,
        'N': np.ones(len(tt_beamgas.name)) * atomic_density
    }
)

######################################################
# Beam-gas simulation
######################################################
twiss = line.twiss4d()

beamgas_manager = xc.BeamGasManager(
    line=line,
    twiss=twiss,
    gas_density=gas_density,
    bunch_population=bunch_population,
    nemitt_x=nemitt_x, nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    process='coulomb',
    coulomb_theta_max=50e-3,
    # process='brems',
    # brems_energy_cut=1e6,
)

beamgas_manager.initialise_beamgas()

# Build a CPU tracker with OpenMP multithreading to speed up tracking
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

# For each BeamGasScattering element:
#   1. Generate beam-gas-scattered macro-particles at that element
#   2. Track them around the ring for `nturns` turns, starting and ending at that element
particles_list = []
for ii, element in enumerate(tt_beamgas.name):
    s_start_elem = tab.rows[tab.name == element].s[0]

    # Generate beam-gas-scattered macro-particles
    particles = line[element].scatter(num_particles=10000)

    # Track
    print(f"\nTracking particles scattered at {element} (s = {s_start_elem:.2f} m)")
    line.track(particles, ele_start=element, ele_stop=element, num_turns=100, with_progress=1)

    particles_list.append(particles)

######################################################
# Optional: Refine loss location to improve loss map accuracy
######################################################
# NOTE: we refine each macro-particle set separately (rather than refining a
# single merged collection). The refinement is performed per particle and is
# independent, so this is physically identical to refining the merged set, but
# it keeps the per-source sets distinct so we can color them in the loss map.
loss_loc_refinement = xt.LossLocationRefinement(line,
    n_theta = 360, # Angular resolution in the polygonal approximation of the aperture
    r_max = 0.5,   # Maximum transverse aperture in m
    dr = 50e-6,    # Transverse loss refinement accuracy [m]
    ds = 0.1,      # Longitudinal loss refinement accuracy [m]
    )

# The same refinement object can be reused for every particle set: the
# interpolated aperture model is built from the line, not from the particles.
for particles in particles_list:
    loss_loc_refinement.refine_loss_location(particles)

######################################################
# Compute lifetime
######################################################
# Keep only the lost particles (state == 0) in each per-source set
lost_list = [p.filter(p.state == 0) for p in particles_list]

# Total loss rate summed over all scattering sources
loss_rate = sum(np.sum(lost.weight) for lost in lost_list)
# Compute lifetime
lifetime = bunch_population / loss_rate

######################################################
# Plot: loss map with per-source (stacked) contributions
######################################################
circumference = line.get_length()
binwidth = 0.1 # m
bins = np.arange(0, circumference + binwidth, binwidth)

# Per-source data for the stacked histogram (loss rate converted to kHz)
s_per_source = [lost.s for lost in lost_list]
w_per_source = [lost.weight * 1e-3 for lost in lost_list]
labels = [f'{nn} @ s={ss:.1f} m'
          for nn, ss in zip(tt_beamgas.name, tt_beamgas.s)]

# A distinct color per scattering source
n_src = len(lost_list)
if n_src <= 10:
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(n_src)]
elif n_src <= 20:
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(n_src)]
else:
    cmap = plt.get_cmap('turbo')
    colors = [cmap(x) for x in np.linspace(0, 1, n_src)]

plt.close('all')
plt.figure(figsize=(12, 6))
plt.title(f'Toy ring Coulomb loss map (Coulomb lifetime: {lifetime/60:.2f} min)')
plt.hist(s_per_source, bins=bins, weights=w_per_source,
         stacked=True, color=colors, label=labels)
plt.xlabel('s [m]')
plt.ylabel('Loss rate [kHz]')
plt.legend(title='Scattering source', fontsize=7, ncol=2, loc='upper right')
plt.grid()
plt.tight_layout()
plt.savefig('plot.png', dpi=300)