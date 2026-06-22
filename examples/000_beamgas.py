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

theta_max = [10e-3, 20e-3, 30e-3, 40e-3, 50e-3, 60e-3, 70e-3, 80e-3, 90e-3, 100e-3, 200e-3, 400e-3]

xsec = []
for th in theta_max:
    beamgas_manager = xc.BeamGasManager(
        line=line,
        twiss=twiss,
        gas_density=gas_density,
        bunch_population=bunch_population,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        process='coulomb',
        coulomb_theta_max=th,
        # process='brems',
        # brems_energy_cut=1e6,
    )

    calculator =  beamgas_manager.coulomb['N']

    xsec.append(calculator.compute_xsec())

theta_max = np.array(theta_max)
xsec = np.array(xsec)

# Plot xsec vs theta_max
plt.close('all')
plt.plot(theta_max*1e3, xsec*1e28)
plt.xlabel('theta_max [mrad]')
plt.ylabel('xsec [barn]')
# plt.yscale('log')
plt.savefig('plot.png', dpi=300)