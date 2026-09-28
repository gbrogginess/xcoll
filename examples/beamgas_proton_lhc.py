# copyright ############################### #
# This file is part of the Xcoll Package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""
Proton beam-gas loss map of the LHC (beam 1, 6.8 TeV).

A 100 m section of the arc 67, about 1.8 km upstream of the IR7 primary
collimators, is filled with a relatively high residual-gas pressure for the
LHC; everywhere else the vacuum is perfect, so beam-gas scattering elements
are only needed in that section. Two gas configurations are simulated:

* 'uniform': constant H2, CH4, CO and CO2 densities in the section;
* 'profile': a pressure profile read from a file
  (``data/beamgas_pressure_profile_lhc_arc67.csv``, synthetic).

For each, the protons that interact with the gas (absorption, elastic with
Coulomb and interference, quasi-elastic, single diffraction and knock-on
electrons) are tracked with the Everest collimators; the protons pushed out
of the RF bucket are followed with an accelerated energy loss to the
momentum cleaning. The result is the beam-gas lifetime, the rates per
process, a loss map (absorption in the gas, collimators, aperture) and the
table of absorbed protons, i.e. the source of the hadronic showers for a
FLUKA simulation.

Runs in a few minutes on a multi-core CPU. Run it from the examples folder
(OpenMP kernels cannot be compiled from the root of the Xcoll repository).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import xcoll as xc

start_time = time.time()

path_in = Path(__file__).parent
path_out = Path.cwd()

GAS_CASES = ['uniform', 'profile']

######################################################
# Beam parameters (Run 3, flat top)
######################################################
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
sigma_z = 0.09
sigma_delta = 1.1e-4
bunch_intensity = 1.4e11

######################################################
# Target section
######################################################
s_start = 18000.   # m, arc 67 of beam 1
s_end = 18100.
n_scattering_elements = 20

# Uniform case: molecular densities [molecules/m^3]
uniform_density = {'H2': 1e15, 'CH4': 1e14, 'CO': 3e14, 'CO2': 1e14}

######################################################
# Load the machine and install the collimators
######################################################
env = xt.load(path_in / 'machines' / 'lhc_run3_b1.json')
line = env['lhcb1']
circumference = line.get_length()

colldb = xc.CollimatorDatabase.from_yaml(
    path_in / 'colldbs' / 'lhc_run3.yaml', beam=1)
colldb.install_everest_collimators(line=line, verbose=False)

######################################################
# Install the beam-gas scattering centres
######################################################
# Each BeamGasScattering element represents the section between the previous
# one and itself. The first one also stands for everything upstream of the
# target section, where the gas density is zero.
placements = []
for ii, ss in enumerate(np.linspace(s_start, s_end,
                                    n_scattering_elements + 1)[1:]):
    name = f'beamgas.{ii}'
    env.elements[name] = xc.BeamGasScattering()
    placements.append(env.place(name, at=ss))
line.insert(placements)

line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))
line.xcoll.collimators.assign_optics()
print(f"Machine ready in {time.time() - start_time:.0f}s.")


######################################################
# Gas-density profiles
######################################################
def with_perfect_vacuum_outside(s, molecular):
    """Pad a profile of the target section with zero density elsewhere
    (the profile is interpolated linearly, and clamped beyond its ends)."""
    eps = 1e-6
    s_pad = np.concatenate(([0.0, s[0] - eps], s, [s[-1] + eps,
                                                   circumference]))
    out = {'s': s_pad}
    for kk, vv in molecular.items():
        out[kk] = np.concatenate(([0.0, 0.0], vv, [0.0, 0.0]))
    return out


def uniform_gas():
    s = np.array([s_start, s_end])
    molecular = {kk: np.full(2, vv) for kk, vv in uniform_density.items()}
    return xc.beamgas.molecular_to_atomic_density(
        with_perfect_vacuum_outside(s, molecular))


def profile_gas():
    data = pd.read_csv(
        path_in / 'data' / 'beamgas_pressure_profile_lhc_arc67.csv',
        comment='#')
    molecular = {kk: data[kk].to_numpy() for kk in data.columns
                 if kk != 's'}
    return xc.beamgas.molecular_to_atomic_density(
        with_perfect_vacuum_outside(data['s'].to_numpy(), molecular))


######################################################
# Beam-gas studies
######################################################
tab = line.get_table()
tab_coll = tab.rows[tab.element_type == 'EverestCollimator']
# Loss position of a collimator: its centre
collimator_s = {nn: ss + 0.5*line[nn].length
                for nn, ss in zip(tab_coll.name, tab_coll.s)}

results = {}
for case in GAS_CASES:
    gas_density = uniform_gas() if case == 'uniform' else profile_gas()

    study = line.xcoll.beamgas_configure(
        gas_density=gas_density,
        process='all',
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        sigma_delta=sigma_delta,
        bunch_intensity=bunch_intensity,
        # Per scattering element and per process
        n_scattering_events=100,
        # Wide Coulomb window: the log-uniform sampling makes it cheap
        coulomb_theta=(1e-7, 50e-3),
        seed=20260928,
        verbose=False,
    )

    line.xcoll.scattering.enable()
    result = study.run(
        track=True,
        n_turns=20,
        keep_particles=True,
        # Protons pushed out of the RF bucket are brought to the momentum
        # cleaning with an accelerated synchrotron-radiation energy loss
        out_of_bucket='drift',
        drift_per_turn=5e-6,
    )
    line.xcoll.scattering.disable()
    results[case] = (study, result)

    print(f"\n######## Gas case: {case} ########")
    print(result.process_rates.cols['xsec_mb interaction_rate '
                                    'rate_tracking rate_tracking_error '
                                    'loss_fraction'])
    print(f"Beam-gas interaction rate: {result.rate_scattering:.4g} 1/s")
    print(f"Beam-gas loss rate:        {result.rate_tracking:.4g} "
          f"+- {result.rate_tracking_error:.2g} 1/s")
    print(f"Beam-gas lifetime:         {result.lifetime_tracking/3600:.4g} "
          f"+- {result.lifetime_tracking_error/3600:.2g} h "
          f"(target section only)")
    print(f"Elastic rate above coulomb_theta[1]: "
          f"{result.rate_above_theta_max:.2g} 1/s")
    print(f"Still outside the RF bucket after the drift: "
          f"{result.rate_out_of_bucket:.2g} 1/s")

    # Source of the hadronic showers: the absorbed protons
    source = study.absorption_events(result)
    source.to_pandas().to_csv(
        path_out / f'beamgas_absorption_events_{case}.csv', index=False)
    print(f"Wrote {len(source.s)} absorbed protons "
          f"({source.weight.sum():.4g} 1/s) to "
          f"beamgas_absorption_events_{case}.csv")

print(f"\nTotal time: {time.time() - start_time:.0f}s.")


######################################################
# Loss map
######################################################
def loss_table(result):
    """Weighted losses by type: absorption in the gas, collimators and
    aperture, with their s position [m]."""
    part = result.particles
    lost = part.filter((part.particle_id >= 0) & (part.state <= 0))
    names = np.array(line.element_names)[lost.at_element]
    gas = lost.state == xc.constants.LOST_ON_BEAMGAS
    coll = np.isin(lost.state, [xc.constants.LOST_ON_MATERIAL,
                                xc.constants.LOST_ON_MATERIAL_SEC])
    aper = lost.state == 0
    s_coll = np.array([collimator_s.get(nn, np.nan) for nn in names[coll]])
    return {'gas': (lost.s[gas], lost.weight[gas]),
            'collimator': (s_coll, lost.weight[coll], names[coll]),
            'aperture': (lost.s[aper], lost.weight[aper])}


fig, axes = plt.subplots(len(GAS_CASES), 1, figsize=(12, 4*len(GAS_CASES)),
                         sharex=True)
axes = np.atleast_1d(axes)
binwidth = 10.
bins = np.arange(0, circumference + binwidth, binwidth)
for ax, case in zip(axes, GAS_CASES):
    study, result = results[case]
    losses = loss_table(result)
    total = result.rate_tracking
    for kind, color in (('gas', 'tab:green'), ('aperture', 'tab:blue'),
                        ('collimator', 'black')):
        s, w = losses[kind][:2]
        ax.hist(s, bins=bins, weights=w/total, color=color, label=kind,
                histtype='stepfilled' if kind == 'collimator' else 'step')
    ax.set_yscale('log')
    # The importance weights of the rare far-tail events are tiny
    ax.set_ylim(1e-7, 1)
    ax.set_ylabel('fraction of the loss rate per 10 m')
    ax.set_title(f"LHC B1 6.8 TeV, beam-gas in arc 67 ({case}): lifetime "
                 f"{result.lifetime_tracking/3600:.3g} h")
    ax.legend()
    ax.grid()

    # Summary of the main loss locations
    s_coll, w_coll, n_coll = losses['collimator']
    uniq, inv = np.unique(n_coll, return_inverse=True)
    per_coll = np.bincount(inv, weights=w_coll)
    order = np.argsort(per_coll)[::-1][:8]
    print(f"\nMain collimator losses ({case}):")
    for ii in order:
        print(f"  {uniq[ii]:20s} {per_coll[ii]/total:8.2%}")
    for kind in ('gas', 'aperture'):
        print(f"  {kind:20s} {losses[kind][1].sum()/total:8.2%}")
axes[-1].set_xlabel('s [m]')
plt.tight_layout()
plt.savefig(path_out / 'beamgas_proton_lhc_lossmap.png', dpi=120)
plt.show()
