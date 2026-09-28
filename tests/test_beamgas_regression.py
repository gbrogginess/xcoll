# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""
Regression test freezing the electron/positron beam-gas results.

The reference file was produced with the e± implementation as it was before
the proton beam-gas physics was added (branch ``beamgas``, commit
``0832d895``). Any later change must leave these numbers untouched. To
regenerate the reference (only if an intentional e± change is made), run
``python test_beamgas_regression.py`` from the ``tests`` directory.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import constants as sc

import xtrack as xt
import xcoll as xc


REFERENCE_FILE = Path(__file__).parent / 'data' / 'beamgas' / \
    'electron_positron_reference.npz'

ATOMIC_DENSITY = 2 * 1e-7 * 1e2 / (sc.Boltzmann * 293.15)

CASES = {
    'electron_brems': dict(particle='electron', process='brems',
                           brems_energy_cut=1e6),
    'electron_coulomb': dict(particle='electron', process='coulomb',
                             coulomb_theta=(1e-6, 0.3)),
    'positron_coulomb': dict(particle='positron', process='coulomb',
                             coulomb_theta=(1e-6, 0.3)),
}

N_EVENTS = 50
N_TURNS = 10
SEED = 20260928
COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta', 'weight')


def _build_toy_ring(particle):
    env = xt.Environment()
    line = env.new_line(components=[
        env.new('mqf.1', xt.Quadrupole, length=0.3, k1=0.1),
        env.new('d1.1', xt.Drift, length=1),
        env.new('mb1.1', xt.Bend, length=3, angle=np.pi/2),
        env.new('d2.1', xt.Drift, length=1),
        env.new('mqd.1', xt.Quadrupole, length=0.3, k1=-0.7),
        env.new('d3.1', xt.Drift, length=1),
        env.new('mb2.1', xt.Bend, length=3, angle=np.pi/2),
        env.new('d4.1', xt.Drift, length=1),
        env.new('mqf.2', xt.Quadrupole, length=0.3, k1=0.1),
        env.new('d1.2', xt.Drift, length=1),
        env.new('mb1.2', xt.Bend, length=3, angle=np.pi/2),
        env.new('d2.2', xt.Drift, length=1),
        env.new('mqd.2', xt.Quadrupole, length=0.3, k1=-0.7),
        env.new('d3.2', xt.Drift, length=1),
        env.new('mb2.2', xt.Bend, length=3, angle=np.pi/2),
        env.new('d4.2', xt.Drift, length=1),
    ])
    line.set_particle_ref(particle, p0c=1e9)
    line.configure_bend_model(core='full', edge=None)

    circumference = line.get_length()
    placements = []
    for ii, ss in enumerate(np.linspace(0, circumference, 9)[1:]):
        env.elements[f'BeamGasScattering.{ii}'] = xc.BeamGasScattering()
        placements.append(env.place(f'BeamGasScattering.{ii}', at=ss))
    line.insert(placements)

    tab = line.get_table()
    needs_aperture = tab.rows.match_not(
        element_type='Drift.*|Marker|BeamGasScattering|').name
    env.new('aper', xt.LimitRect, min_x=-0.04, max_x=0.04,
            min_y=-0.04, max_y=0.04)
    placements = []
    for nn in needs_aperture:
        env.new(f'{nn}_aper_entry', 'aper')
        env.new(f'{nn}_aper_exit', 'aper')
        placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))
        placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))
    line.insert(placements)
    line.build_tracker()

    tab = line.get_table()
    tt = tab.rows[tab.element_type == 'BeamGasScattering']
    gas_density = xt.Table({
        'name': tt.name,
        's': tt.s,
        'N': np.ones(len(tt.name))*ATOMIC_DENSITY,
    })
    return line, gas_density


def _make_study(line, gas_density, case):
    kwargs = {kk: vv for kk, vv in CASES[case].items() if kk != 'particle'}
    study = xc.BeamGasStudy(
        line=line, gas_density=gas_density,
        nemitt_x=1e-5, nemitt_y=1e-7, sigma_z=4e-3, sigma_delta=1e-3,
        bunch_intensity=4e9, n_scattering_events=N_EVENTS, seed=SEED,
        method='4d', **kwargs)
    study.initialise_beamgas(verbose=False)
    return study


def _sorted_by_id(particles, names):
    allocated = particles.particle_id >= 0
    order = np.argsort(particles.particle_id[allocated])
    return {nn: np.asarray(getattr(particles, nn))[allocated][order]
            for nn in names}


def _compute(case):
    """Return a flat dict of the quantities frozen by this test."""
    line, gas_density = _build_toy_ring(CASES[case]['particle'])
    out = {}

    study = _make_study(line, gas_density, case)
    for kk in sorted(study.xsecs):
        out[f'xsec/{kk}'] = np.array(study.xsecs[kk])
    for nn, particles in study.generate_particles().items():
        out[f'rate/{nn}'] = np.array(line[nn].interaction_rate)
        for kk, vv in _sorted_by_id(particles, COORDS).items():
            out[f'generated/{nn}/{kk}'] = vv
        log = line[nn].scatter_log
        out[f'log/{nn}/theta'] = np.asarray(log['theta'], dtype=float)
        out[f'log/{nn}/photon_energy'] = np.asarray(log['photon_energy'],
                                                    dtype=float)

    study = _make_study(line, gas_density, case)
    result = study.run(track=True, n_turns=N_TURNS, keep_particles=True)
    out['result/rate_scattering'] = np.array(result.rate_scattering)
    out['result/rate_tracking'] = np.array(result.rate_tracking)
    out['result/rate_tracking_error'] = np.array(result.rate_tracking_error)
    out['result/cutoff_scan_rate'] = np.asarray(
        result.cutoff_scan.rate_tracking)
    for nn, particles in result.particles_by_element.items():
        for kk, vv in _sorted_by_id(
                particles, COORDS + ('state', 'at_element', 'at_turn')).items():
            out[f'tracked/{nn}/{kk}'] = vv
    return out


@pytest.fixture(scope='module')
def reference():
    if not REFERENCE_FILE.exists():
        pytest.fail(f"Missing reference file {REFERENCE_FILE}.")
    with np.load(REFERENCE_FILE) as data:
        return {kk: data[kk] for kk in data.files}


@pytest.mark.parametrize('case', list(CASES))
def test_electron_positron_results_are_unchanged(case, reference):
    new = _compute(case)
    ref = {kk.split(':', 1)[1]: vv for kk, vv in reference.items()
           if kk.startswith(f'{case}:')}
    assert set(new) == set(ref)
    for kk in sorted(ref):
        if kk.startswith('tracked/') and kk.split('/')[-1] in (
                'state', 'at_element', 'at_turn'):
            np.testing.assert_array_equal(new[kk], ref[kk], err_msg=kk)
        else:
            np.testing.assert_allclose(new[kk], ref[kk], rtol=1e-12,
                                       atol=0, err_msg=kk)


if __name__ == '__main__':
    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for case in CASES:
        for kk, vv in _compute(case).items():
            data[f'{case}:{kk}'] = vv
    np.savez_compressed(REFERENCE_FILE, **data)
    print(f"Wrote {len(data)} arrays to {REFERENCE_FILE}")
