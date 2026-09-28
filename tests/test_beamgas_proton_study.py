# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""
Integration tests of the proton beam-gas study on a toy ring.
"""
import warnings

import numpy as np
import pytest
from scipy import constants as sc

import xtrack as xt
import xcoll as xc
from xcoll.beamgas import proton_cross_sections as pcs
from xcoll.beamgas.proton_study import ProtonBeamGasStudy


NEMITT = 2.5e-6
SIGMA_Z = 0.05
SIGMA_DELTA = 1e-4
BUNCH_INTENSITY = 1.15e11
N_DENSITY = 1e15  # N atoms/m^3
H_DENSITY = 2e15  # H atoms/m^3


def _toy_ring(p0c=450e9, aperture=0.04, rf_voltage=0.0):
    """FODO-like toy ring with 8 scattering elements, as in the e± tests,
    optionally with an RF cavity (harmonic 100)."""
    env = xt.Environment()
    components = [
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
    ]
    line = env.new_line(components=components)
    line.set_particle_ref('proton', p0c=p0c)
    line.configure_bend_model(core='full', edge=None)

    circumference = line.get_length()
    placements = []
    if rf_voltage > 0:
        f_rev = line.particle_ref.beta0[0]*sc.c/circumference
        # Above transition: stable phase at pi
        env.new('cav', xt.Cavity, voltage=rf_voltage, frequency=100*f_rev,
                phase=np.pi)
        placements.append(env.place('cav', at=0.5))
    for ii, ss in enumerate(np.linspace(0, circumference, 9)[1:]):
        env.elements[f'BeamGasScattering.{ii}'] = xc.BeamGasScattering()
        placements.append(env.place(f'BeamGasScattering.{ii}', at=ss))
    line.insert(placements)

    tab = line.get_table()
    needs_aperture = tab.rows.match_not(
        element_type='Drift.*|Marker|BeamGasScattering|Cavity|').name
    env.new('aper', xt.LimitRect, min_x=-aperture, max_x=aperture,
            min_y=-aperture, max_y=aperture)
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
        'N': np.ones(len(tt.name))*N_DENSITY,
        'H': np.ones(len(tt.name))*H_DENSITY,
    })
    return line, gas_density


def _study(line, gas_density, **kwargs):
    kwargs.setdefault('process', 'all')
    kwargs.setdefault('n_scattering_events', 300)
    kwargs.setdefault('seed', 2026)
    kwargs.setdefault('method', '4d')
    study = xc.BeamGasStudy(
        line=line, gas_density=gas_density, nemitt_x=NEMITT, nemitt_y=NEMITT,
        sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
        bunch_intensity=BUNCH_INTENSITY, **kwargs)
    study.initialise_beamgas(verbose=False)
    return study


@pytest.fixture(scope='module')
def ring():
    return _toy_ring()


def _expected_rate(line, calculators_by_species):
    """N f_rev L sum_species n sigma for the uniform gas of the toy ring."""
    f_rev = line.particle_ref.beta0[0]*sc.c/line.get_length()
    density = {'N': N_DENSITY, 'H': H_DENSITY}
    return BUNCH_INTENSITY*f_rev*line.get_length()*sum(
        density[kk]*cc.compute_xsec()
        for kk, cc in calculators_by_species.items())


#############################################################
# Construction and validation
#############################################################
class TestConstruction:

    def test_dispatches_on_the_beam_species(self, ring):
        study = _study(*ring)
        assert isinstance(study, ProtonBeamGasStudy)
        assert study.process == ('absorption', 'elastic', 'quasi_elastic',
                                 'diffractive', 'knock_on')
        assert study.q0 == 1.0

    def test_process_selection(self, ring):
        study = _study(*ring, process=['Absorption', 'coulomb', 'coulomb'])
        assert study.process == ('absorption', 'coulomb')
        study = _study(*ring, process='quasi_elastic')
        assert study.process == ('quasi_elastic',)

    @pytest.mark.parametrize('process, match', [
        ('brems', 'electron and positron'),
        ('photonuclear', 'Unknown beam-gas process'),
        (['elastic', 'coulomb'], 'cannot be combined'),
        ([], 'At least one')])
    def test_raises_on_invalid_process(self, ring, process, match):
        with pytest.raises(ValueError, match=match):
            _study(*ring, process=process)

    def test_raises_on_electron_only_options(self, ring):
        with pytest.raises(ValueError, match='brems_energy_cut'):
            _study(*ring, brems_energy_cut=1e6)

    def test_raises_below_91_GeV(self):
        line, gas = _toy_ring(p0c=50e9)
        with pytest.raises(ValueError, match='91 GeV'):
            _study(line, gas)

    def test_raises_on_wrong_charge(self, ring):
        line = ring[0].copy()
        line.particle_ref = xt.Particles(pdg_id=2212, q0=-1, p0c=450e9,
                                         mass0=xt.PROTON_MASS_EV)
        with pytest.raises(ValueError, match='q0'):
            xc.BeamGasStudy(line=line, gas_density=ring[1],
                            nemitt_x=NEMITT, nemitt_y=NEMITT,
                            sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
                            n_scattering_events=10)

    def test_events_per_process(self, ring):
        n = {'absorption': 10, 'elastic': 50}
        study = _study(*ring, process=['absorption', 'elastic'],
                       n_scattering_events=n)
        particles = study.generate_particles()[study.elements[0]]
        log = {kk: np.asarray(vv) for kk, vv in
               study.line[study.elements[0]].scatter_log.items()}
        assert np.sum(log['process'] == 'absorption') == 10
        assert np.sum(log['process'] == 'elastic') == 50
        assert np.sum(particles.particle_id >= 0) == 60
        with pytest.raises(ValueError, match='misses the processes'):
            _study(*ring, process=['absorption', 'elastic'],
                   n_scattering_events={'absorption': 10})

    def test_sd_scale_per_species(self, ring):
        study = _study(*ring, process=['absorption', 'diffractive'],
                       sd_scale={'N': 0.1})
        assert study.sd_scale == {'N': 0.1, 'H': pcs.DEFAULT_SD_SCALE}
        for kk, Z in (('N', 7), ('H', 1)):
            sd = study.calculators['diffractive'][kk]
            assert sd.sd_scale == study.sd_scale[kk]
            # The inelastic cross section is split, not changed
            xs = sd.cross_sections
            assert np.isclose(
                study.process_xsecs['absorption'][kk]
                + study.process_xsecs['diffractive'][kk],
                xs.inelastic - xs.quasi_elastic)
        with pytest.raises(ValueError, match='not in the gas'):
            _study(*ring, sd_scale={'Ar': 0.1})

    def test_line_facade(self, ring):
        study = ring[0].xcoll.beamgas_configure(
            gas_density=ring[1], nemitt_x=NEMITT, nemitt_y=NEMITT,
            sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA, n_scattering_events=10,
            method='4d', verbose=False)
        assert isinstance(study, ProtonBeamGasStudy)
        assert 'absorption' in study.process


#############################################################
# Rates and generated particles
#############################################################
class TestRates:

    def test_process_rates_match_the_cross_sections(self, ring):
        line, gas = ring
        study = _study(line, gas)
        total = 0.0
        for pp in study.process:
            rate = sum(line[nn]._process_rates[pp] for nn in study.elements)
            assert np.isclose(rate, _expected_rate(line,
                                                   study.calculators[pp]))
            total += rate
        assert np.isclose(total, sum(line[nn].interaction_rate
                                     for nn in study.elements))
        table = study.local_rates()
        assert np.allclose(sum(table[f'interaction_rate_{pp}']
                               for pp in study.process),
                           table.interaction_rate)

    def test_weights_sum_to_the_process_rates(self, ring):
        line, gas = ring
        study = _study(line, gas, n_scattering_events=2000)
        for nn, particles in study.generate_particles().items():
            log = {kk: np.asarray(vv)
                   for kk, vv in line[nn].scatter_log.items()}
            for pp in study.process:
                mask = log['process'] == pp
                rate = line[nn]._process_rates[pp]
                if pp in ('elastic', 'knock_on'):
                    # Importance-sampled: equal on average
                    assert np.isclose(log['weight'][mask].sum(), rate,
                                      rtol=0.25)
                else:
                    assert np.isclose(log['weight'][mask].sum(), rate,
                                      rtol=1e-12)
            allocated = particles.particle_id >= 0
            assert np.allclose(particles.s[allocated], line[nn].s)
            assert np.all(particles.at_element[allocated]
                          == line.element_names.index(nn))

    def test_absorbed_protons_are_lost_at_the_element(self, ring):
        line, gas = ring
        study = _study(line, gas)
        nn = study.elements[3]
        particles = line[nn].scatter(rng=study.rng)
        log = line[nn].scatter_log
        ids = particles.particle_id
        state = {int(ii): int(ss) for ii, ss in zip(ids, particles.state)
                 if ii >= 0}
        for pid, absorbed, process in zip(log['particle_id'],
                                          log['absorbed'], log['process']):
            assert absorbed == (process == 'absorption')
            expected = xc.constants.LOST_ON_BEAMGAS if absorbed else 1
            assert state[int(pid)] == expected

    def test_species_are_drawn_according_to_the_rates(self, ring):
        line, gas = ring
        study = _study(line, gas, process='absorption',
                       n_scattering_events=20_000)
        nn = study.elements[0]
        line[nn].scatter(rng=study.rng)
        gas_log = np.asarray(line[nn].scatter_log['gas'])
        sigma = {kk: cc.compute_xsec()
                 for kk, cc in study.calculators['absorption'].items()}
        p_n = N_DENSITY*sigma['N']/(N_DENSITY*sigma['N']
                                    + H_DENSITY*sigma['H'])
        assert np.isclose(np.mean(gas_log == 'N'), p_n, atol=0.01)

    def test_energy_scaling(self):
        rates = {}
        for p0c in (450e9, 6.8e12):
            line, gas = _toy_ring(p0c=p0c)
            study = _study(line, gas, process='absorption')
            rates[p0c] = sum(line[nn].interaction_rate
                             for nn in study.elements)
        expected = {p0c: _expected_rate(
            _toy_ring(p0c=p0c)[0],
            {kk: pcs.ProtonAbsorptionCalculator(Z, p0c)
             for kk, Z in (('N', 7), ('H', 1))})
            for p0c in (450e9, 6.8e12)}
        assert np.isclose(rates[6.8e12]/rates[450e9],
                          expected[6.8e12]/expected[450e9])
        assert 1.05 < rates[6.8e12]/rates[450e9] < 1.15


#############################################################
# Lifetime and bookkeeping
#############################################################
class TestLifetime:

    def test_absorption_lifetime(self, ring):
        # Uniform gas, absorption only: every event is a loss, and the
        # lifetime is exactly 1/(c sum_i n_i sigma_abs,i)
        line, gas = ring
        study = _study(line, gas, process='absorption')
        result = study.run(track=True, n_turns=1)
        sigma = {kk: cc.compute_xsec()
                 for kk, cc in study.calculators['absorption'].items()}
        beta_c = line.particle_ref.beta0[0]*sc.c
        tau = 1/(beta_c*(N_DENSITY*sigma['N'] + H_DENSITY*sigma['H']))
        assert np.isclose(result.lifetime_tracking, tau, rtol=1e-10)
        # No Monte Carlo noise (up to rounding)
        assert result.rate_tracking_error < 1e-8*result.rate_tracking

    def test_inelastic_lifetime(self):
        # With an aperture that stops every scattered proton, absorption,
        # quasi-elastic and diffractive events are all losses: the lifetime
        # is 1/(c sum_i n_i sigma_inel,i)
        line, gas = _toy_ring(aperture=1e-7)
        study = _study(line, gas, process=['absorption', 'quasi_elastic',
                                           'diffractive'])
        result = study.run(track=True, n_turns=1)
        beta_c = line.particle_ref.beta0[0]*sc.c
        sigma_inel = {kk: pcs.ProtonNucleusCrossSections(Z, 450e9).inelastic
                      for kk, Z in (('N', 7), ('H', 1))}
        tau = 1/(beta_c*(N_DENSITY*sigma_inel['N']
                         + H_DENSITY*sigma_inel['H']))
        assert np.isclose(result.lifetime_tracking, tau, rtol=1e-10)

    def test_no_aperture_only_absorption_is_lost(self):
        line, gas = _toy_ring(aperture=10.0)
        study = _study(line, gas, process=['absorption', 'elastic',
                                           'quasi_elastic'])
        result = study.run(track=True, n_turns=5)
        rates = result.process_rates
        absorption = rates['interaction_rate', 'absorption']
        assert np.isclose(result.rate_tracking, absorption, rtol=1e-6)
        # Only the elastic scattering off hydrogen at the largest |t|, with
        # a recoil of a sizeable fraction of the energy, is lost
        assert rates['rate_tracking', 'elastic'] < 1e-6*absorption
        assert rates['rate_tracking', 'quasi_elastic'] == 0

    def test_all_processes(self, ring):
        line, gas = ring
        study = _study(line, gas, n_scattering_events=1000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = study.run(track=True, n_turns=10, keep_particles=True)
        rates = result.process_rates
        assert list(rates.process) == list(study.process)
        assert np.isclose(rates.interaction_rate.sum(),
                          result.rate_scattering)
        assert np.isclose(rates.rate_tracking.sum(), result.rate_tracking)
        assert np.isclose(rates['loss_fraction', 'absorption'], 1.0)
        assert 0 < result.rate_tracking_error < 0.1*result.rate_tracking
        # rate error combines the independent strata of the processes
        assert np.isclose(result.rate_tracking_error,
                          np.sqrt(np.sum(rates.rate_tracking_error**2)))
        assert set(result.cutoff_scan) == {'elastic', 'knock_on'}
        assert result.rate_above_theta_max >= 0
        # No RF in the toy ring: no bucket
        assert result.rate_out_of_bucket is None
        log = result.interaction_log
        for col in ('process', 't', 'energy_loss', 'mass_x2',
                    'coulomb_fraction', 'absorbed'):
            assert col in log._col_names
        assert np.all(np.isfinite(log.mass_x2[log.process == 'diffractive']))
        assert np.all(np.isnan(log.mass_x2[log.process != 'diffractive']))

    def test_absorption_events(self, ring):
        line, gas = ring
        study = _study(line, gas, process=['absorption', 'elastic'])
        result = study.run(track=True, n_turns=2, keep_particles=True)
        table = study.absorption_events(result)
        assert len(table.x) == len(study.elements)*300
        assert np.isclose(table.weight.sum(),
                          result.process_rates['interaction_rate',
                                               'absorption'])
        assert set(np.unique(table.gas)) == {'N', 'H'}
        assert np.all(table.Z[table.gas == 'N'] == 7)
        assert np.all(table.A[table.gas == 'N'] == 14)
        assert np.allclose(table.pc, 450e9*(1 + table.delta))
        with pytest.raises(ValueError, match='keep_particles'):
            study.absorption_events(study.run(track=True, n_turns=1))


#############################################################
# Protons pushed out of the RF bucket
#############################################################
@pytest.fixture(scope='module')
def rf_ring():
    return _toy_ring(rf_voltage=20e6)


class TestOutOfBucket:

    def test_bucket(self, rf_ring):
        line, gas = rf_ring
        study = _study(line, gas, process='diffractive', method='6d')
        delta_max, f_rf = study._rf_bucket()
        twiss = study.twiss
        harmonic = f_rf*twiss.T_rev0
        # Small-amplitude synchrotron tune of the stationary bucket
        assert np.isclose(delta_max, 2*twiss.qs/(harmonic*twiss.slip_factor),
                          rtol=0.02)

    def test_generated_protons_start_inside_the_bucket(self, rf_ring):
        line, gas = rf_ring
        study = _study(line, gas, process='absorption', method='6d',
                       n_scattering_events=5000)
        nn = study.elements[0]
        particles = line[nn].scatter(rng=study.rng)
        # The absorbed protons keep their initial coordinates
        allocated = particles.particle_id >= 0
        particles.state[allocated] = 1
        assert not np.any(study._out_of_bucket(
            particles, line[nn], study._rf_bucket()))

    def test_report_and_drift(self, rf_ring):
        line, gas = rf_ring
        study = _study(line, gas, process='diffractive', method='6d',
                       n_scattering_events=300)
        with pytest.warns(UserWarning, match='outside the RF bucket'):
            report = study.run(track=True, n_turns=20)
        assert report.rate_out_of_bucket > 0
        study = _study(line, gas, process='diffractive', method='6d',
                       n_scattering_events=300)
        drift = study.run(track=True, n_turns=20, out_of_bucket='drift',
                          drift_per_turn=2e-4, max_drift_turns=2000)
        # The same events (same seed): the out-of-bucket protons are now lost
        assert drift.rate_out_of_bucket < 1e-3*report.rate_out_of_bucket
        assert np.isclose(drift.rate_tracking,
                          report.rate_tracking + report.rate_out_of_bucket,
                          rtol=1e-6)

    def test_raises_on_invalid_mode(self, rf_ring):
        study = _study(*rf_ring, process='absorption', method='6d')
        with pytest.raises(ValueError, match='out_of_bucket'):
            study.run(track=True, n_turns=1, out_of_bucket='ignore')
