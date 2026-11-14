# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import constants as sc

import xobjects as xo
import xtrack as xt
import xcoll as xc
from xcoll.thermal_photons.cross_sections import (ThermalComptonCalculator,
                                                  photon_density)


P0C = 45.6e9
TEMPERATURE = 297.15
ENERGY_LOSS_CUT = 5e-3


@pytest.fixture(scope='module', autouse=True)
def allow_kernel_compilation():
    # The scattering kernel of ThermalPhotonMarker is compiled on first use
    # when the prebuilt kernels do not contain it
    with xo.settings.override(allow_kernel_compilation=True):
        yield


def _ring():
    """Four-cell electron ring of the example, without markers."""
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
    line.set_particle_ref('electron', p0c=P0C)
    line.configure_bend_model(core='full', edge=None)
    return line


BEAM = dict(nemitt_x=1e-5, nemitt_y=1e-7, sigma_z=4e-3, sigma_delta=1e-3,
            bunch_intensity=4e9)


def _ring_with_markers():
    """The ring with eight equally spaced markers, ring closed."""
    line = _ring()
    length = line.get_length()
    line.xcoll.thermal_photons.install_markers(
        at=list(np.arange(1, 8) * length / 8))
    return line


@pytest.fixture(scope='module')
def ring(allow_kernel_compilation):
    """The ring with markers and its twiss, shared by the tests."""
    line = _ring_with_markers()
    line.build_tracker()
    return line, line.twiss(method='4d')


def _configure(line, **kwargs):
    config = dict(temperature=TEMPERATURE, energy_loss_cut=ENERGY_LOSS_CUT,
                  n_scattering_events=200, seed=1997, method='4d',
                  verbose=False, **BEAM)
    config.update(kwargs)
    line.xcoll.thermal_photons.configure(**config)


def _place_markers(line, positions):
    env = line.env
    names = [f'tp.{ii}' for ii in range(len(positions))]
    for nn in names:
        env.elements[nn] = xc.ThermalPhotonMarker()
    line.insert([env.place(nn, at=ss) for nn, ss in zip(names, positions)])
    return names


#############################################################
# Generator and markers
#############################################################
def test_facade_returns_a_generator_bound_to_the_line():
    line = _ring()
    tp = line.xcoll.thermal_photons
    assert isinstance(tp, xc.ThermalPhotonGenerator)
    assert tp.line is line


def test_generator_needs_a_line():
    with pytest.raises(TypeError, match='xtrack.Line'):
        xc.ThermalPhotonGenerator(None)


def test_marker_names_are_in_line_order():
    line = _ring()
    assert line.xcoll.thermal_photons.marker_names == []

    length = line.get_length()
    names = _place_markers(line, [length, 3.0, 0.5 * length])
    assert line.xcoll.thermal_photons.marker_names == [
        names[1], names[2], names[0]]


def test_markers_are_distinct_from_beamgas_markers():
    line = _ring()
    line.xcoll.beamgas.install_markers(at=[1.0, 5.0])
    line.xcoll.thermal_photons.install_markers(at=[2.0, 6.0])
    assert line.xcoll.beamgas.marker_names == [
        'beamgas.0', 'beamgas.1', 'beamgas.2']
    assert line.xcoll.thermal_photons.marker_names == [
        'thermal_photons.0', 'thermal_photons.1', 'thermal_photons.2']


def test_repeated_marker_is_refused():
    line = _ring()
    env = line.env
    env.elements['tp'] = xc.ThermalPhotonMarker()
    line.insert([env.place('tp', at=1.0), env.place('tp', at=2.0)])
    with pytest.raises(ValueError, match='more than once'):
        line.xcoll.thermal_photons.marker_names


# The same positions given to env.place directly and to install_markers
POSITIONS = {
    'number':         lambda env: dict(at=3.0),
    'element':        lambda env: dict(at='mb1.1'),
    'start':          lambda env: dict(at='mqf.2@start'),
    'end':            lambda env: dict(at='mqd.1@end'),
    'from':           lambda env: dict(at=0.5, from_='mb2.1'),
    'from_anchor':    lambda env: dict(at=-0.1, from_='mb2.1',
                                       from_anchor='end'),
    'expression':     lambda env: dict(at='2 * s_tp'),
    'reference':      lambda env: dict(at=3 * env.ref['s_tp']),
}


@pytest.mark.parametrize('position', POSITIONS.values(), ids=POSITIONS.keys())
def test_install_markers_places_like_env_place(position):
    manual = _ring()
    env = manual.env
    env['s_tp'] = 2.6
    env.elements['tp'] = xc.ThermalPhotonMarker()
    manual.insert([env.place('tp', **position(env))])

    line = _ring()
    line.env['s_tp'] = 2.6
    names = line.xcoll.thermal_photons.install_markers(
        close_ring=False, **position(line.env))

    assert names == ['thermal_photons.0']
    assert isinstance(line['thermal_photons.0'], xc.ThermalPhotonMarker)
    tab, tab_manual = line.get_table(), manual.get_table()
    assert tab['s', 'thermal_photons.0'] == tab_manual['s', 'tp']
    # The rest of the line is cut in the same way
    np.testing.assert_array_equal(tab.s, tab_manual.s)
    np.testing.assert_array_equal(tab.element_type, tab_manual.element_type)


def test_install_markers_closes_the_ring():
    line = _ring()
    tp = line.xcoll.thermal_photons

    names = tp.install_markers(at=[1.0, 5.0])
    assert names == ['thermal_photons.0', 'thermal_photons.1',
                     'thermal_photons.2']
    assert tp.marker_names == names
    assert line.get_table()['s', 'thermal_photons.2'] == line.get_length()

    # The ring is already closed: no other marker at the end
    assert tp.install_markers(at=8.0) == ['thermal_photons.3']
    assert tp.marker_names[-1] == 'thermal_photons.2'


def test_install_markers_without_closing_the_ring():
    line = _ring()
    names = line.xcoll.thermal_photons.install_markers(
        at=[1.0, 5.0], close_ring=False)
    assert names == ['thermal_photons.0', 'thermal_photons.1']
    assert line.get_table()['s', 'thermal_photons.1'] < line.get_length()


def test_install_markers_names():
    line = _ring()
    tp = line.xcoll.thermal_photons
    line.env.elements['thermal_photons.1'] = xt.Marker()

    assert tp.install_markers(at=[1.0, 2.0], close_ring=False) == [
        'thermal_photons.0', 'thermal_photons.2']
    assert tp.install_markers(
        at=3.0, names='my_tp', close_ring=False) == ['my_tp']
    assert tp.install_markers(
        at=4.0, prefix='tp', close_ring=False) == ['tp.0']
    with pytest.raises(ValueError, match='already used'):
        tp.install_markers(at=5.0, names='my_tp')
    with pytest.raises(ValueError, match='repeated'):
        tp.install_markers(at=[5.0, 6.0], names=['a', 'a'])


def test_install_markers_checks_its_arguments():
    line = _ring()
    tp = line.xcoll.thermal_photons
    with pytest.raises(ValueError, match='`from_` has 1 entries'):
        tp.install_markers(at=[1.0, 2.0], from_=['mb1.1'])
    with pytest.raises(ValueError, match='at least one'):
        tp.install_markers(at=[])
    assert tp.marker_names == []


#############################################################
# Configuration
#############################################################
def test_configure_stores_the_configuration_on_the_markers(ring):
    line, tw = ring
    _configure(line, twiss=tw)
    tp = line.xcoll.thermal_photons
    names = tp.marker_names
    length = line.get_length()
    tab = line.get_table()
    ref = line.particle_ref
    beta_gamma = ref.beta0[0] * ref.gamma0[0]
    calc = ThermalComptonCalculator(P0C, TEMPERATURE,
                                    energy_loss_cut=ENERGY_LOSS_CUT)

    s_start = 0.0
    for nn in names:
        marker = line[nn]
        s = tab['s', nn]
        assert marker.p0c == P0C
        assert marker.ds == pytest.approx(s - s_start)
        for kk in ('alfx', 'betx', 'alfy', 'bety', 'dx', 'dpx', 'dy',
                   'dpy'):
            assert getattr(marker, kk) == tw[kk, nn]
        for kk in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
            assert getattr(marker, f'{kk}_co') == tw[kk, nn]
        assert marker.gemitt_x == pytest.approx(BEAM['nemitt_x'] / beta_gamma)
        assert marker.gemitt_y == pytest.approx(BEAM['nemitt_y'] / beta_gamma)
        assert marker.sigma_z == BEAM['sigma_z']
        assert marker.sigma_delta == BEAM['sigma_delta']
        assert marker.bunch_intensity == BEAM['bunch_intensity']
        assert marker.n_scattering_events == 200
        assert marker.seed == 1997
        assert marker.temperature == TEMPERATURE
        assert marker.energy_loss_cut == ENERGY_LOSS_CUT

        # N * c/C * n sigma_eff ds
        expected = (BEAM['bunch_intensity'] * sc.c / length * (s - s_start)
                    * photon_density(TEMPERATURE) * calc.compute_xsec())
        assert marker.interaction_rate == pytest.approx(expected, rel=1e-12)
        assert marker.mean_free_path == pytest.approx(
            calc.beta0 / (photon_density(TEMPERATURE) * calc.compute_xsec()))
        s_start = s


def test_configure_does_not_modify_the_temperature_table(ring):
    line, tw = ring
    profile = xt.Table({'name': np.array(['a', 'b']),
                        's': np.array([0.0, line.get_length()]),
                        'temperature': np.array([280.0, 320.0])})
    before = {kk: np.array(profile[kk]) for kk in ('s', 'temperature')}
    _configure(line, twiss=tw, temperature=profile)
    for kk, vv in before.items():
        np.testing.assert_array_equal(profile[kk], vv)


def test_configure_stores_a_drawn_seed(ring):
    line, tw = ring
    _configure(line, twiss=tw, seed=None)
    seeds = {line[nn].seed for nn in line.xcoll.thermal_photons.marker_names}
    assert len(seeds) == 1
    assert seeds.pop() > 0
    with pytest.raises(TypeError, match='integer'):
        _configure(line, twiss=tw, seed=1.5)
    with pytest.raises(ValueError, match='non-negative'):
        _configure(line, twiss=tw, seed=-1)


def test_configure_a_subset_of_the_markers(ring):
    line, tw = ring
    _configure(line, twiss=tw)
    tp = line.xcoll.thermal_photons
    names = tp.marker_names
    rates = [line[nn].interaction_rate for nn in names]

    _configure(line, twiss=tw, at=[names[5], names[2]], temperature=350.,
               n_scattering_events=17)
    for ii, nn in enumerate(names):
        if ii in (2, 5):
            assert line[nn].temperature == 350.
            assert line[nn].n_scattering_events == 17
            assert line[nn].interaction_rate > rates[ii]
        else:
            assert line[nn].temperature == TEMPERATURE
            assert line[nn].interaction_rate == rates[ii]


def test_configure_computes_the_twiss(ring):
    line, tw = ring
    _configure(line)
    nn = line.xcoll.thermal_photons.marker_names[3]
    assert line[nn].betx == pytest.approx(tw['betx', nn], rel=1e-9)
    assert line[nn].dx == pytest.approx(tw['dx', nn], rel=1e-9)


def test_configure_requires_the_last_marker_at_the_end_of_the_line():
    line = _ring()
    line.xcoll.thermal_photons.install_markers(at=[1.0, 5.0],
                                               close_ring=False)
    with pytest.raises(ValueError, match='must sit at the end of the line'):
        _configure(line)


#############################################################
# Generation
#############################################################
COORDINATES = ('x', 'px', 'y', 'py', 'zeta', 'delta', 'weight', 's',
               'particle_id')


def _assert_same_particles(part, other):
    alive, alive_other = part.state > 0, other.state > 0
    assert np.sum(alive) == np.sum(alive_other) > 0
    for nn in COORDINATES:
        np.testing.assert_array_equal(
            getattr(part, nn)[alive], getattr(other, nn)[alive_other],
            err_msg=f'{nn} differs')


def test_generate_at_one_or_several_markers(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tp = line.xcoll.thermal_photons
    names = tp.marker_names

    assert isinstance(tp.generate(at=names[1]), xt.Particles)
    assert list(tp.generate(at=[names[3], names[1]])) == [names[1], names[3]]
    particles, logs = tp.generate(return_log=True)
    assert list(particles) == list(logs) == names
    part, log = tp.generate(at=names[1], return_log=True)
    assert isinstance(log, xt.Table)
    assert line.element_dict[names[1]].scatter_log is not None


def test_generated_particles(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tp = line.xcoll.thermal_photons
    ref = line.particle_ref

    tab = line.get_table()
    for nn in tp.marker_names:
        part, log = tp.generate(at=nn, return_log=True)
        alive = part.state > 0
        assert np.sum(alive) == 200
        assert np.all(part.s[alive] == tab['s', nn])
        assert np.all(part.at_element[alive]
                      == line.element_names.index(nn))
        assert part.q0 == ref.q0 and part.mass0 == ref.mass0
        assert np.all(part.pdg_id[alive] == ref.pdg_id[0])
        # The weights carry the interaction rate of the section
        assert np.sum(part.weight[alive]) == pytest.approx(
            line[nn].interaction_rate, rel=1e-12)
        # One interaction per particle, above the cut
        np.testing.assert_array_equal(np.sort(log.particle_id),
                                      np.sort(part.particle_id[alive]))
        assert np.all(log.name == nn)
        assert np.all(log.energy_loss >= ENERGY_LOSS_CUT)
        assert np.all(part.delta[alive] < 0)


def test_generation_does_not_depend_on_the_other_markers(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tp = line.xcoll.thermal_photons
    names = tp.marker_names

    single_job = tp.generate()
    for nn in reversed(names):
        _assert_same_particles(tp.generate(at=nn), single_job[nn])
    for nn, part in tp.generate(at=names[::-2]).items():
        _assert_same_particles(part, single_job[nn])


def test_replicas(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tp = line.xcoll.thermal_photons
    nn = tp.marker_names[2]

    one = tp.generate(at=nn)
    first = tp.generate(at=nn, replica=0, n_replicas=3)
    replicas = [first] + [tp.generate(at=nn, replica=rr, n_replicas=3)
                          for rr in (1, 2)]

    # The same stream, weights divided by the number of replicas
    alive = one.state > 0
    np.testing.assert_array_equal(first.x[first.state > 0], one.x[alive])
    np.testing.assert_array_equal(first.delta[first.state > 0],
                                  one.delta[alive])
    np.testing.assert_allclose(first.weight[first.state > 0],
                               one.weight[alive] / 3, rtol=1e-15)
    # Independent samples, which together carry the rate once
    assert not np.array_equal(replicas[1].x[:10], replicas[2].x[:10])
    assert not np.array_equal(replicas[1].delta[:10], replicas[2].delta[:10])
    total = sum(np.sum(pp.weight[pp.state > 0]) for pp in replicas)
    assert total == pytest.approx(line[nn].interaction_rate, rel=1e-12)


def test_seed_of_the_generation(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tp = line.xcoll.thermal_photons
    nn = tp.marker_names[0]
    _assert_same_particles(tp.generate(at=nn, seed=1997), tp.generate(at=nn))
    other = tp.generate(at=nn, seed=5)
    assert not np.array_equal(other.delta[:10], tp.generate(at=nn).delta[:10])


def test_generation_from_a_reloaded_line(ring, tmp_path):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    line.to_json(tmp_path / 'line.json')
    loaded = xt.load(tmp_path / 'line.json')

    generated = line.xcoll.thermal_photons.generate()
    reloaded = loaded.xcoll.thermal_photons.generate()
    assert list(generated) == list(reloaded)
    for nn in generated:
        _assert_same_particles(reloaded[nn], generated[nn])


def test_generate_checks_its_arguments(ring):
    line = _ring_with_markers()
    tp = line.xcoll.thermal_photons
    nn = tp.marker_names[0]
    with pytest.raises(ValueError, match='not configured'):
        tp.generate(at=nn)

    _configure(line, twiss=ring[1])
    with pytest.raises(ValueError, match='n_replicas'):
        tp.generate(at=nn, n_replicas=0)
    with pytest.raises(ValueError, match='replica'):
        tp.generate(at=nn, replica=2, n_replicas=2)
    with pytest.raises(TypeError, match='not a ThermalPhotonMarker'):
        tp.generate(at='mqf.1')

    _configure(line, twiss=ring[1], at=nn, n_scattering_events=0)
    with pytest.raises(ValueError, match='must be positive'):
        tp.generate(at=nn)

    line.set_particle_ref('electron', p0c=2e9)
    with pytest.raises(ValueError, match='configure the markers again'):
        tp.generate(at=tp.marker_names[1])


#############################################################
# Analysis
#############################################################
def _fake_track(part, seed):
    """
    Stand-in for tracking: half of the particles are lost, on an aperture
    (state 0) or in a collimator (LOST_ON_MATERIAL = -330), and the first
    lost one produces a lost secondary, which carries its weight.
    """
    rng = np.random.default_rng(seed)
    alive = np.flatnonzero(part.state > 0)
    lost = alive[rng.random(len(alive)) < 0.5]
    part.state[lost] = rng.choice([0, -330], size=len(lost))
    part.at_turn[alive] = 10

    free = np.flatnonzero(part.particle_id < 0)[0]
    with part._bypass_linked_vars():
        for _, nn in xt.Particles.per_particle_vars:
            getattr(part, nn)[free] = getattr(part, nn)[lost[0]]
    part.particle_id[free] = np.max(part.particle_id) + 1
    part.parent_particle_id[free] = part.particle_id[lost[0]]


def _lost_contributions(part):
    primary = (part.parent_particle_id == part.particle_id) & (
        part.particle_id >= 0)
    lost = (part.state <= 0) & primary
    return part.weight[primary] * lost[primary]


def test_analyse_without_particles(ring):
    line, tw = ring
    _configure(line, twiss=tw)
    tp = line.xcoll.thermal_photons
    names = tp.marker_names

    result = tp.analyse()
    assert isinstance(result, xc.ThermalPhotonResult)
    assert result.element_names == names
    assert result.tracked is False
    assert result.rate_generated is None and result.rate_tracking is None
    assert result.rate_scattering == pytest.approx(
        sum(line[nn].interaction_rate for nn in names))
    assert result.rate_total == pytest.approx(
        sum(line[nn].total_rate for nn in names))
    assert result.lifetime_total < result.lifetime_scattering

    # A subset of the markers
    subset = tp.analyse(at=names[2:4])
    assert subset.element_names == names[2:4]
    assert subset.rate_scattering == pytest.approx(
        line[names[2]].interaction_rate + line[names[3]].interaction_rate)


def test_analyse_merged_particles_and_samples_by_marker(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tp = line.xcoll.thermal_photons
    names = tp.marker_names

    samples = tp.generate()
    for ii, part in enumerate(samples.values()):
        _fake_track(part, seed=ii)
    contributions = {nn: _lost_contributions(pp)
                     for nn, pp in samples.items()}
    expected = sum(np.sum(cc) for cc in contributions.values())
    variance = sum(len(cc) / (len(cc) - 1)
                   * (np.sum(cc**2) - np.sum(cc)**2 / len(cc))
                   for cc in contributions.values())

    by_marker = tp.analyse(samples)
    assert by_marker.tracked is True
    assert by_marker.rate_tracking == pytest.approx(expected, rel=1e-12)
    assert by_marker.rate_tracking_error == pytest.approx(
        np.sqrt(variance), rel=1e-9)
    assert by_marker.rate_generated == pytest.approx(
        by_marker.rate_scattering, rel=1e-12)
    # The secondaries are not counted
    np.testing.assert_array_equal(by_marker.local_rates.num_particles, 200)

    merged = tp.analyse(xt.Particles.merge(list(samples.values())))
    assert merged.rate_tracking == pytest.approx(expected, rel=1e-12)
    assert merged.rate_tracking_error is None
    assert merged.cutoff_scan is None

    single = tp.analyse(samples[names[1]], at=names[1])
    assert single.rate_tracking == pytest.approx(
        np.sum(contributions[names[1]]), rel=1e-12)
    assert single.rate_tracking_error > 0


def test_analyse_with_the_records_of_the_interactions(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tp = line.xcoll.thermal_photons
    names = tp.marker_names

    samples, logs = tp.generate(return_log=True)
    for ii, part in enumerate(samples.values()):
        _fake_track(part, seed=ii)

    # Half of the events are lost whatever their energy loss: the cut
    # visibly truncates the losses
    with pytest.warns(UserWarning, match='energy-loss cut'):
        result = tp.analyse(samples, logs=logs)

    scan = result.cutoff_scan
    assert scan.num_events.sum() == 200 * len(names)
    assert scan.rate_tracking[0] == pytest.approx(result.rate_tracking,
                                                  rel=1e-12)
    assert scan.rate_tracking_error[0] == pytest.approx(
        result.rate_tracking_error, rel=1e-9)

    log = result.interaction_log
    assert len(log.particle_id) == 200 * len(names)
    for col in ('name', 'replica', 's', 'particle_id', 'energy_loss',
                'photon_energy', 'theta', 'incident_photon_energy',
                'weight'):
        assert col in log._col_names
    assert np.sum(log.weight) == pytest.approx(result.rate_scattering,
                                               rel=1e-12)

    # The records also as data frames or dicts of arrays
    frames = {nn: ll.to_pandas() for nn, ll in logs.items()}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        other = tp.analyse(samples, logs=frames)
    np.testing.assert_array_equal(other.cutoff_scan.num_lost, scan.num_lost)


def test_analyse_checks_its_arguments(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tp = line.xcoll.thermal_photons
    names = tp.marker_names
    samples, logs = tp.generate(return_log=True)

    with pytest.raises(ValueError, match='`at` cannot be given'):
        tp.analyse(samples, at=names[0])
    with pytest.raises(TypeError, match='xtrack.Particles'):
        tp.analyse({names[0]: 'particles'})
    with pytest.raises(ValueError, match='marker by marker'):
        tp.analyse(samples[names[0]], logs=logs)
    with pytest.raises(ValueError, match='same markers'):
        tp.analyse(samples, logs={names[0]: logs[names[0]]})
    with pytest.raises(ValueError, match='does not match'):
        tp.analyse({names[0]: samples[names[0]]},
                   logs={names[0]: logs[names[1]]})

    _configure(line, twiss=ring[1], at=names[0], bunch_intensity=1.0)
    with pytest.raises(ValueError, match='different bunch intensities'):
        tp.analyse()


#############################################################
# Running the simulation
#############################################################
def test_run_tracks_each_sample_from_its_marker(ring, monkeypatch):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tp = line.xcoll.thermal_photons
    names = tp.marker_names

    calls = []

    def fake_track(particles, **kwargs):
        calls.append(kwargs)
        _fake_track(particles, seed=len(calls))

    monkeypatch.setattr(line, 'track', fake_track)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # see the test with the records
        result = tp.run(n_turns=7, at=names[1:4], n_replicas=2,
                        keep_particles=True)

    assert [cc['ele_start'] for cc in calls] == 2 * names[1:4]
    assert all(cc['ele_stop'] == cc['ele_start'] and cc['num_turns'] == 7
               for cc in calls)

    # The explicit steps give the same result
    samples = {nn: [] for nn in names[1:4]}
    for rr in range(2):
        for nn in names[1:4]:
            part = tp.generate(at=nn, replica=rr, n_replicas=2)
            _fake_track(part, seed=3 * rr + names[1:4].index(nn) + 1)
            samples[nn].append(part)
    explicit = tp.analyse(samples)
    assert result.element_names == explicit.element_names
    assert result.rate_tracking == pytest.approx(explicit.rate_tracking,
                                                 rel=1e-12)
    assert result.rate_tracking_error == pytest.approx(
        explicit.rate_tracking_error, rel=1e-12)
    assert result.cutoff_scan is not None
    assert len(result.interaction_log.particle_id) == 2 * 3 * 200

    assert list(result.particles_by_element) == names[1:4]
    assert np.all(result.lost_particles.state <= 0)


def test_run_tracks_the_particles(ring):
    line, tw = ring
    # No apertures in this ring: nothing is lost
    _configure(line, twiss=tw)
    result = line.xcoll.thermal_photons.run(n_turns=3, keep_particles=True)

    assert result.tracked is True
    assert result.rate_tracking == 0
    assert result.lifetime_tracking == np.inf
    assert result.lost_particles is None
    particles = result.particles
    assert np.all(particles.at_turn[particles.state > 0] == 3)


#############################################################
# Independent jobs
#############################################################
def _assert_same_tracked_particles(part, other):
    """The same particles, in any order (matched by particle_id)."""
    def allocated_by_id(pp):
        index = np.flatnonzero(pp.particle_id >= 0)
        return index[np.argsort(pp.particle_id[index])]

    index, index_other = allocated_by_id(part), allocated_by_id(other)
    assert len(index) == len(index_other) > 0
    for nn in COORDINATES + ('state', 'at_turn', 'parent_particle_id'):
        np.testing.assert_array_equal(
            getattr(part, nn)[index], getattr(other, nn)[index_other],
            err_msg=f'{nn} differs')


def test_jobs_from_a_saved_line_merge_like_a_single_job(ring, tmp_path):
    """
    Independent jobs, each loading the saved line, generating one marker and
    replica, tracking and saving the particles and the record of their
    interactions, give the particles and the result of a single job.
    """
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    line.to_json(tmp_path / 'line.json')
    tp = line.xcoll.thermal_photons
    names = tp.marker_names
    n_replicas = 2

    single_job = {nn: [] for nn in names}
    single_job_logs = {nn: [] for nn in names}
    for rr in range(n_replicas):
        particles, logs = tp.generate(replica=rr, n_replicas=n_replicas,
                                      return_log=True)
        for nn, part in particles.items():
            _fake_track(part, seed=rr)
            single_job[nn].append(part)
            single_job_logs[nn].append(logs[nn])

    # The jobs, in reverse order, exchanging only files
    jobs = [(nn, rr) for rr in range(n_replicas) for nn in names][::-1]
    for job_id, (nn, rr) in enumerate(jobs):
        job_line = xt.load(tmp_path / 'line.json')
        part, log = job_line.xcoll.thermal_photons.generate(
            at=nn, replica=rr, n_replicas=n_replicas, return_log=True)
        _fake_track(part, seed=rr)
        part.to_json(tmp_path / f'job.{job_id}.json')
        log.to_pandas().to_csv(tmp_path / f'job.{job_id}.csv', index=False)

    jobs_output = {nn: [None] * n_replicas for nn in names}
    jobs_logs = {nn: [None] * n_replicas for nn in names}
    for job_id, (nn, rr) in enumerate(jobs):
        with open(tmp_path / f'job.{job_id}.json') as fid:
            jobs_output[nn][rr] = xt.Particles.from_dict(json.load(fid))
        jobs_logs[nn][rr] = pd.read_csv(tmp_path / f'job.{job_id}.csv',
                                        float_precision='round_trip')

    for nn in names:
        for rr in range(n_replicas):
            _assert_same_tracked_particles(
                jobs_output[nn][rr], single_job[nn][rr])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # see the test with the records
        expected = tp.analyse(single_job, logs=single_job_logs)
        result = tp.analyse(jobs_output, logs=jobs_logs)

    # Same particles, summed in another order (the loaded ones are
    # reorganized): equal to rounding
    assert result.rate_tracking == pytest.approx(
        expected.rate_tracking, rel=1e-12)
    assert result.rate_tracking_error == pytest.approx(
        expected.rate_tracking_error, rel=1e-9)
    for col in ('n_replicas', 'num_particles', 'num_lost_particles'):
        np.testing.assert_array_equal(
            result.local_rates[col], expected.local_rates[col])
    for col in ('sum_weight', 'sum_lost_weight'):
        np.testing.assert_allclose(
            result.local_rates[col], expected.local_rates[col], rtol=1e-12)
    for col in ('num_events', 'num_lost'):
        np.testing.assert_array_equal(
            result.cutoff_scan[col], expected.cutoff_scan[col])
    np.testing.assert_allclose(result.cutoff_scan.rate_tracking,
                               expected.cutoff_scan.rate_tracking, rtol=1e-12)
    for col in ('name', 'replica', 'particle_id', 'energy_loss',
                'photon_energy', 'weight'):
        np.testing.assert_array_equal(
            result.interaction_log[col], expected.interaction_log[col])
    merged = tp.analyse(xt.Particles.merge(
        [pp for vv in jobs_output.values() for pp in vv]))
    assert merged.rate_tracking == pytest.approx(
        expected.rate_tracking, rel=1e-12)
