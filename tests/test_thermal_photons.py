# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
import warnings

import numpy as np
import pytest
from scipy import constants as sc
from scipy import stats
from scipy.integrate import quad

import xobjects as xo
import xtrack as xt
import xcoll as xc
from xcoll.thermal_photons.cross_sections import (
    BOLTZMANN_EV_K, ELECTRON_MASS_EV, ZETA_3, ZETA_4,
    ThermalComptonCalculator, klein_nishina_partial_ratio,
    klein_nishina_ratio, klein_nishina_weight, mean_photon_energy,
    photon_density, planck_spectrum, thomson_xsec, _energy_loss_dxsec_ratio)


#############################################################
# Shared parameters
#############################################################
NEMITT_X = 1e-5
NEMITT_Y = 1e-7
SIGMA_Z = 4e-3
SIGMA_DELTA = 1e-3
BUNCH_INTENSITY = 4e9
# LEP at the Z pole, with the temperature of Burkhardt's note (24 C)
P0C = 45.6e9
TEMPERATURE = 297.15
ENERGY_LOSS_CUT = 5e-3

MUON_MASS_EV = 105.6583755e6


@pytest.fixture(scope='module', autouse=True)
def allow_kernel_compilation():
    # The scattering kernel of ThermalPhotonMarker is compiled on first use
    # when the prebuilt kernels do not contain it
    with xo.settings.override(allow_kernel_compilation=True):
        yield


def calculator(energy=None, temperature=TEMPERATURE, cut=0.0, **kwargs):
    """Calculator for electrons of the given total energy [eV]."""
    p0c = P0C if energy is None else np.sqrt(energy**2 - ELECTRON_MASS_EV**2)
    return ThermalComptonCalculator(p0c, temperature, energy_loss_cut=cut,
                                    **kwargs)


def four_momentum(calc, px, py, delta):
    """Energy and momentum components [eV] of particles of the calculator."""
    pc = calc.p0c * (1 + delta)
    pz = np.sqrt(pc**2 - (calc.p0c*px)**2 - (calc.p0c*py)**2)
    return (np.sqrt(pc**2 + calc.mass0**2), calc.p0c*px, calc.p0c*py, pz)


def check_four_momentum_conservation(calc, px, py, delta, px_out, py_out,
                                     delta_out, energy_loss, photon_energy,
                                     incident_photon_energy, incident_cos_psi):
    """
    Check P + K = P' + K' on the particle coordinates and the event record.

    Energy: E - E' = k' - k. Momentum transfer: (P - P')^2 = -2 K.K', with
    K.K' = y P.K and P.K = k (E - p cos(psi)), which involves the energy
    loss, the transverse kick and the incident photon at the same time.
    """
    e0, x0, y0, z0 = four_momentum(calc, px, py, delta)
    e1, x1, y1, z1 = four_momentum(calc, px_out, py_out, delta_out)
    np.testing.assert_allclose(e0 - e1, photon_energy - incident_photon_energy,
                               rtol=1e-6)

    pc = calc.p0c * (1 + delta)
    p_dot_k = incident_photon_energy * (e0 - pc*incident_cos_psi)
    transfer = (e0 - e1)**2 - (x0 - x1)**2 - (y0 - y1)**2 - (z0 - z1)**2
    np.testing.assert_allclose(transfer, -2*energy_loss*p_dot_k, rtol=2e-2)


def brown_steinke_f0(u):
    """Fraction of the scatterings above u, eq. 3.35 of Brown and Steinke."""
    return quad(lambda x: (x*x - 3*u*x + 3*u*u*np.log(x/u) + 2*u**3/x)
                / np.expm1(x), u, 80)[0] / (2*ZETA_3)


def brown_steinke_f1(u):
    """First-order correction, eq. 3.36 of Brown and Steinke."""
    return quad(lambda x: (x**3 - 4.5*u*u*x - u**3 + 6*u**3*np.log(x/u)
                           + 4.5*u**4/x) / np.expm1(x), u, 80)[0] / (6*ZETA_4)


#############################################################
# Module-level fixture: toy ring with thermal-photon markers
#############################################################
@pytest.fixture(scope='module')
def toy_ring(allow_kernel_compilation):
    """
    Build a FODO-like toy ring with ThermalPhotonMarker elements and
    apertures.

    Yields a dict with keys ``line`` and ``twiss``.
    """
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

    circumference = line.get_length()
    placements = []
    for ii, ss in enumerate(np.linspace(0, circumference, 9)[1:]):
        env.elements[f'ThermalPhotonMarker.{ii}'] = xc.ThermalPhotonMarker()
        placements.append(env.place(f'ThermalPhotonMarker.{ii}', at=ss))
    line.insert(placements)

    tab = line.get_table()
    needs_aperture = tab.rows.match_not(
        element_type='Drift.*|Marker|ThermalPhotonMarker|').name
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
    twiss = line.twiss(method='4d')

    yield {'line': line, 'twiss': twiss}


def configure(toy_ring, line=None, **kwargs):
    """Configure the markers of the toy ring with its defaults."""
    config = dict(temperature=TEMPERATURE, energy_loss_cut=ENERGY_LOSS_CUT,
                  n_scattering_events=200, seed=1997,
                  twiss=toy_ring['twiss'], nemitt_x=NEMITT_X,
                  nemitt_y=NEMITT_Y, sigma_z=SIGMA_Z,
                  sigma_delta=SIGMA_DELTA, bunch_intensity=BUNCH_INTENSITY,
                  method='4d', verbose=False)
    config.update(kwargs)
    line = toy_ring['line'] if line is None else line
    line.xcoll.thermal_photons.configure(**config)
    return line


def markers(line):
    return line.xcoll.thermal_photons.marker_names


#############################################################
# Photon field
#############################################################
class TestPhotonField:

    def test_photon_density(self):
        # Telnov, eq. 7: n = 20.2 T^3 cm^-3
        assert np.isclose(photon_density(300.), 20.2e6*300.**3, rtol=5e-3)
        # Burkhardt: 5.329e14 m^-3 at 24 C
        assert np.isclose(photon_density(297.15), 5.329e14, rtol=2e-3)
        assert np.isclose(photon_density(600.)/photon_density(300.), 8.0)

    def test_mean_photon_energy(self):
        assert np.isclose(mean_photon_energy(TEMPERATURE),
                          2.701178*BOLTZMANN_EV_K*TEMPERATURE, rtol=1e-6)
        # Burkhardt: 0.0692 eV at 24 C
        assert np.isclose(mean_photon_energy(297.15), 0.0692, rtol=1e-3)

    def test_planck_spectrum_moments(self):
        kT = BOLTZMANN_EV_K * TEMPERATURE
        density = quad(planck_spectrum, 0, 80*kT, args=(TEMPERATURE,))[0]
        energy = quad(lambda k: k*planck_spectrum(k, TEMPERATURE),
                      0, 80*kT)[0]
        assert np.isclose(density, photon_density(TEMPERATURE), rtol=1e-9)
        assert np.isclose(energy/density, mean_photon_energy(TEMPERATURE),
                          rtol=1e-9)


#############################################################
# Klein-Nishina cross section
#############################################################
class TestKleinNishina:

    def test_thomson_xsec(self):
        assert np.isclose(thomson_xsec(), 0.6652e-28, rtol=1e-4)
        # Classical radius scales as q^2 m_e / M
        assert np.isclose(thomson_xsec(MUON_MASS_EV, 1.0),
                          thomson_xsec()*(ELECTRON_MASS_EV/MUON_MASS_EV)**2)
        assert np.isclose(thomson_xsec(q0=2.0), 16*thomson_xsec())

    @pytest.mark.parametrize('eps', [0.02, 0.3, 1.0, 10.0, 300.0])
    def test_total_matches_published_forms(self, eps):
        x = 2*eps
        # Telnov, eq. 2
        telnov = 0.75/x*((1 - 4/x - 8/x**2)*np.log1p(x) + 0.5 + 8/x
                         - 0.5/(1 + x)**2)
        # Di Domenico, after eq. 10
        di_domenico = 0.75*((1/x - 4/x**2 - 8/x**3)*np.log1p(x)
                            + (1 + x/2)/(1 + x)**2 + 8/x**2)
        assert np.isclose(klein_nishina_ratio(eps), telnov, rtol=1e-11)
        assert np.isclose(klein_nishina_ratio(eps), di_domenico, rtol=1e-9)

    def test_total_at_low_energy(self):
        eps = np.array([1e-9, 1e-6, 1e-4, 1e-3])
        np.testing.assert_allclose(klein_nishina_ratio(eps),
                                   1 - 2*eps + 5.2*eps**2 - 13.3*eps**3,
                                   rtol=1e-10)
        # The series and the closed form join smoothly
        assert np.isclose(klein_nishina_ratio(0.01*(1 - 1e-9)),
                          klein_nishina_ratio(0.01*(1 + 1e-9)), rtol=1e-10)

    @pytest.mark.parametrize('eps', [1e-8, 1e-3, 0.05, 1.0, 100.0])
    def test_quadrature_reproduces_the_total(self, eps):
        assert np.isclose(klein_nishina_partial_ratio(eps),
                          klein_nishina_ratio(eps), rtol=1e-12)

    @pytest.mark.parametrize('eps,cut', [(0.02, 0.01), (0.02, 0.03),
                                         (0.5, 0.2), (5.0, 0.5),
                                         (1e-4, 1e-4)])
    def test_partial_matches_numerical_integral(self, eps, cut):
        # Energy loss above the cut: cos(theta) < 1 - a/eps
        cos_max = 1 - cut/(1 - cut)/eps
        expected = quad(lambda c: 0.375*(1 + c*c)*klein_nishina_weight(eps, c),
                        -1, cos_max)[0]
        assert np.isclose(klein_nishina_partial_ratio(eps, cut), expected,
                          rtol=1e-10)

    def test_partial_vanishes_below_threshold(self):
        # Backscattering gives y_max = 2 eps / (1 + 2 eps)
        eps = 0.01
        y_max = 2*eps/(1 + 2*eps)
        assert klein_nishina_partial_ratio(eps, 1.001*y_max) == 0
        assert klein_nishina_partial_ratio(eps, 0.999*y_max) > 0

    def test_weight_is_at_most_one(self):
        eps, cos_theta = np.meshgrid(np.geomspace(1e-6, 1e3, 40),
                                     np.linspace(-1, 1, 41))
        weight = klein_nishina_weight(eps, cos_theta)
        assert np.all(weight > 0) and np.all(weight <= 1 + 1e-15)
        np.testing.assert_allclose(klein_nishina_weight(1e-12, cos_theta), 1)

    @pytest.mark.parametrize('eps,cut', [(0.02, 0.01), (0.5, 0.2)])
    def test_energy_loss_spectrum_integrates_to_partial(self, eps, cut):
        # Telnov's eq. 4 is the Klein-Nishina formula in the energy loss
        y_max = 2*eps/(1 + 2*eps)
        integral = quad(lambda y: _energy_loss_dxsec_ratio(eps, y),
                        cut, y_max)[0]
        assert np.isclose(integral, klein_nishina_partial_ratio(eps, cut),
                          rtol=1e-9)


#############################################################
# Scattering rate
#############################################################
class TestRate:

    @pytest.mark.parametrize('flux_weighting', [True, False])
    @pytest.mark.parametrize('beta_gamma', [0.05, 0.5, 3.0, 1e3])
    def test_thomson_rate_for_any_velocity(self, beta_gamma, flux_weighting):
        # The flux factor averages to one over an isotropic gas: the rate of
        # a heavy particle is c n sigma_T whatever its velocity
        mass0 = 1e15
        calc = ThermalComptonCalculator(
            beta_gamma*mass0, 300., mass0=mass0,
            flux_weighting=flux_weighting)
        assert np.isclose(calc.compute_fraction(), 1.0, rtol=1e-9)
        assert np.isclose(calc.compute_rate(),
                          sc.c*photon_density(300.)*thomson_xsec(mass0, -1),
                          rtol=1e-9)

    @pytest.mark.parametrize('energy', [5e9, 20e9])
    def test_first_order_klein_nishina_correction(self, energy):
        # Brown and Steinke, eq. 2.18, with s = 2 E T / m^2 = eps_T
        calc = calculator(energy, 291.)
        s = calc.eps_T
        assert np.isclose(s, 2*energy*calc.kT/ELECTRON_MASS_EV**2, rtol=1e-6)
        first_order = 1 - 4*s*ZETA_4/ZETA_3
        assert abs(calc.compute_fraction() - first_order) < 30*s**2

    @pytest.mark.parametrize('u', [0.1, 0.69, 2.0, 6.0])
    def test_fraction_above_cut_in_the_thomson_limit(self, u):
        # Brown and Steinke, eq. 3.35, obtained by integrating in a
        # different order
        mass0 = 1e12
        p0c = 100*mass0
        eps_T = ThermalComptonCalculator(p0c, 300., mass0=mass0).eps_T
        cut = 2*u*eps_T/(1 + 2*u*eps_T)
        calc = ThermalComptonCalculator(p0c, 300., mass0=mass0,
                                        energy_loss_cut=cut)
        assert np.isclose(calc.xi_min, u)
        assert np.isclose(calc.compute_fraction(), brown_steinke_f0(u),
                          rtol=1e-7)

    @pytest.mark.parametrize('energy', [5e9, 46.1e9])
    @pytest.mark.parametrize('u', [0.3, 1.0])
    def test_fraction_above_cut_to_first_order(self, energy, u):
        # Brown and Steinke, eq. 3.34
        s = calculator(energy, 291.).eps_T
        calc = calculator(energy, 291., cut=2*u*s/(1 + 2*u*s))
        first_order = (brown_steinke_f0(u)
                       - 4*s*ZETA_4/ZETA_3*brown_steinke_f1(u))
        assert abs(calc.compute_fraction() - first_order) < 30*s**2

    def test_telnov_scaling_law(self):
        # Telnov, eq. 15: for small energies the fraction above the cut
        # depends only on cut / (E T)
        a = calculator(1e9, 300., cut=1e-4)
        b = calculator(2e9, 150., cut=1e-4)
        c = calculator(2e9, 300., cut=2e-4)
        assert np.isclose(a.compute_fraction(), b.compute_fraction(),
                          rtol=1e-6)
        assert np.isclose(a.compute_fraction(), c.compute_fraction(),
                          rtol=5e-3)

    @pytest.mark.parametrize('flux_weighting', [True, False])
    def test_spectrum_integrates_to_fraction(self, flux_weighting):
        calc = calculator(cut=0.01, flux_weighting=flux_weighting)
        integral = quad(calc.compute_spectrum, 0.01, 0.6, epsrel=1e-9,
                        limit=200)[0]
        assert np.isclose(integral, calc.compute_fraction(), rtol=1e-6)

    @pytest.mark.parametrize('y', [1e-3, 0.01, 0.05, 0.15])
    def test_spectrum_matches_jones_kernel(self, y):
        # Spectrum of the photons scattered by an ultra-relativistic electron
        # in an isotropic photon gas (Jones 1968; Blumenthal and Gould 1970,
        # eq. 2.48), integrated over the Planck spectrum
        calc = calculator(cut=0.0)

        def jones(kappa):
            big_gamma = 2*kappa*calc.eps_T      # 4 gamma k / m
            q = y/(big_gamma*(1 - y))
            if q >= 1:
                return 0.0
            kernel = (2*q*np.log(q) + (1 + 2*q)*(1 - q)
                      + 0.5*(big_gamma*q)**2/(1 + big_gamma*q)*(1 - q))
            return kappa**2/(2*ZETA_3*np.expm1(kappa)) * 3/big_gamma * kernel

        kappa_min = y/(1 - y)/(2*calc.eps_T)
        expected = quad(jones, kappa_min, kappa_min + 80, epsrel=1e-10,
                        limit=200)[0]
        assert np.isclose(calc.compute_spectrum(y), expected, rtol=1e-6)

    def test_rate_xsec_and_fraction_are_consistent(self):
        calc = calculator(cut=0.01)
        fraction = calc.compute_fraction()
        assert 0 < fraction < calculator().compute_fraction() < 1
        assert np.isclose(calc.compute_xsec(), thomson_xsec()*fraction)
        assert np.isclose(calc.compute_rate(),
                          sc.c*photon_density(TEMPERATURE)*calc.compute_xsec())

    def test_rate_scales_with_mass_and_charge(self):
        # Same velocity and same eps_T: a particle 10 times heavier at a
        # temperature 10 times higher, and with twice the charge
        calc = calculator(cut=0.01)
        heavy = ThermalComptonCalculator(
            10*P0C, 10*TEMPERATURE, mass0=10*ELECTRON_MASS_EV, q0=2.0,
            energy_loss_cut=0.01)
        assert np.isclose(heavy.eps_T, calc.eps_T)
        assert np.isclose(heavy.compute_fraction(), calc.compute_fraction())
        assert np.isclose(heavy.compute_xsec(), calc.compute_xsec()*16/100)

    def test_raises_on_invalid_arguments(self):
        with pytest.raises(ValueError, match='p0c'):
            ThermalComptonCalculator(0.0, 300.)
        with pytest.raises(ValueError, match='temperature'):
            ThermalComptonCalculator(P0C, 0.0)
        with pytest.raises(ValueError, match='q0'):
            ThermalComptonCalculator(P0C, 300., q0=0.0)
        with pytest.raises(ValueError, match='energy_loss_cut'):
            ThermalComptonCalculator(P0C, 300., energy_loss_cut=1.0)
        with pytest.raises(ValueError, match='energy_loss_cut'):
            ThermalComptonCalculator(P0C, 300., energy_loss_cut=-0.1)


#############################################################
# Published numbers
#############################################################
# Table 2 of Burkhardt's note: energy acceptance [%], fraction of collisions
# leading to a loss and lifetime [h], at 45.6 GeV and 90 GeV
BURKHARDT_TABLE_2 = {
    45.6e9: [(0.2, .7375, 36.33), (0.4, .6028, 44.44), (0.6, .5067, 52.87),
             (0.8, .4322, 61.98), (1.0, .3722, 71.97), (1.2, .3225, 83.07),
             (1.4, .2808, 95.41), (1.6, .2453, 109.22), (1.8, .2150, 124.63),
             (2.0, .1886, 142.03)],
    90.0e9: [(0.2, .8311, 32.94), (0.4, .7329, 37.35), (0.6, .6577, 41.63),
             (0.8, .5959, 45.94), (1.0, .5435, 50.37), (1.2, .4980, 54.97),
             (1.4, .4580, 59.77), (1.6, .4223, 64.82), (1.8, .3904, 70.13),
             (2.0, .3615, 75.73)],
}
BURKHARDT_MEAN_WEIGHT = {45.6e9: 0.9767, 90.0e9: 0.9558}


class TestPublishedNumbers:

    def test_burkhardt_normalisation(self):
        # Rate of collisions per second, metre and mA of beam current, and
        # minimum lifetime, with the Thomson cross section at 24 C
        density = photon_density(297.15)
        assert np.isclose(density*thomson_xsec()*1e-3/sc.e, 221.0, rtol=2e-3)
        assert np.isclose(1/(sc.c*density*thomson_xsec())/3600, 26.17,
                          rtol=2e-3)

    @pytest.mark.parametrize('energy', [45.6e9, 90.0e9])
    def test_burkhardt_table_without_flux_weighting(self, energy):
        # The published numbers are those of photon directions drawn
        # isotropically, without the relative-velocity factor
        mean_weight = calculator(energy, flux_weighting=False
                                 ).compute_fraction()
        assert abs(mean_weight - BURKHARDT_MEAN_WEIGHT[energy]) < 1.5e-3
        for cut, fraction, lifetime in BURKHARDT_TABLE_2[energy]:
            calc = calculator(energy, cut=cut/100, flux_weighting=False)
            assert np.isclose(calc.compute_fraction()/mean_weight, fraction,
                              rtol=0.03)
            assert np.isclose(1/calc.compute_rate()/3600, lifetime,
                              rtol=0.03)

    @pytest.mark.parametrize('energy', [45.6e9, 90.0e9])
    def test_flux_weighting_shortens_the_lifetime(self, energy):
        # With the factor, head-on photons are favoured: the energy losses
        # are larger and the lifetimes shorter than in Burkhardt's table
        for cut, _, lifetime in BURKHARDT_TABLE_2[energy]:
            calc = calculator(energy, cut=cut/100)
            assert 1/calc.compute_rate()/3600 < 0.94*lifetime
        ratio = (1/calculator(energy, cut=0.02).compute_rate()/3600
                 / BURKHARDT_TABLE_2[energy][-1][2])
        assert np.isclose(ratio, {45.6e9: 0.712, 90.0e9: 0.781}[energy],
                          atol=5e-3)

    def test_di_domenico_photons_per_bunch_passing(self):
        # Photons scattered over 600 m by a bunch of 1.37e11 electrons,
        # at 291 K (section 5 of Di Domenico)
        for energy, expected in ((45.6e9, 2.65), (100e9, 2.56)):
            calc = calculator(energy, 291.)
            number = calc.compute_rate()/(calc.beta0*sc.c)*600*1.37e11
            assert np.isclose(number, expected, rtol=5e-3)

    def test_telnov_lifetime(self):
        # Eq. 9 of Telnov: 92000 s at 300 K with the Thomson cross section
        assert np.isclose(
            1/(sc.c*photon_density(300.)*thomson_xsec()), 92000, rtol=1e-2)
        # Fig. 3 of Telnov at an energy loss of 1 %, read off the figure
        for energy, lifetime in ((15e9, 164), (30e9, 72), (50e9, 52),
                                 (100e9, 41)):
            calc = calculator(energy, 300., cut=0.01)
            assert np.isclose(1/calc.compute_rate()/3600, lifetime, rtol=0.05)

    def test_brown_steinke_lifetime(self):
        # Analytic result for LEP: 46.1 GeV, 291 K, energy loss above 1.2 %
        calc = calculator(46.1e9, 291., cut=0.012)
        assert np.isclose(calc.eps_T, 0.00886, rtol=2e-3)
        assert np.isclose(calc.xi_min, 0.69, rtol=1e-2)
        assert np.isclose(1/calc.compute_rate()/3600, 68, rtol=0.02)
        # ... to be compared with the 90 h of Di Domenico, reproduced
        # without the flux factor
        no_flux = calculator(46.1e9, 291., cut=0.012, flux_weighting=False)
        assert np.isclose(1/no_flux.compute_rate()/3600, 90, rtol=0.02)

    def test_brown_steinke_mean_energy_loss(self):
        # Eq. 3.41: <E - E'>/E = 1.80 s (1 - 8.5 s), i.e. 1.5 % at LEP,
        # against the 1.1 % of the unweighted sampling
        for flux_weighting, expected in ((True, 0.015), (False, 0.011)):
            calc = calculator(46.1e9, 291., flux_weighting=flux_weighting)
            mean = quad(lambda y: y*calc.compute_spectrum(y), 0, 0.6,
                        epsrel=1e-8, limit=200)[0]/calc.compute_fraction()
            assert np.isclose(mean, expected, atol=5e-4)
        s = calculator(5e9, 291.).eps_T
        calc = calculator(5e9, 291.)
        mean = quad(lambda y: y*calc.compute_spectrum(y), 0, 0.3,
                    epsrel=1e-9, limit=200)[0]/calc.compute_fraction()
        assert np.isclose(mean, 2*s*ZETA_4/ZETA_3*(1 - 8.5*s), rtol=1e-3)


#############################################################
# Reference sampler (numpy)
#############################################################
class TestSampler:

    @pytest.mark.parametrize('flux_weighting', [True, False])
    def test_events_are_above_the_cut_with_unit_weights(self,
                                                        flux_weighting):
        calc = calculator(cut=0.01, flux_weighting=flux_weighting)
        zeros = np.zeros(2000)
        sample = calc.sample_deflections(zeros, zeros, zeros,
                                         np.random.default_rng(1))
        assert np.all(sample.weight == 1)
        assert np.all(sample.energy_loss >= 0.01)
        assert np.all(sample.delta < -0.0099)
        assert np.all(np.abs(sample.cos_theta_rest) <= 1)

    @pytest.mark.parametrize('flux_weighting', [True, False])
    @pytest.mark.parametrize('cut', [1e-6, 5e-3, 0.03])
    def test_energy_loss_distribution(self, cut, flux_weighting):
        # The events above a higher cut are a fraction of the sample given
        # by the ratio of the rates
        n = 100_000
        calc = calculator(cut=cut, flux_weighting=flux_weighting)
        zeros = np.zeros(n)
        sample = calc.sample_deflections(zeros, zeros, zeros,
                                         np.random.default_rng(2))
        for higher in (0.01, 0.03, 0.06):
            if higher <= cut:
                continue
            expected = (calculator(cut=higher, flux_weighting=flux_weighting
                                   ).compute_fraction()
                        / calc.compute_fraction())
            sigma = np.sqrt(expected*(1 - expected)/n)
            assert abs(np.mean(sample.energy_loss > higher)
                       - expected) < 4*sigma

    def test_four_momentum_conservation(self):
        n = 20_000
        rng = np.random.default_rng(3)
        calc = calculator(cut=1e-3)
        px, py = 1e-4*rng.standard_normal((2, n))
        delta = 1e-3*rng.standard_normal(n)
        sample = calc.sample_deflections(px, py, delta, rng)
        check_four_momentum_conservation(
            calc, px, py, delta, sample.px, sample.py, sample.delta,
            sample.energy_loss, sample.photon_energy,
            sample.incident_photon_energy, sample.incident_cos_psi)
        # The invariant energy loss is that of the particle up to 1/gamma
        energy = four_momentum(calc, px, py, delta)[0]
        energy_out = four_momentum(calc, sample.px, sample.py,
                                   sample.delta)[0]
        np.testing.assert_allclose(1 - energy_out/energy, sample.energy_loss,
                                   rtol=1e-3)
        # The scattered photon is emitted within a few 1/gamma
        assert np.isclose(np.median(sample.theta)*calc.gamma0, 1, rtol=0.3)

    @pytest.mark.parametrize('gamma', [1.05, 3.0, 300.0])
    def test_mean_energy_loss_is_the_inverse_compton_power(self, gamma):
        # Thomson limit, any velocity: the energy lost per scattering is
        # (4/3) beta^2 gamma^2 <k>, i.e. P = (4/3) sigma_T c beta^2 gamma^2 U
        n = 400_000
        mass0 = 1e9
        calc = ThermalComptonCalculator(
            np.sqrt(gamma**2 - 1)*mass0, 300., mass0=mass0)
        zeros = np.zeros(n)
        sample = calc.sample_deflections(zeros, zeros, zeros,
                                         np.random.default_rng(4))
        loss = calc.energy0 - np.sqrt(
            (calc.p0c*(1 + sample.delta))**2 + mass0**2)
        expected = 4/3*calc.beta0**2*gamma**2*mean_photon_energy(300.)
        assert abs(loss.mean() - expected) < 4*loss.std()/np.sqrt(n)

    @pytest.mark.parametrize('gamma', [1.5, 300.0])
    def test_incident_photons_carry_the_flux_factor(self, gamma):
        # Thomson limit: the photons that scatter follow the Planck spectrum,
        # with directions distributed as 1 - beta cos(psi)
        n = 400_000
        mass0 = 1e9
        zeros = np.zeros(n)
        for flux_weighting in (True, False):
            calc = ThermalComptonCalculator(
                np.sqrt(gamma**2 - 1)*mass0, 300., mass0=mass0,
                flux_weighting=flux_weighting)
            sample = calc.sample_deflections(zeros, zeros, zeros,
                                             np.random.default_rng(5))
            mean_cos_psi = -calc.beta0/3 if flux_weighting else 0.0
            assert abs(sample.incident_cos_psi.mean()
                       - mean_cos_psi) < 4*0.58/np.sqrt(n)
            assert np.isclose(sample.incident_photon_energy.mean(),
                              mean_photon_energy(300.), rtol=4*0.66/np.sqrt(n))

    def test_sampler_follows_the_energy_of_each_particle(self):
        # A particle with a 10 % higher momentum scatters like a beam of
        # that momentum
        n = 100_000
        calc = calculator(cut=0.01)
        sample = calc.sample_deflections(np.zeros(n), np.zeros(n),
                                         np.full(n, 0.1),
                                         np.random.default_rng(6))
        higher = ThermalComptonCalculator(1.1*P0C, TEMPERATURE,
                                          energy_loss_cut=0.01)
        expected = (ThermalComptonCalculator(
                        1.1*P0C, TEMPERATURE, energy_loss_cut=0.03
                        ).compute_fraction()/higher.compute_fraction())
        sigma = np.sqrt(expected*(1 - expected)/n)
        assert abs(np.mean(sample.energy_loss > 0.03) - expected) < 4*sigma


#############################################################
# Scattering kernel of the marker
#############################################################
def kernel_scatter(p0c, temperature, cut, n, seed, *, mass0=ELECTRON_MASS_EV,
                   sigma_p=0.0, sigma_delta=0.0, delta=0.0):
    """Scatter n particles with the kernel of a ThermalPhotonMarker."""
    rng = np.random.default_rng(seed)
    marker = xc.ThermalPhotonMarker(temperature=temperature,
                                    energy_loss_cut=cut)
    px, py = sigma_p*rng.standard_normal((2, n))
    delta = delta + sigma_delta*rng.standard_normal(n)
    particles = xt.Particles(_capacity=2*n, p0c=p0c, mass0=mass0, q0=-1,
                             px=px, py=py, delta=delta)
    particles._init_random_number_generator(
        seeds=rng.integers(1, 2**32 - 1, size=2*n, dtype=np.uint32))
    record = marker._scatter_particles(particles, n)
    order = np.argsort(particles.particle_id[:n])
    assert np.array_equal(particles.particle_id[:n][order], np.arange(n))
    return {'px': px, 'py': py, 'delta': delta,
            'px_out': particles.px[:n][order],
            'py_out': particles.py[:n][order],
            'delta_out': particles.delta[:n][order],
            'particles': particles, **record}


class TestScatteringKernel:

    @pytest.mark.parametrize('energy,temperature,cut', [
        (45.6e9, 297.15, 5e-3),     # LEP / FCC-ee Z
        (182.5e9, 300., 0.02),      # FCC-ee ttbar
        (45.6e9, 297.15, 1e-6),     # nearly all the scatterings
        (7e9, 300., 0.01),          # steep spectrum: truncated envelope
    ])
    def test_kernel_matches_the_rate_and_the_reference(self, energy,
                                                       temperature, cut):
        n = 200_000
        calc = calculator(energy, temperature, cut=cut)
        out = kernel_scatter(calc.p0c, temperature, cut, n, seed=7,
                             sigma_p=1e-4, sigma_delta=1e-3)
        assert np.all(out['energy_loss'] >= cut)
        assert np.all(out['n_trials'] >= 1)

        # Against the deterministic rate
        for factor in (1.5, 3.0):
            expected = (calculator(energy, temperature, cut=factor*cut
                                   ).compute_fraction()
                        / calc.compute_fraction())
            sigma = np.sqrt(expected*(1 - expected)/n)
            assert abs(np.mean(out['energy_loss'] > factor*cut)
                       - expected) < 4*sigma + 2e-3*expected

        # Against the numpy implementation of the same algorithm
        sample = calc.sample_deflections(out['px'], out['py'], out['delta'],
                                         np.random.default_rng(8))
        angle = out['px_out']/(1 + out['delta_out'])
        angle_ref = sample.px/(1 + sample.delta)
        for kernel, reference in (
                (out['energy_loss'], sample.energy_loss),
                (out['photon_energy'], sample.photon_energy),
                (out['theta'], sample.theta),
                (out['incident_photon_energy'],
                 sample.incident_photon_energy),
                (out['incident_cos_psi'], sample.incident_cos_psi),
                (out['cos_theta_rest'], sample.cos_theta_rest),
                (out['delta_out'] - out['delta'],
                 sample.delta - out['delta']),
                (angle - out['px']/(1 + out['delta']),
                 angle_ref - out['px']/(1 + out['delta']))):
            assert stats.ks_2samp(kernel, reference).pvalue > 1e-4

    def test_kernel_conserves_four_momentum(self):
        calc = calculator(cut=1e-3)
        out = kernel_scatter(P0C, TEMPERATURE, 1e-3, 20_000, seed=9,
                             sigma_p=1e-4, sigma_delta=1e-3)
        check_four_momentum_conservation(
            calc, out['px'], out['py'], out['delta'], out['px_out'],
            out['py_out'], out['delta_out'], out['energy_loss'],
            out['photon_energy'], out['incident_photon_energy'],
            out['incident_cos_psi'])

    def test_kernel_keeps_the_longitudinal_variables_consistent(self):
        # delta is updated together with the other energy variables
        out = kernel_scatter(P0C, TEMPERATURE, 1e-3, 1000, seed=10)
        particles = out['particles']
        reference = xt.Particles(p0c=P0C, mass0=ELECTRON_MASS_EV,
                                 delta=particles.delta[:1000])
        np.testing.assert_allclose(particles.ptau[:1000], reference.ptau,
                                   rtol=1e-12)
        np.testing.assert_allclose(particles.rvv[:1000], reference.rvv,
                                   rtol=1e-12)
        np.testing.assert_allclose(particles.rpp[:1000], reference.rpp,
                                   rtol=1e-12)
        assert np.all(particles.zeta[:1000] == 0)
        assert np.all(particles.state[:1000] == 1)

    def test_kernel_is_reproducible(self):
        a = kernel_scatter(P0C, TEMPERATURE, 5e-3, 500, seed=11)
        b = kernel_scatter(P0C, TEMPERATURE, 5e-3, 500, seed=11)
        c = kernel_scatter(P0C, TEMPERATURE, 5e-3, 500, seed=12)
        assert np.array_equal(a['delta_out'], b['delta_out'])
        assert np.array_equal(a['px_out'], b['px_out'])
        assert not np.array_equal(a['delta_out'], c['delta_out'])

    def test_kernel_follows_the_energy_of_each_particle(self):
        n = 100_000
        out = kernel_scatter(P0C, TEMPERATURE, 0.01, n, seed=13, delta=0.1)
        higher = ThermalComptonCalculator(1.1*P0C, TEMPERATURE,
                                          energy_loss_cut=0.01)
        expected = (ThermalComptonCalculator(
                        1.1*P0C, TEMPERATURE, energy_loss_cut=0.03
                        ).compute_fraction()/higher.compute_fraction())
        sigma = np.sqrt(expected*(1 - expected)/n)
        assert abs(np.mean(out['energy_loss'] > 0.03) - expected) < 4*sigma

    def test_kernel_with_muons(self):
        # Same eps_T as 45.6 GeV electrons at 297.15 K: the energy loss
        # distribution only depends on eps_T and on the cut
        n = 100_000
        ratio = MUON_MASS_EV/ELECTRON_MASS_EV
        p0c = np.sqrt((45.6e9*ratio)**2 - MUON_MASS_EV**2)
        calc = ThermalComptonCalculator(p0c, TEMPERATURE*ratio,
                                        mass0=MUON_MASS_EV,
                                        energy_loss_cut=0.01)
        assert np.isclose(calc.eps_T, calculator(45.6e9).eps_T, rtol=1e-9)
        out = kernel_scatter(p0c, TEMPERATURE*ratio, 0.01, n, seed=14,
                             mass0=MUON_MASS_EV)
        expected = (calculator(45.6e9, cut=0.03).compute_fraction()
                    / calculator(45.6e9, cut=0.01).compute_fraction())
        sigma = np.sqrt(expected*(1 - expected)/n)
        assert abs(np.mean(out['energy_loss'] > 0.03) - expected) < 4*sigma

    def test_kernel_at_low_velocity(self):
        # beta = 0.6: the exact range of the photon energy matters, and
        # the particle can also gain energy
        n = 200_000
        mass0 = 1e9
        gamma = 1.25
        p0c = np.sqrt(gamma**2 - 1)*mass0
        out = kernel_scatter(p0c, 300., 1e-12, n, seed=15, mass0=mass0)
        loss = gamma*mass0 - np.sqrt((p0c*(1 + out['delta_out']))**2
                                     + mass0**2)
        expected = 4/3*0.6**2*gamma**2*mean_photon_energy(300.)
        # The cut removes a negligible part of the scatterings
        assert abs(loss.mean() - expected) < 4*loss.std()/np.sqrt(n)
        assert np.any(loss < 0)
        assert abs(out['incident_cos_psi'].mean() + 0.2) < 4*0.57/np.sqrt(n)

    def test_raises_if_not_configured(self):
        particles = xt.Particles(p0c=P0C, x=[0.0, 0.0])
        particles._init_random_number_generator()
        with pytest.raises(ValueError, match='temperature'):
            xc.ThermalPhotonMarker()._scatter_particles(particles, 2)
        with pytest.raises(ValueError, match='energy_loss_cut'):
            xc.ThermalPhotonMarker(temperature=300.)._scatter_particles(
                particles, 2)


#############################################################
# Input validation of the configuration
#############################################################
class TestConfigureInputs:

    def test_normalised_emittances(self, toy_ring):
        line = configure(toy_ring)
        beta_gamma = (line.particle_ref.beta0[0]*line.particle_ref.gamma0[0])
        for nn in markers(line):
            assert np.isclose(line[nn].gemitt_x, NEMITT_X/beta_gamma)
            assert np.isclose(line[nn].gemitt_y, NEMITT_Y/beta_gamma)

    def test_geometric_emittances(self, toy_ring):
        line = configure(toy_ring, nemitt_x=None, nemitt_y=None,
                         gemitt_x=1e-9, gemitt_y=1e-11)
        for nn in markers(line):
            assert np.isclose(line[nn].gemitt_x, 1e-9)
            assert np.isclose(line[nn].gemitt_y, 1e-11)

    @pytest.mark.parametrize('missing_key', ['temperature', 'sigma_z',
                                             'sigma_delta',
                                             'n_scattering_events'])
    def test_raises_on_missing_required_kwarg(self, toy_ring, missing_key):
        with pytest.raises(ValueError, match=missing_key):
            configure(toy_ring, **{missing_key: None})

    def test_raises_on_both_nemitt_and_gemitt(self, toy_ring):
        with pytest.raises(ValueError, match='not both'):
            configure(toy_ring, gemitt_x=1e-9, gemitt_y=1e-11)

    def test_raises_on_neither_nemitt_nor_gemitt(self, toy_ring):
        with pytest.raises(ValueError, match='must provide'):
            configure(toy_ring, nemitt_x=None, nemitt_y=None)

    def test_needs_exactly_one_definition_of_the_cut(self, toy_ring):
        lma = xt.Table({'name': np.array(['a', 'b']),
                        's': np.array([0.0, 21.2]),
                        'delta_neg': np.array([-0.01, -0.01]),
                        'delta_pos': np.array([0.01, 0.01])})
        with pytest.raises(ValueError, match='exactly one'):
            configure(toy_ring, energy_loss_cut=None)
        with pytest.raises(ValueError, match='exactly one'):
            configure(toy_ring, local_momentum_acceptance=lma)

    @pytest.mark.parametrize('cut', [0.0, -0.01, 1.0])
    def test_raises_on_invalid_cut(self, toy_ring, cut):
        with pytest.raises(ValueError, match='energy_loss_cut'):
            configure(toy_ring, energy_loss_cut=cut)

    @pytest.mark.parametrize('temperature', [0.0, -5.0, np.inf])
    def test_raises_on_invalid_temperature(self, toy_ring, temperature):
        with pytest.raises(ValueError, match='temperature'):
            configure(toy_ring, temperature=temperature)

    def test_raises_on_wrong_temperature_type(self, toy_ring):
        with pytest.raises(TypeError, match='temperature'):
            configure(toy_ring, temperature='warm')

    def test_raises_on_bad_temperature_table(self, toy_ring):
        with pytest.raises(ValueError, match='missing columns'):
            configure(toy_ring, temperature=xt.Table(
                {'name': np.array(['a']), 's': np.array([0.0])}))
        for bad_value in (np.nan, np.inf, 0.0, -1.0):
            with pytest.raises(ValueError, match='temperature'):
                configure(toy_ring, temperature=xt.Table(
                    {'name': np.array(['a', 'b']),
                     's': np.array([0.0, 21.2]),
                     'temperature': np.array([300.0, bad_value])}))
        with pytest.raises(ValueError, match='non-decreasing'):
            configure(toy_ring, temperature=xt.Table(
                {'name': np.array(['a', 'b']), 's': np.array([5.0, 1.0]),
                 'temperature': np.array([300.0, 300.0])}))

    def test_raises_on_neutral_beam(self, toy_ring):
        line = toy_ring['line'].copy()
        line.particle_ref = xt.Particles(mass0=939.565e6, q0=0, p0c=P0C)
        with pytest.raises(ValueError, match='neutral'):
            configure(toy_ring, line=line)

    @pytest.mark.parametrize('particle', ['positron', 'muon', 'proton'])
    def test_accepts_any_charged_beam(self, toy_ring, particle):
        line = toy_ring['line'].copy()
        line.set_particle_ref(particle, p0c=P0C)
        configure(toy_ring, line=line, n_scattering_events=50)
        nn = markers(line)[0]
        ref = line.particle_ref
        expected = BUNCH_INTENSITY*sc.c/line.get_length()*line[nn].ds*(
            photon_density(TEMPERATURE)*ThermalComptonCalculator(
                P0C, TEMPERATURE, mass0=float(ref.mass0), q0=float(ref.q0),
                energy_loss_cut=ENERGY_LOSS_CUT).compute_xsec())
        assert np.isclose(line[nn].interaction_rate, expected, rtol=1e-9)
        if particle == 'positron':
            particles = line.xcoll.thermal_photons.generate(at=nn)
            assert np.all(particles.q0 == 1)
            assert particles.pdg_id[0] == -11

    def test_raises_when_no_markers(self, toy_ring):
        line = xt.Line(elements=[xt.Drift(length=1.0)])
        line.set_particle_ref('electron', p0c=P0C)
        with pytest.raises(ValueError, match='does not contain any'):
            configure(toy_ring, line=line)


#############################################################
# Configuration of the markers
#############################################################
class TestConfigure:

    @pytest.fixture(autouse=True)
    def _configure(self, toy_ring):
        self.toy_ring = toy_ring
        self.line = configure(toy_ring)
        self.calc = calculator(cut=ENERGY_LOSS_CUT)

    def test_sections_cover_the_whole_line(self):
        table = self.line.xcoll.thermal_photons.analyse().local_rates
        assert np.isclose(table.ds.sum(), self.line.get_length())

    def test_all_elements_are_configured(self):
        for nn in markers(self.line):
            elem = self.line[nn]
            assert elem.p0c > 0
            assert elem.betx > 0 and elem.bety > 0
            assert elem.interaction_rate > 0
            assert elem.total_rate > elem.interaction_rate
            assert elem.n_scattering_events == 200
            assert elem.temperature == TEMPERATURE
            assert elem.energy_loss_cut == ENERGY_LOSS_CUT

    def test_interaction_rate_matches_analytic_formula(self):
        # rate = N_bunch * (scattering rate of one particle), independent of
        # the circumference for a uniform temperature
        total = sum(self.line[nn].interaction_rate
                    for nn in markers(self.line))
        assert np.isclose(total, BUNCH_INTENSITY*self.calc.compute_rate(),
                          rtol=1e-9)
        total = sum(self.line[nn].total_rate for nn in markers(self.line))
        assert np.isclose(total, BUNCH_INTENSITY*calculator().compute_rate(),
                          rtol=1e-9)

    def test_mean_free_path(self):
        expected = self.calc.beta0/(photon_density(TEMPERATURE)
                                    * self.calc.compute_xsec())
        for nn in markers(self.line):
            assert np.isclose(self.line[nn].mean_free_path, expected)
        # The local rate per unit length integrates to the interaction rate
        table = self.line.xcoll.thermal_photons.analyse().local_rates
        np.testing.assert_allclose(table.local_rate*table.ds,
                                   table.interaction_rate, rtol=1e-9)

    def test_lifetime_limits(self):
        result = self.line.xcoll.thermal_photons.analyse()
        # 26.17 h of Burkhardt with the Thomson cross section, times the
        # ratio of the Thomson to the Compton cross section
        assert np.isclose(result.lifetime_total/3600,
                          26.17/calculator().compute_fraction(), rtol=2e-3)
        assert result.lifetime_scattering > result.lifetime_total
        assert np.isclose(result.lifetime_scattering,
                          1/self.calc.compute_rate())

    def test_partial_configuration_single_element(self):
        nn = markers(self.line)[2]
        configure(self.toy_ring, at=nn, n_scattering_events=17)
        assert self.line[nn].n_scattering_events == 17
        assert self.line[markers(self.line)[1]].n_scattering_events == 200

    def test_partial_configuration_raises_on_wrong_element(self):
        with pytest.raises(TypeError, match='not a ThermalPhotonMarker'):
            configure(self.toy_ring, at='mqf.1')
        with pytest.raises(ValueError, match='not present'):
            configure(self.toy_ring, at='nope')

    def test_cut_from_the_local_momentum_acceptance(self):
        line = self.line
        length = line.get_length()
        # Acceptance of 1 % in the first half of the ring, narrowing
        # linearly to 0.6 % at its end
        lma = xt.Table({
            'name': np.array(['a', 'b', 'c']),
            's': np.array([0.0, length/2, length]),
            'delta_neg': np.array([-0.01, -0.01, -0.006]),
            'delta_pos': np.array([0.01, 0.01, 0.006])})
        before = np.asarray(lma.delta_neg).copy()
        configure(self.toy_ring, energy_loss_cut=None,
                  local_momentum_acceptance=lma,
                  local_momentum_acceptance_scale=0.8)
        assert np.array_equal(lma.delta_neg, before)
        tab = line.get_table()
        for nn in markers(line):
            s = float(tab['s', nn])
            # Smallest acceptance of the section: at the marker itself
            acceptance = np.interp(s, [0, length/2, length],
                                   [0.01, 0.01, 0.006])
            assert np.isclose(line[nn].energy_loss_cut, 0.8*acceptance)
            expected = (BUNCH_INTENSITY*line[nn].ds/length
                        * calculator(cut=0.8*acceptance).compute_rate())
            assert np.isclose(line[nn].interaction_rate, expected, rtol=1e-9)

    def test_cut_uses_the_smallest_acceptance_of_the_section(self):
        line = self.line
        names = markers(line)
        tab = line.get_table()
        s = [float(tab['s', nn]) for nn in names]
        # A dip of the acceptance in the middle of the section of marker 3
        s_dip = 0.5*(s[2] + s[3])
        lma = xt.Table({
            'name': np.array(['a', 'b', 'c', 'd', 'e']),
            's': np.array([0.0, s[2], s_dip, s[3], line.get_length()]),
            'delta_neg': -np.array([0.01, 0.01, 0.004, 0.01, 0.01]),
            'delta_pos': np.array([0.01, 0.01, 0.004, 0.01, 0.01])})
        configure(self.toy_ring, energy_loss_cut=None,
                  local_momentum_acceptance=lma)
        cuts = [line[nn].energy_loss_cut for nn in names]
        assert np.isclose(cuts[3], 0.85*0.004)
        np.testing.assert_allclose(np.delete(cuts, 3), 0.85*0.01)

    def test_temperature_profile(self):
        line = self.line
        length = line.get_length()
        # Linear from 250 K to 350 K over the ring
        profile = xt.Table({'name': np.array(['a', 'b']),
                            's': np.array([0.0, length]),
                            'temperature': np.array([250.0, 350.0])})
        configure(self.toy_ring, temperature=profile)

        def rate_density(s, cut):
            temperature = 250.0 + 100.0*s/length
            return calculator(temperature=temperature, cut=cut).compute_rate()

        tab = line.get_table()
        s_start = 0.0
        for nn in markers(line):
            s = float(tab['s', nn])
            for attribute, cut in (('interaction_rate', ENERGY_LOSS_CUT),
                                   ('total_rate', 0.0)):
                expected = BUNCH_INTENSITY/length*quad(
                    rate_density, s_start, s, args=(cut,), epsrel=1e-10)[0]
                assert np.isclose(getattr(line[nn], attribute), expected,
                                  rtol=1e-8)
            # Rate-weighted mean temperature of the section
            assert 250 + 100*s_start/length < line[nn].temperature
            assert line[nn].temperature < 250 + 100*s/length
            midpoint = 250 + 100*0.5*(s_start + s)/length
            assert 0 < line[nn].temperature - midpoint < 1
            s_start = s

    def test_cold_section(self):
        line = self.line
        names = markers(line)
        tab = line.get_table()
        s = [float(tab['s', nn]) for nn in names]
        # The sections of markers 2 and 3 are at 4.5 K, with sharp
        # transitions
        profile = xt.Table({
            'name': np.array(['a', 'b', 'c', 'd', 'e', 'f']),
            's': np.array([0.0, s[1], s[1], s[3], s[3], line.get_length()]),
            'temperature': np.array([300., 300., 4.5, 4.5, 300., 300.])})
        configure(self.toy_ring, temperature=profile)
        warm = BUNCH_INTENSITY*line[names[0]].ds/line.get_length()*(
            calculator(temperature=300., cut=ENERGY_LOSS_CUT).compute_rate())
        for ii, nn in enumerate(names):
            if ii in (2, 3):
                assert line[nn].temperature == 4.5
                assert 0 < line[nn].interaction_rate < 1e-12*warm
                assert np.isclose(line[nn].total_rate/warm, (4.5/300.)**3
                                  / calculator(temperature=300.,
                                               cut=ENERGY_LOSS_CUT
                                               ).compute_fraction(),
                                  rtol=1e-2)
            else:
                assert line[nn].temperature == 300.
                assert np.isclose(line[nn].interaction_rate, warm, rtol=1e-9)

        # The cold sections are not generated
        generator = line.xcoll.thermal_photons
        assert list(generator.generate()) == [
            nn for ii, nn in enumerate(names) if ii not in (2, 3)]
        with pytest.raises(ValueError, match='zero or negligible'):
            generator.generate(at=names[2])
        result = generator.run(n_turns=2)
        assert result.element_names == [
            nn for ii, nn in enumerate(names) if ii not in (2, 3)]


#############################################################
# Particle generation
#############################################################
class TestGenerateParticles:

    @pytest.fixture(autouse=True)
    def _configure(self, toy_ring):
        self.toy_ring = toy_ring
        self.line = configure(toy_ring, n_scattering_events=500)

    def test_weights_sum_to_the_interaction_rate(self):
        line = self.line
        for nn, particles in line.xcoll.thermal_photons.generate().items():
            allocated = particles.particle_id >= 0
            assert np.sum(allocated) == 500
            assert np.isclose(particles.weight[allocated].sum(),
                              line[nn].interaction_rate, rtol=1e-12)
            assert np.all(particles.weight[allocated]
                          == particles.weight[allocated][0])

    def test_every_particle_interacts_once(self):
        particles, logs = self.line.xcoll.thermal_photons.generate(
            return_log=True)
        for nn, part in particles.items():
            log = logs[nn]
            assert len(log.particle_id) == int(np.sum(part.particle_id >= 0))
            assert np.all(log.energy_loss >= ENERGY_LOSS_CUT)
            assert np.all(log.photon_energy > log.incident_photon_energy)
            assert np.all(log.theta > 0)
            assert self.line.element_dict[nn].scatter_log is log

    def test_log_matches_the_particles(self):
        line = self.line
        nn = markers(line)[0]
        particles, log = line.xcoll.thermal_photons.generate(
            at=nn, return_log=True)
        order = np.argsort(particles.particle_id[:500])
        assert np.array_equal(particles.particle_id[:500][order],
                              log.particle_id)
        # The energy carried away by the photon is lost by the particle
        energy = np.sqrt((P0C*(1 + particles.delta[:500][order]))**2
                         + ELECTRON_MASS_EV**2)
        # ... which started within a few sigma of the reference energy
        assert np.allclose(energy + log.photon_energy,
                           np.sqrt(P0C**2 + ELECTRON_MASS_EV**2),
                           rtol=6*SIGMA_DELTA)
        assert np.array_equal(particles.weight[:500][order], log.weight)

    def test_generated_distribution_matches_the_optics(self, toy_ring):
        line = configure(toy_ring, n_scattering_events=20_000)
        nn = markers(line)[0]
        particles, log = line.xcoll.thermal_photons.generate(
            at=nn, return_log=True)
        allocated = particles.particle_id >= 0
        elem = line[nn]

        # x, y and zeta are not changed by the scattering
        assert np.isclose(particles.zeta[allocated].std(), SIGMA_Z, rtol=0.05)
        sigma_x = np.sqrt(elem.gemitt_x*elem.betx
                          + (elem.dx*SIGMA_DELTA)**2)
        assert np.isclose(particles.x[allocated].std(), sigma_x, rtol=0.05)
        assert np.isclose(particles.x[allocated].mean(), elem.x_co,
                          atol=0.1*sigma_x)
        sigma_y = np.sqrt(elem.gemitt_y*elem.bety)
        assert np.isclose(particles.y[allocated].std(), sigma_y, rtol=0.05)

        # delta is the matched one minus the energy loss
        order = np.argsort(particles.particle_id[allocated])
        delta_before = ((1 + particles.delta[allocated][order])
                        / (1 - log.energy_loss) - 1)
        assert np.isclose(delta_before.std(), SIGMA_DELTA, rtol=0.05)
        assert abs(delta_before.mean()) < 0.1*SIGMA_DELTA

    def test_particles_start_at_the_element(self):
        line = self.line
        tab = line.get_table()
        for nn, particles in line.xcoll.thermal_photons.generate().items():
            allocated = particles.particle_id >= 0
            assert np.allclose(particles.s[allocated], tab['s', nn])
            assert np.all(particles.at_element[allocated]
                          == line.element_names.index(nn))
            assert np.all(particles.state[allocated] == 1)
            assert len(particles.particle_id) == 4*500

    def test_particles_carry_a_random_generator(self):
        nn = markers(self.line)[0]
        particles = self.line.xcoll.thermal_photons.generate(at=nn)
        assert particles._has_valid_rng_state()

    def test_raises_if_not_configured(self):
        line = xt.Line(elements=[xc.ThermalPhotonMarker()],
                       element_names=['tp'])
        line.set_particle_ref('electron', p0c=P0C)
        with pytest.raises(ValueError, match='not configured'):
            line.xcoll.thermal_photons.generate()


#############################################################
# Running the simulation
#############################################################
class TestRun:

    @pytest.fixture(autouse=True)
    def _configure(self, toy_ring):
        self.toy_ring = toy_ring
        self.line = configure(toy_ring)
        self.tp = self.line.xcoll.thermal_photons

    def test_analyse_without_tracking(self):
        result = self.tp.analyse()
        assert result.tracked is False
        assert result.rate_tracking is None
        assert result.particles is None
        assert result.cutoff_scan is None
        assert result.rate_scattering > 0
        assert np.isclose(result.lifetime_scattering,
                          BUNCH_INTENSITY/result.rate_scattering)
        assert np.isclose(result.lifetime_total,
                          BUNCH_INTENSITY/result.rate_total)

    def test_run_with_tracking(self):
        result = self.tp.run(n_turns=20, keep_particles=True)
        assert result.tracked is True
        assert 0 < result.rate_tracking <= result.rate_scattering
        assert np.isclose(result.rate_generated, result.rate_scattering)
        assert np.isclose(result.lifetime_tracking,
                          BUNCH_INTENSITY/result.rate_tracking)
        assert result.lifetime_tracking >= result.lifetime_scattering
        assert result.particles is not None
        assert result.lost_particles is not None
        assert np.all(result.lost_particles.state <= 0)
        assert np.all(result.lost_particles.particle_id >= 0)
        assert np.isclose(result.lost_particles.weight.sum(),
                          result.rate_tracking)

    def test_lifetime_is_independent_of_bunch_intensity(self, toy_ring):
        kwargs = dict(n_scattering_events=200, seed=7)
        configure(toy_ring, **kwargs)
        a = self.tp.run(n_turns=20)
        configure(toy_ring, bunch_intensity=10*BUNCH_INTENSITY, **kwargs)
        b = self.tp.run(n_turns=20)
        assert np.isclose(a.rate_tracking*10, b.rate_tracking)
        assert np.isclose(a.lifetime_tracking, b.lifetime_tracking)

    def test_local_rates_columns(self):
        result = self.tp.run(n_turns=5, keep_particles=True)
        table = result.local_rates
        for col in ('name', 's', 'ds', 'temperature', 'energy_loss_cut',
                    'mean_free_path', 'local_rate', 'interaction_rate',
                    'total_rate', 'num_particles', 'sum_weight',
                    'num_lost_particles', 'sum_lost_weight'):
            assert col in table._col_names
        assert np.isclose(table.sum_weight.sum(), result.rate_scattering)
        assert np.isclose(table.sum_lost_weight.sum(), result.rate_tracking)

    def test_interaction_log(self):
        result = self.tp.run(n_turns=5)
        log = result.interaction_log
        assert set(np.unique(log.name)) == set(markers(self.line))
        assert np.all(log.energy_loss >= ENERGY_LOSS_CUT)
        assert np.all(log.replica == 0)
        assert np.isclose(log.weight.sum(), result.rate_scattering)

    def test_reproducible_with_seed(self, toy_ring):
        rates = []
        for seed in (11, 11, 12):
            configure(toy_ring, seed=seed)
            rates.append(self.tp.run(n_turns=10).rate_tracking)
        assert rates[0] == rates[1]
        assert rates[0] != rates[2]

    def test_statistical_error(self):
        result = self.tp.run(n_turns=20)
        assert result.rate_tracking_error > 0
        assert result.rate_tracking_error < result.rate_tracking
        assert np.isclose(
            result.lifetime_tracking_error/result.lifetime_tracking,
            result.rate_tracking_error/result.rate_tracking)

    def test_replicas(self):
        result = self.tp.run(n_turns=5, n_replicas=3)
        assert np.all(result.local_rates.n_replicas == 3)
        assert np.all(result.local_rates.num_particles == 600)
        assert np.isclose(result.rate_generated, result.rate_scattering)
        assert set(np.unique(result.interaction_log.replica)) == {0, 1, 2}

    def test_cutoff_scan(self, toy_ring):
        # Losses only start above about 1.1 times this cut
        configure(toy_ring, n_scattering_events=1000)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = self.tp.run(n_turns=20)
        scan = result.cutoff_scan
        assert np.isclose(scan.cut_scale[0], 1.0)
        assert np.all(np.diff(scan.cut_scale) > 0)
        assert scan.num_events.sum() == 8*1000
        assert scan.num_lost.sum() == result.local_rates.num_lost_particles.sum()
        # Every lost event is above the first cut: the first row is the
        # result itself
        assert np.isclose(scan.rate_tracking[0], result.rate_tracking)
        assert np.isclose(scan.rate_tracking_error[0],
                          result.rate_tracking_error)
        assert np.isclose(scan.lifetime_tracking[0], result.lifetime_tracking)
        # The events just above the cut are not lost, and all of them are
        # far above it
        assert scan.loss_probability[0] == 0
        assert scan.loss_probability[np.searchsorted(scan.cut_scale, 3.0)] == 1
        assert np.all(np.diff(scan.rate_tracking) <= 0)

    def test_warns_when_the_cut_truncates_the_losses(self, toy_ring):
        # Above this cut every particle is lost
        configure(toy_ring, energy_loss_cut=0.02)
        with pytest.warns(UserWarning, match='energy-loss cut'):
            result = self.tp.run(n_turns=20)
        assert result.cutoff_scan.loss_probability[0] == 1
        assert np.isclose(result.rate_tracking, result.rate_scattering)

    def test_raises_on_invalid_run_arguments(self):
        with pytest.raises(ValueError, match='n_turns'):
            self.tp.run(n_turns=0)
        with pytest.raises(ValueError, match='n_replicas'):
            self.tp.run(n_turns=1, n_replicas=0)


#############################################################
# Serialisation
#############################################################
class TestThermalPhotonMarkerSerialization:
    """A configured ThermalPhotonMarker keeps its configuration in JSON."""

    def test_configured_markers_survive_json(self, toy_ring, tmp_path):
        line = configure(toy_ring)
        names = markers(line)
        for ii, nn in enumerate(names):
            # Distinct non-zero values, else the comparison would be vacuous
            for jj, cc in enumerate(('x_co', 'px_co', 'y_co', 'py_co',
                                     'zeta_co', 'delta_co')):
                setattr(line[nn], cc, 1e-6 * (ii + 1) * (jj + 1))
            line[nn].seed = 1997 + ii
            line[nn].temperature = 290.0 + ii

        line.to_json(tmp_path / 'line.json')
        loaded = xt.load(tmp_path / 'line.json')

        for nn in names:
            marker = loaded[nn]
            assert isinstance(marker, xc.ThermalPhotonMarker)
            assert line[nn].betx > 0 and line[nn].interaction_rate > 0
            for field in xc.ThermalPhotonMarker._xofields:
                np.testing.assert_array_equal(
                    getattr(marker, field), getattr(line[nn], field),
                    err_msg=f'{nn}: {field} changed in the JSON round trip')


#############################################################
# Passive tracking
#############################################################
class TestPassiveTracking:

    def test_tracking_is_not_affected(self, toy_ring):
        line = toy_ring['line']
        particles = line.build_particles(x=[1e-4, 2e-4], px=[1e-6, -1e-6])
        reference = particles.copy()
        line.track(particles, num_turns=5)

        line_no_markers = line.copy()
        line_no_markers.discard_tracker()
        for nn in [nn for nn in line_no_markers.element_names
                   if isinstance(line_no_markers[nn],
                                 xc.ThermalPhotonMarker)]:
            line_no_markers.element_dict[nn] = xt.Marker()
        line_no_markers.build_tracker()
        line_no_markers.track(reference, num_turns=5)

        assert np.allclose(particles.x, reference.x, rtol=0, atol=1e-15)
        assert np.allclose(particles.px, reference.px, rtol=0, atol=1e-15)

    def test_element_is_not_collective(self, toy_ring):
        assert not toy_ring['line']['ThermalPhotonMarker.0'].iscollective
