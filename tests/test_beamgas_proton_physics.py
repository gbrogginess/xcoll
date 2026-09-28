# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""
Unit tests of the proton-gas physics models.

The Geant4 reference values in ``data/beamgas/geant4/proton_reference.json``
were produced with Geant4 11.4.2 by ``make_proton_reference.cc`` in the same
directory, which calls the Geant4 classes that the models are ported from.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from xcoll.beamgas import proton_cross_sections as pcs
from xcoll.beamgas import proton_data

MB = pcs.MILLIBARN

REFERENCE = json.loads((Path(__file__).parent / 'data' / 'beamgas' / 'geant4'
                        / 'proton_reference.json').read_text())


def _quad_log(func, x1, x2, n_sub=60):
    """Adaptive quadrature in log(x) on sub-intervals."""
    edges = np.linspace(np.log(x1), np.log(x2), n_sub + 1)
    return sum(quad(lambda u: func(np.exp(u))*np.exp(u), a, b, limit=200)[0]
               for a, b in zip(edges[:-1], edges[1:]))


#############################################################
# Hadronic cross sections
#############################################################
class TestHadronicCrossSections:

    def test_proton_nucleon_cross_sections_match_geant4(self):
        for ip, p in enumerate(REFERENCE['p_GeV']):
            tot, el, _ = pcs.proton_nucleon_cross_sections(p, 'proton')
            assert np.isclose(tot, REFERENCE['ns_pp_total_mb'][ip], rtol=1e-6)
            assert np.isclose(el, REFERENCE['ns_pp_elastic_mb'][ip],
                              rtol=1e-6)
            tot, el, _ = pcs.proton_nucleon_cross_sections(p, 'neutron')
            assert np.isclose(tot, REFERENCE['ns_pn_total_mb'][ip], rtol=1e-6)
            assert np.isclose(el, REFERENCE['ns_pn_elastic_mb'][ip],
                              rtol=1e-6)

    def test_bgg_cross_sections_match_geant4(self):
        n_z = len(REFERENCE['Z'])
        for ip, p in enumerate(REFERENCE['p_GeV']):
            for iz, Z in enumerate(REFERENCE['Z']):
                Z = int(Z)
                k = ip*n_z + iz
                xs = pcs.ProtonNucleusCrossSections(Z, p*1e9)
                assert np.isclose(xs.inelastic/MB,
                                  REFERENCE['bgg_inelastic_mb'][k], rtol=1e-6)
                assert np.isclose(xs.elastic/MB,
                                  REFERENCE['bgg_elastic_mb'][k], rtol=1e-6)
                if Z > 1:
                    for attr, key in (('gg_total', 'gg_total_mb'),
                                      ('gg_inelastic', 'gg_inelastic_mb'),
                                      ('gg_production', 'gg_production_mb'),
                                      ('gg_elastic', 'gg_elastic_mb')):
                        assert np.isclose(getattr(xs, attr)/MB,
                                          REFERENCE[key][k], rtol=1e-6)

    def test_quasi_elastic_is_glauber_gribov_inelastic_minus_production(self):
        for Z in (2, 7, 18, 54):
            xs = pcs.ProtonNucleusCrossSections(Z, 6.8e12)
            factor = proton_data.BGG_GLAUBER_FACTOR_INELASTIC[Z]
            assert np.isclose(xs.quasi_elastic,
                              factor*(xs.gg_inelastic - xs.gg_production))
            assert 0 < xs.quasi_elastic < 0.15*xs.inelastic
        assert pcs.ProtonNucleusCrossSections(1, 6.8e12).quasi_elastic == 0

    def test_energy_dependence(self):
        p = np.geomspace(100, 1e4, 20)*1e9
        for Z in (1, 2, 6, 7, 8, 18):
            inel = [pcs.ProtonNucleusCrossSections(Z, pp).inelastic for pp in p]
            el = [pcs.ProtonNucleusCrossSections(Z, pp).elastic for pp in p]
            assert np.all(np.diff(inel) > 0)
            assert np.all(np.diff(el) > 0)
        # 450 GeV -> 6.8 TeV: +15% for hydrogen, +7% for argon
        ratio = {Z: (pcs.ProtonNucleusCrossSections(Z, 6.8e12).inelastic
                     / pcs.ProtonNucleusCrossSections(Z, 450e9).inelastic)
                 for Z in (1, 7, 18)}
        assert np.isclose(ratio[1], 1.153, atol=2e-3)
        assert np.isclose(ratio[7], 1.093, atol=2e-3)
        assert np.isclose(ratio[18], 1.071, atol=2e-3)

    def test_proton_proton_agrees_with_the_everest_fits(self):
        # Independent fits used by Everest (properties.h)
        for p in (450., 6800.):
            s = 2*pcs._MP*p
            pptot = 1e3*(0.041084 - 0.0023302*np.log(s)
                         + 0.00031514*np.log(s)**2)
            ppel = 11.7 - 1.59*np.log(s) + 0.134*np.log(s)**2
            tot, el, _ = pcs.proton_nucleon_cross_sections(p)
            assert np.isclose(tot, pptot, rtol=0.02)
            assert np.isclose(el, ppel, rtol=0.06)

    def test_rho_parameter(self):
        # Measured rho(pp) is ~0.02 at sqrt(s) = 29 GeV and ~0.1 at 113 GeV
        assert np.isclose(pcs.proton_proton_rho(450.), 0.02, atol=0.01)
        assert np.isclose(pcs.proton_proton_rho(6800.), 0.106, atol=0.01)

    @pytest.mark.parametrize('molecule, composition, design_report', [
        ('H2', {1: 2}, 94.), ('CH4', {6: 1, 1: 4}, 566.),
        ('H2O', {8: 1, 1: 2}, 565.), ('CO', {6: 1, 8: 1}, 870.),
        ('CO2', {6: 1, 8: 2}, 1317.)])
    def test_lhc_design_report_totals(self, molecule, composition,
                                      design_report):
        # The LHC Design Report (vol. 1, table 12.1) nuclear cross sections at
        # 7 TeV are total hadronic ones (inelastic + nuclear elastic)
        total = sum(nn*(pcs.ProtonNucleusCrossSections(Z, 7e12).inelastic
                        + pcs.ProtonNucleusCrossSections(Z, 7e12).elastic)
                    for Z, nn in composition.items())/MB
        assert np.isclose(total, design_report, rtol=0.04)

    def test_lhc_design_report_helium_is_lower(self):
        # Known difference: the Design Report value for helium (126 mb) lies
        # between the BGG inelastic (116 mb) and total (149 mb)
        xs = pcs.ProtonNucleusCrossSections(2, 7e12)
        assert xs.inelastic/MB < 126. < (xs.inelastic + xs.elastic)/MB

    def test_raises_below_the_glauber_energy(self):
        with pytest.raises(ValueError, match='91 GeV'):
            pcs.ProtonNucleusCrossSections(7, 50e9)

    def test_raises_on_unsupported_element(self):
        with pytest.raises(ValueError, match='out of the supported range'):
            pcs.ProtonNucleusCrossSections(93, 450e9)


#############################################################
# CHIPS elastic distribution
#############################################################
class TestChipsElastic:

    @staticmethod
    def _cases():
        for ip, p in enumerate(REFERENCE['chips_p_GeV']):
            for iz, Z in enumerate(REFERENCE['chips_Z']):
                yield ip, iz, p, int(Z)

    def test_cross_section_matches_geant4(self):
        # Exercises the parameter port and the table interpolation below
        # e^8 GeV/c
        n_z = len(REFERENCE['chips_Z'])
        for ip, iz, p, Z in self._cases():
            N = proton_data.BGG_MASS_NUMBER[Z] - Z
            chips = pcs.ChipsElasticDistribution(Z, N, p)
            assert np.isclose(chips.sigma_chips,
                              REFERENCE['chips_elastic_mb'][ip*n_z + iz],
                              rtol=1e-9)

    def test_distribution_matches_geant4_samples(self):
        probs = np.array(REFERENCE['chips_probabilities'])
        n_z = len(REFERENCE['chips_Z'])
        quantiles = np.array(REFERENCE['chips_t_quantiles_GeV2']).reshape(
            len(REFERENCE['chips_p_GeV']), n_z, len(probs))
        for ip, iz, p, Z in self._cases():
            N = proton_data.BGG_MASS_NUMBER[Z] - Z
            chips = pcs.ChipsElasticDistribution(Z, N, p)
            cdf = np.array([chips.probability(0, tt)
                            for tt in quantiles[ip, iz]])
            # 2e5 Geant4 samples: statistical spread of the quantiles
            assert np.max(np.abs(cdf - probs)) < 0.006

    @pytest.mark.parametrize('Z', [1, 2, 7, 18])
    def test_pdf_is_normalised(self, Z):
        N = proton_data.BGG_MASS_NUMBER[Z] - Z
        chips = pcs.ChipsElasticDistribution(Z, N, 6800.)
        integral = _quad_log(chips.pdf, 1e-12, chips.t_max)
        assert np.isclose(integral, 1.0, rtol=1e-6)
        assert np.isclose(chips.probability(0, chips.t_max), 1.0)

    @pytest.mark.parametrize('Z, t_lim', [(1, None), (7, None), (2, None),
                                          (7, (0.05, 0.3)), (18, (1e-4, 1e-3))])
    def test_sampler_matches_pdf(self, Z, t_lim):
        rng = np.random.default_rng(42)
        N = proton_data.BGG_MASS_NUMBER[Z] - Z
        chips = pcs.ChipsElasticDistribution(Z, N, 450.)
        t = np.sort(chips.sample(100_000, rng, t_lim=t_lim))
        t1, t2 = (0.0, chips.t_max) if t_lim is None else t_lim
        assert t[0] >= t1 and t[-1] <= t2
        grid = np.quantile(t, np.linspace(0.01, 0.99, 50))
        norm = chips.probability(t1, t2)
        cdf = np.array([chips.probability(t1, gg)/norm for gg in grid])
        empirical = np.searchsorted(t, grid)/t.size
        assert np.max(np.abs(cdf - empirical)) < 0.006


#############################################################
# Coulomb scattering off the nucleus
#############################################################
class TestCoulomb:

    def test_majorant_and_acceptance_match_geant4(self):
        theta = REFERENCE['coulomb_theta_edges']
        k = 0
        for p in REFERENCE['coulomb_p_GeV']:
            for Z in REFERENCE['coulomb_Z']:
                coul = pcs.WentzelCoulombCrossSection(int(Z), p*1e9)
                for th1, th2 in zip(theta[:-1], theta[1:]):
                    majorant = coul.majorant_xsec(th1, th2)
                    # Geant4 takes the difference of cosines, which loses
                    # ~1e-16/(1 - cos(theta1)) of relative precision
                    rtol = 1e-8 + 2.2e-16/(2*np.sin(0.5*th1)**2)
                    assert np.isclose(majorant,
                                      REFERENCE['coulomb_majorant_mb'][k],
                                      rtol=rtol)
                    z1, z2 = (2*np.sin(0.5*th1)**2, 2*np.sin(0.5*th2)**2)
                    effective = pcs._integrate_log(coul.dxsec_dz, z1, z2,
                                                   nodes_per_decade=4)
                    accepted = REFERENCE['coulomb_acceptance'][k]
                    # 1e6 Geant4 samples per window
                    sigma = np.sqrt(max(accepted*(1 - accepted), 1e-6)/1e6)
                    assert abs(effective/majorant - accepted) < 5*sigma + rtol
                    k += 1

    def test_rutherford_limit(self):
        # Between screening and nuclear size, dsigma/d|t| = 4 pi (Z alpha
        # hbar c)^2/(beta^2 t^2)
        coul = pcs.WentzelCoulombCrossSection(7, 450e9)
        t = 1e-6
        beta2 = 1/coul._invbeta2
        expected = 4*np.pi*(7*pcs.ALPHA)**2*pcs.HBARC2_GEV2_MB/beta2/t**2
        assert np.isclose(coul.dxsec_dt(t), expected, rtol=1e-3)

    def test_form_factor_suppresses_large_angles(self):
        coul = pcs.WentzelCoulombCrossSection(18, 6.8e12)
        z = 0.5*(10e-6)**2   # 10 urad
        # ~20% suppression for argon at 6.8 TeV and 10 urad
        assert 0.75 < coul.acceptance(z) < 0.85
        assert np.all(coul.acceptance(np.geomspace(1e-20, 1e-2, 50)) <= 1)


#############################################################
# Elastic channel: hadronic + Coulomb + interference
#############################################################
class TestElastic:

    @pytest.mark.parametrize('p0c', [450e9, 6.8e12])
    @pytest.mark.parametrize('Z', [1, 7, 18])
    def test_xsec_matches_numerical_integral(self, p0c, Z):
        el = pcs.ProtonElasticCalculator(Z, p0c, theta_lim=(1e-6, 50e-3))
        integral = _quad_log(el.dxsec_dt, *el.t_lim)
        assert np.isclose(el.compute_xsec()/MB, integral, rtol=1e-8)

    def test_components_add_up(self):
        kw = dict(theta_lim=(1e-6, 50e-3))
        full = pcs.ProtonElasticCalculator(7, 6.8e12, **kw)
        no_int = pcs.ProtonElasticCalculator(7, 6.8e12, interference=False,
                                             **kw)
        coul = pcs.ProtonElasticCalculator(7, 6.8e12, nuclear=False, **kw)
        nuc = pcs.ProtonElasticCalculator(7, 6.8e12, coulomb=False, **kw)
        assert coul.process == 'coulomb'
        assert nuc.process == 'nuclear_elastic'
        assert np.isclose(no_int.compute_xsec(),
                          coul.compute_xsec() + nuc.compute_xsec())
        # rho > 0: destructive interference, a few percent here
        interference = full.compute_xsec() - no_int.compute_xsec()
        assert -0.1*full.compute_xsec() < interference < 0

    def test_nuclear_elastic_without_window_is_the_bgg_cross_section(self):
        el = pcs.ProtonElasticCalculator(7, 450e9, coulomb=False,
                                         theta_lim=(0.0, np.pi))
        assert np.isclose(el.compute_xsec(), el.cross_sections.elastic)
        t, weight = el.sample_t(1000, np.random.default_rng(0))
        assert np.allclose(weight, 1.0)

    def test_interference_formula(self):
        el = pcs.ProtonElasticCalculator(1, 6.8e12, theta_lim=(1e-6, 50e-3))
        t = np.array([1e-4, 1e-3, 1e-2])
        fc = el.coulomb_xs.dxsec_dt(t)
        fn = el.sigma_nuclear*el.chips.pdf(t)
        a_phi = -pcs.ALPHA*(np.euler_gamma + np.log(0.5*el.chips.slope*t))
        expected = (fc + fn - 2*np.sqrt(fc*fn)*(el.rho*np.cos(a_phi)
                                                + np.sin(a_phi))
                    / np.sqrt(1 + el.rho**2))
        assert np.allclose(el.dxsec_dt(t), expected)
        # |F_C + F_N|^2 >= 0
        tt = np.geomspace(*el.t_lim, 1000)
        assert np.all(el.dxsec_dt(tt) >= 0)

    @pytest.mark.parametrize('Z', [1, 18])
    def test_weighted_sample_reproduces_the_cross_section(self, Z):
        rng = np.random.default_rng(7)
        el = pcs.ProtonElasticCalculator(Z, 6.8e12, theta_lim=(1e-6, 1e-2))
        t, weight = el.sample_t(1_000_000, rng)
        assert np.isclose(weight.mean(), 1.0,
                          atol=5*weight.std()/np.sqrt(weight.size))
        edges = np.geomspace(*el.t_lim, 13)
        sigma = el.compute_xsec()/MB
        for a, b in zip(edges[:-1], edges[1:]):
            mask = (t >= a) & (t < b)
            estimate = weight[mask].sum()/t.size*sigma
            error = np.sqrt((weight[mask]**2).sum())/t.size*sigma
            exact = _quad_log(el.dxsec_dt, a, b, n_sub=4)
            assert abs(estimate - exact) < 5*error + 1e-9*sigma

    def test_raises_on_invalid_window(self):
        with pytest.raises(ValueError, match='coulomb_theta'):
            pcs.ProtonElasticCalculator(7, 450e9, theta_lim=(1e-3, 1e-4))
        with pytest.raises(ValueError, match='theta_min > 0'):
            pcs.ProtonElasticCalculator(7, 450e9, theta_lim=(0.0, 1e-3))
        with pytest.raises(ValueError, match='At least one'):
            pcs.ProtonElasticCalculator(7, 450e9, coulomb=False,
                                        nuclear=False)


#############################################################
# Quasi-elastic, diffraction, absorption and knock-on
#############################################################
class TestOtherProcesses:

    def test_quasi_elastic_uses_the_pp_slope(self):
        rng = np.random.default_rng(1)
        qe = pcs.ProtonQuasiElasticCalculator(7, 450e9)
        pp = pcs.ChipsElasticDistribution(1, 0, qe.p)
        t = np.sort(qe.chips.sample(50_000, rng))
        grid = np.quantile(t, np.linspace(0.05, 0.95, 19))
        cdf = np.array([pp.probability(0, gg) for gg in grid])
        assert np.max(np.abs(cdf - np.searchsorted(t, grid)/t.size)) < 0.01
        assert qe.compute_xsec() == qe.cross_sections.quasi_elastic
        assert pcs.ProtonQuasiElasticCalculator(1, 450e9).compute_xsec() == 0

    def test_diffraction_cross_section(self):
        for p0c in (450e9, 6.8e12):
            s = 2*pcs._MP*p0c*1e-9
            sd_h = pcs.ProtonDiffractionCalculator(1, p0c)
            assert sd_h.n_eff == 1.0
            assert np.isclose(sd_h.compute_xsec()/MB, 4.3 + 0.3*np.log(s))
            sd_n = pcs.ProtonDiffractionCalculator(7, p0c, sd_scale=0.5)
            n_eff = 1.6177*proton_data.NIST_ATOMIC_MASS[7]**(1/3)
            assert np.isclose(sd_n.compute_xsec()/MB,
                              0.5*n_eff*(4.3 + 0.3*np.log(s)), rtol=1e-4)

    @pytest.mark.parametrize('p0c', [450e9, 6.8e12])
    def test_diffraction_kinematics(self, p0c):
        rng = np.random.default_rng(3)
        n = 200_000
        sd = pcs.ProtonDiffractionCalculator(7, p0c)
        zeros = np.zeros(n)
        sample = sd.sample_deflections(zeros, zeros, zeros, rng)
        m2 = sample.mass_x2
        assert m2.min() >= sd.m2_lim[0] and m2.max() <= sd.m2_lim[1]
        # dN/dM^2 ~ 1/M^2: log(M^2) uniform
        u = np.log(m2/sd.m2_lim[0])/np.log(sd.m2_lim[1]/sd.m2_lim[0])
        assert np.all(np.abs(np.histogram(u, bins=10, range=(0, 1))[0]/n
                             - 0.1) < 0.005)
        # Energy loss nu = (M^2 - m_p^2 + |t|)/(2 m_p)
        nu = (m2 - pcs._MP**2 + sample.t)/(2*pcs._MP)
        assert np.allclose(sample.energy_loss, nu*1e9)
        E = np.sqrt(sd.p**2 + pcs._MP**2)
        p_out = np.sqrt((E - nu)**2 - pcs._MP**2)
        assert np.allclose((1 + sample.delta)*sd.p, p_out, rtol=1e-12)
        # The relative momentum loss is (M^2 - m_p^2 + |t|)/s ~ M^2/s
        assert np.allclose(-sample.delta, (m2 - pcs._MP**2 + sample.t)/sd.s,
                           rtol=1e-4)
        assert np.all(sample.theta > 0)

    def test_absorption(self):
        for Z in (1, 7, 18):
            ab = pcs.ProtonAbsorptionCalculator(Z, 6.8e12)
            sd = pcs.ProtonDiffractionCalculator(Z, 6.8e12)
            xs = ab.cross_sections
            assert np.isclose(ab.compute_xsec(), xs.inelastic
                              - xs.quasi_elastic - sd.compute_xsec())
            px = np.array([1e-5, -2e-6])
            sample = ab.sample_deflections(px, px, px,
                                           np.random.default_rng(0))
            assert np.all(sample.absorbed)
            assert np.array_equal(sample.px, px)
            assert np.array_equal(sample.delta, px)
        with pytest.raises(ValueError, match='sd_scale'):
            pcs.ProtonAbsorptionCalculator(1, 6.8e12, sd_scale=10)

    def test_knock_on_matches_geant4(self):
        k = 0
        for p in REFERENCE['knock_on_p_GeV']:
            for Z in REFERENCE['knock_on_Z']:
                for cut in REFERENCE['knock_on_energy_fraction_cut']:
                    ko = pcs.ProtonKnockOnCalculator(int(Z), p*1e9, cut=cut)
                    assert np.isclose(ko.bare_xsec(*ko.energy_lim),
                                      REFERENCE['knock_on_mb'][k], rtol=1e-6)
                    # The proton form factor only removes events
                    assert ko.compute_xsec()/MB <= ko.bare_xsec(
                        *ko.energy_lim)*(1 + 1e-9)
                    k += 1

    def test_knock_on_sample(self):
        rng = np.random.default_rng(9)
        n = 500_000
        ko = pcs.ProtonKnockOnCalculator(7, 450e9)
        zeros = np.zeros(n)
        sample = ko.sample_deflections(zeros, zeros, zeros, rng)
        # The deflection of a proton by an electron is at most m_e/m_p
        assert sample.theta.max() <= pcs._ME/pcs._MP*(1 + 1e-6)
        assert np.allclose(sample.t, 2*pcs._ME*sample.energy_loss*1e-9)
        weight = sample.weight
        assert np.isclose(weight.mean(), 1.0,
                          atol=5*weight.std()/np.sqrt(n))
        # Weighted rate above 1e-3 of the energy
        sigma = ko.compute_xsec()/MB
        above = sample.energy_loss*1e-9 > 1e-3*ko.energy
        estimate = weight[above].sum()/n*sigma
        error = np.sqrt((weight[above]**2).sum())/n*sigma
        exact = pcs._integrate_log(ko.dxsec_dT, 1e-3*ko.energy, ko.t_kin_max)
        assert abs(estimate - exact) < 5*error


#############################################################
# Kinematics common to all processes
#############################################################
@pytest.mark.parametrize('calculator', [
    lambda: pcs.ProtonElasticCalculator(7, 450e9, theta_lim=(1e-6, 50e-3)),
    lambda: pcs.ProtonQuasiElasticCalculator(7, 6.8e12),
    lambda: pcs.ProtonDiffractionCalculator(18, 6.8e12),
    lambda: pcs.ProtonKnockOnCalculator(1, 450e9)])
def test_scattered_momentum_is_consistent(calculator):
    calc = calculator()
    rng = np.random.default_rng(2)
    n = 20_000
    px = rng.normal(0, 1e-5, n)
    py = rng.normal(0, 1e-5, n)
    delta = rng.normal(0, 1e-4, n)
    sample = calc.sample_deflections(px, py, delta, rng)
    # Momentum modulus from the energy loss
    p_in = calc.p*(1 + delta)
    E_in = np.sqrt(p_in**2 + pcs._MP**2)
    p_out = np.sqrt((E_in - sample.energy_loss*1e-9)**2 - pcs._MP**2)
    assert np.allclose((1 + sample.delta)*calc.p, p_out, rtol=1e-12)
    # Angle between the incoming and outgoing momenta
    def unit(px, py, delta):
        pz = np.sqrt((1 + delta)**2 - px**2 - py**2)
        v = np.stack((px, py, pz), axis=1)
        return v/np.linalg.norm(v, axis=1)[:, None]
    cross = np.linalg.norm(np.cross(unit(px, py, delta),
                                    unit(sample.px, sample.py, sample.delta)),
                           axis=1)
    assert np.allclose(np.arcsin(cross), sample.theta, rtol=1e-6, atol=1e-12)
