# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""
Single-interaction models for a relativistic proton beam on residual gas.

This module provides the cross sections and the Monte Carlo samplers used by
:class:`xcoll.BeamGasStudy` for proton beams (LHC energies, ~100 GeV to
10 TeV). Each gas atom is treated independently, consistently with the
electron/positron models of :mod:`xcoll.beamgas.cross_sections`.

Processes
---------
* ``'absorption'``: inelastic proton-nucleus interaction that removes the
  beam proton (non-diffractive production, projectile and double
  diffraction). Its cross section is the inelastic one minus the
  quasi-elastic and target-diffractive parts, which are generated
  separately because the beam proton survives them.
* ``'elastic'``: coherent elastic scattering off the whole nucleus (proton-
  proton elastic scattering for hydrogen), including single Coulomb
  scattering off the nucleus and the Coulomb-nuclear interference. The
  variants ``'nuclear_elastic'`` and ``'coulomb'`` keep only the hadronic or
  the Coulomb amplitude.
* ``'quasi_elastic'``: elastic scattering off a single bound nucleon, with
  break-up of the nucleus. Not present for hydrogen.
* ``'diffractive'``: single diffraction with dissociation of the target
  nucleon, ``p N -> p X``. The beam proton survives with a momentum loss of
  order ``M_X^2/s``.
* ``'knock_on'``: elastic scattering off an atomic electron with a large
  energy transfer (hard tail of the ionisation energy loss).

Acknowledgements
----------------
The hadronic and electromagnetic models are transcriptions of the Geant4
models that the FTFP_BERT physics list uses for protons, as found in
Geant4 11.4.2 (geant4-11-04-patch-02). The provenance of each piece is given
in the individual docstrings:

* Inelastic and elastic proton-nucleus cross sections: Barashenkov-Glauber-
  Gribov, ``G4BGGNucleonInelasticXS``, ``G4BGGNucleonElasticXS``,
  ``G4ComponentGGHadronNucleusXsc``, ``G4HadronNucleonXsc`` and
  ``G4NuclearRadii``. Constants that depend on the Barashenkov data tables
  are tabulated in :mod:`xcoll.beamgas.proton_data`.
* Quasi-elastic cross section: Glauber-Gribov inelastic minus production
  cross section, from ``G4ComponentGGHadronNucleusXsc``.
* Elastic momentum transfer: CHIPS parametrisation of
  ``G4ChipsProtonElasticXS``, as used by ``G4ChipsElasticModel``.
* Single Coulomb scattering: ``G4WentzelOKandVIxSection``, as used by
  ``G4eCoulombScatteringModel`` for protons.
* Knock-on electrons: ``G4BetheBlochModel``.

Geant4 has no standalone single-diffraction model. The target-dissociation
model is the one of the Everest scattering routine of Xcoll
(``xcoll/scattering_routines/everest``, K2/SixTrack heritage), with the
exact two-body kinematics of the scattered proton.

The Coulomb-nuclear interference follows the standard West-Yennie treatment,
with the forward real-to-imaginary ratio ``rho`` derived from the same PDG
parametrisation of the proton-proton total cross section that Geant4 uses.

References
----------
.. [1] S. Agostinelli et al. (Geant4 Collaboration), Nucl. Instrum. Meth. A
   **506**, 250 (2003). https://doi.org/10.1016/S0168-9002(03)01368-8
.. [2] V. M. Grichine, "A simplified Glauber model for hadron-nucleus cross
   sections", Eur. Phys. J. C **62**, 399 (2009).
   https://doi.org/10.1140/epjc/s10052-009-1033-0
.. [3] M. V. Kossov, "Approximation of photonuclear interaction cross-
   sections", Eur. Phys. J. A **14**, 377 (2002).
   https://doi.org/10.1140/epja/i2001-10223-x
.. [4] C. Patrignani et al. (Particle Data Group), Chin. Phys. C **40**,
   100001 (2016), section "Plots of cross sections and related quantities".
.. [5] G. B. West and D. R. Yennie, "Coulomb interference in high-energy
   scattering", Phys. Rev. **172**, 1413 (1968).
   https://doi.org/10.1103/PhysRev.172.1413
.. [6] M. M. Block and R. N. Cahn, "High-energy pp and p̄p forward elastic
   scattering and total cross sections", Rev. Mod. Phys. **57**, 563 (1985).
   https://doi.org/10.1103/RevModPhys.57.563
.. [7] C. Tambasco, "An improved scattering routine for collimation tracking
   studies at LHC", MSc thesis, Università di Roma La Sapienza (2014).
"""

from dataclasses import dataclass

import numpy as np
from scipy import constants as sc

import xtrack as xt

from .cross_sections import _momentum_frame
from . import proton_data


# ############################################################ #
# Constants
# ############################################################ #
PROTON_MASS_EV = xt.PROTON_MASS_EV
ELECTRON_MASS_EV = xt.ELECTRON_MASS_EV
NEUTRON_MASS_EV = sc.physical_constants[
    'neutron mass energy equivalent in MeV'][0]*1e6
ATOMIC_MASS_CONSTANT_EV = sc.m_u*sc.c**2/sc.e
ALPHA = sc.alpha
# Proton magnetic moment in units of the nuclear magneton
PROTON_MAGNETIC_MOMENT = sc.physical_constants[
    'proton mag. mom. to nuclear magneton ratio'][0]
CLASSICAL_ELECTRON_RADIUS = sc.value('classical electron radius')

# Masses in GeV, the unit of the hadronic parametrisations
_MP = PROTON_MASS_EV*1e-9
_MN = NEUTRON_MASS_EV*1e-9
_ME = ELECTRON_MASS_EV*1e-9
_AMU = ATOMIC_MASS_CONSTANT_EV*1e-9

# (hbar c)^2 in GeV^2 mb, and hbar c in GeV fm
HBARC2_GEV2_MB = (sc.hbar*sc.c/sc.e*1e-9)**2*1e31
HBARC_GEV_FM = sc.hbar*sc.c/sc.e*1e-9*1e15

# One millibarn in m^2
MILLIBARN = 1e-31

# Processes available for proton beams
PROTON_PROCESSES = ('absorption', 'elastic', 'nuclear_elastic', 'coulomb',
                    'quasi_elastic', 'diffractive', 'knock_on')
# The three variants of the elastic channel are mutually exclusive
ELASTIC_VARIANTS = ('elastic', 'nuclear_elastic', 'coulomb')
# Processes simulated when process='all'
ALL_PROTON_PROCESSES = ('absorption', 'elastic', 'quasi_elastic',
                        'diffractive', 'knock_on')

# Gauss-Legendre nodes used for the integrals in log(t) or log(T), per decade
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)


def _check_Z(Z):
    Z = int(Z)
    if Z < 1 or Z > proton_data.MAX_Z:
        raise ValueError(f"Atomic number Z={Z} is out of the supported range "
                         f"[1, {proton_data.MAX_Z}].")
    return Z


def _integrate_log(func, x1, x2, nodes_per_decade=1):
    """
    Integrate ``func(x)`` from ``x1`` to ``x2`` in ``log(x)``.

    A 32-point Gauss-Legendre rule is applied on each decade (or on each
    fraction of a decade, see ``nodes_per_decade``), which integrates the
    smooth, many-decade integrands of this module to near machine precision.

    Parameters
    ----------
    func : callable
        Vectorised integrand.
    x1, x2 : float
        Integration limits, ``0 < x1 <= x2``.
    nodes_per_decade : int, optional
        Number of 32-point sub-intervals per decade.

    Returns
    -------
    integral : float
        Value of the integral.
    """
    if x2 <= x1:
        return 0.0
    u1, u2 = np.log(x1), np.log(x2)
    n_sub = max(int(np.ceil((u2 - u1)/np.log(10)*nodes_per_decade)), 1)
    edges = np.linspace(u1, u2, n_sub + 1)
    half = 0.5*np.diff(edges)
    mid = 0.5*(edges[1:] + edges[:-1])
    u = mid[:, None] + half[:, None]*_GL_NODES[None, :]
    x = np.exp(u)
    return float(np.sum(half[:, None]*_GL_WEIGHTS[None, :]*func(x)*x))


def _mandelstam_s(p, m1, m2):
    """Centre-of-mass energy squared [GeV^2] for momentum ``p`` [GeV/c] of a
    projectile of mass ``m1`` on a target of mass ``m2`` at rest."""
    return m1*m1 + m2*m2 + 2.0*m2*np.sqrt(p*p + m1*m1)


def _proton_kinematics(pc_ev):
    """Return (p, E) in GeV for a proton of momentum ``pc_ev`` [eV]."""
    p = np.asarray(pc_ev, dtype=float)*1e-9
    return p, np.sqrt(p*p + _MP*_MP)


def _scattering_z(t, E, p, energy_loss, mass=_MP):
    """
    Exact ``z = 1 - cos(theta)`` of a projectile scattered with momentum
    transfer ``|t|`` and energy loss ``energy_loss``.

    Starting from ``t = 2 m^2 - 2 (E E' - p p' cos(theta))``, the quantity
    ``D = E E' - p p' - m^2 = m^2 dE^2/(E E' + p p' - m^2)`` is evaluated
    without cancellation, so that ``z = (|t|/2 - D)/(p p')`` keeps full
    relative precision down to the smallest angles.

    Parameters
    ----------
    t : ndarray
        Momentum transfer ``|t|`` [GeV^2].
    E, p : ndarray
        Total energy [GeV] and momentum [GeV/c] of the projectile.
    energy_loss : ndarray
        Energy lost by the projectile [GeV].
    mass : float, optional
        Projectile mass [GeV].

    Returns
    -------
    z : ndarray
        ``1 - cos(theta)``, clipped to ``[0, 2]``.
    p_out : ndarray
        Momentum of the scattered projectile [GeV/c].
    """
    E_out = E - energy_loss
    p_out = np.sqrt(np.maximum(E_out*E_out - mass*mass, 0.0))
    D = mass*mass*energy_loss**2/(E*E_out + p*p_out - mass*mass)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (0.5*t - D)/(p*p_out)
    return np.clip(np.nan_to_num(z, nan=0.0), 0.0, 2.0), p_out


def _scatter_momenta(px, py, delta, theta, p_out, p0c, rng):
    """
    Rotate the momentum of each particle by ``theta`` (random azimuth) and
    set its modulus to ``p_out``.

    Parameters
    ----------
    px, py, delta : ndarray
        Normalised momenta before the interaction.
    theta : ndarray
        Polar scattering angle [rad].
    p_out : ndarray
        Momentum after the interaction [GeV/c].
    p0c : float
        Reference momentum [eV].
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    px, py, delta : ndarray
        Normalised momenta after the interaction.
    """
    n = px.size
    phi = rng.uniform(0.0, 2.0*np.pi, n)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    u_hat, v_hat, p_hat, _ = _momentum_frame(px, py, delta)
    direction = ((sin_theta*np.cos(phi))[:, None]*u_hat
                 + (sin_theta*np.sin(phi))[:, None]*v_hat
                 + cos_theta[:, None]*p_hat)
    p_vec = (p_out*1e9/p0c)[:, None]*direction
    return p_vec[:, 0], p_vec[:, 1], np.linalg.norm(p_vec, axis=1) - 1.0


# ############################################################ #
# Sample
# ############################################################ #
@dataclass
class ProtonScatteringSample:
    """
    Outcome of a batch of proton-gas interactions.

    Momenta are normalised to the reference momentum ``p0c``, consistently
    with the Xsuite convention and with
    :class:`xcoll.beamgas.ScatteringSample`.

    Parameters
    ----------
    px, py : ndarray
        Normalised transverse momenta after the interaction.
    delta : ndarray
        Relative momentum deviation after the interaction.
    weight : ndarray
        Importance-sampling weight of each event, normalised to the total
        cross section returned by ``compute_xsec``, so that its expectation
        value is one.
    theta : ndarray
        Polar scattering angle of the proton [rad]. Zero for absorption.
    t : ndarray
        Four-momentum transfer ``|t|`` [GeV^2]. ``NaN`` for absorption.
    energy_loss : ndarray
        Energy lost by the proton [eV]. For knock-on electrons it is the
        kinetic energy of the electron, for diffraction the energy given to
        the dissociated system. ``NaN`` for absorption.
    mass_x2 : ndarray
        Squared mass of the diffractive system ``M_X^2`` [GeV^2]. ``NaN``
        for the other processes.
    coulomb_fraction : ndarray
        Fraction of the elastic differential cross section at the sampled
        ``|t|`` that is due to the Coulomb amplitude alone. ``NaN`` for the
        other processes.
    absorbed : ndarray of bool
        Whether the proton is removed by the interaction.
    """
    px: np.ndarray
    py: np.ndarray
    delta: np.ndarray
    weight: np.ndarray
    theta: np.ndarray
    t: np.ndarray
    energy_loss: np.ndarray
    mass_x2: np.ndarray
    coulomb_fraction: np.ndarray
    absorbed: np.ndarray

    @classmethod
    def _new(cls, n, px, py, delta, **kwargs):
        nan = np.full(n, np.nan)
        fields = dict(px=px, py=py, delta=delta, weight=np.ones(n),
                      theta=np.zeros(n), t=nan, energy_loss=nan.copy(),
                      mass_x2=nan.copy(), coulomb_fraction=nan.copy(),
                      absorbed=np.zeros(n, dtype=bool))
        fields.update(kwargs)
        return cls(**fields)


# ############################################################ #
# Hadron-nucleon cross sections
# ############################################################ #
def proton_nucleon_cross_sections(p, target='proton'):
    """
    Total, elastic and inelastic proton-nucleon cross sections.

    Port of the proton branch of ``G4HadronNucleonXsc::HadronNucleonXscNS``
    for ``p >= 10 GeV/c``: the total cross section is the PDG-2017
    parametrisation (``HadronNucleonXscPDG``), the elastic one the N. Starkov
    parametrisation tuned to the TOTEM data above 373 GeV/c.

    Parameters
    ----------
    p : float
        Projectile momentum in the target rest frame [GeV/c]. Must be at
        least 10 GeV/c.
    target : {'proton', 'neutron'}, optional
        Target nucleon.

    Returns
    -------
    total, elastic, inelastic : float
        Cross sections [mb].
    """
    if p < 10.0:
        raise ValueError("The proton-nucleon cross sections are only "
                         "implemented above 10 GeV/c.")
    if target == 'proton':
        mt, P, R1, R2 = _MP, 34.41, 13.07, -7.394
    elif target == 'neutron':
        mt, P, R1, R2 = _MN, 34.71, 12.52, -6.66
    else:
        raise ValueError(f"Unknown target nucleon {target!r}.")

    s = _mandelstam_s(p, _MP, mt)
    x = _MP + mt + 2.1206
    blog = np.log(s/(x*x))
    total = (0.272*blog*blog + P + R1*np.exp(-0.4473*blog)
             + R2*np.exp(-0.5486*blog))

    if p >= 373.0:
        elastic = (6.5 + 0.308*np.log(s/400.0)**1.65
                   + 9.19*np.exp(-0.458*np.log(s)))
    elif p >= 100.0:
        elastic = (5.53 + 0.308*np.log(s/28.9)**1.1
                   + 9.19*np.exp(-0.458*np.log(s)))
    else:
        elastic = 6.0 + 20.0/((np.log(p) - 0.182)**2 + 1.0)
    elastic = min(elastic, total)
    return float(total), float(elastic), float(max(total - elastic, 0.0))


def proton_proton_rho(p):
    """
    Forward real-to-imaginary ratio of the proton-proton elastic amplitude.

    Obtained from the PDG high-energy parametrisation of the total cross
    section used by ``G4HadronNucleonXsc::HadronNucleonXscPDG``,
    ``sigma = H L^2 + P + R1 e^(-eta1 L) + R2 e^(-eta2 L)`` with
    ``L = log(s/s_M)``, through the derivative dispersion relations:
    ``rho sigma = pi H L - R1 e^(-eta1 L) tan(pi eta1/2)
    + R2 e^(-eta2 L) cot(pi eta2/2)``, see [4]_ of the module.

    Parameters
    ----------
    p : float
        Proton momentum in the target rest frame [GeV/c].

    Returns
    -------
    rho : float
        Real-to-imaginary ratio (about 0.02 at 450 GeV/c and 0.11 at
        6.8 TeV/c).
    """
    s = _mandelstam_s(p, _MP, _MP)
    x = 2*_MP + 2.1206
    blog = np.log(s/(x*x))
    H, P, R1, R2, eta1, eta2 = 0.272, 34.41, 13.07, -7.394, 0.4473, 0.5486
    total = (H*blog*blog + P + R1*np.exp(-eta1*blog)
             + R2*np.exp(-eta2*blog))
    real = (np.pi*H*blog - R1*np.exp(-eta1*blog)*np.tan(0.5*np.pi*eta1)
            + R2*np.exp(-eta2*blog)/np.tan(0.5*np.pi*eta2))
    return float(real/total)


# ############################################################ #
# Proton-nucleus cross sections
# ############################################################ #
def _radius_hngg(A):
    """Nuclear radius [fm] of ``G4NuclearRadii::RadiusHNGG``."""
    if A > 20:
        return 1.08*A**(1/3)*(0.8 + 0.2*np.exp(-(A - 20)/20))
    return 1.08*A**(1/3)*(1.0 + 0.1*np.exp(-(A - 20)/20))


class ProtonNucleusCrossSections:
    """
    Proton cross sections on the nucleus of a gas atom.

    Port of the Barashenkov-Glauber-Gribov cross sections that FTFP_BERT
    uses for protons above 91 GeV: ``G4BGGNucleonInelasticXS`` and
    ``G4BGGNucleonElasticXS``. For ``Z > 1`` they are the Glauber-Gribov
    cross sections of ``G4ComponentGGHadronNucleusXsc::ComputeCrossSections``
    times per-element factors that join them to the Barashenkov cross
    sections at 91 GeV; for hydrogen they are ``1.0115`` times the proton-
    proton cross sections of ``G4HadronNucleonXsc::HadronNucleonXscNS``.

    The quasi-elastic cross section, i.e. elastic scattering off a bound
    nucleon with break-up of the nucleus, is the Glauber-Gribov inelastic
    minus production cross section, scaled by the same factor as the
    inelastic cross section. It is part of the inelastic cross section.

    Parameters
    ----------
    Z : int
        Atomic number of the target, between 1 and 92.
    p0c : float
        Momentum of the proton [eV]. Must be above 91 GeV (the Barashenkov
        part of the BGG cross sections is not implemented).

    Attributes
    ----------
    Z, A, N : int
        Atomic number, mass number and neutron number of the target nucleus,
        as used by the BGG cross sections.
    target_mass : float
        Mass of the target nucleus [eV] (proton mass for hydrogen).
    inelastic, elastic, quasi_elastic, production : float
        Cross sections [m^2]. ``elastic`` is the coherent elastic
        (proton-proton elastic for hydrogen).
    gg_total, gg_inelastic, gg_production, gg_elastic : float
        Unscaled Glauber-Gribov cross sections [m^2] (``Z > 1`` only).
    """

    _GLAUBER_ENERGY = 91.0  # GeV, kinetic energy

    def __init__(self, Z, p0c):
        self.Z = _check_Z(Z)
        self.A = int(proton_data.BGG_MASS_NUMBER[self.Z])
        self.N = self.A - self.Z
        self.p0c = float(p0c)
        p, E = _proton_kinematics(self.p0c)
        if E - _MP <= self._GLAUBER_ENERGY:
            raise ValueError(
                f"The proton-gas cross sections are only implemented above "
                f"{self._GLAUBER_ENERGY:.0f} GeV of kinetic energy (got "
                f"{E - _MP:.3g} GeV).")

        if self.Z == 1:
            self.target_mass = PROTON_MASS_EV
        else:
            self.target_mass = (proton_data.NIST_ATOMIC_MASS[self.Z]
                                * ATOMIC_MASS_CONSTANT_EV
                                - self.Z*ELECTRON_MASS_EV)

        pp_tot, pp_el, pp_in = proton_nucleon_cross_sections(p, 'proton')
        if self.Z == 1:
            factor = 1.0115
            self.gg_total = self.gg_inelastic = np.nan
            self.gg_production = self.gg_elastic = np.nan
            self.inelastic = factor*pp_in*MILLIBARN
            self.elastic = factor*pp_el*MILLIBARN
            self.production = self.inelastic
            self.quasi_elastic = 0.0
            return

        pn_tot, _, pn_in = proton_nucleon_cross_sections(p, 'neutron')
        Z, N = self.Z, self.N
        R = _radius_hngg(self.A)
        # 2 pi R^2, with 1 fm^2 = 10 mb
        nucleus_square = 2.0*np.pi*R*R*10.0
        ratio = (Z*pp_tot + N*pn_tot)/nucleus_square
        cof_inelastic = 2.4
        bar_tot = proton_data.GG_PROTON_BARYON_CORRECTION_TOTAL[Z]
        bar_in = proton_data.GG_PROTON_BARYON_CORRECTION_INELASTIC[Z]
        gg_total = nucleus_square*np.log(1.0 + ratio)*bar_tot
        gg_inelastic = (nucleus_square*np.log(1.0 + cof_inelastic*ratio)
                        / cof_inelastic*bar_in)
        gg_elastic = max(gg_total - gg_inelastic, 0.0)
        xratio = (Z*pp_in + N*pn_in)/nucleus_square
        gg_production = min(nucleus_square*np.log(1.0 + cof_inelastic*xratio)
                            * bar_in/cof_inelastic, gg_inelastic)

        self.gg_total = gg_total*MILLIBARN
        self.gg_inelastic = gg_inelastic*MILLIBARN
        self.gg_production = gg_production*MILLIBARN
        self.gg_elastic = gg_elastic*MILLIBARN

        f_in = proton_data.BGG_GLAUBER_FACTOR_INELASTIC[Z]
        f_el = proton_data.BGG_GLAUBER_FACTOR_ELASTIC[Z]
        self.inelastic = f_in*self.gg_inelastic
        self.elastic = f_el*self.gg_elastic
        self.production = f_in*self.gg_production
        self.quasi_elastic = self.inelastic - self.production

    def __repr__(self):
        return (f"<ProtonNucleusCrossSections Z={self.Z} A={self.A}, "
                f"p0c={self.p0c:.3e} eV: inelastic="
                f"{self.inelastic/MILLIBARN:.1f} mb, elastic="
                f"{self.elastic/MILLIBARN:.1f} mb>")


# ############################################################ #
# CHIPS elastic differential cross section
# ############################################################ #
_CHIPS_PP_PARAMETERS = np.array([
    2.865, 18.9, .6461, 3., 9., .425, .4276, .0022, 5., 74., 3., 3.4, .2, .17,
    .001, 8., .055, 3.64, 5.e-5, 4000., 1500., .46, 1.2e6, 3.5e6, 5.e-5,
    1.e10, 8.5e8, 1.e10, 1.1, 3.4e6, 6.8e6, 0.])

# Momentum table of G4ChipsProtonElasticXS: log(p/GeV) from -8 to 8 in 127
# steps. Below the top of the table the parameters are linearly interpolated
# between the nodes; above it they are evaluated directly.
_CHIPS_LP_MIN = -8.0
_CHIPS_LP_MAX = 8.0
_CHIPS_N_LAST = 127
_CHIPS_DLP = (_CHIPS_LP_MAX - _CHIPS_LP_MIN)/_CHIPS_N_LAST


def _chips_parameters(Z, N):
    """
    Parameters of the CHIPS elastic cross section for a proton on (Z, N).

    Port of ``G4ChipsProtonElasticXS::GetPTables`` (parameter part).
    """
    if Z == 1 and N == 0:
        return _CHIPS_PP_PARAMETERS
    a = float(Z + N)
    sa = np.sqrt(a)
    ssa = np.sqrt(sa)
    asa = a*sa
    a2 = a*a
    a3 = a2*a
    a4 = a3*a
    a5 = a4*a
    a6 = a4*a2
    a7 = a6*a
    a8 = a7*a
    a9 = a8*a
    a10 = a5*a5
    a12 = a6*a6
    a14 = a7*a7
    a16 = a8*a8
    a17 = a16*a
    a20 = a16*a4
    a32 = a16*a16
    par = np.zeros(52)
    par[0] = 5./(1. + 22./asa)
    par[1] = 4.8*a**1.14/(1. + 3.6/a3)
    par[2] = 1./(1. + 4.E-3*a4) + 2.E-6*a3/(1. + 1.3E-6*a3)
    par[3] = 1.3*a
    par[4] = 3.E-8*a3/(1. + 4.E-7*a4)
    par[5] = .07*asa/(1. + .009*a2)
    par[6] = (3. + 3.E-16*a20)/(1. + a20*(2.E-16/a + 3.E-19*a))
    par[7] = (5.E-9*a4*sa + .27/a)/(1. + 5.E16/a20)/(1. + 6.E-9*a4) + .015/a2
    par[8] = (.001*a + .07/a)/(1. + 5.E13/a16 + 5.E-7*a3) + .0003/sa
    if a < 6.5:
        a28 = a16*a12
        par[9] = 4000*a
        par[10] = 1.2e7*a8 + 380*a17
        par[11] = .7/(1. + 4.e-12*a16)
        par[12] = 2.5/a8/(a4 + 1.e-16*a32)
        par[13] = .28*a
        par[14] = 1.2*a2 + 2.3
        par[15] = 3.8/a
        par[16] = .01/(1. + .0024*a5)
        par[17] = .2*a
        par[18] = 9.e-7/(1. + .035*a5)
        par[19] = (42. + 2.7e-11*a16)/(1. + .14*a)
        par[20] = 2.25*a3
        par[21] = 18.
        par[22] = 2.4e-3*a8/(1. + 2.6e-4*a7)
        par[23] = 3.5e-36*a32*a8/(1. + 5.e-15*a32/a)
        par[24] = 1.e5/(a8 + 2.5e12/a16)
        par[25] = 8.e7/(a12 + 1.e-27*a28*a28)
        par[26] = .0006*a3
        par[27] = 10. + 4.e-8*a12*a
        par[28] = .114
        par[29] = .003
        par[30] = 2.e-23
        par[31] = 1./(1. + .0001*a8)
        par[32] = 1.5e-4/(1. + 5.e-6*a12)
        par[33] = .03
        par[34] = a/2
        par[35] = 2.e-7*a4
        par[36] = 4.
        par[37] = 64./a3
        par[38] = 1.e8*np.exp(.32*asa)
        par[39] = 20.*np.exp(.45*asa)
        par[40] = 7.e3 + 2.4e6/a5
        par[41] = 2.5e5*np.exp(.085*a3)
        par[42] = 2.5*a
        par[43] = 920. + .03*a8*a3
        par[44] = 93. + .0023*a12
    else:
        p1a10 = 2.2e-28*a10
        r4a16 = 6.e14/a16
        s4a16 = r4a16*r4a16
        par[9] = 4.5*a**1.15
        par[10] = .06*a**.6
        par[11] = .6*a/(1. + 2.e15/a16)
        par[12] = .17/(a + 9.e5/a3 + 1.5e33/a32)
        par[13] = (.001 + 7.e-11*a5)/(1. + 4.4e-11*a5)
        par[14] = (p1a10*p1a10 + 2.e-29)/(1. + 2.e-22*a12)
        par[15] = 400./a12 + 2.e-22*a9
        par[16] = 1.e-32*a12/(1. + 5.e22/a14)
        par[17] = 1000./a2 + 9.5*sa*ssa
        par[18] = 4.e-6*a*asa + 1.e11/a16
        par[19] = (120./a + .002*a2)/(1. + 2.e14/a16)
        par[20] = 9. + 100./a
        par[21] = .002*a3 + 3.e7/a6
        par[22] = 7.e-15*a4*asa
        par[23] = 9000./a4
        par[24] = .0011*asa/(1. + 3.e34/a32/a4)
        par[25] = 1.e-5*a2 + 2.e14/a16
        par[26] = 1.2e-11*a2/(1. + 1.5e19/a12)
        par[27] = .016*asa/(1. + 5.e16/a16)
        par[28] = .002*a4/(1. + 7.e7/(a - 6.83)**14)
        par[29] = 2.e6/a6 + 7.2/a**.11
        par[30] = 11.*a3/(1. + 7.e23/a16/a8)
        par[31] = 100./asa
        par[32] = (.1 + 4.4e-5*a2)/(1. + 5.e5/a4)
        par[33] = 3.5e-4*a2/(1. + 1.e8/a8)
        par[34] = 1.3 + 3.e5/a4
        par[35] = 500./(a2 + 50.) + 3
        par[36] = 1.e-9/a + s4a16*s4a16
        par[37] = .4*asa + 3.e-9*a6
        par[38] = .0005*a5
        par[39] = .002*a5
        par[40] = 10.
        par[41] = .05 + .005*a
        par[42] = 7.e-8/sa
        par[43] = .8*sa
        par[44] = .02*sa
        par[45] = 1.e8/a3
        par[46] = 3.e32/(a32 + 1.e32)
        par[47] = 24.
        par[48] = 20./sa
        par[49] = 7.e3*a/(sa + 1.)
        par[50] = 900.*sa/(1. + 500./a3)
    par[51] = 1.e15 + 2.e27/a4/(1. + 2.e-18*a16)
    return par


def _chips_tab_values(lp, Z, N, par):
    """
    CHIPS elastic parameters at ``lp = log(p/GeV)``.

    Port of ``G4ChipsProtonElasticXS::GetTabValues``.

    Returns
    -------
    values : ndarray
        ``[sigma, SS, S1, B1, S2, B2, S3, B3, S4, B4]`` with ``sigma`` the
        CHIPS elastic cross section [mb].
    """
    p = np.exp(lp)
    sp = np.sqrt(p)
    p2 = p*p
    p3 = p2*p
    p4 = p3*p
    if Z == 1 and N == 0:
        p2s = p2*sp
        dl2 = lp - par[8]
        SS = par[31]
        S1 = ((par[9] + par[10]*dl2*dl2)/(1. + par[11]/p4/p)
              + (par[12]/p2 + par[13]*p)/(p4 + par[14]*sp))
        B1 = par[15]*p**par[16]/(1. + par[17]/p3)
        S2 = par[18] + par[19]/(p4 + par[20]*p)
        B2 = par[21] + par[22]/(p4 + par[23]/sp)
        S3 = par[24] + par[25]/(p4*p4 + par[26]*p2 + par[27])
        B3 = par[28] + par[29]/(p4 + par[30])
        S4 = 0.
        B4 = 0.
        dl1 = lp - par[3]
        sigma = (par[0]/p2s/(1. + par[7]/p2s)
                 + (par[1] + par[2]*dl1*dl1 + par[4]/p)
                 / (1. + par[5]*lp)/(1. + par[6]/p4))
        return np.array([sigma, SS, S1, B1, S2, B2, S3, B3, S4, B4])

    p5 = p4*p
    p6 = p5*p
    p8 = p6*p2
    p10 = p8*p2
    p12 = p10*p2
    p16 = p8*p8
    dl = lp - 5.
    a = float(Z + N)
    if a < 6.5:
        pah = p**(a/2)
        pa = pah*pah
        pa2 = pa*pa
        S1 = (par[9]/(1. + par[10]*p4*pa) + par[11]/(p4 + par[12]*p4/pa2)
              + (par[13]*dl*dl + par[14])/(1. + par[15]/p2))
        B1 = (par[16] + par[17]*p2)/(p4 + par[18]/pah) + par[19]
        SS = par[20]/(1. + par[21]/p2) + par[22]/(p6/pa + par[23]/p16)
        S2 = par[24]/(pa/p2 + par[25]/p4) + par[26]
        B2 = par[27]*p**par[28] + par[29]/(p8 + par[30]/p16)
        S3 = par[31]/(pa*p + par[32]/pa) + par[33]
        B3 = par[34]/(p3 + par[35]/p6) + par[36]/(1. + par[37]/p2)
        S4 = p2*(pah*par[38]*np.exp(-pah*par[39])
                 + par[40]/(1. + par[41]*p**par[42]))
        B4 = par[43]*pa/p2/(1. + pa*par[44])
    else:
        S1 = (par[9]/(1. + par[10]/p4) + par[11]/(p4 + par[12]/p2)
              + par[13]/(p5 + par[14]/p16))
        B1 = ((par[15]/p8 + par[19])/(p + par[16]/p**par[20])
              + par[17]/(1. + par[18]/p4))
        SS = par[21]/(p4/p**par[23] + par[22]/p4)
        S2 = par[24]/p4/(p**par[25] + par[26]/p12) + par[27]
        B2 = par[28]/p**par[29] + par[30]/p**par[31]
        S3 = (par[32]/p**par[35]/(1. + par[36]/p12)
              + par[33]/(1. + par[34]/p6))
        B3 = par[37]/p8 + par[38]/p2 + par[39]/(1. + par[40]/p8)
        S4 = ((par[41]/p4 + par[46]/p)/(1. + par[42]/p10)
              + (par[43] + par[44]*dl*dl)/(1. + par[45]/p12))
        B4 = par[47]/(1. + par[48]/p) + par[49]*p4/(1. + par[50]*p5)
    sigma = ((par[0]*dl*dl + par[1])/(1. + par[2]/p + par[5]/p6)
             + par[3]/(p3 + par[4]/p3)
             + par[7]/(p4 + (par[8]/p)**par[6]))
    return np.array([sigma, SS, S1, B1, S2, B2, S3, B3, S4, B4])


class ChipsElasticDistribution:
    """
    Distribution of the momentum transfer in proton elastic scattering.

    Port of the CHIPS parametrisation of ``G4ChipsProtonElasticXS``
    (``GetPTables``, ``GetTabValues``, ``GetExchangeT`` and ``GetQ2max``),
    which ``G4ChipsElasticModel`` uses for protons in FTFP_BERT. The
    distribution is a mixture of three (proton target) or four (nuclear
    target) components, each sampled by inverting a generalised
    exponential; this class evaluates the resulting probability density and
    samples it, optionally restricted to a window of ``|t|``.

    As in Geant4, below ``p = e^8 GeV/c`` the parameters are linearly
    interpolated in ``log(p)`` on the 128-node table of the Geant4 class,
    and evaluated directly above.

    Parameters
    ----------
    Z, N : int
        Proton and neutron numbers of the target (``Z=1, N=0`` for a free
        proton).
    p : float
        Projectile momentum [GeV/c].

    Attributes
    ----------
    sigma_chips : float
        CHIPS integrated elastic cross section [mb]. Not used for the
        normalisation (as in Geant4, where the BGG cross section is used).
    t_max : float
        Maximum ``|t|`` [GeV^2].
    slope : float
        Slope ``B1`` of the main (forward) component [GeV^-2].
    """

    def __init__(self, Z, N, p):
        self.Z = int(Z)
        self.N = int(N)
        self.p = float(p)
        par = _chips_parameters(self.Z, self.N)
        lp = np.log(self.p)
        if _CHIPS_LP_MIN < lp <= _CHIPS_LP_MAX:
            shift = (lp - _CHIPS_LP_MIN)/_CHIPS_DLP
            low = min(max(int(shift), 0), _CHIPS_N_LAST - 1)
            shift -= low
            v_low = _chips_tab_values(_CHIPS_LP_MIN + low*_CHIPS_DLP,
                                      self.Z, self.N, par)
            v_high = _chips_tab_values(_CHIPS_LP_MIN + (low + 1)*_CHIPS_DLP,
                                       self.Z, self.N, par)
            values = v_low + shift*(v_high - v_low)
        else:
            values = _chips_tab_values(lp, self.Z, self.N, par)
        (sigma, self._SS, self._S1, self._B1, self._S2, self._B2, self._S3,
         self._B3, self._S4, self._B4) = values
        self.sigma_chips = max(float(sigma), 0.0)
        self.slope = float(self._B1)
        self.t_max = self._compute_t_max()
        self._build_components()

    def _compute_t_max(self):
        """Port of ``G4ChipsProtonElasticXS::GetQ2max`` [GeV^2]."""
        p2 = self.p*self.p
        if self.Z == 1 and self.N == 0:
            t_mid = np.sqrt(p2 + _MP*_MP)*_MP - _MP*_MP
            return float(2*t_mid)
        # Nuclear mass from the mass number (the exact isotope mass only
        # changes the far tail, well beyond any aperture)
        mt = (self.Z + self.N)*_AMU
        dmt = 2*mt
        mds = dmt*np.sqrt(p2 + _MP*_MP) + _MP*_MP + mt*mt
        return float(dmt*dmt*p2/mds)

    def _build_components(self):
        """
        Set up the mixture components of ``GetExchangeT``.

        Each component ``i`` has a weight ``I_i`` and a monotonic function
        ``y_i(t)``, with ``y_i`` exponentially distributed on
        ``[0, y_i(t_max)]``.
        """
        tm = self.t_max
        a = self.Z + self.N
        comps = []
        if self.Z == 1 and self.N == 0:
            # (kind, B, extra, weight)
            B1, B2, B3 = self._B1, self._B2, self._B3
            R1 = -np.expm1(-tm*B1)
            R2 = -np.expm1(-(tm*B2)**3)
            R3 = -np.expm1(-tm*B3)
            comps.append(('lin', B1, 0.0, R1*self._S1/B1))
            comps.append(('cube_scaled', B2, 0.0, R2*self._S2))
            comps.append(('lin', B3, 0.0, R3*self._S3))
        else:
            B1, SS = self._B1, self._SS
            R1 = -np.expm1(-tm*(B1 + tm*SS))
            comps.append(('quad', B1, SS, R1*self._S1))
            k2 = 5 if a > 6.5 else 3
            R2 = -np.expm1(-self._B2*tm**k2)
            comps.append((f'pow{k2}', self._B2, 0.0, R2*self._S2))
            if a > 6.5:
                R3 = -np.expm1(-self._B3*tm**7)
                comps.append(('pow7', self._B3, 0.0, R3*self._S3))
            else:
                R3 = -np.expm1(-self._B3*tm)
                comps.append(('lin', self._B3, 0.0, R3*self._S3))
            R4 = -np.expm1(-self._B4*tm)
            comps.append(('rev' if a < 6.5 else 'lin', self._B4, 0.0,
                          R4*self._S4))
        weights = np.array([max(cc[3], 0.0) for cc in comps])
        self._components = [cc[:3] for cc in comps]
        self._weights = weights/weights.sum()
        # Normalisation of each component on [0, t_max]: y is zero at one
        # end of the range (t=0, or t=t_max for the backward component)
        self._norms = np.array([
            -np.expm1(-max(self._y(ii, np.array(0.0)),
                           self._y(ii, np.array(tm))))
            for ii in range(len(self._components))])

    def _y(self, ii, t):
        """Exponent ``y_i(t)`` of component ``ii``."""
        kind, B, SS = self._components[ii]
        if kind == 'lin':
            return B*t
        if kind == 'quad':
            # Geant4 neglects the quadratic term when |2 SS| <= 1e-7
            return B*t + (SS*t*t if abs(2*SS) > 1e-7 else 0.0)
        if kind == 'cube_scaled':
            return (B*t)**3
        if kind == 'pow3':
            return B*t**3
        if kind == 'pow5':
            return B*t**5
        if kind == 'pow7':
            return B*t**7
        if kind == 'rev':
            return B*(self.t_max - t)
        raise ValueError(kind)

    def _dy_dt(self, ii, t):
        """Derivative ``|dy_i/dt|`` of component ``ii``."""
        kind, B, SS = self._components[ii]
        if kind in ('lin', 'rev'):
            return np.full_like(t, B, dtype=float)
        if kind == 'quad':
            return B + (2*SS*t if abs(2*SS) > 1e-7 else 0.0)
        if kind == 'cube_scaled':
            return 3*B**3*t*t
        if kind == 'pow3':
            return 3*B*t*t
        if kind == 'pow5':
            return 5*B*t**4
        if kind == 'pow7':
            return 7*B*t**6
        raise ValueError(kind)

    def _t_from_y(self, ii, y):
        """Inverse ``t_i(y)`` of component ``ii``."""
        kind, B, SS = self._components[ii]
        if kind == 'lin':
            return y/B
        if kind == 'quad':
            if abs(2*SS) > 1e-7:
                return (np.sqrt(B*B + 4*SS*y) - B)/(2*SS)
            return y/B
        if kind == 'cube_scaled':
            return np.cbrt(y)/B
        if kind == 'pow3':
            return np.cbrt(y/B)
        if kind == 'pow5':
            return (y/B)**0.2
        if kind == 'pow7':
            return (y/B)**(1/7)
        if kind == 'rev':
            return self.t_max - y/B
        raise ValueError(kind)

    def _component_mass(self, ii, t1, t2):
        """Probability of component ``ii`` in ``[t1, t2]``."""
        y1 = self._y(ii, np.asarray(t1, dtype=float))
        y2 = self._y(ii, np.asarray(t2, dtype=float))
        lo, hi = np.minimum(y1, y2), np.maximum(y1, y2)
        return (np.exp(-lo) - np.exp(-hi))/self._norms[ii]

    def pdf(self, t):
        """
        Probability density of ``|t|`` on ``[0, t_max]`` [GeV^-2].

        Parameters
        ----------
        t : ndarray
            Momentum transfer ``|t|`` [GeV^2].

        Returns
        -------
        pdf : ndarray
            Probability density, zero outside ``[0, t_max]``.
        """
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        inside = (t >= 0) & (t <= self.t_max)
        tt = t[inside]
        for ii in range(len(self._components)):
            out[inside] += (self._weights[ii]*np.exp(-self._y(ii, tt))
                            * self._dy_dt(ii, tt)/self._norms[ii])
        return out

    def probability(self, t1, t2):
        """Probability that ``|t|`` lies in ``[t1, t2]``."""
        t1 = max(float(t1), 0.0)
        t2 = min(float(t2), self.t_max)
        if t2 <= t1:
            return 0.0
        return float(sum(self._weights[ii]*self._component_mass(ii, t1, t2)
                         for ii in range(len(self._components))))

    def sample(self, n, rng, t_lim=None):
        """
        Sample ``|t|`` from the distribution, optionally within a window.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : numpy.random.Generator
            Random number generator.
        t_lim : tuple of float, optional
            Window ``(t1, t2)`` [GeV^2]; the distribution is truncated to it
            by inverting each component within the window.

        Returns
        -------
        t : ndarray
            Sampled ``|t|`` [GeV^2].
        """
        t1, t2 = (0.0, self.t_max) if t_lim is None else t_lim
        t1 = max(float(t1), 0.0)
        t2 = min(float(t2), self.t_max)
        n_comp = len(self._components)
        masses = np.array([self._weights[ii]
                           * self._component_mass(ii, t1, t2)
                           for ii in range(n_comp)])
        if masses.sum() <= 0:
            raise ValueError("The |t| window has zero probability.")
        i_comp = rng.choice(n_comp, size=n, p=masses/masses.sum())
        u = rng.random(n)
        t = np.empty(n)
        for ii in range(n_comp):
            mask = i_comp == ii
            if not mask.any():
                continue
            y1 = self._y(ii, np.array(t1))
            y2 = self._y(ii, np.array(t2))
            lo, hi = min(y1, y2), max(y1, y2)
            # y exponentially distributed within [lo, hi]
            e_lo, e_hi = np.exp(-lo), np.exp(-hi)
            y = -np.log(e_lo - u[mask]*(e_lo - e_hi))
            t[mask] = self._t_from_y(ii, y)
        return np.clip(t, t1, t2)


# ############################################################ #
# Single Coulomb scattering off the nucleus
# ############################################################ #
class WentzelCoulombCrossSection:
    """
    Single Coulomb scattering of a proton off a screened nucleus.

    Port of ``G4WentzelOKandVIxSection`` for a proton projectile, as used by
    ``G4eCoulombScatteringModel`` in the standard EM physics of FTFP_BERT:
    the Wentzel (screened Rutherford) majorant

    .. math::
        \\frac{d\\sigma}{dz} = \\frac{2 \\pi Z^2 (\\alpha \\hbar c)^2}
            {\\beta^2 p^2} \\frac{1}{(z + z_s)^2}, \\quad z = 1 - \\cos\\theta,

    with the screening parameter ``z_s`` of ``SetupTarget``, multiplied by
    the acceptance of ``SampleSingleScattering``: the exponential nuclear
    form factor ``(1 + A_F z)^{-2}`` (``R = 1.27 fm A^0.27``), the spin
    (Mott) correction and the nuclear recoil factor, capped at one. Only the
    nucleus is included here; the atomic electrons are a separate process
    (:class:`ProtonKnockOnCalculator`).

    Parameters
    ----------
    Z : int
        Atomic number of the target.
    p0c : float
        Proton momentum [eV].

    Attributes
    ----------
    screen_z : float
        Screening parameter ``z_s`` (in units of ``1 - cos(theta)``).
    form_factor_a : float
        Form-factor coefficient ``A_F``.
    """

    def __init__(self, Z, p0c):
        self.Z = _check_Z(Z)
        self.p0c = float(p0c)
        p, E = _proton_kinematics(self.p0c)
        self.p = float(p)
        mom2 = self.p*self.p
        invbeta2 = 1.0 + _MP*_MP/mom2
        self._invbeta2 = invbeta2
        # Thomas-Fermi screening radii (InitialiseA), in GeV^2
        a0 = _ME/0.88534
        afact = 0.5*ALPHA*ALPHA*a0*a0
        if self.Z == 1:
            self.screen_z = afact/mom2
            # 3.097e-6 MeV^-2
            form_factor = 3.097
        else:
            screen_r2 = afact*(1 + np.exp(-self.Z*self.Z*0.001))*self.Z**(2/3)
            self.screen_z = (min(self.Z*1.13,
                                 1.13 + 3.76*self.Z**2*invbeta2*ALPHA**2)
                             * screen_r2/mom2)
            a27 = proton_data.NIST_ATOMIC_MASS[self.Z]**0.27
            # 6.937e-6 MeV^-2 A^0.54
            form_factor = 6.937*a27*a27
        self.form_factor_a = form_factor*mom2
        target_mass = (_MP if self.Z == 1 else
                       proton_data.NIST_ATOMIC_MASS[self.Z]*_AMU)
        self._fact_b = 0.5/invbeta2
        self._fact_b1 = 0.5*np.pi*ALPHA
        self._fact_d = self.p/target_mass
        # 2 pi Z^2 (alpha hbar c)^2/(beta^2 p^2) [mb]
        self._kin_factor = (2*np.pi*self.Z**2*ALPHA**2*HBARC2_GEV2_MB
                            * invbeta2/mom2)

    def majorant_dxsec_dz(self, z):
        """Wentzel majorant ``dsigma/dz`` [mb]."""
        return self._kin_factor/(z + self.screen_z)**2

    def acceptance(self, z):
        """Form-factor, spin and recoil factor of ``SampleSingleScattering``,
        capped at one."""
        fm = 1.0/(1.0 + self.form_factor_a*z)**2
        grej = ((1.0 - z*self._fact_b
                 + self._fact_b1*self.Z*np.sqrt(z*self._fact_b)*(2.0 - z))
                * fm/(1.0 + z*self._fact_d))
        return np.minimum(grej, 1.0)

    def dxsec_dz(self, z):
        """Effective single-scattering ``dsigma/dz`` [mb]."""
        return self.majorant_dxsec_dz(z)*self.acceptance(z)

    def dxsec_dt(self, t):
        """Effective ``dsigma/d|t|`` [mb/GeV^2], with ``z = |t|/(2 p^2)``."""
        return self.dxsec_dz(0.5*t/self.p**2)/(2*self.p**2)

    def majorant_xsec(self, theta1, theta2):
        """
        Majorant cross section between two angles [mb], as
        ``ComputeNuclearCrossSection``.

        ``1 - cos(theta)`` is evaluated as ``2 sin^2(theta/2)``, which keeps
        full precision at the small angles where the difference of cosines
        used by Geant4 cancels.
        """
        z1 = 2.0*np.sin(0.5*theta1)**2
        z2 = 2.0*np.sin(0.5*theta2)**2
        return float(self._kin_factor*(z2 - z1)
                     / ((z1 + self.screen_z)*(z2 + self.screen_z)))


# ############################################################ #
# Calculators
# ############################################################ #
class _ProtonCalculatorBase:
    """
    Common interface of the proton-gas calculators.

    Parameters
    ----------
    Z : int
        Atomic number of the gas species.
    p0c : float
        Reference momentum [eV].
    cross_sections : ProtonNucleusCrossSections, optional
        Shared hadronic cross sections of this species, to avoid recomputing
        them for every process.
    """

    process = None

    def __init__(self, Z, p0c, cross_sections=None):
        self.Z = _check_Z(Z)
        self.p0c = float(p0c)
        if cross_sections is None:
            cross_sections = ProtonNucleusCrossSections(self.Z, self.p0c)
        self.cross_sections = cross_sections
        self.A = cross_sections.A
        self.N = cross_sections.N
        self.p, self.energy = (float(v) for v in _proton_kinematics(self.p0c))

    def __repr__(self):
        return (f"<{type(self).__name__} Z={self.Z}, p0c={self.p0c:.3e} eV, "
                f"xsec={self.compute_xsec()/MILLIBARN:.4g} mb>")

    def _particle_kinematics(self, delta):
        """Momentum and energy of each macro-particle [GeV]."""
        p = self.p*(1.0 + delta)
        return p, np.sqrt(p*p + _MP*_MP)


class ProtonAbsorptionCalculator(_ProtonCalculatorBase):
    """
    Inelastic proton-nucleus interaction in which the beam proton is lost.

    The cross section is the BGG inelastic cross section minus the
    quasi-elastic and target-diffractive cross sections, whose events keep
    the beam proton and are generated by :class:`ProtonQuasiElasticCalculator`
    and :class:`ProtonDiffractionCalculator`. No final state is generated:
    the proton is flagged as absorbed at the interaction point, so that the
    hadronic shower can be simulated externally.

    Parameters
    ----------
    Z : int
        Atomic number of the gas species.
    p0c : float
        Reference momentum [eV].
    sd_scale : float, optional
        Scale factor of the single-diffraction cross section, see
        :class:`ProtonDiffractionCalculator`.
    cross_sections : ProtonNucleusCrossSections, optional
        Shared cross sections of this species.
    """

    process = 'absorption'

    def __init__(self, Z, p0c, sd_scale=1.0, cross_sections=None):
        super().__init__(Z, p0c, cross_sections)
        self.sd_scale = float(sd_scale)
        sigma_sd = ProtonDiffractionCalculator(
            Z, p0c, sd_scale=sd_scale,
            cross_sections=self.cross_sections).compute_xsec()
        self._xsec = (self.cross_sections.inelastic
                      - self.cross_sections.quasi_elastic - sigma_sd)
        if self._xsec < 0:
            raise ValueError(
                f"The quasi-elastic plus single-diffraction cross sections "
                f"exceed the inelastic one for Z={self.Z}; reduce "
                f"`sd_scale`.")

    def compute_xsec(self):
        """Absorption cross section per atom [m^2]."""
        return self._xsec

    def sample_deflections(self, px, py, delta, rng):
        """
        Flag a batch of protons as absorbed; their momenta are unchanged.

        Parameters
        ----------
        px, py, delta : ndarray
            Normalised momenta before the interaction.
        rng : numpy.random.Generator
            Random number generator (unused).

        Returns
        -------
        sample : ProtonScatteringSample
        """
        n = px.size
        return ProtonScatteringSample._new(
            n, px.copy(), py.copy(), delta.copy(),
            absorbed=np.ones(n, dtype=bool))


class ProtonElasticCalculator(_ProtonCalculatorBase):
    """
    Elastic proton scattering off the nucleus: hadronic, Coulomb, and their
    interference.

    The differential cross section in ``|t|`` is

    .. math::
        \\frac{d\\sigma}{d|t|} = |F_C + F_N|^2
            = \\frac{d\\sigma_C}{d|t|} + \\frac{d\\sigma_N}{d|t|}
            - 2 \\sqrt{\\frac{d\\sigma_C}{d|t|}\\frac{d\\sigma_N}{d|t|}}
              \\frac{\\rho \\cos(\\alpha\\Phi) + \\sin(\\alpha\\Phi)}
                    {\\sqrt{1 + \\rho^2}},

    where

    * ``dsigma_N/d|t|`` is the coherent hadronic elastic cross section: the
      BGG elastic cross section times the CHIPS distribution of ``|t|``
      (:class:`ChipsElasticDistribution`), as ``G4ChipsElasticModel``;
    * ``dsigma_C/d|t|`` is single Coulomb scattering off the nucleus
      (:class:`WentzelCoulombCrossSection`), with ``z = |t|/(2 p^2)``;
    * ``F_N = (rho + i) |F_N|/sqrt(1 + rho^2)`` has the constant phase of
      the forward amplitude, with ``rho`` from :func:`proton_proton_rho`
      (``rho_pA`` is approximated by ``rho_pp``), and ``F_C`` the repulsive
      Coulomb amplitude with the West-Yennie phase
      ``alpha Phi = -Z alpha [gamma_E + log(B |t|/2)]``, ``B`` being the
      CHIPS forward slope.

    Events are generated within a window of ``|t|`` corresponding to the
    polar-angle window ``theta_lim`` (``|t| = 2 p^2 (1 - cos(theta))``). The
    proposal is a mixture of a log-uniform density in ``|t|`` (which
    over-samples the large Coulomb angles responsible for losses, as for the
    e± Coulomb model) and of the CHIPS distribution restricted to the
    window, with mixing fractions proportional to the Coulomb and hadronic
    cross sections in the window. The importance weights are bounded and
    restore the exact differential cross section above.

    The kinematics is exact: the proton loses the recoil energy
    ``|t|/(2 M)`` and its polar angle follows from ``|t|``.

    Parameters
    ----------
    Z : int
        Atomic number of the gas species.
    p0c : float
        Reference momentum [eV].
    theta_lim : tuple of float, optional
        Window of the generated polar angle [rad]. The hadronic part below
        the lower limit (negligible for any sensible window) and all the
        scattering above the upper limit are not generated.
    coulomb, nuclear, interference : bool, optional
        Include the Coulomb amplitude, the hadronic amplitude and their
        interference. The interference needs both amplitudes.
    cross_sections : ProtonNucleusCrossSections, optional
        Shared cross sections of this species.
    """

    process = 'elastic'

    def __init__(self, Z, p0c, theta_lim=(1e-7, 50e-3), coulomb=True,
                 nuclear=True, interference=True, cross_sections=None):
        super().__init__(Z, p0c, cross_sections)
        if not (coulomb or nuclear):
            raise ValueError("At least one of `coulomb` and `nuclear` must "
                             "be enabled.")
        self.theta_lim = (float(theta_lim[0]), float(theta_lim[1]))
        if not 0.0 <= self.theta_lim[0] < self.theta_lim[1] <= np.pi:
            raise ValueError(
                "`coulomb_theta` must be a pair (theta_min, theta_max) with "
                "0 <= theta_min < theta_max <= pi.")
        if coulomb and self.theta_lim[0] <= 0:
            raise ValueError("Coulomb scattering needs theta_min > 0.")
        self.coulomb = bool(coulomb)
        self.nuclear = bool(nuclear)
        self.interference = bool(interference and coulomb and nuclear)
        if not self.nuclear:
            self.process = 'coulomb'
        elif not self.coulomb:
            self.process = 'nuclear_elastic'

        self.target_mass = self.cross_sections.target_mass*1e-9
        self.chips = ChipsElasticDistribution(self.Z, self.N, self.p)
        self.coulomb_xs = WentzelCoulombCrossSection(self.Z, self.p0c)
        self.rho = proton_proton_rho(self.p)
        self.sigma_nuclear = self.cross_sections.elastic/MILLIBARN

        p2 = self.p*self.p
        z_lim = 2.0*np.sin(0.5*np.array(self.theta_lim))**2
        self.t_lim = (float(2*p2*z_lim[0]),
                      float(min(2*p2*z_lim[1], self.chips.t_max)))
        if self.t_lim[1] <= self.t_lim[0]:
            raise ValueError("The |t| window is empty.")

        # Cross sections of the proposal components in the window [mb]
        t1, t2 = self.t_lim
        self._sigma_c = (_integrate_log(self.coulomb_xs.dxsec_dt, t1, t2)
                         if self.coulomb else 0.0)
        self._sigma_n = (self.sigma_nuclear*self.chips.probability(t1, t2)
                         if self.nuclear else 0.0)
        sigma_i = (_integrate_log(self._interference, t1, t2,
                                  nodes_per_decade=2)
                   if self.interference else 0.0)
        self._sigma = self._sigma_c + self._sigma_n + sigma_i
        self._frac_c = self._sigma_c/(self._sigma_c + self._sigma_n)
        self._window_n = (self.chips.probability(t1, t2)
                          if self.nuclear else 0.0)

    def _interference(self, t):
        """Coulomb-nuclear interference term [mb/GeV^2]."""
        fc = self.coulomb_xs.dxsec_dt(t)
        fn = self.sigma_nuclear*self.chips.pdf(t)
        with np.errstate(divide='ignore'):
            a_phi = -self.Z*ALPHA*(np.euler_gamma
                                   + np.log(0.5*self.chips.slope*t))
        return (-2.0*np.sqrt(fc*fn)
                * (self.rho*np.cos(a_phi) + np.sin(a_phi))
                / np.sqrt(1.0 + self.rho**2))

    def dxsec_dt(self, t):
        """
        Differential cross section ``dsigma/d|t|`` [mb/GeV^2].

        Parameters
        ----------
        t : ndarray
            Momentum transfer ``|t|`` [GeV^2].

        Returns
        -------
        dxsec : ndarray
            Coulomb, hadronic and interference terms, as enabled.
        """
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        if self.coulomb:
            out += self.coulomb_xs.dxsec_dt(t)
        if self.nuclear:
            out += self.sigma_nuclear*self.chips.pdf(t)
        if self.interference:
            out += self._interference(t)
        return out

    def compute_xsec(self):
        """Elastic cross section per atom within the window [m^2]."""
        return self._sigma*MILLIBARN

    def _proposal_pdf(self, t):
        """Density of the proposal mixture [GeV^-2]."""
        t1, t2 = self.t_lim
        out = np.zeros_like(t)
        if self.coulomb:
            out += self._frac_c/(t*np.log(t2/t1))
        if self.nuclear:
            out += (1 - self._frac_c)*self.chips.pdf(t)/self._window_n
        return out

    def sample_t(self, n, rng):
        """
        Sample ``|t|`` with importance weights.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        t : ndarray
            Momentum transfer ``|t|`` [GeV^2].
        weight : ndarray
            Importance weights, normalised so that their expectation value
            is one.
        """
        t1, t2 = self.t_lim
        from_coulomb = rng.random(n) < self._frac_c
        t = np.empty(n)
        n_c = int(from_coulomb.sum())
        if n_c:
            t[from_coulomb] = t1*np.exp(rng.random(n_c)*np.log(t2/t1))
        if n - n_c:
            t[~from_coulomb] = self.chips.sample(n - n_c, rng,
                                                 t_lim=self.t_lim)
        weight = self.dxsec_dt(t)/(self._sigma*self._proposal_pdf(t))
        return t, weight

    def sample_deflections(self, px, py, delta, rng):
        """
        Sample elastic scattering events for a batch of protons.

        Parameters
        ----------
        px, py, delta : ndarray
            Normalised momenta before the interaction.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        sample : ProtonScatteringSample
        """
        n = px.size
        t, weight = self.sample_t(n, rng)
        p, E = self._particle_kinematics(delta)
        energy_loss = 0.5*t/self.target_mass
        z, p_out = _scattering_z(t, E, p, energy_loss)
        theta = 2.0*np.arcsin(np.sqrt(0.5*z))
        px_new, py_new, delta_new = _scatter_momenta(
            px, py, delta, theta, p_out, self.p0c, rng)
        coulomb_fraction = (self.coulomb_xs.dxsec_dt(t)/self.dxsec_dt(t)
                            if self.coulomb else np.zeros(n))
        return ProtonScatteringSample._new(
            n, px_new, py_new, delta_new, weight=weight, theta=theta, t=t,
            energy_loss=energy_loss*1e9, coulomb_fraction=coulomb_fraction)


class ProtonQuasiElasticCalculator(_ProtonCalculatorBase):
    """
    Quasi-elastic scattering off a single bound nucleon.

    The cross section is the Glauber-Gribov quasi-elastic cross section of
    :class:`ProtonNucleusCrossSections` (zero for hydrogen). The momentum
    transfer follows the CHIPS proton-proton elastic distribution (the same
    shape is used for protons and neutrons of the nucleus), and the
    kinematics is that of elastic scattering on a free nucleon at rest:
    Fermi motion and nuclear binding (below 3e-5 in momentum at 450 GeV) are
    neglected.

    Parameters
    ----------
    Z : int
        Atomic number of the gas species.
    p0c : float
        Reference momentum [eV].
    cross_sections : ProtonNucleusCrossSections, optional
        Shared cross sections of this species.
    """

    process = 'quasi_elastic'

    def __init__(self, Z, p0c, cross_sections=None):
        super().__init__(Z, p0c, cross_sections)
        self.chips = ChipsElasticDistribution(1, 0, self.p)

    def compute_xsec(self):
        """Quasi-elastic cross section per atom [m^2]."""
        return self.cross_sections.quasi_elastic

    def sample_deflections(self, px, py, delta, rng):
        """
        Sample quasi-elastic events for a batch of protons.

        Parameters
        ----------
        px, py, delta : ndarray
            Normalised momenta before the interaction.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        sample : ProtonScatteringSample
        """
        n = px.size
        t = self.chips.sample(n, rng)
        p, E = self._particle_kinematics(delta)
        energy_loss = 0.5*t/_MP
        z, p_out = _scattering_z(t, E, p, energy_loss)
        theta = 2.0*np.arcsin(np.sqrt(0.5*z))
        px_new, py_new, delta_new = _scatter_momenta(
            px, py, delta, theta, p_out, self.p0c, rng)
        return ProtonScatteringSample._new(
            n, px_new, py_new, delta_new, theta=theta, t=t,
            energy_loss=energy_loss*1e9)


class ProtonDiffractionCalculator(_ProtonCalculatorBase):
    """
    Single diffraction with dissociation of a target nucleon, ``p N -> p X``.

    The model is the single-diffractive channel of the Everest scattering
    routine of Xcoll (``nuclear_interaction.h`` and ``properties.h``, K2/
    SixTrack heritage [7]_):

    * cross section ``N_eff (4.3 + 0.3 log(s)) mb`` with ``s = 2 m_p p``
      [GeV^2] and ``N_eff = 2 pi 0.415 fm (3/(4 pi))^(1/3) A^(1/3)`` the
      number of nucleons participating in the interaction (``N_eff = 1``
      for hydrogen), times ``sd_scale``;
    * diffractive mass distributed as ``dN/dM^2 ~ 1/M^2`` between the pion
      production threshold ``(m_p + m_pi)^2`` and ``0.15 s``;
    * ``|t|`` exponential with the slope ``b(M^2)`` of Everest, derived from
      the proton-proton slope ``b_pp = 7.156 + 1.439 log(sqrt(s))``,
      above the kinematic minimum ``|t|_min(M^2)``.

    The kinematics of the scattered proton is exact: it loses the energy
    ``nu = (M^2 - m_p^2 + |t|)/(2 m_p)`` (so ``Delta p/p ~ -M^2/s``) and its
    polar angle follows from ``|t|`` and ``nu``.

    Parameters
    ----------
    Z : int
        Atomic number of the gas species.
    p0c : float
        Reference momentum [eV].
    sd_scale : float, optional
        Scale factor of the cross section. The Everest normalisation follows
        the two-arm single-diffractive data, so a value around 0.5 may be
        appropriate for the target dissociation alone.
    xi_max : float, optional
        Maximum ``M^2/s``. Default 0.15.
    cross_sections : ProtonNucleusCrossSections, optional
        Shared cross sections of this species.
    """

    process = 'diffractive'

    _M2_MIN = (0.93827208816 + 0.13957039)**2   # GeV^2, (m_p + m_pi+)^2

    def __init__(self, Z, p0c, sd_scale=1.0, xi_max=0.15,
                 cross_sections=None):
        super().__init__(Z, p0c, cross_sections)
        self.sd_scale = float(sd_scale)
        if self.sd_scale < 0:
            raise ValueError("`sd_scale` must be non-negative.")
        self.xi_max = float(xi_max)
        # Everest/K2 definitions (properties.h): s = 2 m_p p, in GeV^2
        self.s = 2*_MP*self.p
        self.b_pp = 7.156 + 1.439*np.log(np.sqrt(self.s))
        sigma_pp = 4.3 + 0.3*np.log(self.s)
        if self.Z == 1:
            self.n_eff = 1.0
        else:
            A = proton_data.NIST_ATOMIC_MASS[self.Z]
            self.n_eff = (2*np.pi*0.415*(3/4/np.pi)**(1/3)*A**(1/3))
        self._xsec = self.sd_scale*self.n_eff*sigma_pp*MILLIBARN
        self.m2_lim = (self._M2_MIN, self.xi_max*self.s)
        if self.m2_lim[1] <= self.m2_lim[0]:
            raise ValueError("The beam energy is too low for single "
                             "diffraction with this `xi_max`.")

    def compute_xsec(self):
        """Single-diffractive cross section per atom [m^2]."""
        return self._xsec

    def slope(self, m2):
        """Slope ``b(M^2)`` of the ``|t|`` distribution [GeV^-2], from
        Everest ``nuclear_interaction.h``."""
        m2 = np.asarray(m2, dtype=float)
        b = np.where(m2 < 2.0, 2*self.b_pp,
                     np.where(m2 <= 5.0, (106.0 - 17.0*m2)*self.b_pp/36.0,
                              7*self.b_pp/12.0))
        return b

    def sample_deflections(self, px, py, delta, rng):
        """
        Sample single-diffractive events for a batch of protons.

        Parameters
        ----------
        px, py, delta : ndarray
            Normalised momenta before the interaction.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        sample : ProtonScatteringSample
        """
        n = px.size
        m2_1, m2_2 = self.m2_lim
        m2 = m2_1*np.exp(rng.random(n)*np.log(m2_2/m2_1))
        p, E = self._particle_kinematics(delta)
        # Kinematic minimum of |t| (z >= 0): fixed-point iteration of
        # |t| = 2 D(nu(|t|))
        t_min = np.zeros(n)
        for _ in range(4):
            nu = (m2 - _MP*_MP + t_min)/(2*_MP)
            E_out = E - nu
            p_out = np.sqrt(E_out*E_out - _MP*_MP)
            t_min = 2*_MP*_MP*nu*nu/(E*E_out + p*p_out - _MP*_MP)
        t = t_min + rng.exponential(1.0, n)/self.slope(m2)
        nu = (m2 - _MP*_MP + t)/(2*_MP)
        z, p_out = _scattering_z(t, E, p, nu)
        theta = 2.0*np.arcsin(np.sqrt(0.5*z))
        px_new, py_new, delta_new = _scatter_momenta(
            px, py, delta, theta, p_out, self.p0c, rng)
        return ProtonScatteringSample._new(
            n, px_new, py_new, delta_new, theta=theta, t=t,
            energy_loss=nu*1e9, mass_x2=m2)


class ProtonKnockOnCalculator(_ProtonCalculatorBase):
    """
    Hard elastic scattering off atomic electrons (knock-on electrons).

    Port of ``G4BetheBlochModel`` for a proton: the differential cross
    section per atom in the electron kinetic energy ``T`` is

    .. math::
        \\frac{d\\sigma}{dT} = \\frac{2 \\pi r_e^2 m_e c^2 Z}{\\beta^2}
            \\left[\\frac{1}{T^2} - \\frac{\\beta^2}{T T_{max}}
            + \\frac{1}{2 E^2}\\right] g(T),

    where the first factor is ``ComputeCrossSectionPerElectron`` and ``g``
    the acceptance of ``SampleSecondaries``: the proton form factor
    ``(1 + 2 m_e T/(0.8426 GeV)^2)^{-2}`` with the magnetic-moment
    correction, which suppresses the hardest transfers at LHC energies.
    Atomic binding is neglected (``T`` is far above the ionisation
    potentials).

    Energies ``T`` from ``cut*E`` to ``T_max`` are generated, log-uniformly
    with importance weights, since the losses come from the rare hard
    transfers. The proton kinematics is exact: it loses ``T``, and its polar
    angle follows from ``|t| = 2 m_e T`` (at most ``m_e/m_p``).

    Parameters
    ----------
    Z : int
        Atomic number of the gas species (number of electrons per atom).
    p0c : float
        Reference momentum [eV].
    cut : float, optional
        Minimum generated energy transfer as a fraction of the total proton
        energy. Default 1e-6.
    cross_sections : ProtonNucleusCrossSections, optional
        Shared cross sections of this species.
    """

    process = 'knock_on'

    def __init__(self, Z, p0c, cut=1e-6, cross_sections=None):
        super().__init__(Z, p0c, cross_sections)
        self.cut = float(cut)
        E, p = self.energy, self.p
        self.beta2 = (p/E)**2
        ratio = _ME/_MP
        tau = (E - _MP)/_MP
        self.t_kin_max = (2.0*_ME*tau*(tau + 2.0)
                          / (1.0 + 2.0*(tau + 1.0)*ratio + ratio*ratio))
        self.energy_lim = (self.cut*E, self.t_kin_max)
        if not 0 < self.energy_lim[0] < self.energy_lim[1]:
            raise ValueError("`knock_on_cut` must be positive and below the "
                             "maximum energy transfer.")
        # 2 pi r_e^2 m_e c^2 [mb GeV]
        self._prefactor = (2*np.pi*CLASSICAL_ELECTRON_RADIUS**2/MILLIBARN
                           * _ME*self.Z/self.beta2)
        self._form_factor = 2.0*_ME/0.8426**2
        self._mag_moment2 = PROTON_MAGNETIC_MOMENT**2 - 1.0
        self._sigma = _integrate_log(self.dxsec_dT, *self.energy_lim,
                                     nodes_per_decade=2)

    def bare_xsec(self, T1, T2):
        """
        Cross section between ``T1`` and ``T2`` [mb] without the form-factor
        acceptance, as ``G4BetheBlochModel::ComputeCrossSectionPerAtom``.
        """
        E2 = self.energy**2
        cross = ((T2 - T1)/(T1*T2)
                 - self.beta2*np.log(T2/T1)/self.t_kin_max
                 + 0.5*(T2 - T1)/E2)
        return float(self._prefactor*cross)

    def acceptance(self, T):
        """Form-factor and magnetic-moment acceptance of
        ``SampleSecondaries``."""
        T = np.asarray(T, dtype=float)
        E2 = self.energy**2
        f1 = 0.5*T*T/E2
        f = 1.0 - self.beta2*T/self.t_kin_max + f1
        x = self._form_factor*T
        grej = 1.0/(1.0 + x)**2
        x2 = 0.5*_ME*T/_MP**2
        grej *= 1.0 + self._mag_moment2*(x2 - f1/f)/(1.0 + x2)
        return np.where(x > 1e-6, grej, 1.0)

    def dxsec_dT(self, T):
        """Differential cross section ``dsigma/dT`` [mb/GeV]."""
        T = np.asarray(T, dtype=float)
        E2 = self.energy**2
        bracket = 1.0/T**2 - self.beta2/(T*self.t_kin_max) + 0.5/E2
        return self._prefactor*bracket*self.acceptance(T)

    def compute_xsec(self):
        """Knock-on cross section per atom above the cut [m^2]."""
        return self._sigma*MILLIBARN

    def sample_deflections(self, px, py, delta, rng):
        """
        Sample knock-on events for a batch of protons.

        Parameters
        ----------
        px, py, delta : ndarray
            Normalised momenta before the interaction.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        sample : ProtonScatteringSample
        """
        n = px.size
        T1, T2 = self.energy_lim
        log_ratio = np.log(T2/T1)
        T = T1*np.exp(rng.random(n)*log_ratio)
        weight = self.dxsec_dT(T)*T*log_ratio/self._sigma
        p, E = self._particle_kinematics(delta)
        t = 2*_ME*T
        z, p_out = _scattering_z(t, E, p, T)
        theta = 2.0*np.arcsin(np.sqrt(0.5*z))
        px_new, py_new, delta_new = _scatter_momenta(
            px, py, delta, theta, p_out, self.p0c, rng)
        return ProtonScatteringSample._new(
            n, px_new, py_new, delta_new, weight=weight, theta=theta, t=t,
            energy_loss=T*1e9)
