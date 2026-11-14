# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""
Blackbody photon field and Compton scattering of beam particles on it.

This module provides the photon density of the thermal radiation that fills
an accelerator vacuum chamber, the Klein-Nishina cross section, the rate of
Compton scattering of a beam particle on the thermal photons (optionally
restricted to the events with an energy loss above a cut), and a Monte Carlo
sampler of the scattering events. It is used by
:class:`xcoll.ThermalPhotonGenerator`.

Physics summary
---------------
A particle of mass ``M``, charge ``q`` and velocity ``beta`` moves through an
isotropic blackbody photon gas at temperature ``T``. The number of scatterings
per unit time on the photons of energy ``k`` that travel at an angle ``psi``
to the particle is

    dW = c sigma(k*) (1 - beta cos(psi)) dn(k, psi)

where ``k* = gamma k (1 - beta cos(psi))`` is the photon energy in the rest
frame of the particle, ``sigma`` the Klein-Nishina cross section and
``(1 - beta cos(psi))`` the relative-velocity (flux) factor of the invariant
collision rate. Changing variables from ``(k, psi)`` to

    xi = k (1 - beta cos(psi)) / ((1 + beta) k_B T),    kappa = k / (k_B T)

the rest-frame photon energy is ``k* = xi eps_T M`` with
``eps_T = (1 + beta) gamma k_B T / M``, and the rate factorises as

    dW = c n_gamma (1 + beta)^2 / (4 beta zeta(3))
         * sigma(xi eps_T) xi dxi * dkappa / (exp(kappa) - 1),
         xi <= kappa <= Gamma^2 xi,    Gamma^2 = (1 + beta) / (1 - beta)

so that ``xi`` is distributed as
``xi [L(xi) - L(Gamma^2 xi)] sigma(xi eps_T)`` with
``L(x) = -ln(1 - exp(-x))``, and ``kappa`` given ``xi`` as
``1 / (exp(kappa) - 1)``. The scattering itself is sampled in the rest frame
from the Klein-Nishina angular distribution, and the particle momentum
follows from four-momentum conservation. Nothing is approximated: the
expressions hold for any ``beta``.

The energy loss is measured by the Lorentz invariant

    y = 1 - (P'.K) / (P.K) = 1 - k*'/k*

(``P``, ``P'`` the four-momenta of the particle before and after, ``K`` that
of the incident photon), which is the fractional energy that the photon loses
in the rest frame, and equals the relative energy loss of an ultra-relativistic
beam particle, ``(E - E')/E``, up to corrections of order ``1/gamma``.

Validity
--------
* The Klein-Nishina cross section holds for a point-like spin-1/2 particle
  (electrons, positrons, muons), with the classical radius scaled as
  ``q^2 m_e / M``. For hadrons and fully stripped ions only its low-energy
  (Thomson) limit is model independent, which is the only regime reached with
  thermal photons. Partially stripped ions (resonant scattering on the bound
  electrons) are not described.
* The photon field is an isotropic blackbody at a single temperature. See
  :class:`xcoll.ThermalPhotonGenerator` for the treatment of temperature
  profiles.
* Unpolarised photons; the polarisation of the beam is then irrelevant for
  the cross section.

References
----------
.. [1] H. Burkhardt, "Monte Carlo simulation of scattering of beam particles
   and thermal photons", CERN SL/Note 93-73 (OP) (1993).
.. [2] V. I. Telnov, "Scattering of electrons on thermal radiation photons in
   electron-positron storage rings", Nucl. Instrum. Meth. A **260**, 304
   (1987). https://doi.org/10.1016/0168-9002(87)90093-3
.. [3] A. Di Domenico, "Inverse Compton scattering of thermal radiation at
   LEP and LEP-200", Part. Accel. **39**, 137 (1992).
.. [4] O. Klein and Y. Nishina, "Ueber die Streuung von Strahlung durch freie
   Elektronen nach der neuen relativistischen Quantendynamik von Dirac",
   Z. Phys. **52**, 853 (1929). https://doi.org/10.1007/BF01366453
.. [5] L. D. Landau and E. M. Lifshitz, "The Classical Theory of Fields",
   section 12 (invariant collision rate).
.. [6] G. R. Blumenthal and R. J. Gould, "Bremsstrahlung, synchrotron
   radiation, and Compton scattering of high-energy electrons traversing
   dilute gases", Rev. Mod. Phys. **42**, 237 (1970).
   https://doi.org/10.1103/RevModPhys.42.237
.. [7] F. C. Jones, "Calculated spectrum of inverse-Compton-scattered
   photons", Phys. Rev. **167**, 1159 (1968).
   https://doi.org/10.1103/PhysRev.167.1159
.. [8] A. Natochii, "Thermal Compton scattering of electron beams on
   blackbody photons: a Monte Carlo event generator for multi-turn tracking
   at the Electron-Ion Collider", BNL-229489-2026-TECH / EIC-ADD-TN-159
   (2026).
"""

from dataclasses import dataclass

import numpy as np
from scipy import constants as sc
from scipy.special import spence

import xtrack as xt


# ############################################################ #
# Constants
# ############################################################ #
ELECTRON_MASS_EV = xt.ELECTRON_MASS_EV

BOLTZMANN_EV_K = sc.k / sc.e
HBAR_C_EV_M = sc.hbar * sc.c / sc.e
CLASSICAL_ELECTRON_RADIUS = sc.value('classical electron radius')

# Riemann zeta(3) and zeta(4)
ZETA_3 = 1.2020569031595942
ZETA_4 = np.pi**4 / 90.0

# Mean photon energy of the Planck number spectrum, in units of k_B T
MEAN_PHOTON_ENERGY_KT = 3.0 * ZETA_4 / ZETA_3

# Below this rest-frame photon energy (in units of the particle mass) the
# total Klein-Nishina cross section is evaluated with its power series
_KN_SERIES_LIMIT = 1e-2

# Value of xi_min above which the truncated exponential envelope is used to
# sample xi, instead of rejecting the draws of the untruncated distribution
_XI_SWITCH = 1.0

# The index of the series used to sample the Planck-like distributions is not
# searched beyond this value (reached with a probability of 2.5e-8)
_MAX_SERIES_INDEX = 4096

_MAX_REJECTION_ITERATIONS = 100000

# Gauss-Legendre nodes and weights on [0, 1]
_GL_ORDER = 48
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_ORDER)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W

# Panels of the integration over xi - xi_min: geometric below 1 (to resolve
# the logarithmic singularity of the integrand at xi = 0), then doubling
_XI_PANELS = np.concatenate((
    [0.0], np.geomspace(1e-12, 1.0, 25), 2.0**np.arange(1, 11)))


# ############################################################ #
# Scattering sample
# ############################################################ #
@dataclass
class ComptonSample:
    """
    Outcome of a batch of Compton scatterings on thermal photons.

    All momentum components are normalised to the reference momentum ``p0c``,
    consistently with the Xsuite convention (``px = Px/p0c``,
    ``delta = (p - p0)/p0``).

    Parameters
    ----------
    px, py : ndarray
        Normalised transverse momenta after the interaction.
    delta : ndarray
        Relative momentum deviation after the interaction.
    weight : ndarray
        Importance-sampling weight of each event. The sampler is unbiased, so
        every weight is one.
    energy_loss : ndarray
        Relative energy loss ``y = 1 - k*'/k*`` of each event (Lorentz
        invariant; equal to the relative energy loss of the beam particle up
        to corrections of order ``1/gamma``).
    photon_energy : ndarray
        Energy of the scattered photon in the laboratory frame [eV].
    theta : ndarray
        Angle of the scattered photon to the direction of the incoming beam
        particle, in the laboratory frame [rad].
    incident_photon_energy : ndarray
        Energy of the thermal photon in the laboratory frame [eV].
    incident_cos_psi : ndarray
        Cosine of the angle between the thermal photon and the beam particle
        in the laboratory frame (``-1`` for a head-on collision).
    cos_theta_rest : ndarray
        Cosine of the scattering angle of the photon in the rest frame of the
        beam particle.

    Attributes
    ----------
    px, py, delta, weight, energy_loss, photon_energy, theta,
    incident_photon_energy, incident_cos_psi, cos_theta_rest : ndarray
        See above.
    """
    px: np.ndarray
    py: np.ndarray
    delta: np.ndarray
    weight: np.ndarray
    energy_loss: np.ndarray
    photon_energy: np.ndarray
    theta: np.ndarray
    incident_photon_energy: np.ndarray
    incident_cos_psi: np.ndarray
    cos_theta_rest: np.ndarray


# ############################################################ #
# Helpers
# ############################################################ #
def _log1mexp(x):
    """
    Return ``L(x) = -ln(1 - exp(-x))`` for ``x >= 0``, without cancellation.

    ``L`` is positive, decreasing, diverges logarithmically at ``x = 0`` and
    behaves as ``exp(-x)`` at large ``x``. It is its own inverse:
    ``L(L(x)) = x``.

    Parameters
    ----------
    x : float or ndarray
        Non-negative argument.

    Returns
    -------
    L : float or ndarray
        ``-ln(1 - exp(-x))``; ``inf`` at ``x = 0``.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', over='ignore'):
        return np.where(x > np.log(2.0),
                        -np.log1p(-np.exp(-x)),
                        -np.log(-np.expm1(-x)))


def _log1mexp_times_exp(x):
    """
    Return ``L(x) exp(x)``, which tends to one at large ``x``.

    Parameters
    ----------
    x : float or ndarray
        Positive argument.

    Returns
    -------
    value : float or ndarray
        ``-ln(1 - z)/z`` with ``z = exp(-x)``.
    """
    z = np.exp(-np.asarray(x, dtype=float))
    small = z < 1e-8
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(small, 1.0 + 0.5*z, -np.log1p(-z)/np.where(small, 1.0, z))


def _momentum_frame(px, py, delta):
    """
    Build a right-handed orthonormal frame attached to the particle momentum.

    The returned basis ``(u_hat, v_hat, p_hat)`` has ``p_hat`` along the
    particle momentum. The azimuthal orientation of ``u_hat`` is arbitrary,
    which is irrelevant because the azimuths are sampled uniformly.

    Parameters
    ----------
    px, py : ndarray
        Normalised transverse momenta.
    delta : ndarray
        Relative momentum deviation.

    Returns
    -------
    u_hat, v_hat, p_hat : ndarray
        Basis vectors, each with shape ``(n, 3)``.
    """
    pz = np.sqrt(np.maximum((1.0 + delta)**2 - px**2 - py**2, 0.0))
    p_vec = np.stack((px, py, pz), axis=1)
    p_hat = p_vec / np.linalg.norm(p_vec, axis=1)[:, None]

    # u_hat = z_hat x p_hat, normalised; degenerate when p_hat is along z_hat
    u_hat = np.stack((-p_hat[:, 1], p_hat[:, 0], np.zeros_like(px)), axis=1)
    u_norm = np.linalg.norm(u_hat, axis=1)
    degenerate = u_norm < 1e-12
    u_hat[degenerate] = (1.0, 0.0, 0.0)
    u_norm[degenerate] = 1.0
    u_hat /= u_norm[:, None]

    v_hat = np.cross(p_hat, u_hat)

    return u_hat, v_hat, p_hat


def _uniform_open(rng, n):
    """
    Uniform random numbers in ``(0, 1]``, safe as arguments of a logarithm.
    """
    return 1.0 - rng.random(n)


def _sample_series_index(n, rng):
    """
    Draw integers ``m >= 1`` with probability ``1/(m^3 zeta(3))``.

    Parameters
    ----------
    n : int
        Number of integers to draw.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    m : ndarray
        Drawn integers, as floats, capped at ``_MAX_SERIES_INDEX``.
    """
    target = rng.random(n) * ZETA_3
    m = np.ones(n)
    cumulative = np.ones(n)
    pending = target > cumulative
    while pending.any():
        m[pending] += 1.0
        cumulative[pending] += 1.0 / m[pending]**3
        pending = (target > cumulative) & (m < _MAX_SERIES_INDEX)
    return m


def _sample_thomson_cos_theta(u):
    """
    Invert the cumulative distribution of the Thomson angular distribution.

    The density ``3/8 (1 + c^2)`` on ``[-1, 1]`` has the cumulative
    distribution ``1/2 + (3 c + c^3)/8``. Setting it equal to ``u`` gives the
    depressed cubic ``c^3 + 3 c - 2 h = 0`` with ``h = 4 u - 2``, whose only
    real root is ``c = w - 1/w`` with ``w^3 = h + sqrt(h^2 + 1)`` (Cardano).

    Parameters
    ----------
    u : ndarray
        Uniform random numbers in ``[0, 1]``.

    Returns
    -------
    cos_theta : ndarray
        Cosines distributed as ``1 + cos^2(theta)``.
    """
    h = 4.0 * np.asarray(u, dtype=float) - 2.0
    root = np.sqrt(h * h + 1.0)
    # h + root, written without cancellation for negative h
    w = np.cbrt(np.where(h >= 0.0, h + root, 1.0 / (root - h)))
    return np.clip(w - 1.0 / w, -1.0, 1.0)


# ############################################################ #
# Photon field
# ############################################################ #
def photon_density(temperature):
    """
    Number density of the photons of a blackbody radiation field.

    ``n_gamma = (2 zeta(3) / pi^2) (k_B T / (hbar c))^3``, i.e.
    ``2.029e7 T^3`` photons per m^3 with ``T`` in kelvin.

    Parameters
    ----------
    temperature : float or ndarray
        Temperature [K].

    Returns
    -------
    density : float or ndarray
        Photon density [1/m^3].
    """
    kT = BOLTZMANN_EV_K * np.asarray(temperature, dtype=float)
    return 2.0 * ZETA_3 / np.pi**2 * (kT / HBAR_C_EV_M)**3


def mean_photon_energy(temperature):
    """
    Mean energy of the photons of a blackbody radiation field.

    ``<k> = (pi^4 / (30 zeta(3))) k_B T = 2.701 k_B T``.

    Parameters
    ----------
    temperature : float or ndarray
        Temperature [K].

    Returns
    -------
    energy : float or ndarray
        Mean photon energy [eV].
    """
    return (MEAN_PHOTON_ENERGY_KT * BOLTZMANN_EV_K
            * np.asarray(temperature, dtype=float))


def planck_spectrum(photon_energy, temperature):
    """
    Planck number spectrum of a blackbody radiation field.

    ``dn/dk = k^2 / (pi^2 (hbar c)^3 (exp(k / k_B T) - 1))``.

    Parameters
    ----------
    photon_energy : float or ndarray
        Photon energy [eV].
    temperature : float
        Temperature [K].

    Returns
    -------
    spectrum : float or ndarray
        Number of photons per unit volume and photon energy [1/(m^3 eV)].
    """
    k = np.asarray(photon_energy, dtype=float)
    kT = BOLTZMANN_EV_K * float(temperature)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        out = k**2 / (np.pi**2 * HBAR_C_EV_M**3 * np.expm1(k / kT))
    return np.where(k > 0, out, 0.0)


# ############################################################ #
# Cross sections
# ############################################################ #
def thomson_xsec(mass0=ELECTRON_MASS_EV, q0=1.0):
    """
    Thomson cross section of a point-like charged particle.

    ``sigma_T = (8 pi / 3) r_0^2`` with the classical radius of the particle
    ``r_0 = q0^2 r_e m_e / mass0``. For an electron it is 0.6652 barn.

    Parameters
    ----------
    mass0 : float, optional
        Rest mass of the particle [eV]. Default: the electron mass.
    q0 : float, optional
        Charge of the particle [e]. Only its square matters.

    Returns
    -------
    xsec : float
        Thomson cross section [m^2].
    """
    r0 = float(q0)**2 * CLASSICAL_ELECTRON_RADIUS * ELECTRON_MASS_EV / float(mass0)
    return 8.0 * np.pi / 3.0 * r0**2


def klein_nishina_weight(eps, cos_theta):
    """
    Ratio of the Klein-Nishina to the Thomson differential cross section.

    With ``rho = k*'/k* = 1 / (1 + eps (1 - cos(theta)))`` the ratio of the
    scattered to the incident photon energy in the rest frame,

        (dsigma_KN/dOmega) / (dsigma_T/dOmega)
            = rho^2 (rho + 1/rho - 1 + cos^2(theta)) / (1 + cos^2(theta))

    which never exceeds one. It is the weight used by Burkhardt [1] to turn
    Thomson-distributed angles into Klein-Nishina ones by rejection.

    Parameters
    ----------
    eps : float or ndarray
        Incident photon energy in the rest frame of the particle, in units
        of the particle rest energy.
    cos_theta : float or ndarray
        Cosine of the scattering angle of the photon in the rest frame.

    Returns
    -------
    weight : float or ndarray
        Ratio of the differential cross sections, in ``(0, 1]``.
    """
    eps = np.asarray(eps, dtype=float)
    c = np.asarray(cos_theta, dtype=float)
    rho = 1.0 / (1.0 + eps * (1.0 - c))
    return rho**2 * (rho + 1.0/rho - 1.0 + c**2) / (1.0 + c**2)


def klein_nishina_ratio(eps):
    """
    Total Klein-Nishina cross section in units of the Thomson cross section.

    With ``x = 2 eps``,

        sigma_KN / sigma_T = (3 / (4 x)) [ (1 - 4/x - 8/x^2) ln(1 + x)
                                          + 1/2 + 8/x - 1/(2 (1 + x)^2) ]

    (the form of Telnov [2], eq. 2), evaluated with its power series
    ``1 - 2 eps + 5.2 eps^2 - 13.3 eps^3 + ...`` at small ``eps``, where the
    closed form suffers from cancellations.

    Parameters
    ----------
    eps : float or ndarray
        Incident photon energy in the rest frame of the particle, in units
        of the particle rest energy.

    Returns
    -------
    ratio : float or ndarray
        ``sigma_KN / sigma_T``, in ``(0, 1]``.
    """
    eps = np.asarray(eps, dtype=float)
    small = eps < _KN_SERIES_LIMIT
    x = 2.0 * np.where(small, 1.0, eps)
    closed = 0.75 / x * ((1.0 - 4.0/x - 8.0/x**2) * np.log1p(x)
                         + 0.5 + 8.0/x - 0.5/(1.0 + x)**2)
    e = np.where(small, eps, 0.0)
    series = (1.0 + e*(-2.0 + e*(26.0/5.0 + e*(-133.0/10.0 + e*(1144.0/35.0
              + e*(-544.0/7.0 + e*(3784.0/21.0 + e*(-6176.0/15.0))))))))
    return np.where(small, series, closed)


def klein_nishina_partial_ratio(eps, energy_loss_cut=0.0):
    """
    Klein-Nishina cross section for an energy loss above a cut, in units of
    the Thomson cross section.

    The events with ``y = 1 - k*'/k* >= energy_loss_cut`` are those with
    ``eps (1 - cos(theta)) >= a``, ``a = cut/(1 - cut)``. In the variable
    ``s = ln(1 + eps (1 - cos(theta)))``,

        sigma(y >= cut) / sigma_T
            = (3 / (8 eps)) int_{ln(1 + a)}^{ln(1 + 2 eps)}
              [rho^2 + 1 - rho + rho cos^2(theta)] ds,    rho = exp(-s)

    whose integrand is smooth for any ``eps``; the integral is evaluated with
    a Gauss-Legendre rule. It vanishes for ``eps <= a/2``, where even
    backscattering does not reach the cut.

    Parameters
    ----------
    eps : float or ndarray
        Incident photon energy in the rest frame of the particle, in units
        of the particle rest energy.
    energy_loss_cut : float, optional
        Lowest relative energy loss ``y`` that is counted, in ``[0, 1)``.
        Default ``0``: the total cross section.

    Returns
    -------
    ratio : float or ndarray
        ``sigma(y >= cut) / sigma_T``.
    """
    eps = np.asarray(eps, dtype=float)
    shape = eps.shape
    eps = np.atleast_1d(eps)
    a = energy_loss_cut / (1.0 - energy_loss_cut)

    open_ = eps > 0.5 * a
    e = np.where(open_, eps, 1.0)
    s_lo = np.log1p(a)
    s_hi = np.where(open_, np.log1p(2.0 * e), s_lo)
    s = s_lo + (s_hi - s_lo)[..., None] * _GL_X
    rho = np.exp(-s)
    cos_theta = 1.0 - np.expm1(s) / e[..., None]
    integrand = rho**2 + 1.0 - rho + rho * cos_theta**2
    integral = (s_hi - s_lo) * np.sum(_GL_W * integrand, axis=-1)
    out = np.where(open_, 0.375 / e * integral, 0.0)
    return out.reshape(shape)


def _energy_loss_dxsec_ratio(eps, y):
    """
    Differential Klein-Nishina cross section in the relative energy loss, in
    units of the Thomson cross section.

    ``(1/sigma_T) dsigma/dy = (3 / (8 eps)) [1/(1 - y) + 1 - y - t (2 - t)]``
    with ``t = 1 - cos(theta) = y / ((1 - y) eps)``, for ``t <= 2``. It is
    Telnov's eq. 4 [2] written with ``x = 2 eps``.

    Parameters
    ----------
    eps : ndarray
        Incident photon energy in the rest frame, in units of the particle
        rest energy.
    y : ndarray
        Relative energy loss.

    Returns
    -------
    ratio : ndarray
        ``(1/sigma_T) dsigma/dy``; zero where ``y`` cannot be reached.
    """
    eps = np.asarray(eps, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = y / ((1.0 - y) * eps)
        out = 0.375 / eps * (1.0/(1.0 - y) + 1.0 - y - t*(2.0 - t))
    return np.where((t <= 2.0) & (eps > 0), out, 0.0)


# ############################################################ #
# Distribution of the rest-frame photon energy
# ############################################################ #
def _debye_like(xi):
    """
    Return ``D(xi) = int_xi^inf kappa / (exp(kappa) - 1) dkappa``.

    ``D(xi) = xi L(xi) + Li_2(exp(-xi))``, with ``D(0) = pi^2 / 6``.
    """
    xi = np.asarray(xi, dtype=float)
    z = np.exp(-xi)
    with np.errstate(invalid='ignore'):
        first = np.where(xi > 0, xi * _log1mexp(xi), 0.0)
    return first + spence(1.0 - z)


def _xi_density(xi, beta, flux_weighting=True):
    """
    Distribution of the scaled rest-frame photon energy ``xi`` of the
    scatterings in the Thomson limit, normalised to one.

    With the flux factor it is
    ``(1 + beta)^2 / (4 beta zeta(3)) xi [L(xi) - L(Gamma^2 xi)]``; without
    it (photon directions drawn isotropically, as described in [1]) it is
    ``(1 + beta) / (4 beta zeta(3)) [D(xi) - D(Gamma^2 xi)]``. The actual
    distribution is this density times ``sigma_KN(xi eps_T)/sigma_T``.

    Parameters
    ----------
    xi : ndarray
        Scaled rest-frame photon energy.
    beta : float
        Velocity of the particle, in units of the speed of light.
    flux_weighting : bool, optional
        Whether the flux factor ``1 - beta cos(psi)`` is included.

    Returns
    -------
    density : ndarray
        Density of ``xi``.
    """
    xi = np.asarray(xi, dtype=float)
    gamma2_ratio = (1.0 + beta) / (1.0 - beta)
    with np.errstate(over='ignore', invalid='ignore'):
        xi_hi = gamma2_ratio * xi
        if flux_weighting:
            out = ((1.0 + beta)**2 / (4.0 * beta * ZETA_3)
                   * np.where(xi > 0,
                              xi * (_log1mexp(xi) - _log1mexp(xi_hi)), 0.0))
        else:
            out = ((1.0 + beta) / (4.0 * beta * ZETA_3)
                   * (_debye_like(xi) - _debye_like(xi_hi)))
    return out


# ############################################################ #
# Compton scattering on thermal photons
# ############################################################ #
class ThermalComptonCalculator:
    """
    Compton scattering of a beam particle on blackbody photons.

    The calculator gives the scattering rate of a particle of momentum
    ``p0c`` in a blackbody photon gas of temperature ``temperature``,
    restricted to the events with a relative energy loss above
    ``energy_loss_cut``, the spectrum of the energy loss, and Monte Carlo
    samples of the scattering events. The rate is a one-dimensional integral
    (see the module docstring), evaluated with a Gauss-Legendre rule; no
    Monte Carlo is involved in it.

    Only events above ``energy_loss_cut`` are generated; ``compute_xsec``
    returns the cross section restricted to the same range. Choosing
    ``energy_loss_cut`` just below the momentum acceptance of the machine
    therefore concentrates the Monte Carlo statistics on the events that can
    actually lead to a loss.

    Parameters
    ----------
    p0c : float
        Momentum of the beam particle times the speed of light [eV].
    temperature : float
        Temperature of the photon gas [K].
    mass0 : float, optional
        Rest mass of the beam particle [eV]. Default: the electron mass.
    q0 : float, optional
        Charge of the beam particle [e]. Only its square matters.
    energy_loss_cut : float, optional
        Lowest relative energy loss ``y = 1 - k*'/k*`` that is generated, in
        ``[0, 1)``. Default ``0``: all the scatterings.
    flux_weighting : bool, optional
        Whether the relative-velocity factor ``1 - beta cos(psi)`` weights
        the incident photons. ``True`` (default) is the physical rate.
        ``False`` draws the photon direction isotropically, as described in
        Burkhardt's note [1]: it is not physical and is provided only to
        reproduce the tables published there.

    Attributes
    ----------
    p0c, temperature, mass0, q0, energy_loss_cut, flux_weighting
        See above.
    energy0 : float
        Total energy of the beam particle [eV].
    gamma0, beta0 : float
        Lorentz factor and velocity of the beam particle.
    kT : float
        Thermal energy ``k_B T`` [eV].
    eps_T : float
        ``(1 + beta0) gamma0 k_B T / mass0``: energy of a head-on photon of
        laboratory energy ``k_B T`` in the rest frame of the particle, in
        units of the particle rest energy. Its smallness measures how close
        the scattering is to the Thomson limit.
    xi_min : float
        Smallest scaled rest-frame photon energy that can cause an energy
        loss above the cut, ``cut / ((1 - cut) 2 eps_T)``.
    photon_density : float
        Density of the photon gas [1/m^3].
    thomson_xsec : float
        Thomson cross section of the beam particle [m^2].

    Notes
    -----
    In the Thomson limit (``eps_T << 1``) and for ultra-relativistic
    particles the fraction of the scatterings above the cut depends on the
    parameters only through ``xi_min``, i.e. through
    ``cut / (gamma0 T)``: this is the scaling law of Telnov [2], eq. 15.

    References
    ----------
    See the module docstring.
    """

    def __init__(self, p0c, temperature, *, mass0=ELECTRON_MASS_EV, q0=-1.0,
                 energy_loss_cut=0.0, flux_weighting=True):
        self.p0c = float(p0c)
        self.temperature = float(temperature)
        self.mass0 = float(mass0)
        self.q0 = float(q0)
        self.energy_loss_cut = float(energy_loss_cut)
        self.flux_weighting = bool(flux_weighting)

        if not self.p0c > 0.0:
            raise ValueError("`p0c` must be positive.")
        if not self.temperature > 0.0:
            raise ValueError("`temperature` must be positive.")
        if not self.mass0 > 0.0:
            raise ValueError("`mass0` must be positive.")
        if self.q0 == 0.0:
            raise ValueError(
                "`q0` must be non-zero: neutral particles do not scatter "
                "photons in this model.")
        if not 0.0 <= self.energy_loss_cut < 1.0:
            raise ValueError("`energy_loss_cut` must be in [0, 1).")

        self.energy0 = np.sqrt(self.p0c**2 + self.mass0**2)
        self.gamma0 = self.energy0 / self.mass0
        self.beta0 = self.p0c / self.energy0
        self.kT = BOLTZMANN_EV_K * self.temperature
        self.eps_T = (1.0 + self.beta0) * self.gamma0 * self.kT / self.mass0
        self.xi_min = (self.energy_loss_cut / (1.0 - self.energy_loss_cut)
                       / (2.0 * self.eps_T))
        self.photon_density = float(photon_density(self.temperature))
        self.thomson_xsec = thomson_xsec(self.mass0, self.q0)

    def __repr__(self):
        return (f"<ThermalComptonCalculator p0c={self.p0c:.3e} eV, "
                f"T={self.temperature:.2f} K, "
                f"energy_loss_cut={self.energy_loss_cut:.3e}>")

    def _xi_nodes(self, xi_min):
        """
        Nodes and weights of the integration over ``xi`` from ``xi_min``.
        """
        edges = xi_min + _XI_PANELS
        width = np.diff(edges)
        xi = (edges[:-1, None] + width[:, None] * _GL_X).ravel()
        weight = (width[:, None] * _GL_W).ravel()
        return xi, weight

    def compute_fraction(self):
        """
        Compute the scattering rate above the energy-loss cut, in units of
        the Thomson rate ``c n_gamma sigma_T``.

        ``F = int_{xi_min}^inf w(xi) sigma(xi eps_T; y >= cut)/sigma_T dxi``,
        with ``w`` the distribution of the scaled rest-frame photon energy
        in the Thomson limit. Without cut, ``F`` is the mean ratio of the
        Compton to the Thomson cross section (``w_mean`` in [1]).

        Parameters
        ----------
        None

        Returns
        -------
        fraction : float
            Dimensionless rate, in ``[0, 1]``.
        """
        xi, weight = self._xi_nodes(self.xi_min)
        integrand = (_xi_density(xi, self.beta0, self.flux_weighting)
                     * klein_nishina_partial_ratio(xi * self.eps_T,
                                                   self.energy_loss_cut))
        return float(np.sum(weight * integrand))

    def compute_xsec(self):
        """
        Compute the effective cross section for scattering on the thermal
        photons with an energy loss above the cut.

        It is defined by the rate per particle
        ``W = c n_gamma sigma_eff``, i.e. it is the Klein-Nishina cross
        section averaged over the photon gas with the flux factor,
        ``sigma_eff = sigma_T F`` (:meth:`compute_fraction`). The scattering
        probability per unit path length is ``n_gamma sigma_eff / beta0``.

        Parameters
        ----------
        None

        Returns
        -------
        xsec : float
            Effective cross section [m^2].
        """
        return self.thomson_xsec * self.compute_fraction()

    def compute_rate(self):
        """
        Compute the scattering rate of one beam particle above the
        energy-loss cut.

        ``W = c n_gamma sigma_eff``. Its inverse is the beam lifetime of a
        ring at uniform temperature in which every scattering above the cut,
        and none below, leads to a loss.

        Parameters
        ----------
        None

        Returns
        -------
        rate : float
            Scattering rate per particle [1/s].
        """
        return sc.c * self.photon_density * self.compute_xsec()

    def compute_spectrum(self, energy_loss):
        """
        Compute the spectrum of the relative energy loss.

        ``(1/sigma_T) dsigma_eff/dy``: its integral above ``y`` is the
        fraction returned by :meth:`compute_fraction` for a cut at ``y``.
        The cut of the calculator is not applied.

        Parameters
        ----------
        energy_loss : float or ndarray
            Relative energy loss ``y``, in ``(0, 1)``.

        Returns
        -------
        spectrum : float or ndarray
            Differential scattering rate in ``y``, in units of the Thomson
            rate ``c n_gamma sigma_T``.
        """
        y = np.asarray(energy_loss, dtype=float)
        shape = y.shape
        y = np.atleast_1d(y)
        out = np.zeros(y.size)
        for ii, yy in enumerate(y.ravel()):
            if not 0.0 < yy < 1.0:
                continue
            xi, weight = self._xi_nodes(yy / (1.0 - yy) / (2.0 * self.eps_T))
            out[ii] = np.sum(
                weight * _xi_density(xi, self.beta0, self.flux_weighting)
                * _energy_loss_dxsec_ratio(xi * self.eps_T, yy))
        return out.reshape(shape) if shape else float(out[0])

    def sample_deflections(self, px, py, delta, rng):
        """
        Sample Compton scatterings on thermal photons for a batch of beam
        particles.

        Each particle scatters once, with a relative energy loss above the
        cut. The thermal photon and the scattering angles are drawn for the
        actual energy of each particle, and the outgoing momentum follows
        from exact four-momentum conservation, so the particle acquires both
        an energy loss and a (small) angular deflection.

        This is the numpy implementation of the algorithm of the
        ``ThermalPhotonMarker`` C kernel, used to validate it.

        Parameters
        ----------
        px, py : ndarray
            Normalised transverse momenta before the interaction.
        delta : ndarray
            Relative momentum deviation before the interaction.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        sample : ComptonSample
            Outgoing momenta, unit weights and the record of the events.
        """
        px = np.atleast_1d(np.asarray(px, dtype=float))
        py = np.atleast_1d(np.asarray(py, dtype=float))
        delta = np.atleast_1d(np.asarray(delta, dtype=float))
        n = px.size

        pc = self.p0c * (1.0 + delta)
        energy = np.sqrt(pc**2 + self.mass0**2)
        gamma = energy / self.mass0
        beta = pc / energy

        if self.flux_weighting:
            xi, kappa, cos_theta = self._sample_photons(gamma, beta, rng)
        else:
            xi, kappa, cos_theta = self._sample_photons_unweighted(
                gamma, beta, rng)
        phi_k = 2.0 * np.pi * rng.random(n)
        phi = 2.0 * np.pi * rng.random(n)

        # Incident photon in the laboratory frame
        k = kappa * self.kT
        q = np.clip((1.0 + beta) * xi / kappa, 0.0, None)
        cos_psi = np.clip((1.0 - q) / beta, -1.0, 1.0)
        sin_psi = np.sqrt(1.0 - cos_psi**2)

        # Incident photon in the rest frame: energy and aberrated direction
        k_rest = gamma * k * q
        eps = k_rest / self.mass0
        cos_zeta = np.clip((cos_psi - beta) / q, -1.0, 1.0)
        sin_zeta = sin_psi / (gamma * q)

        # Scattering in the rest frame
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        rho = 1.0 / (1.0 + eps * (1.0 - cos_theta))
        k_rest_out = rho * k_rest
        # Components of the scattered direction along the incident photon
        # (n), in the plane of n and the boost axis (e1) and normal to it (e2)
        n_z = cos_theta * cos_zeta - sin_theta * np.cos(phi) * sin_zeta
        n_1 = cos_theta * sin_zeta + sin_theta * np.cos(phi) * cos_zeta
        n_2 = sin_theta * np.sin(phi)

        # Scattered photon in the laboratory frame
        k_out = gamma * k_rest_out * (1.0 + beta * n_z)
        k_out_z = gamma * k_rest_out * (n_z + beta)
        k_out_1 = k_rest_out * n_1
        k_out_2 = k_rest_out * n_2

        # Momentum transferred to the particle, in the frame of its momentum
        dp_1 = k * sin_psi - k_out_1
        dp_2 = -k_out_2
        dp_x = dp_1 * np.cos(phi_k) - dp_2 * np.sin(phi_k)
        dp_y = dp_1 * np.sin(phi_k) + dp_2 * np.cos(phi_k)
        dp_z = k * cos_psi - k_out_z

        # Energy conservation, and the momentum modulus without cancellation
        energy_out = energy + k - k_out
        pc_out = np.sqrt(energy_out**2 - self.mass0**2)
        dpc = (energy_out - energy) * (energy_out + energy) / (pc_out + pc)

        u_hat, v_hat, p_hat = _momentum_frame(px, py, delta)
        dp_vec = (dp_x[:, None] * u_hat + dp_y[:, None] * v_hat
                  + dp_z[:, None] * p_hat) / self.p0c

        cos_chi = np.clip(k_out_z / k_out, -1.0, 1.0)
        sin_chi = np.hypot(k_out_1, k_out_2) / k_out

        return ComptonSample(
            px=px + dp_vec[:, 0],
            py=py + dp_vec[:, 1],
            delta=delta + dpc / self.p0c,
            weight=np.ones(n),
            energy_loss=1.0 - rho,
            photon_energy=k_out,
            theta=np.arctan2(sin_chi, cos_chi),
            incident_photon_energy=k,
            incident_cos_psi=cos_psi,
            cos_theta_rest=cos_theta,
        )

    def _sample_photons(self, gamma, beta, rng):
        """
        Sample the thermal photon and the scattering angle of each event.

        The scaled rest-frame photon energy ``xi`` and the cosine of the
        rest-frame scattering angle are drawn together by rejection: ``xi``
        from its Thomson-limit distribution ``xi L(xi)`` above ``xi_min``
        and the angle from the Thomson distribution, accepted with the
        probability ``[1 - L(Gamma^2 xi)/L(xi)]`` (exact kinematic range of
        the photon energy) times the Klein-Nishina weight, if the energy loss
        is above the cut. ``xi L(xi)`` is the mixture
        ``sum_m (1/m^3) m^2 xi exp(-m xi)``, so it is drawn exactly as a
        Gamma(2) variable of rate ``m``, with ``m`` drawn with probability
        ``1/m^3``; for ``xi_min > 1`` the envelope
        ``xi exp(-xi) L(xi_min) exp(xi_min)`` is used instead, to avoid
        discarding the draws below ``xi_min``.

        The laboratory photon energy ``kappa = k / k_B T`` is then drawn
        from ``1/(exp(kappa) - 1)`` on ``[xi, Gamma^2 xi]`` by inversion:
        ``L(kappa) = (1 - u) L(xi) + u L(Gamma^2 xi)``, and
        ``kappa = L(L(kappa))``.

        Parameters
        ----------
        gamma, beta : ndarray
            Lorentz factor and velocity of each particle.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        xi, kappa, cos_theta : ndarray
            Scaled rest-frame photon energy, laboratory photon energy in
            units of ``k_B T``, and cosine of the rest-frame scattering
            angle.
        """
        n = gamma.size
        eps_T_all = (1.0 + beta) * gamma * self.kT / self.mass0
        gamma2_ratio_all = (1.0 + beta) / (1.0 - beta)
        a = self.energy_loss_cut / (1.0 - self.energy_loss_cut)
        xi_min_all = a / (2.0 * eps_T_all)

        xi = np.empty(n)
        cos_theta = np.empty(n)
        pending = np.arange(n)
        for _ in range(_MAX_REJECTION_ITERATIONS):
            m = pending.size
            if m == 0:
                break
            eps_T = eps_T_all[pending]
            xi_min = xi_min_all[pending]
            trial = np.empty(m)
            accept = np.ones(m, dtype=bool)

            # Draws of the untruncated distribution, discarded below xi_min
            low = xi_min <= _XI_SWITCH
            n_low = int(low.sum())
            if n_low:
                index = _sample_series_index(n_low, rng)
                trial[low] = -np.log(_uniform_open(rng, n_low)
                                     * _uniform_open(rng, n_low)) / index
                accept[low] = trial[low] >= xi_min[low]

            # Truncated exponential envelope above xi_min
            high = ~low
            n_high = int(high.sum())
            if n_high:
                x0 = xi_min[high]
                t = -np.log(_uniform_open(rng, n_high))
                second = rng.random(n_high) * (1.0 + x0) >= x0
                t = t - np.where(second,
                                 np.log(_uniform_open(rng, n_high)), 0.0)
                trial[high] = x0 + t
                accept[high] = (
                    rng.random(n_high) * _log1mexp_times_exp(x0)
                    < _log1mexp_times_exp(trial[high]))

            # Exact kinematic range of the laboratory photon energy
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = (_log1mexp(gamma2_ratio_all[pending] * trial)
                         / _log1mexp(trial))
            accept &= rng.random(m) >= np.nan_to_num(ratio)

            # Thomson angle, energy-loss cut and Klein-Nishina weight
            c = _sample_thomson_cos_theta(rng.random(m))
            eps = trial * eps_T
            accept &= eps * (1.0 - c) >= a
            accept &= rng.random(m) < klein_nishina_weight(eps, c)

            done = pending[accept]
            xi[done] = trial[accept]
            cos_theta[done] = c[accept]
            pending = pending[~accept]
        else:
            raise RuntimeError(
                "Rejection sampling of the thermal photons did not converge "
                f"after {_MAX_REJECTION_ITERATIONS} iterations.")

        u = rng.random(n)
        with np.errstate(over='ignore'):
            kappa = _log1mexp((1.0 - u) * _log1mexp(xi)
                              + u * _log1mexp(gamma2_ratio_all * xi))
        return xi, kappa, cos_theta

    def _sample_photons_unweighted(self, gamma, beta, rng):
        """
        Sample the thermal photon and the scattering angle of each event
        without the flux factor.

        This follows the description of Burkhardt's note [1]: the photon
        energy is drawn from the Planck number spectrum and its direction
        isotropically; the scattering angle is drawn from the Thomson
        distribution and accepted with the Klein-Nishina weight. Events below
        the energy-loss cut are discarded. It is used only to reproduce the
        numbers published in [1].

        Parameters
        ----------
        gamma, beta : ndarray
            Lorentz factor and velocity of each particle.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        xi, kappa, cos_theta : ndarray
            Scaled rest-frame photon energy, laboratory photon energy in
            units of ``k_B T``, and cosine of the rest-frame scattering
            angle.
        """
        n = gamma.size
        eps_T_all = (1.0 + beta) * gamma * self.kT / self.mass0
        a = self.energy_loss_cut / (1.0 - self.energy_loss_cut)

        xi = np.empty(n)
        kappa = np.empty(n)
        cos_theta = np.empty(n)
        pending = np.arange(n)
        for _ in range(_MAX_REJECTION_ITERATIONS):
            m = pending.size
            if m == 0:
                break
            b = beta[pending]
            index = _sample_series_index(m, rng)
            kap = -np.log(_uniform_open(rng, m) * _uniform_open(rng, m)
                          * _uniform_open(rng, m)) / index
            cos_psi = 2.0 * rng.random(m) - 1.0
            trial = kap * (1.0 - b * cos_psi) / (1.0 + b)
            c = _sample_thomson_cos_theta(rng.random(m))
            eps = trial * eps_T_all[pending]
            accept = eps * (1.0 - c) >= a
            accept &= rng.random(m) < klein_nishina_weight(eps, c)

            done = pending[accept]
            xi[done] = trial[accept]
            kappa[done] = kap[accept]
            cos_theta[done] = c[accept]
            pending = pending[~accept]
        else:
            raise RuntimeError(
                "Rejection sampling of the thermal photons did not converge "
                f"after {_MAX_REJECTION_ITERATIONS} iterations.")

        return xi, kappa, cos_theta
