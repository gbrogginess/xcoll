# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
from scipy.constants import c as C_LIGHT
from scipy.constants import k as BOLTZMANN_SI
from scipy.constants import hbar as HBAR_SI
from scipy.constants import e as QELEM_SI
from scipy.constants import physical_constants

import xobjects as xo
import xtrack as xt

THOMSON_CROSS_SECTION = physical_constants['Thomson cross section'][0]  # [m^2]
APERY_ZETA3 = 1.2020569031595943
APERY_ZETA4 = 1.0823232337111382


def blackbody_photon_density(temperature):
    """
    Number density of the blackbody photon gas.

    The photon-number density of an isotropic blackbody field at temperature
    ``T`` is

    .. math::
        n_\\gamma = \\frac{2\\zeta(3)}{\\pi^2}
                    \\left(\\frac{k_B T}{\\hbar c}\\right)^3
                  = 16 \\pi \\zeta(3) \\left(\\frac{k_B T}{h c}\\right)^3 ,

    Parameters
    ----------
    temperature : float
        Temperature of the vacuum chamber [K].

    Returns
    -------
    photon_density : float
        Photon number density [1/m^3].
    """
    return (2*APERY_ZETA3/np.pi**2
            * (BOLTZMANN_SI*float(temperature)/(HBAR_SI*C_LIGHT))**3)


def blackbody_mean_photon_energy(temperature):
    """
    Mean energy of a blackbody photon.

    The photon-number weighted mean energy is
    :math:`\\langle k \\rangle = 3\\zeta(4)/\\zeta(3)\\, k_B T
    \\simeq 2.7011\\, k_B T`.

    Parameters
    ----------
    temperature : float
        Temperature of the vacuum chamber [K].

    Returns
    -------
    mean_photon_energy : float
        Mean photon energy [eV].
    """
    kT_eV = BOLTZMANN_SI*float(temperature)/QELEM_SI
    return 3*APERY_ZETA4/APERY_ZETA3*kT_eV


class ThermalComptonScattering(xt.BeamElement):
    """
    Beam element that generates Compton scattering events of the stored beam on
    the thermal (blackbody) photons radiated by the vacuum chamber, at a single
    location in a lattice.

    Each element is one scattering centre and represents a section of the ring
    of length ``section_length``.  When :meth:`scatter` is called, a sample of
    macro-leptons is drawn from the local (Gaussian) beam distribution, each
    macro-lepton is given ``n_trials`` independent scattering trials
    representing the section length, and the macro-particles whose momentum
    deviation changed by more than ``delta_threshold`` are returned with an
    absolute rate weight in Hz.  The element is *passive* during tracking
    (``track`` is a no-op): all the physics happens in :meth:`scatter`.

    The Monte Carlo kernel is implemented in C99 (see
    ``thermal_compton_src/thermal_compton.h``) and follows the proposal method
    of H. Burkhardt [1]: trial angles are drawn from the Thomson angular
    distribution and accepted with the Klein-Nishina to Thomson ratio, so that
    accepted events follow the Klein-Nishina distribution while the absolute
    normalisation is set by the (energy independent) Thomson cross section.

    Parameters
    ----------
    s : float, optional
        Longitudinal position of the element in the lattice [m].
    particle_ref : xtrack.Particles, optional
        Reference particle.  Only leptons are supported (electrons or
        positrons); ``p0c`` and ``mass0`` are taken from it.
    element_index : int, optional
        Index of this element in the line.
    temperature : float, optional
        Temperature of the vacuum chamber [K].  Default 300 K.
    section_length : float, optional
        Length of the lattice section represented by this element [m].
    section_rate : float, optional
        Total *trial* scattering rate of the section [1/s], i.e.
        ``n_particles * (section_length/circumference) * c * n_gamma *
        sigma_T``.  It is normally set by :class:`ThermalComptonStudy`.
    alfx, betx, alfy, bety : float, optional
        Twiss parameters at the element.
    dx, dpx, dy, dpy : float, optional
        Dispersion functions at the element.
    x_co, px_co, y_co, py_co, zeta_co, delta_co : float, optional
        Closed-orbit coordinates at the element.
    gemitt_x, gemitt_y : float, optional
        Horizontal and vertical geometric emittances [m rad].
    sigma_z : float, optional
        RMS bunch length [m].
    sigma_delta : float, optional
        RMS relative momentum spread.
    n_macroparticles : int, optional
        Number of macro-leptons drawn from the local beam distribution.
    n_trials : int, optional
        Number of scattering trials per macro-lepton.  Each trial represents a
        path length ``section_length/n_trials``.
    delta_threshold : float, optional
        Thinning threshold on the *change* of the relative momentum deviation
        ``|delta_out - delta_in|``.  Only events above this threshold are
        returned for tracking.  It is a numerical importance threshold and must
        be chosen well below the ring momentum acceptance; it is *not* the
        machine acceptance.
    max_events_per_macro : int, optional
        Capacity of the per-macro-particle output slice.  Events beyond this
        number are counted in :attr:`n_dropped` and lost; a warning is issued.
    enable_tail_veto : bool, optional
        Enable the exact early veto of the trials that cannot reach
        ``delta_threshold``.  Speeds up the generation without changing the
        result; switch it off only for debugging.
    n_sigma_cut : float or None, optional
        Truncation of the Gaussian beam distribution used to draw the
        macro-leptons.  ``None`` for no truncation.

    Attributes
    ----------
    rate_scattering : float
        Total Compton scattering rate represented by the last call to
        :meth:`scatter` [1/s], i.e. the sum of the weights of *all* accepted
        Compton events, including those below ``delta_threshold``.
    rate_tail : float
        Rate carried by the returned (above threshold) particles [1/s].
    energy_loss_rate : float
        Energy carried away by the scattered photons per unit time in this
        section [eV/s].  Can be compared with the analytic inverse-Compton
        power :math:`P = (4/3)\\,\\sigma_T c\\,\\gamma^2\\beta^2 U_\\gamma`.
    n_dropped : int
        Number of tail events that could not be stored (output slice full, or
        backward-going lepton).

    Notes
    -----
    **Physics summary**

    For each trial:

    1. A thermal photon energy is drawn from the blackbody *photon-number*
       spectrum :math:`p(k)\\propto k^2/(e^{k/k_BT}-1)`, using the exact
       Gamma-mixture representation.
    2. The photon direction is drawn from the *flux-weighted* distribution
       :math:`p(\\mu) = (1-\\beta\\mu)/2`, with :math:`\\mu` the cosine of the
       angle between the photon and the lepton momentum.  The factor
       :math:`(1-\\beta\\mu)` is the Moeller relative-velocity factor of the
       collision rate; it integrates to one over an isotropic gas (so the total
       rate is unaffected) but it strongly favours head-on collisions, which
       are exactly the ones producing the large momentum deviations relevant
       for beam losses.
    3. The photon is boosted to the lepton rest frame, where
       :math:`k^* = \\gamma k (1-\\beta\\mu)`.
    4. A polar angle is drawn from the Thomson distribution
       :math:`\\propto 1+\\cos^2\\theta` (exact inverse CDF) and accepted with
       probability :math:`(d\\sigma_{KN}/d\\Omega)/(d\\sigma_T/d\\Omega)\\le 1`;
       the accepted sample therefore follows Klein-Nishina and the acceptance
       fraction is :math:`\\sigma_{KN}/\\sigma_T`.
    5. The scattered photon is boosted back to the laboratory and the outgoing
       lepton is obtained by 4-momentum conservation.

    The macro-particle weights are absolute rates in Hz: their sum over the
    whole ring is the loss rate, and the lifetime follows from
    :math:`\\tau = N_{\\rm beam}/R_{\\rm loss}`.

    References
    ----------
    .. [1] H. Burkhardt, "Monte Carlo simulation of scattering of beam
       particles and thermal photons", CERN SL/Note 93-73 (OP) (1993).
       https://cds.cern.ch/record/703373
    .. [2] V. Telnov, "Scattering of electrons on thermal radiation photons in
       electron-positron storage rings", Nucl. Instrum. Meth. A **260** (1987)
       304. https://doi.org/10.1016/0168-9002(87)90093-3
    .. [3] A. Di Domenico, "Inverse Compton Scattering of Thermal Radiation at
       LEP and LEP-200", Particle Accelerators **39** (1992) 137.
    .. [4] A. Natochii, "Thermal Compton Scattering of Electron Beams on
       Blackbody Photons: A Monte Carlo Event Generator for Multi-Turn Tracking
       at the Electron-Ion Collider", BNL-229489-2026-TECH, EIC-ADD-TN-159
       (2026), and https://github.com/eic/thermal-compton-mc
    """

    _xofields = {
        'p0c': xo.Float64,
        'mass0': xo.Float64,
        'temperature': xo.Float64,
        'section_length': xo.Float64,
        'section_rate': xo.Float64,
        'gemitt_x': xo.Float64,
        'gemitt_y': xo.Float64,
        'alfx': xo.Float64,
        'betx': xo.Float64,
        'alfy': xo.Float64,
        'bety': xo.Float64,
        'dx': xo.Float64,
        'dpx': xo.Float64,
        'dy': xo.Float64,
        'dpy': xo.Float64,
        'x_co': xo.Float64,
        'px_co': xo.Float64,
        'y_co': xo.Float64,
        'py_co': xo.Float64,
        'zeta_co': xo.Float64,
        'delta_co': xo.Float64,
        'sigma_z': xo.Float64,
        'sigma_delta': xo.Float64,
        'n_trials': xo.Int64,
        'delta_threshold': xo.Float64,
        'enable_tail_veto': xo.Int64,
    }

    isthick = False
    behaves_like_drift = False

    _depends_on = [xt.RandomUniformAccurate]

    _extra_c_sources = [
        '#include "xcoll/beam_elements/thermal_compton_src/thermal_compton.h"'
    ]

    _per_particle_kernels = {
        '_scatter': xo.Kernel(
            c_name='ThermalComptonScatter',
            args=[
                xo.Arg(xo.Float64, name='x_out', pointer=True),
                xo.Arg(xo.Float64, name='px_out', pointer=True),
                xo.Arg(xo.Float64, name='y_out', pointer=True),
                xo.Arg(xo.Float64, name='py_out', pointer=True),
                xo.Arg(xo.Float64, name='zeta_out', pointer=True),
                xo.Arg(xo.Float64, name='delta_out', pointer=True),
                xo.Arg(xo.Float64, name='photon_energy_out', pointer=True),
                xo.Arg(xo.Float64, name='weight_out', pointer=True),
                xo.Arg(xo.Int64,   name='n_events_out', pointer=True),
                xo.Arg(xo.Int64,   name='n_dropped_out', pointer=True),
                xo.Arg(xo.Float64, name='rate_scattering_out', pointer=True),
                xo.Arg(xo.Float64, name='energy_loss_rate_out', pointer=True),
                xo.Arg(xo.Float64, name='weight_per_trial'),
                xo.Arg(xo.Int64,   name='max_events'),
            ],
        ),
    }

    def __init__(self, s=0.0,
                 particle_ref=None,
                 element_index=0,
                 temperature=300.0,
                 section_length=0.0,
                 section_rate=0.0,
                 alfx=0.0, betx=1.0, alfy=0.0, bety=1.0,
                 dx=0.0, dpx=0.0, dy=0.0, dpy=0.0,
                 x_co=0.0, px_co=0.0, y_co=0.0, py_co=0.0,
                 zeta_co=0.0, delta_co=0.0,
                 gemitt_x=0.0, gemitt_y=0.0,
                 sigma_z=0.0, sigma_delta=0.0,
                 n_macroparticles=1000,
                 n_trials=100,
                 delta_threshold=1e-3,
                 max_events_per_macro=None,
                 enable_tail_veto=True,
                 n_sigma_cut=5.0,
                 **kwargs):
        """
        Create a thermal Compton scattering element.

        Most of the local optics and beam parameters are normally supplied
        later by :class:`xcoll.ThermalComptonStudy`; direct construction is
        mainly used to place scattering markers in a line before configuring a
        study.

        Parameters
        ----------
        See the class docstring.

        Returns
        -------
        None
        """
        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        super().__init__(**kwargs)

        if particle_ref is None:
            particle_ref = xt.Particles(_context=self._buffer.context)

        self.s = s
        self.element_index = element_index
        self.particle_ref = particle_ref
        if temperature < 0:
            raise ValueError('`temperature` must be non-negative.')
        self.temperature = float(temperature)
        self.section_length = section_length
        self.section_rate = section_rate
        self.alfx = alfx
        self.betx = betx
        self.alfy = alfy
        self.bety = bety
        self.dx = dx
        self.dpx = dpx
        self.dy = dy
        self.dpy = dpy
        self.x_co = x_co
        self.px_co = px_co
        self.y_co = y_co
        self.py_co = py_co
        self.zeta_co = zeta_co
        self.delta_co = delta_co
        self.gemitt_x = gemitt_x
        self.gemitt_y = gemitt_y
        self.sigma_z = sigma_z
        self.sigma_delta = sigma_delta
        self.n_trials = n_trials
        self.delta_threshold = delta_threshold
        self.enable_tail_veto = int(bool(enable_tail_veto))

        # Python-side configuration (not needed by the kernel)
        self.n_macroparticles = int(n_macroparticles)
        self.max_events_per_macro = max_events_per_macro
        self.n_sigma_cut = n_sigma_cut

        # Diagnostics filled by scatter()
        self.rate_scattering = 0.0
        self.rate_tail = 0.0
        self.energy_loss_rate = 0.0
        self.n_dropped = 0

    ############################################################
    # Properties
    ############################################################
    @property
    def photon_density(self):
        """
        Blackbody photon number density at the configured temperature.

        Returns
        -------
        photon_density : float
            Photon number density [1/m^3].
        """
        return blackbody_photon_density(self.temperature)

    @property
    def particle_ref(self):
        """
        Reference particle of the element.

        Returns
        -------
        particle_ref : xtrack.Particles
            Reference particle.
        """
        return self._particle_ref

    @particle_ref.setter
    def particle_ref(self, value):
        """
        Set the reference particle and propagate ``p0c`` and ``mass0``.

        Parameters
        ----------
        value : xtrack.Particles
            Reference particle.  A warning is issued if it is not a lepton,
            since the Klein-Nishina cross section used here assumes a lepton
            (the Thomson cross section scales as the inverse squared mass).

        Returns
        -------
        None
        """
        self._particle_ref = value
        if value is not None and len(np.atleast_1d(value.p0c)) > 0:
            self.p0c = float(value.p0c[0])
            self.mass0 = float(value.mass0)
            if abs(self.mass0 - xt.ELECTRON_MASS_EV)/xt.ELECTRON_MASS_EV > 1e-6:
                import warnings
                warnings.warn(
                    "ThermalComptonScattering assumes a lepton beam: the "
                    "Thomson/Klein-Nishina cross sections used here are the "
                    "electron ones. Results for other species are not valid.",
                    UserWarning, stacklevel=2)

    @property
    def max_events_per_macro(self):
        """
        Capacity of the per-macro-particle output slice.

        If ``None`` was given at construction, a default is derived from
        ``n_trials`` when :meth:`scatter` is called.

        Returns
        -------
        max_events_per_macro : int or None
            Output capacity per macro-particle.
        """
        return self._max_events_per_macro

    @max_events_per_macro.setter
    def max_events_per_macro(self, value):
        """
        Set the capacity of the per-macro-particle output slice.

        Parameters
        ----------
        value : int or None
            Output capacity per macro-particle.  ``None`` selects the default.

        Returns
        -------
        None
        """
        self._max_events_per_macro = None if value is None else int(value)

    @property
    def mean_photon_energy(self):
        """
        Mean blackbody photon energy at the configured temperature.

        Returns
        -------
        mean_photon_energy : float
            Mean photon energy [eV].
        """
        return blackbody_mean_photon_energy(self.temperature)

    @property
    def weight_per_trial(self):
        """
        Absolute rate weight carried by a single scattering trial.

        The section trial rate ``section_rate`` is shared equally among the
        ``n_macroparticles * n_trials`` trials.

        Returns
        -------
        weight_per_trial : float
            Rate weight per trial [1/s].
        """
        n_tot = self.n_macroparticles*self.n_trials
        if n_tot == 0:
            return 0.0
        return self.section_rate/n_tot

    ############################################################
    # Configuration
    ############################################################
    def _configure(self, **kwargs):
        config_allowed = {
            's', 'particle_ref', 'element_index', 'temperature',
            'section_length', 'section_rate',
            'alfx', 'betx', 'alfy', 'bety',
            'dx', 'dpx', 'dy', 'dpy',
            'x_co', 'px_co', 'y_co', 'py_co', 'zeta_co', 'delta_co',
            'gemitt_x', 'gemitt_y', 'sigma_z', 'sigma_delta',
            'n_macroparticles', 'n_trials', 'delta_threshold',
            'max_events_per_macro', 'enable_tail_veto', 'n_sigma_cut',
        }
        unknown = set(kwargs) - config_allowed
        if unknown:
            bad = ', '.join(sorted(unknown))
            raise KeyError(f"Unsupported _configure() keys: {bad}")

        enable_tail_veto = kwargs.pop('enable_tail_veto', None)
        if enable_tail_veto is not None:
            self.enable_tail_veto = int(bool(enable_tail_veto))

        for kk, vv in kwargs.items():
            setattr(self, kk, vv)

    ############################################################
    # Local beam distribution
    ############################################################
    def generate_macroparticles(self, n_macroparticles=None, rng=None,
                                _context=None):
        """
        Draw a sample of macro-leptons from the local beam distribution.

        The distribution is the (optionally truncated) 6D Gaussian defined by
        the local Twiss parameters, the geometric emittances, the bunch length
        and the momentum spread, displaced to the local closed orbit and
        including the local dispersion.

        Parameters
        ----------
        n_macroparticles : int or None, optional
            Number of macro-leptons.  Defaults to
            :attr:`n_macroparticles`.
        rng : numpy.random.Generator or None, optional
            Random generator used for the beam distribution (independent of
            the RNG used inside the scattering kernel).
        _context : xobjects.Context or None, optional
            Context in which the particles are created.  Defaults to the
            context of the element.

        Returns
        -------
        particles : xtrack.Particles
            Macro-leptons representing the local beam distribution, with
            ``particle_id`` equal to their index.
        """
        if n_macroparticles is None:
            n_macroparticles = self.n_macroparticles
        n_macroparticles = int(n_macroparticles)
        if rng is None:
            rng = np.random.default_rng()
        if _context is None:
            _context = self._context

        cut = self.n_sigma_cut

        def _normal(size):
            out = rng.standard_normal(size)
            if cut is not None:
                bad = np.abs(out) > cut
                while bad.any():
                    out[bad] = rng.standard_normal(int(bad.sum()))
                    bad = np.abs(out) > cut
            return out

        n1, n2, n3, n4, n5, n6 = (_normal(n_macroparticles) for _ in range(6))

        delta = self.sigma_delta*n6
        zeta = self.sigma_z*n5

        sqrt_ex = np.sqrt(self.gemitt_x)
        sqrt_ey = np.sqrt(self.gemitt_y)
        x = sqrt_ex*np.sqrt(self.betx)*n1
        px = -sqrt_ex/np.sqrt(self.betx)*(self.alfx*n1 - n2)
        y = sqrt_ey*np.sqrt(self.bety)*n3
        py = -sqrt_ey/np.sqrt(self.bety)*(self.alfy*n3 - n4)

        # Dispersion and closed orbit
        x += self.dx*delta + self.x_co
        px += self.dpx*delta + self.px_co
        y += self.dy*delta + self.y_co
        py += self.dpy*delta + self.py_co
        zeta += self.zeta_co
        delta += self.delta_co

        return xt.Particles(_context=_context,
                            p0c=self.p0c,
                            mass0=self.particle_ref.mass0,
                            q0=self.particle_ref.q0,
                            pdg_id=self.particle_ref.pdg_id,
                            x=x, px=px, y=y, py=py,
                            zeta=zeta, delta=delta)

    ############################################################
    # Event generation
    ############################################################
    def scatter(self, n_macroparticles=None, seed=None, _macroparticles=None,
                _capacity_factor=2):
        """
        Generate weighted thermal-Compton-scattered macro-particles.

        A sample of macro-leptons is drawn from the local beam distribution
        (unless one is supplied), each is given :attr:`n_trials` scattering
        trials, and the events whose relative momentum deviation changed by
        more than :attr:`delta_threshold` are returned as an
        :class:`xtrack.Particles` object with absolute rate weights [1/s].

        Parameters
        ----------
        n_macroparticles : int or None, optional
            Number of macro-leptons to draw.  Defaults to
            :attr:`n_macroparticles`.
        seed : int or None, optional
            Seed of the random-number generators.  If ``None``, a seed is drawn
            with :mod:`numpy.random`.
        _macroparticles : xtrack.Particles or None, optional
            Explicit macro-lepton sample, e.g. a tracked distribution.  Its
            ``particle_id`` must be ``0 ... n-1``.
        _capacity_factor : int, optional
            Capacity of the returned :class:`xtrack.Particles` object, as a
            multiple of the number of generated particles.

        Returns
        -------
        particles : xtrack.Particles
            Scattered macro-particles above the momentum-deviation threshold,
            weighted so that the sum of their weights is the section rate
            represented by them [1/s].
        """
        context = self._context

        if seed is None:
            seed = int(np.random.randint(1, 2**31 - 1))
        rng = np.random.default_rng(seed)

        if _macroparticles is None:
            particles = self.generate_macroparticles(
                n_macroparticles=n_macroparticles, rng=rng, _context=context)
        else:
            particles = _macroparticles
            if particles._context is not context:
                particles = particles.copy(_context=context)

        n_macro = int(len(particles.x))
        if not np.array_equal(context.nparray_from_context_array(
                particles.particle_id), np.arange(n_macro)):
            raise ValueError(
                "The macro-particles must have `particle_id` equal to their "
                "index (0 ... n-1); they are used to address the output "
                "slices of the kernel.")
        self.n_macroparticles = n_macro

        seeds = rng.integers(1, 2**32 - 1,
                             size=particles._capacity).astype(np.uint32)
        particles._init_random_number_generator(seeds=seeds)

        max_events = self.max_events_per_macro
        if max_events is None:
            # Generous default: the number of tail events per macro-particle is
            # binomial with mean << n_trials; 32 + 10% of the trials is safe
            # for any realistic threshold, and cheap in memory.
            max_events = int(32 + 0.1*self.n_trials)
        max_events = max(1, int(max_events))

        n_out = n_macro*max_events
        x_out = context.zeros(shape=(n_out,), dtype=np.float64)
        px_out = context.zeros(shape=(n_out,), dtype=np.float64)
        y_out = context.zeros(shape=(n_out,), dtype=np.float64)
        py_out = context.zeros(shape=(n_out,), dtype=np.float64)
        zeta_out = context.zeros(shape=(n_out,), dtype=np.float64)
        delta_out = context.zeros(shape=(n_out,), dtype=np.float64)
        photon_energy_out = context.zeros(shape=(n_out,), dtype=np.float64)
        weight_out = context.zeros(shape=(n_out,), dtype=np.float64)
        n_events_out = context.zeros(shape=(n_macro,), dtype=np.int64)
        n_dropped_out = context.zeros(shape=(n_macro,), dtype=np.int64)
        rate_out = context.zeros(shape=(n_macro,), dtype=np.float64)
        eloss_out = context.zeros(shape=(n_macro,), dtype=np.float64)

        self._scatter(particles=particles,
                      x_out=x_out, px_out=px_out,
                      y_out=y_out, py_out=py_out,
                      zeta_out=zeta_out, delta_out=delta_out,
                      photon_energy_out=photon_energy_out,
                      weight_out=weight_out,
                      n_events_out=n_events_out,
                      n_dropped_out=n_dropped_out,
                      rate_scattering_out=rate_out,
                      energy_loss_rate_out=eloss_out,
                      weight_per_trial=self.weight_per_trial,
                      max_events=max_events)

        n_events = context.nparray_from_context_array(n_events_out)
        n_dropped = int(np.sum(context.nparray_from_context_array(n_dropped_out)))

        self.rate_scattering = float(np.sum(
            context.nparray_from_context_array(rate_out)))
        self.energy_loss_rate = float(np.sum(
            context.nparray_from_context_array(eloss_out)))
        self.n_dropped = n_dropped
        if n_dropped > 0:
            import warnings
            warnings.warn(
                f"{n_dropped} events were dropped at element "
                f"'{getattr(self, 'element_name', '?')}' "
                f"(output capacity {max_events} per macro-particle, or "
                f"backward-going leptons). The represented rate is truncated: "
                f"increase `max_events_per_macro` or reduce `n_trials`.",
                UserWarning, stacklevel=2)

        # Gather the (sparse) per-macro-particle slices
        mask = (np.arange(n_out) % max_events) < np.repeat(n_events, max_events)
        idx = np.where(mask)[0]
        n = idx.size

        part = xt.Particles(
            _context=context,
            _capacity=max(1, _capacity_factor*n),
            p0c=self.p0c,
            mass0=self.particle_ref.mass0,
            q0=self.particle_ref.q0,
            pdg_id=self.particle_ref.pdg_id,
            x=x_out[idx], px=px_out[idx],
            y=y_out[idx], py=py_out[idx],
            zeta=zeta_out[idx], delta=delta_out[idx],
            weight=weight_out[idx],
            s=getattr(self, 's', 0.0))
        part.at_element = self.element_index

        self.rate_tail = float(np.sum(
            context.nparray_from_context_array(weight_out[idx])))
        # Diagnostics on the last generated sample
        self.photon_energy_log = context.nparray_from_context_array(
            photon_energy_out[idx]).copy()

        return part

    def track(self, particles):
        """
        Track particles through the element without applying a kick.

        Thermal Compton scattering is generated explicitly by :meth:`scatter`;
        during normal lattice tracking this element behaves as a passive
        marker.

        Parameters
        ----------
        particles : xtrack.Particles
            Particles tracked through the passive marker element.

        Returns
        -------
        None
        """
        super().track(particles)