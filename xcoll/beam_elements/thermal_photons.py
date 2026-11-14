# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xtrack as xt

from ..general import _pkg_root
from .beamgas import _resolve_n_scattering_events


# Number of values that the scattering kernel records for each event
_RECORD_SIZE = 7
_RECORD_COLUMNS = ('energy_loss', 'photon_energy', 'theta',
                   'incident_photon_energy', 'incident_cos_psi',
                   'cos_theta_rest', 'n_trials')


class ThermalPhotonMarker(xt.BeamElement):
    """
    Passive marker of a thermal-photon scattering location.

    Each marker represents the section of lattice that precedes it, up to the
    previous marker (or the start of the line for the first one). It holds the
    configuration needed to generate there the Compton scatterings of the beam
    particles on the thermal (blackbody) photons that fill the vacuum chamber:
    the local optics and closed orbit, the beam parameters, the temperature,
    the energy-loss cut and the interaction rate of the section. All of it is
    stored in xofields, so a configured line can be saved to JSON and loaded
    back, e.g. by independent jobs.

    The marker does nothing during tracking. The markers are installed,
    configured and used through :class:`xcoll.ThermalPhotonGenerator`,
    available as ``line.xcoll.thermal_photons``.

    Parameters
    ----------
    ds : float, optional
        Length of the lattice section represented by this marker [m].
    p0c : float, optional
        Reference momentum at configuration [eV].
    bunch_intensity : float, optional
        Number of particles in the bunch.
    alfx, betx : float, optional
        Horizontal Twiss parameters at the marker.
    alfy, bety : float, optional
        Vertical Twiss parameters at the marker.
    dx, dpx : float, optional
        Horizontal dispersion and its derivative at the marker.
    dy, dpy : float, optional
        Vertical dispersion and its derivative at the marker.
    x_co, px_co : float, optional
        Horizontal closed-orbit position and normalised momentum.
    y_co, py_co : float, optional
        Vertical closed-orbit position and normalised momentum.
    zeta_co, delta_co : float, optional
        Longitudinal closed-orbit coordinate and relative momentum deviation.
    gemitt_x, gemitt_y : float, optional
        Horizontal and vertical geometric emittances [m rad].
    sigma_z : float, optional
        RMS bunch length [m].
    sigma_delta : float, optional
        RMS relative momentum spread.
    n_scattering_events : int, optional
        Number of scatterings generated at this marker.
    temperature : float, optional
        Temperature of the blackbody radiation from which the photons of the
        events generated at this marker are drawn [K]. For a section of
        non-uniform temperature it is the mean temperature weighted with the
        local interaction rate.
    energy_loss_cut : float, optional
        Lowest relative energy loss generated at this marker, in ``(0, 1)``.
        The energy loss is the Lorentz invariant ``y = 1 - k*'/k*``, the
        fraction of its energy that the photon loses in the rest frame of the
        particle; it equals the relative energy loss of an ultra-relativistic
        beam particle up to corrections of order ``1/gamma``.
    interaction_rate : float, optional
        Scattering rate of the represented section [1/s], restricted to the
        energy losses above ``energy_loss_cut``. It is the total weight of
        the particles generated at this marker.
    total_rate : float, optional
        Scattering rate of the represented section for all the energy losses
        [1/s]. Most of it does not perturb the beam.
    mean_free_path : float, optional
        Local mean free path for a scattering above ``energy_loss_cut`` [m].
    seed : int, optional
        Seed of the random streams of the marker.

    Attributes
    ----------
    scatter_log : xtrack.Table or None
        Record of the interactions of the last generation at this marker; see
        :meth:`xcoll.ThermalPhotonGenerator.generate`. It is kept in memory
        only. ``None`` before the first generation.

    Notes
    -----
    **Physics summary**

    For each of the ``n_scattering_events`` generated macro-particles:

    1. Initial coordinates are drawn from the local matched Gaussian
       distribution, built from the Twiss parameters, the geometric
       emittances, ``sigma_z`` and ``sigma_delta``, and shifted onto the local
       closed orbit. Betatron and dispersive contributions are included;
       transverse coupling is not.
    2. A thermal photon is drawn among those that scatter on the particle:
       its energy follows the Planck spectrum at ``temperature`` and its
       direction is isotropic, both weighted with the relative-velocity
       factor ``1 - beta cos(psi)`` of the collision rate and with the
       Klein-Nishina cross section.
    3. The photon scatters in the rest frame of the particle with the
       Klein-Nishina angular distribution, restricted to the angles for which
       the energy loss exceeds ``energy_loss_cut``, and the outgoing momentum
       of the particle follows from four-momentum conservation.
    4. The macro-particle is given the weight
       ``interaction_rate/n_scattering_events``, so that the total weight of
       the sample is the section interaction rate [1/s].

    Steps 2 and 3 are done by the C kernel ``ThermalPhotonMarker_scatter``,
    with the random-number generator of each particle. The models are those
    of :mod:`xcoll.thermal_photons.cross_sections`, where the variables and
    the sampling scheme are documented.

    Every generated macro-particle interacts exactly once. This *forced
    interaction* scheme is the correct estimator in the single-scattering
    regime, where the real probability of interacting in one turn is many
    orders of magnitude below one (about ``1e-9`` per turn in a 100 km ring
    at room temperature).

    References
    ----------
    .. [1] H. Burkhardt, "Monte Carlo simulation of scattering of beam
       particles and thermal photons", CERN SL/Note 93-73 (OP) (1993).
    .. [2] V. I. Telnov, Nucl. Instrum. Meth. A **260**, 304 (1987).
    .. [3] A. Di Domenico, Part. Accel. **39**, 137 (1992).
    """

    _xofields = {
        'p0c': xo.Float64,
        'bunch_intensity': xo.Float64,
        'ds': xo.Float64,
        'interaction_rate': xo.Float64,
        'total_rate': xo.Float64,
        'mean_free_path': xo.Float64,
        'n_scattering_events': xo.Int64,
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
        'temperature': xo.Float64,
        'energy_loss_cut': xo.Float64,
        'seed': xo.Int64,
    }

    isthick = False
    behaves_like_drift = False
    allow_track = True
    skip_in_loss_location_refinement = True

    _depends_on = [xt.RandomUniformAccurate]

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements', 'elements_src',
                           'thermal_photons.h')
    ]

    _per_particle_kernels = {
        '_scatter': xo.Kernel(
            c_name='ThermalPhotonMarker_scatter',
            args=[
                xo.Arg(xo.Float64, name='record', pointer=True),
            ],
        ),
    }

    def __init__(self, ds=0.0,
                 p0c=0.0,
                 bunch_intensity=0.0,
                 alfx=0.0, betx=0.0, alfy=0.0, bety=0.0,
                 dx=0.0, dpx=0.0, dy=0.0, dpy=0.0,
                 x_co=0.0, px_co=0.0, y_co=0.0, py_co=0.0,
                 zeta_co=0.0, delta_co=0.0,
                 gemitt_x=0.0, gemitt_y=0.0,
                 sigma_z=0.0, sigma_delta=0.0,
                 n_scattering_events=None,
                 temperature=0.0,
                 energy_loss_cut=0.0,
                 interaction_rate=0.0,
                 total_rate=0.0,
                 mean_free_path=0.0,
                 seed=0,
                 **kwargs):
        """
        Create a thermal-photon scattering marker.

        Parameters
        ----------
        See the class docstring.

        Returns
        -------
        None
        """
        self.scatter_log = None

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        super().__init__(**kwargs)

        self.ds = ds
        self.p0c = p0c
        self.bunch_intensity = bunch_intensity
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
        self.n_scattering_events = _resolve_n_scattering_events(
            n_scattering_events=n_scattering_events, default=0)
        self.temperature = temperature
        self.energy_loss_cut = energy_loss_cut
        self.interaction_rate = interaction_rate
        self.total_rate = total_rate
        self.mean_free_path = mean_free_path
        self.seed = seed

    def _configure(self, **kwargs):
        """
        Set the configuration supplied by
        :meth:`xcoll.ThermalPhotonGenerator.configure`.

        Parameters
        ----------
        **kwargs
            Any of the constructor parameters.

        Returns
        -------
        None
        """
        config_allowed = {
            'ds', 'p0c', 'bunch_intensity',
            'alfx', 'betx', 'alfy', 'bety',
            'dx', 'dpx', 'dy', 'dpy',
            'x_co', 'px_co', 'y_co', 'py_co',
            'zeta_co', 'delta_co',
            'gemitt_x', 'gemitt_y',
            'sigma_z', 'sigma_delta',
            'n_scattering_events', 'temperature', 'energy_loss_cut',
            'interaction_rate', 'total_rate', 'mean_free_path', 'seed',
        }

        unknown = set(kwargs) - config_allowed
        if unknown:
            bad = ", ".join(sorted(unknown))
            raise KeyError(f"Unsupported configure() keys: {bad}")

        n_scattering_events = kwargs.pop('n_scattering_events', None)
        if n_scattering_events is not None:
            self.n_scattering_events = _resolve_n_scattering_events(
                n_scattering_events=n_scattering_events,
                default=self.n_scattering_events)

        for kk, vv in kwargs.items():
            setattr(self, kk, vv)

    def _generate_primaries(self, n, rng):
        """
        Draw macro-particles from the local matched Gaussian distribution.

        Betatron and dispersive contributions are generated relative to the
        closed orbit, which is then added so that the returned coordinates are
        the physical ones. The interaction is applied afterwards, on the
        physical momentum of each macro-particle.

        Parameters
        ----------
        n : int
            Number of macro-particles to generate.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        x, px, y, py, zeta, delta : ndarray
            Physical coordinates of the generated macro-particles.
        """
        u = rng.standard_normal((6, n))

        delta = self.sigma_delta*u[5]
        zeta = self.sigma_z*u[4]

        x = np.sqrt(self.gemitt_x*self.betx)*u[0] + self.dx*delta
        px = (np.sqrt(self.gemitt_x/self.betx)*(u[1] - self.alfx*u[0])
              + self.dpx*delta)
        y = np.sqrt(self.gemitt_y*self.bety)*u[2] + self.dy*delta
        py = (np.sqrt(self.gemitt_y/self.bety)*(u[3] - self.alfy*u[2])
              + self.dpy*delta)

        return (x + self.x_co, px + self.px_co,
                y + self.y_co, py + self.py_co,
                zeta + self.zeta_co, delta + self.delta_co)

    def _scatter_particles(self, particles, n):
        """
        Scatter each particle once on a thermal photon.

        The particles must have their random-number generator initialised;
        their momenta are modified in place by the C kernel
        ``ThermalPhotonMarker_scatter``.

        Parameters
        ----------
        particles : xtrack.Particles
            Particles to scatter, on the context of the marker, with
            particle ids ``0 ... n - 1``.
        n : int
            Number of particles to scatter.

        Returns
        -------
        record : dict
            Arrays, indexed by particle id, of the quantities recorded for
            each event: ``energy_loss``, ``photon_energy`` [eV], ``theta``
            [rad], ``incident_photon_energy`` [eV], ``incident_cos_psi``,
            ``cos_theta_rest`` and ``n_trials``.
        """
        if not (self.temperature > 0.0 and 0.0 < self.energy_loss_cut < 1.0):
            raise ValueError(
                "The marker needs a positive `temperature` and an "
                "`energy_loss_cut` in (0, 1) to scatter particles.")

        context = self._buffer.context
        record = context.zeros(shape=(_RECORD_SIZE*n,), dtype=np.float64)
        self._scatter(particles=particles, record=record)
        record = context.nparray_from_context_array(record).reshape(
            n, _RECORD_SIZE)

        n_trials = record[:, _RECORD_COLUMNS.index('n_trials')]
        if np.any(n_trials <= 0):
            raise RuntimeError(
                "Rejection sampling of the thermal photons did not converge "
                f"for {int(np.sum(n_trials <= 0))} of {n} particles.")

        return {kk: record[:, ii].copy()
                for ii, kk in enumerate(_RECORD_COLUMNS)}

    def track(self, particles):
        """
        Track particles through the element without applying a kick.

        The scatterings on the thermal photons are generated by
        :meth:`xcoll.ThermalPhotonGenerator.generate`; during tracking this
        element is a passive marker.

        Parameters
        ----------
        particles : xtrack.Particles
            Particles tracked through the passive marker element.

        Returns
        -------
        None
        """
        super().track(particles)
