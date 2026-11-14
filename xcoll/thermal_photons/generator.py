# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from functools import lru_cache
from warnings import warn

import numpy as np
from scipy import constants as sc

import xtrack as xt

from ..beam_elements.beamgas import _resolve_n_scattering_events
from ..beam_elements.thermal_photons import ThermalPhotonMarker
# The marker placement, the random streams and the seeds work as for the
# beam-gas markers
from ..beamgas.generator import (_S_TOL, _free_names, _is_free,
                                 _is_sequence, _per_marker, _resolve_seed,
                                 _stream)
from .analysis import (ThermalPhotonResult, _cut_scale_scan, _event_losses,
                       _lost, _sample_sums, _tracked)
from .cross_sections import ThermalComptonCalculator, photon_density

# Resolution of ThermalPhotonResult.cutoff_scan
_CUTOFF_SCAN_BINS_PER_DECADE = 40
# Warn when more than this fraction of the loss rate comes from events with
# an energy loss within this factor of the cut
_NEAR_CUT_FACTOR = 1.1
_NEAR_CUT_WARN_FRACTION = 1e-3
# Markers whose section contributes less than this fraction of the
# interaction rate of the line are not generated
_NEGLIGIBLE_RATE_FRACTION = 1e-12

# Gauss-Legendre rule on [0, 1] for the integration of the interaction rate
# over a stretch of linearly varying temperature
_GL_X, _GL_W = np.polynomial.legendre.leggauss(6)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


def _beam_particle(particle_ref):
    """
    Mass and charge of the beam particles, checked to be charged.

    Parameters
    ----------
    particle_ref : xtrack.Particles
        Reference particle of the line.

    Returns
    -------
    mass0 : float
        Rest mass [eV].
    q0 : float
        Charge [e].
    """
    mass0 = float(particle_ref.mass0)
    q0 = float(particle_ref.q0)
    if q0 == 0:
        raise ValueError(
            "The reference particle is neutral: only charged particles "
            "scatter on thermal photons in this model.")
    return mass0, q0


@lru_cache(maxsize=None)
def _effective_xsec(p0c, mass0, q0, temperature, energy_loss_cut):
    """
    Effective cross section for scattering on blackbody photons.

    It depends only on these parameters, so it is computed once and shared by
    all the markers.

    Parameters
    ----------
    p0c : float
        Reference momentum [eV].
    mass0 : float
        Rest mass of the beam particles [eV].
    q0 : float
        Charge of the beam particles [e].
    temperature : float
        Temperature of the photon gas [K].
    energy_loss_cut : float
        Lowest relative energy loss that is counted.

    Returns
    -------
    xsec : float
        Cross section [m^2] such that the scattering rate of one particle is
        ``c * photon_density(temperature) * xsec``.
    """
    return ThermalComptonCalculator(
        p0c, temperature, mass0=mass0, q0=q0,
        energy_loss_cut=energy_loss_cut).compute_xsec()


def _check_configured(name, marker, particle_ref):
    """
    Check that a marker is configured for the reference particle of the line.
    """
    if not marker.p0c > 0:
        raise ValueError(
            f"ThermalPhotonMarker '{name}' is not configured: call "
            "`line.xcoll.thermal_photons.configure(...)` first.")
    p0c = particle_ref.p0c[0]
    if abs(marker.p0c - p0c) > 1e-10 * p0c:
        raise ValueError(
            f"ThermalPhotonMarker '{name}' is configured for p0c = "
            f"{marker.p0c:.10g} eV, but the reference particle of the line "
            f"has p0c = {p0c:.10g} eV: configure the markers again.")


def _check_profile(table, column, name, *, positive):
    """
    Validate a table giving a quantity as a function of ``s``.

    Parameters
    ----------
    table : xtrack.Table
        Table with a column ``s`` and a column ``column``.
    column : str
        Name of the column holding the quantity.
    name : str
        Name of the argument, for the error messages.
    positive : bool
        Whether the quantity must be strictly positive.

    Returns
    -------
    s, values : ndarray
        Positions [m] and values of the quantity.
    """
    if not isinstance(table, xt.Table):
        raise TypeError(f"`{name}` must be an `xt.Table` object.")
    missing = {'s', column} - set(table._col_names)
    if missing:
        raise ValueError(f"`{name}` missing columns: {sorted(missing)}")

    arrays = []
    for cc in ('s', column):
        try:
            vals = np.asarray(table[cc], dtype=float)
        except (TypeError, ValueError):
            raise TypeError(
                f"`{cc}` column must be numeric (cannot coerce to float).")
        if np.isnan(vals).any():
            bad = list(np.where(np.isnan(vals))[0][:5])
            raise ValueError(f"`{cc}` contains NaN at indices {bad}.")
        if np.isinf(vals).any():
            bad = list(np.where(np.isinf(vals))[0][:5])
            raise ValueError(f"`{cc}` contains inf at indices {bad}.")
        arrays.append(vals)
    s, values = arrays
    if len(s) == 0:
        raise ValueError(f"`{name}` is empty.")
    if np.any(np.diff(s) < 0):
        raise ValueError(f"`{name}.s` must be non-decreasing.")
    if positive and np.any(values <= 0):
        bad = list(np.where(values <= 0)[0][:5])
        raise ValueError(
            f"`{column}` must be positive (got non-positive values at "
            f"indices {bad}).")
    return s, values


def _section_nodes(s_table, values, s_start, s_end):
    """
    Break points of a piecewise-linear profile inside a section.

    Parameters
    ----------
    s_table, values : ndarray
        Tabulated positions [m] and values of the profile.
    s_start, s_end : float
        Limits of the section [m].

    Returns
    -------
    s_nodes, value_nodes : ndarray
        The two section edges and the tabulated positions between them, with
        the linearly interpolated values of the profile.
    """
    s_inner = s_table[(s_table > s_start) & (s_table < s_end)]
    s_nodes = np.concatenate(([s_start], s_inner, [s_end]))
    return s_nodes, np.interp(s_nodes, s_table, values)


def _integrated_rates(s_nodes, temperature_nodes, *, p0c, mass0, q0,
                      energy_loss_cut):
    """
    Integrate the scattering probability per unit time over a section.

    The temperature profile is piecewise linear in ``s``. On each stretch the
    integrand ``n_gamma(T) sigma_eff(T)`` is evaluated at the temperature of
    Gauss-Legendre nodes; a stretch of constant temperature needs a single
    evaluation.

    Parameters
    ----------
    s_nodes, temperature_nodes : ndarray
        Break points of the temperature profile in the section [m], [K].
    p0c, mass0, q0 : float
        Momentum [eV], rest mass [eV] and charge [e] of the beam particles.
    energy_loss_cut : float
        Lowest relative energy loss that is generated.

    Returns
    -------
    integral : float
        ``int n_gamma sigma_eff ds`` for the energy losses above the cut
        (dimensionless: scattering probability per ``c dt = ds``).
    integral_total : float
        The same for all the energy losses.
    temperature : float
        Mean temperature of the section [K], weighted with the integrand
        above the cut (with the path length if the integrand vanishes).
    """
    def integrand(temperature, cut):
        return float(photon_density(temperature)) * _effective_xsec(
            p0c, mass0, q0, float(temperature), cut)

    integral = 0.0
    integral_total = 0.0
    weighted_temperature = 0.0
    length_temperature = 0.0
    for s0, s1, t0, t1 in zip(s_nodes[:-1], s_nodes[1:],
                              temperature_nodes[:-1],
                              temperature_nodes[1:]):
        ds = s1 - s0
        if ds <= 0:
            continue
        if t0 == t1:
            nodes, weights = np.array([t0]), np.array([1.0])
        else:
            nodes, weights = t0 + (t1 - t0) * _GL_X, _GL_W
        for tt, ww in zip(nodes, weights):
            gg = integrand(tt, energy_loss_cut)
            integral += ww * ds * gg
            integral_total += ww * ds * integrand(tt, 0.0)
            weighted_temperature += ww * ds * gg * tt
        length_temperature += 0.5 * (t0 + t1) * ds

    length = s_nodes[-1] - s_nodes[0]
    if integral > 0:
        temperature = weighted_temperature / integral
    elif length > 0:
        temperature = length_temperature / length
    else:
        temperature = float(temperature_nodes[-1])
    return integral, integral_total, float(temperature)


def _generate_marker(name, marker, *, particle_ref, at_element, s, rng,
                     weight_scale=1.0):
    """
    Generate the particles scattered on thermal photons of a marker.

    Parameters
    ----------
    name : str
        Name of the marker.
    marker : ThermalPhotonMarker
        Configured marker.
    particle_ref : xtrack.Particles
        Reference particle, defining the species of the particles.
    at_element : int
        Index of the marker in the line.
    s : float
        Position of the marker in the line [m].
    rng : numpy.random.Generator
        Random-number generator.
    weight_scale : float, optional
        Factor applied to the weights of the particles.

    Returns
    -------
    particles : xtrack.Particles
        Particles that have undergone one scattering on a thermal photon,
        with a capacity of four times their number to leave room for
        secondaries.
    log : xtrack.Table
        Record of the interactions, one row per particle: ``name``, ``s``,
        ``particle_id``, ``energy_loss``, ``photon_energy``, ``theta``,
        ``incident_photon_energy`` and ``weight``.
    """
    n = int(marker.n_scattering_events)
    if n <= 0:
        raise ValueError(
            f"ThermalPhotonMarker '{name}': `n_scattering_events` must be "
            "positive to generate scattered particles.")
    if not marker.interaction_rate > 0:
        raise ValueError(
            f"No thermal-photon interactions possible at marker '{name}': "
            "the interaction rate of its section is zero.")

    x, px, y, py, zeta, delta = marker._generate_primaries(n, rng)
    weight = np.full(n, marker.interaction_rate / n * weight_scale)

    # Leave room for the secondaries that collimators may produce
    particles = xt.Particles(
        _context=marker._buffer.context,
        _capacity=4*n,
        p0c=marker.p0c,
        mass0=particle_ref.mass0,
        q0=particle_ref.q0,
        pdg_id=particle_ref.pdg_id,
        x=x, px=px, y=y, py=py, zeta=zeta, delta=delta,
        weight=weight,
        s=s,
    )
    particles.at_element = at_element

    # Each particle has its own random-number generator, seeded from the
    # stream of the marker: the scattering does not depend on the number of
    # threads, and the generators are ready for the tracking
    particles._init_random_number_generator(
        seeds=rng.integers(1, 2**32 - 1, size=4*n, dtype=np.uint32))
    record = marker._scatter_particles(particles, n)

    log = xt.Table({
        'name': np.full(n, name, dtype=object),
        's': np.full(n, float(s)),
        'particle_id': np.arange(n, dtype=np.int64),
        'energy_loss': record['energy_loss'],
        'photon_energy': record['photon_energy'],
        'theta': record['theta'],
        'incident_photon_energy': record['incident_photon_energy'],
        'weight': weight,
    })
    return particles, log


def _log_table(name, replica, log):
    """
    Columns of the record of the interactions of a sample.
    """
    particle_id = np.asarray(log['particle_id'], dtype=np.int64)
    n = len(particle_id)
    table = {
        'name': np.full(n, name, dtype=object),
        'replica': np.full(n, replica),
        's': np.asarray(log['s'], dtype=float),
        'particle_id': particle_id,
    }
    for kk in ('energy_loss', 'photon_energy', 'theta',
               'incident_photon_energy', 'weight'):
        table[kk] = np.asarray(log[kk], dtype=float)
    return table


def _warn_on_truncation(cut_scale, contribution):
    """
    Warn when the energy-loss cut visibly truncates the loss rate.

    Parameters
    ----------
    cut_scale : ndarray
        Energy loss of each event divided by the cut of its marker.
    contribution : ndarray
        Loss-rate contribution of each event, i.e. its weight if lost and
        zero otherwise [1/s].

    Returns
    -------
    None
    """
    rate = float(np.sum(contribution))
    if rate <= 0:
        return

    near_cut = float(np.sum(contribution[cut_scale < _NEAR_CUT_FACTOR])) / rate
    if near_cut > _NEAR_CUT_WARN_FRACTION:
        warn(f"{100*near_cut:.2g}% of the tracked loss rate comes from "
             f"events with an energy loss within a factor "
             f"{_NEAR_CUT_FACTOR} of the energy-loss cut. The cut lies "
             f"inside the range of energy losses that cause losses, so the "
             f"losses from below it are missing and the lifetime is "
             f"overestimated. Lower `energy_loss_cut` (or "
             f"`local_momentum_acceptance_scale`), and check that "
             f"`ThermalPhotonResult.cutoff_scan.loss_probability` vanishes "
             f"in its first rows.", stacklevel=3)


class ThermalPhotonGenerator:
    """
    Simulation of the scattering of the beam on the thermal photons of the
    vacuum chamber of a line, based on the
    :class:`xcoll.ThermalPhotonMarker` elements installed in it.

    The walls of a vacuum chamber at temperature ``T`` fill it with blackbody
    radiation, about ``5e14`` photons per m^3 at room temperature with a mean
    energy of 0.07 eV. A beam particle that Compton-scatters on one of them
    loses a fraction of its energy of the order of ``4 gamma k_B T / (m c^2)``
    (about 1.5 % on average for 45.6 GeV electrons), and is lost if this
    exceeds the momentum acceptance. The process sets the upper limit of the
    beam lifetime of high-energy e+ e- rings and produces off-momentum
    particles all around the ring.

    An instance bound to a line is available as
    ``line.xcoll.thermal_photons``. The generator holds no state other than
    the line: the configuration is stored in the markers, so that a line
    saved to JSON after it has been configured generates the same particles
    once loaded again.

    Parameters
    ----------
    line : xtrack.Line
        Line containing, or receiving, the thermal-photon markers.

    Notes
    -----
    **Assumptions and limitations**

    * The radiation is an isotropic blackbody field at the local wall
      temperature. This is exact inside a closed isothermal cavity, whatever
      the shape and the material of the chamber; it is an approximation near
      transitions between sections at different temperatures and near hot
      spots. Synchrotron radiation is not thermal and is not included.
    * The photon density does not depend on the optics or on the beam size:
      the optics only enter through the distribution of the particles before
      the scattering.
    * The Klein-Nishina cross section holds for point-like spin-1/2
      particles (electrons, positrons, muons). For hadrons and fully
      stripped ions only its Thomson limit is model independent, which is
      the only regime reached in practice, and where the process is
      negligible anyway (the cross section scales as ``(q^2 m_e / M)^2``).
      Partially stripped ions are not described.
    * Single scattering: each generated particle scatters once.

    References
    ----------
    .. [1] H. Burkhardt, "Monte Carlo simulation of scattering of beam
       particles and thermal photons", CERN SL/Note 93-73 (OP) (1993).
    .. [2] V. I. Telnov, "Scattering of electrons on thermal radiation
       photons in electron-positron storage rings", Nucl. Instrum. Meth. A
       **260**, 304 (1987).
    .. [3] A. Di Domenico, "Inverse Compton scattering of thermal radiation
       at LEP and LEP-200", Part. Accel. **39**, 137 (1992).
    .. [4] L. S. Brown and R. S. Steinke, "Compton scattering on black body
       photons", Am. J. Phys. **65**, 304 (1997), arXiv:hep-ph/9606303.
    """

    def __init__(self, line):
        if not isinstance(line, xt.Line):
            raise TypeError(
                f"`line` must be an xtrack.Line (got {type(line).__name__}).")
        self._line = line

    @property
    def line(self):
        """
        Line of the thermal-photon simulation.

        Returns
        -------
        line : xtrack.Line
            Line bound to this generator.
        """
        return self._line

    @property
    def marker_names(self):
        """
        Names of the thermal-photon markers of the line.

        Returns
        -------
        names : list of str
            Names of the :class:`xcoll.ThermalPhotonMarker` elements, in line
            order.
        """
        element_dict = self.line.element_dict
        names = [nn for nn in self.line.element_names
                 if isinstance(element_dict[nn], ThermalPhotonMarker)]
        if len(set(names)) != len(names):
            repeated = sorted({nn for nn in names if names.count(nn) > 1})
            raise ValueError(
                f"Thermal-photon markers {repeated} appear more than once in "
                "the line: each marker must be a distinct element.")
        return names

    def install_markers(self, at, *, from_=None, anchor=None,
                        from_anchor=None, names=None,
                        prefix='thermal_photons', close_ring=True):
        """
        Install thermal-photon markers in the line.

        Each marker is a new :class:`xcoll.ThermalPhotonMarker` element,
        placed with ``env.place`` and inserted with ``line.insert``: ``at``,
        ``from_``, ``anchor`` and ``from_anchor`` have the same meaning as
        there. Installing the markers by hand in the same way is equivalent.

        Each marker represents the section of the line from the previous
        marker (from the start of the line for the first one) to itself, and
        its particles are generated with the optics and the momentum
        acceptance at the marker: markers are best placed densely where the
        optics, the momentum acceptance or the temperature change.

        Parameters
        ----------
        at : float, str, xdeps reference, or sequence of them
            Position of each marker, evaluated at installation: an ``s``
            position [m], an expression of the environment variables, the
            name of an element (to place the marker at its centre), or a
            reference ``'name@anchor'`` with ``anchor`` one of ``'start'``,
            ``'center'`` and ``'end'``. With ``from_``, the offset from the
            reference element.
        from_ : str or sequence of str, optional
            Element from which ``at`` is measured. ``'name@anchor'`` selects
            the anchor of the element.
        anchor : {'start', 'center', 'end'} or sequence of them, optional
            Anchor of the marker placed at ``at``. Markers are thin, so this
            only matters to follow the conventions of ``env.place``.
        from_anchor : {'start', 'center', 'end'} or sequence of them, optional
            Anchor of ``from_`` from which ``at`` is measured.
        names : str or sequence of str, optional
            Names of the new markers. By default ``f'{prefix}.{i}'``, with the
            lowest indices whose names are free in the environment.
        prefix : str, optional
            Prefix of the default names.
        close_ring : bool, optional
            If ``True`` (default), a marker is added at the end of the line
            when, after the insertion, the last marker is not there: otherwise
            the part of the line after it would belong to no section and
            :meth:`configure` refuses the line.

        Returns
        -------
        names : list of str
            Names of the installed markers, the closing one included.
        """
        line = self.line
        env = line.env

        at = list(at) if _is_sequence(at) else [at]
        n_markers = len(at)
        if n_markers == 0:
            raise ValueError("`at` must give at least one position.")
        from_ = _per_marker(from_, n_markers, 'from_')
        anchor = _per_marker(anchor, n_markers, 'anchor')
        from_anchor = _per_marker(from_anchor, n_markers, 'from_anchor')

        if names is None:
            names = _free_names(env, prefix, n_markers)
        else:
            names = _per_marker(
                [names] if isinstance(names, str) else names,
                n_markers, 'names')
            if len(set(names)) != len(names):
                raise ValueError("`names` contains repeated names.")
            for nn in names:
                if not _is_free(env, nn):
                    raise ValueError(
                        f"The name '{nn}' is already used in the environment.")

        places = []
        for nn, aa, ff, an, fa in zip(names, at, from_, anchor, from_anchor):
            env.elements[nn] = ThermalPhotonMarker()
            places.append(env.place(
                nn, at=aa, from_=ff, anchor=an, from_anchor=fa))
        line.insert(places)

        installed = list(names)
        if close_ring and not self._last_marker_at_end():
            end_name, = _free_names(env, prefix, 1)
            env.elements[end_name] = ThermalPhotonMarker()
            line.insert(end_name, at=line.get_length())
            installed.append(end_name)

        return installed

    def configure(self, *, temperature=None, energy_loss_cut=None,
                  local_momentum_acceptance=None,
                  local_momentum_acceptance_scale=0.85, sigma_z=None,
                  sigma_delta=None, n_scattering_events=None,
                  nemitt_x=None, nemitt_y=None, gemitt_x=None, gemitt_y=None,
                  bunch_intensity=1.0, seed=None, twiss=None, method='6d',
                  at=None, verbose=True, **kwargs):
        """
        Configure the thermal-photon markers of the line.

        Integrates the scattering rate over the section of each marker, and
        stores on the marker the resulting interaction rate, the temperature,
        the energy-loss cut, the local optics and closed orbit, the beam
        parameters and the Monte Carlo settings. The section of a marker goes
        from the previous marker to it, and the section of the first one from
        ``s = 0``: the last marker must sit at the end of the line, which
        :meth:`install_markers` ensures.

        Only the scatterings with a relative energy loss above a cut are
        generated, and the interaction rate is that of these scatterings: the
        particles that lose less stay inside the momentum acceptance and are
        of no interest. The cut is given either directly, with
        ``energy_loss_cut``, or through the local momentum acceptance, as
        ``local_momentum_acceptance_scale`` times the smallest
        ``|delta_neg|`` over the section of the marker. It must lie below
        the smallest energy loss that can cause a loss, which
        ``ThermalPhotonResult.cutoff_scan`` allows to check after tracking.

        The whole configuration is stored on the markers: a line saved to JSON
        after this call generates the same particles once loaded again.

        Parameters
        ----------
        temperature : float or xtrack.Table
            Temperature of the vacuum chamber [K]: a number for a uniform
            temperature, or a table with the columns ``s`` [m] and
            ``temperature`` [K] for a profile, interpolated linearly in
            ``s`` (constant beyond its ends). The photon density goes like
            the third power of the temperature.
        energy_loss_cut : float, optional
            Lowest relative energy loss generated, in ``(0, 1)``, the same
            for all the markers. Mutually exclusive with
            ``local_momentum_acceptance``.
        local_momentum_acceptance : xtrack.Table, optional
            Table with ``s`` and ``delta_neg`` columns, e.g. from
            ``line.get_local_momentum_acceptance(...)``, interpolated
            linearly in ``s``. It is not modified. Only the negative side
            matters, since the scattering lowers the energy. Mutually
            exclusive with ``energy_loss_cut``.
        local_momentum_acceptance_scale : float, optional
            Factor applied to ``|delta_neg|`` to obtain the energy-loss cut.
            The margin has to cover the momentum spread and the betatron
            amplitudes of the beam. Default 0.85.
        sigma_z : float
            RMS bunch length [m].
        sigma_delta : float
            RMS relative momentum spread.
        n_scattering_events : int
            Number of Monte Carlo interactions generated at each marker.
        nemitt_x, nemitt_y : float, optional
            Normalized emittances. Mutually exclusive with ``gemitt_x`` and
            ``gemitt_y``.
        gemitt_x, gemitt_y : float, optional
            Geometric emittances. Mutually exclusive with ``nemitt_x`` and
            ``nemitt_y``.
        bunch_intensity : float, optional
            Number of particles in the bunch. The interaction and loss rates
            are proportional to it; the lifetime is not. Default 1, which
            makes the rates per stored particle.
        seed : int or None, optional
            Seed of the random-number streams of the markers. If ``None``, a
            seed is drawn now and stored, so that the generation stays
            reproducible.
        twiss : xtrack.TwissTable or None, optional
            Twiss table of the line. If ``None``, it is computed with
            ``line.twiss(method=method, **kwargs)``.
        method : {'6d', '4d'}, optional
            Twiss method, used if ``twiss`` is ``None``.
        at : str, sequence of str, or None, optional
            Markers to configure. If ``None`` (default), all the markers. The
            sections are always defined by all the markers of the line.
        verbose : bool, optional
            Whether to print the progress of the configuration.
        **kwargs
            Keyword arguments forwarded to ``line.twiss`` if ``twiss`` is
            ``None``.

        Returns
        -------
        None
        """
        line = self.line
        if getattr(line, 'particle_ref', None) is None:
            raise ValueError("`line` must have a `particle_ref`.")
        for name, value in (('temperature', temperature),
                            ('sigma_z', sigma_z),
                            ('sigma_delta', sigma_delta),
                            ('n_scattering_events', n_scattering_events)):
            if value is None:
                raise ValueError(f"`{name}` is required.")
        particle_ref = line.particle_ref
        mass0, q0 = _beam_particle(particle_ref)
        n_scattering_events = _resolve_n_scattering_events(
            n_scattering_events=n_scattering_events, default=None)
        bunch_intensity = float(bunch_intensity)
        if bunch_intensity < 0:
            raise ValueError("`bunch_intensity` must be non-negative.")
        seed = _resolve_seed(seed)

        # Temperature profile, as break points of a piecewise-linear function
        if isinstance(temperature, xt.Table):
            s_temperature, temperature_values = _check_profile(
                temperature, 'temperature', 'temperature', positive=True)
        else:
            try:
                temperature_values = np.array([float(temperature)])
            except (TypeError, ValueError):
                raise TypeError(
                    "`temperature` must be a number or an `xt.Table` with "
                    "`s` and `temperature` columns.")
            if not temperature_values[0] > 0 or np.isinf(
                    temperature_values[0]):
                raise ValueError("`temperature` must be positive and finite.")
            s_temperature = np.array([0.0])

        # Energy-loss cut
        if (energy_loss_cut is None) == (local_momentum_acceptance is None):
            raise ValueError(
                "Provide either `energy_loss_cut` OR "
                "`local_momentum_acceptance` (exactly one of them).")
        if energy_loss_cut is not None:
            energy_loss_cut = float(energy_loss_cut)
            if not 0.0 < energy_loss_cut < 1.0:
                raise ValueError("`energy_loss_cut` must be in (0, 1).")
        else:
            s_lma, delta_neg = _check_profile(
                local_momentum_acceptance, 'delta_neg',
                'local_momentum_acceptance', positive=False)
            scale = float(local_momentum_acceptance_scale)
            if not scale > 0:
                raise ValueError(
                    "`local_momentum_acceptance_scale` must be positive.")
            acceptance = np.abs(delta_neg)

        nemitt_given = nemitt_x is not None and nemitt_y is not None
        gemitt_given = gemitt_x is not None and gemitt_y is not None
        if nemitt_given and gemitt_given:
            raise ValueError(
                "Provide either normalised emittances (nemitt_x, nemitt_y) "
                "OR geometric emittances (gemitt_x, gemitt_y), not both.")
        if not (nemitt_given or gemitt_given):
            raise ValueError(
                "You must provide either both normalised emittances "
                "(nemitt_x, nemitt_y) OR both geometric emittances "
                "(gemitt_x, gemitt_y).")
        if nemitt_given:
            beta_gamma = particle_ref.beta0[0] * particle_ref.gamma0[0]
            gemitt_x = float(nemitt_x / beta_gamma)
            gemitt_y = float(nemitt_y / beta_gamma)

        # Check the markers before the (long) twiss
        markers = self._markers()
        self._check_sections_cover_line()
        selected = markers if at is None else self._markers(at)

        if twiss is None:
            twiss = line.twiss(method=method, **kwargs)

        p0c = float(particle_ref.p0c[0])
        beta0 = float(particle_ref.beta0[0])
        # A particle scatters at the rate c*n*sigma during the time
        # ds/(beta0*c) that it spends in the section, at each turn:
        # rate = N * f_rev * int n sigma ds / beta0 = N * c/C * int n sigma ds
        rate_per_integral = bunch_intensity * sc.c / line.get_length()
        tab = line.get_table()
        index = {nn: ii for ii, nn in enumerate(markers)}

        for nn in selected:
            s = float(tab['s', nn])
            ii = index[nn]
            s_start = float(tab['s', markers[ii - 1]]) if ii > 0 else 0.0

            if energy_loss_cut is not None:
                cut = energy_loss_cut
            else:
                # Smallest momentum acceptance over the section
                cut = scale * float(np.min(_section_nodes(
                    s_lma, acceptance, s_start, s)[1]))
                if not 0.0 < cut < 1.0:
                    raise ValueError(
                        f"ThermalPhotonMarker '{nn}': the scaled local "
                        f"momentum acceptance gives an energy-loss cut of "
                        f"{cut:.4g}, which is not in (0, 1).")

            s_nodes, temperature_nodes = _section_nodes(
                s_temperature, temperature_values, s_start, s)
            integral, integral_total, temperature_marker = _integrated_rates(
                s_nodes, temperature_nodes, p0c=p0c, mass0=mass0, q0=q0,
                energy_loss_cut=cut)
            inverse_mfp = float(photon_density(temperature_marker)) / beta0 * (
                _effective_xsec(p0c, mass0, q0, temperature_marker, cut))

            if verbose:
                print(f'Configuring ThermalPhotonMarker {nn}')

            line.element_dict[nn]._configure(
                p0c=p0c,
                bunch_intensity=bunch_intensity,
                ds=s - s_start,
                alfx=twiss['alfx', nn], betx=twiss['betx', nn],
                alfy=twiss['alfy', nn], bety=twiss['bety', nn],
                dx=twiss['dx', nn], dpx=twiss['dpx', nn],
                dy=twiss['dy', nn], dpy=twiss['dpy', nn],
                x_co=twiss['x', nn], px_co=twiss['px', nn],
                y_co=twiss['y', nn], py_co=twiss['py', nn],
                zeta_co=twiss['zeta', nn], delta_co=twiss['delta', nn],
                gemitt_x=gemitt_x, gemitt_y=gemitt_y,
                sigma_z=float(sigma_z), sigma_delta=float(sigma_delta),
                n_scattering_events=n_scattering_events,
                temperature=temperature_marker,
                energy_loss_cut=cut,
                interaction_rate=float(rate_per_integral * integral),
                total_rate=float(rate_per_integral * integral_total),
                mean_free_path=(np.inf if inverse_mfp <= 0
                                else 1.0 / inverse_mfp),
                seed=seed,
            )

    def generate(self, at=None, *, replica=0, n_replicas=1, seed=None,
                 return_log=False):
        """
        Generate particles scattered on thermal photons at the markers.

        At each marker, ``n_scattering_events`` particles are drawn from the
        local matched Gaussian beam and each of them scatters once on a
        thermal photon, with a relative energy loss above the cut of the
        marker. The weights of the particles of a marker are all equal and
        sum to the interaction rate of its section [1/s], divided by
        ``n_replicas``.

        The particles of a marker are tracked from the marker itself, e.g.
        ``line.track(part, ele_start=name, ele_stop=name, num_turns=...)``,
        and the samples of different markers or replicas, generated in the
        same or in different jobs, can then be merged with
        :meth:`xtrack.Particles.merge` and given to :meth:`analyse`.

        Markers whose section contributes a negligible part (less than
        ``1e-12``) of the interaction rate of the line, e.g. in a cryogenic
        section, are skipped; asking for a single such marker is an error.

        Parameters
        ----------
        at : str, sequence of str, or None, optional
            Markers at which particles are generated. If ``None`` (default),
            all the markers of the line.
        replica : int, optional
            Index of the replica: replicas of a marker are independent
            samples of the same section, e.g. generated in different jobs.
            Default ``0``.
        n_replicas : int, optional
            Number of replicas of each marker that are merged. The weights are
            divided by it, so that the merged replicas represent the
            interaction rate once. Default ``1``.
        seed : int or None, optional
            Seed of the random-number streams. If ``None`` (default), the seed
            stored on each marker by :meth:`configure`. Each marker and
            replica has its own stream, derived from the seed, the name of the
            marker and the replica: the particles of a marker do not depend on
            the other markers that are generated, nor on the order.
        return_log : bool, optional
            If ``True``, also return the record of the interactions, needed by
            :meth:`analyse` for the per-event diagnostics. The record of the
            last generation is also kept, in memory, as the ``scatter_log`` of
            each marker.

        Returns
        -------
        particles : xtrack.Particles or dict
            If ``at`` is a string, the particles generated at that marker.
            Otherwise, a mapping ``{name: xtrack.Particles}`` in line order.
            The particles carry an initialised random-number generator.
        log : xtrack.Table or dict
            Only if ``return_log`` is ``True``: the record of the
            interactions, one row per particle, with columns ``name``, ``s``,
            ``particle_id``, ``energy_loss`` (relative energy loss
            ``y = 1 - k*'/k*``), ``photon_energy`` (scattered photon, [eV]),
            ``theta`` (angle of the scattered photon to the incoming
            particle, [rad]), ``incident_photon_energy`` (thermal photon,
            [eV]) and ``weight`` [1/s]; or a mapping ``{name: xtrack.Table}``.
            ``particle_id`` identifies the particle in the sample of its
            marker, and survives tracking.
        """
        line = self.line
        names = self._markers(at)

        if (isinstance(n_replicas, (bool, np.bool_))
                or not isinstance(n_replicas, (int, np.integer))
                or n_replicas < 1):
            raise ValueError("`n_replicas` must be a positive integer.")
        if (isinstance(replica, (bool, np.bool_))
                or not isinstance(replica, (int, np.integer))
                or not 0 <= replica < n_replicas):
            raise ValueError(
                f"`replica` must be an integer in [0, n_replicas) "
                f"(got {replica} with n_replicas={n_replicas}).")
        if seed is not None:
            seed = _resolve_seed(seed)

        particle_ref = line.particle_ref
        _beam_particle(particle_ref)
        markers = {nn: line.element_dict[nn] for nn in names}
        for nn, marker in markers.items():
            _check_configured(nn, marker, particle_ref)

        names = self._non_negligible(names)

        tab = line.get_table()
        element_index = {nn: ii for ii, nn in enumerate(line.element_names)}

        particles, logs = {}, {}
        for nn in names:
            marker = markers[nn]
            particles[nn], logs[nn] = _generate_marker(
                nn, marker,
                particle_ref=particle_ref,
                at_element=element_index[nn],
                s=tab['s', nn],
                rng=_stream(marker.seed if seed is None else seed,
                            nn, replica),
                weight_scale=1.0 / n_replicas,
            )
            marker.scatter_log = logs[nn]

        if isinstance(at, str):
            particles, logs = particles[at], logs[at]
        if return_log:
            return particles, logs
        return particles

    def analyse(self, particles=None, *, at=None, logs=None):
        """
        Analyse the thermal-photon interaction rates and the tracked
        particles.

        Without particles, the result contains the interaction rates
        configured on the markers. With the particles generated at the
        markers and tracked, it also contains the loss rate: the total weight
        of the scattered particles that are lost (any ``state <= 0``, e.g. on
        an aperture or in a collimator). Secondaries produced while tracking
        carry the weight of their parent and are not counted.

        ``particles`` can be the sample of one marker, the particles of
        several markers merged with :meth:`xtrack.Particles.merge`, possibly
        coming from separate jobs, or the samples of each marker separately.
        The particles do not record the marker at which they were generated,
        so the per-marker counts and the Monte Carlo error of the loss rate
        are only available for the samples given separately, or for the
        particles of a single marker selected with ``at``. The per-event
        diagnostics (``cutoff_scan``, ``interaction_log`` and the warning on
        the energy-loss cut) also need the records of the interactions,
        returned by ``generate(..., return_log=True)``, given as ``logs`` with
        the same structure as ``particles``.

        Parameters
        ----------
        particles : xtrack.Particles, dict, or None, optional
            The particles: ``None`` (only the configured rates), an
            ``xtrack.Particles`` object, or a mapping ``{name: particles}``
            from marker names to the ``xtrack.Particles`` of the marker or a
            list of its replicas.
        at : str, sequence of str, or None, optional
            Markers whose particles are given, when ``particles`` is not a
            mapping. If ``None`` (default), all the markers of the line.
        logs : dict or None, optional
            Records of the interactions, as a mapping ``{name: log}`` or
            ``{name: [logs of the replicas]}`` matching ``particles``. Each
            record can be an ``xtrack.Table``, a ``pandas.DataFrame`` or a
            dict of arrays.

        Returns
        -------
        result : ThermalPhotonResult
            Interaction rates and, with tracked particles, the loss rate and
            the thermal-photon lifetime.
        """
        line = self.line

        samples = None
        if isinstance(particles, dict):
            if at is not None:
                raise ValueError(
                    "`at` cannot be given with a mapping of particles: its "
                    "keys select the markers.")
            names = self._markers(list(particles))
            samples = {}
            for nn in names:
                replicas = particles[nn]
                if not isinstance(replicas, (list, tuple)):
                    replicas = [replicas]
                if len(replicas) == 0:
                    raise ValueError(f"No particles given for marker '{nn}'.")
                samples[nn] = list(replicas)
            all_particles = [pp for vv in samples.values() for pp in vv]
        else:
            names = self._markers(at)
            all_particles = [] if particles is None else [particles]
        for pp in all_particles:
            if not isinstance(pp, xt.Particles):
                raise TypeError(
                    "The particles must be `xtrack.Particles` objects "
                    f"(got {type(pp).__name__}).")

        sample_logs = None
        if logs is not None:
            if samples is None:
                raise ValueError(
                    "`logs` need the particles given marker by marker, as a "
                    "mapping `{name: particles}`: the merged particles cannot "
                    "be matched to the records.")
            if set(logs) != set(samples):
                raise ValueError(
                    "`logs` must have the same markers as `particles`.")
            sample_logs = {}
            for nn in names:
                replica_logs = logs[nn]
                if isinstance(replica_logs, (list, tuple)):
                    replica_logs = list(replica_logs)
                else:
                    replica_logs = [replica_logs]
                if len(replica_logs) != len(samples[nn]):
                    raise ValueError(
                        f"Marker '{nn}': {len(samples[nn])} samples but "
                        f"{len(replica_logs)} records of interactions.")
                sample_logs[nn] = replica_logs

        markers = {nn: line.element_dict[nn] for nn in names}
        for nn, marker in markers.items():
            _check_configured(nn, marker, line.particle_ref)
        bunch_intensities = {float(mm.bunch_intensity)
                             for mm in markers.values()}
        if len(bunch_intensities) > 1:
            raise ValueError(
                "The markers are configured with different bunch "
                f"intensities: {sorted(bunch_intensities)}.")
        first = markers[names[0]]
        bunch_intensity = float(first.bunch_intensity)

        f_rev = (line.particle_ref.beta0[0] * sc.c / line.get_length())
        tab = line.get_table()
        mean_free_path = np.array([mm.mean_free_path
                                   for mm in markers.values()])
        with np.errstate(divide='ignore'):
            local_rate = bunch_intensity * f_rev / mean_free_path
        data = {
            'name': np.array(names),
            's': np.array([float(tab['s', nn]) for nn in names]),
            'ds': np.array([mm.ds for mm in markers.values()]),
            'temperature': np.array([mm.temperature
                                     for mm in markers.values()]),
            'energy_loss_cut': np.array([mm.energy_loss_cut
                                         for mm in markers.values()]),
            'mean_free_path': mean_free_path,
            'local_rate': local_rate,
            'interaction_rate': np.array([mm.interaction_rate
                                          for mm in markers.values()]),
            'total_rate': np.array([mm.total_rate
                                    for mm in markers.values()]),
        }
        rate_scattering = float(np.sum(data['interaction_rate']))
        rate_total = float(np.sum(data['total_rate']))

        tracked = any(_tracked(pp) for pp in all_particles)
        rate_generated = None
        rate_tracking = None
        rate_tracking_error = None
        cutoff_scan = None
        interaction_log = None

        if samples is not None:
            sums = {nn: [_sample_sums(pp) for pp in samples[nn]]
                    for nn in names}

            def per_marker(key):
                return np.array([sum(ss[key] for ss in sums[nn])
                                 for nn in names])

            data['n_replicas'] = np.array([len(samples[nn]) for nn in names])
            data['num_particles'] = per_marker('num_particles')
            data['sum_weight'] = per_marker('sum_weight')
            rate_generated = float(np.sum(data['sum_weight']))
            if tracked:
                data['num_lost_particles'] = per_marker('num_lost_particles')
                data['sum_lost_weight'] = per_marker('sum_lost_weight')
                variance = per_marker('lost_weight_variance')
                data['sum_lost_weight_error'] = np.sqrt(variance)
                rate_tracking = float(np.sum(data['sum_lost_weight']))
                rate_tracking_error = float(np.sqrt(np.sum(variance)))
        elif particles is not None:
            sums = _sample_sums(particles)
            rate_generated = sums['sum_weight']
            if tracked:
                rate_tracking = sums['sum_lost_weight']
            if len(names) == 1:
                # Events of a single marker are identically distributed,
                # also when replicas are merged
                data['num_particles'] = np.array([sums['num_particles']])
                data['sum_weight'] = np.array([sums['sum_weight']])
                if tracked:
                    data['num_lost_particles'] = np.array(
                        [sums['num_lost_particles']])
                    data['sum_lost_weight'] = np.array(
                        [sums['sum_lost_weight']])
                    rate_tracking_error = float(
                        np.sqrt(sums['lost_weight_variance']))

        if sample_logs is not None:
            cut_scale, weight, lost, i_sample, tables = [], [], [], [], []
            for nn in names:
                cut = float(markers[nn].energy_loss_cut)
                for rr, (pp, log) in enumerate(zip(samples[nn],
                                                   sample_logs[nn])):
                    xx, ww, ll = _event_losses(pp, log, 'energy_loss')
                    cut_scale.append(xx / cut)
                    weight.append(ww)
                    lost.append(ll)
                    i_sample.append(np.full(len(xx), len(tables)))
                    tables.append(_log_table(nn, rr, log))
            cut_scale, weight, lost, i_sample = (
                np.concatenate(vv)
                for vv in (cut_scale, weight, lost, i_sample))
            interaction_log = xt.Table({
                kk: np.concatenate([tt[kk] for tt in tables])
                for kk in tables[0]})
            if tracked:
                cutoff_scan = _cut_scale_scan(
                    cut_scale, weight, lost, i_sample, bunch_intensity,
                    _CUTOFF_SCAN_BINS_PER_DECADE)
                _warn_on_truncation(cut_scale, weight * lost)

        def lifetime(rate):
            return np.inf if rate == 0 else float(bunch_intensity / rate)

        lifetime_tracking = None
        lifetime_tracking_error = None
        if rate_tracking is not None:
            lifetime_tracking = lifetime(rate_tracking)
            if rate_tracking_error is not None and rate_tracking > 0:
                lifetime_tracking_error = float(
                    lifetime_tracking * rate_tracking_error / rate_tracking)

        return ThermalPhotonResult(
            element_names=names,
            local_rates=xt.Table(data),
            rate_scattering=rate_scattering,
            lifetime_scattering=lifetime(rate_scattering),
            rate_total=rate_total,
            lifetime_total=lifetime(rate_total),
            rate_tracking=rate_tracking,
            lifetime_tracking=lifetime_tracking,
            tracked=tracked,
            rate_generated=rate_generated,
            rate_tracking_error=rate_tracking_error,
            lifetime_tracking_error=lifetime_tracking_error,
            cutoff_scan=cutoff_scan,
            interaction_log=interaction_log,
        )

    def run(self, n_turns, *, at=None, n_replicas=1, keep_particles=False,
            with_progress=False):
        """
        Generate, track and analyse the particles scattered at the markers.

        Convenience helper doing in one call the steps of a thermal-photon
        simulation: the particles of each marker (and replica) are generated
        with :meth:`generate`, tracked from the marker back to it with
        ``line.track(part, ele_start=name, ele_stop=name, num_turns=n_turns)``
        and analysed with :meth:`analyse`, together with the records of the
        interactions. Doing these steps explicitly gives the same result, and
        allows splitting them over separate jobs or refining the loss
        locations before the analysis.

        Parameters
        ----------
        n_turns : int
            Number of turns to track.
        at : str, sequence of str, or None, optional
            Markers at which particles are generated. If ``None`` (default),
            all the markers of the line. Markers with a negligible
            interaction rate are skipped, as in :meth:`generate`.
        n_replicas : int, optional
            Number of replicas of each marker.
        keep_particles : bool, optional
            If ``True``, the result keeps the particles of each marker (its
            replicas merged), all of them merged, and the lost ones (if any).
        with_progress : bool, optional
            Forwarded to ``line.track``.

        Returns
        -------
        result : ThermalPhotonResult
            Result of :meth:`analyse` for the particles of each marker, with
            the records of the interactions.
        """
        if (isinstance(n_turns, (bool, np.bool_))
                or not isinstance(n_turns, (int, np.integer)) or n_turns < 1):
            raise ValueError("`n_turns` must be a positive integer.")
        if (isinstance(n_replicas, (bool, np.bool_))
                or not isinstance(n_replicas, (int, np.integer))
                or n_replicas < 1):
            raise ValueError("`n_replicas` must be a positive integer.")

        line = self.line
        names = self._markers(at)
        samples, logs = {}, {}
        for replica in range(n_replicas):
            generated, records = self.generate(
                at=names, replica=replica, n_replicas=n_replicas,
                return_log=True)
            for nn, part in generated.items():
                line.track(part, ele_start=nn, ele_stop=nn,
                           num_turns=n_turns, with_progress=with_progress)
                samples.setdefault(nn, []).append(part)
                logs.setdefault(nn, []).append(records[nn])

        result = self.analyse(samples, logs=logs)
        if keep_particles:
            result.particles_by_element = {
                nn: vv[0] if len(vv) == 1 else xt.Particles.merge(vv)
                for nn, vv in samples.items()}
            result.particles = xt.Particles.merge(
                list(result.particles_by_element.values()))
            lost = _lost(result.particles)
            if np.any(lost):
                result.lost_particles = result.particles.filter(lost)
        return result

    def _non_negligible(self, names):
        """
        Markers whose section contributes to the interaction rate.

        Parameters
        ----------
        names : list of str
            Selected markers.

        Returns
        -------
        names : list of str
            The selected markers with an interaction rate above
            ``1e-12`` of that of all the markers of the line.
        """
        element_dict = self.line.element_dict
        total = sum(element_dict[nn].interaction_rate
                    for nn in self.marker_names)
        kept = [nn for nn in names
                if element_dict[nn].interaction_rate
                > _NEGLIGIBLE_RATE_FRACTION * total]
        if len(kept) == 0:
            raise ValueError(
                "No thermal-photon interactions possible at the selected "
                "markers: the interaction rates of their sections are zero "
                "or negligible.")
        return kept

    def _check_sections_cover_line(self):
        """
        Check that every part of the line belongs to the section of a marker.

        The section of a marker goes from the previous one to it, and the
        section of the first one from ``s = 0``. So the last marker must sit
        at the end of the line: otherwise the part of the line after it is not
        integrated, and its interaction rate is silently missing.

        Returns
        -------
        None
        """
        if self._last_marker_at_end():
            return
        last = self._markers()[-1]
        tab = self.line.get_table()
        raise ValueError(
            f"The last ThermalPhotonMarker ('{last}') is at "
            f"s = {float(tab['s', last]):.6g} m, but the line ends at "
            f"s = {float(tab.s[-1]):.6g} m: the last ThermalPhotonMarker "
            "must sit at the end of the line, otherwise the part of the line "
            "after it is not accounted for in the interaction rate. "
            "`line.xcoll.thermal_photons.install_markers(...)` adds it, or "
            "place a marker at `at=line.get_length()`.")

    def _last_marker_at_end(self):
        """
        Whether the last marker sits at the end of the line.

        Returns
        -------
        at_end : bool
            ``True`` if the line has markers and the last one is at the end
            of the line.
        """
        names = self.marker_names
        if len(names) == 0:
            return False
        tab = self.line.get_table()
        return abs(float(tab['s', names[-1]]) - float(tab.s[-1])) <= _S_TOL

    def _markers(self, at=None):
        """
        Validated names of the selected markers, in line order.

        Parameters
        ----------
        at : str, sequence of str, or None
            Selected markers. ``None`` selects all the markers of the line.

        Returns
        -------
        names : list of str
            Names of the selected markers, in line order.
        """
        names = self.marker_names
        if at is None:
            if len(names) == 0:
                raise ValueError(
                    "The line does not contain any ThermalPhotonMarker. "
                    "Install them with "
                    "`line.xcoll.thermal_photons.install_markers(...)`.")
            return names

        if isinstance(at, str):
            selected = [at]
        elif hasattr(at, '__iter__'):
            selected = list(at)
        else:
            raise TypeError(
                "Thermal-photon markers are selected by name "
                f"(got {type(at).__name__}).")
        if len(selected) == 0:
            raise ValueError("No thermal-photon marker selected.")
        position = {nn: ii for ii, nn in enumerate(names)}
        for nn in selected:
            if not isinstance(nn, str):
                raise TypeError(
                    "Thermal-photon markers are selected by name "
                    f"(got {type(nn).__name__}).")
            if nn in position:
                continue
            if nn in set(self.line.element_names):
                raise TypeError(
                    f"Element '{nn}' is not a ThermalPhotonMarker "
                    f"(got {type(self.line.element_dict[nn]).__name__}).")
            raise ValueError(f"Element '{nn}' is not present in the line.")

        return sorted(set(selected), key=position.__getitem__)
