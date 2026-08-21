# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.constants import c as C_LIGHT
from scipy.constants import e as QELEM_SI

import xtrack as xt

from ..beam_elements.thermal_compton import (
    ThermalComptonScattering, THOMSON_CROSS_SECTION,
    blackbody_photon_density, blackbody_mean_photon_energy)


@dataclass
class ThermalComptonResult:
    """
    Result returned by :meth:`ThermalComptonStudy.run`.

    The result separates the quantities inferred from the scattering model
    (total Compton interaction rate) from the quantities obtained after
    tracking the generated particles (loss rate and lifetime).  The particle
    samples are stored only when :meth:`ThermalComptonStudy.run` is called with
    ``keep_particles=True``.

    Parameters
    ----------
    element_names : list
        Names of the thermal Compton scattering elements included in the
        result.
    local_rates : xtrack.Table
        Per-element diagnostics table.  It always contains ``name``, ``s``,
        ``section_length``, ``section_rate``, ``rate_scattering``,
        ``delta_neg``, ``delta_pos``, ``n_trials``, ``event_probability``,
        ``rate_tail``, ``energy_loss_rate``, ``num_particles`` and
        ``sum_weight``.  When tracking is enabled it additionally contains
        ``num_lost_particles`` and ``sum_lost_weight``.
    rate_scattering : float
        Total Compton interaction rate on thermal photons around the ring
        [1/s], i.e. counting *every* scattering event, however soft.
    lifetime_scattering : float
        Absolute lower bound on the lifetime, obtained by assuming that every
        Compton scattering leads to a loss [s].  It is *not* the beam
        lifetime; compare with ``lifetime_tracking``.
    rate_tail : float
        Rate carried by the generated macro-particles [1/s], i.e. the rate of
        scattering events that push a lepton outside the local momentum
        acceptance.  It is the loss rate that would be obtained if every such
        particle were lost, and is therefore an upper bound on
        ``rate_tracking``.
    rate_tracking : float or None
        Weighted loss rate obtained by tracking the generated particles [1/s].
        ``None`` when tracking is disabled.
    lifetime_tracking : float or None
        Beam lifetime inferred from ``rate_tracking`` [s].  ``None`` when
        tracking is disabled.
    energy_loss_rate : float
        Total energy carried away by the scattered photons per unit time
        [eV/s], summed over the ring.
    n_particles : float
        Number of stored particles represented by the study.
    particles_by_element : dict or None
        Mapping ``{element_name: xtrack.Particles}`` with the generated
        particles for each scattering element.  ``None`` unless
        ``keep_particles=True``.
    particles : xtrack.Particles or None
        Merged generated particle sample.  ``None`` unless
        ``keep_particles=True``.
    lost_particles : xtrack.Particles or None
        Subset of ``particles`` lost during tracking.  ``None`` unless both
        ``track=True`` and ``keep_particles=True``.
    tracked : bool
        Whether the generated particles were tracked to determine losses.
    """
    element_names: list
    local_rates: xt.Table
    rate_scattering: float
    lifetime_scattering: float
    rate_tail: float
    rate_tracking: float | None
    lifetime_tracking: float | None
    energy_loss_rate: float
    n_particles: float
    tracked: bool
    particles_by_element: dict | None = None
    particles: xt.Particles | None = None
    lost_particles: xt.Particles | None = None


class ThermalComptonStudy:
    def __init__(self, line=None, elements=None, twiss=None,
                 temperature=300.0,
                 n_particles=None, bunch_intensity=None, n_bunches=1,
                 beam_current=None,
                 nemitt_x=None, nemitt_y=None,
                 gemitt_x=None, gemitt_y=None,
                 sigma_z=None, sigma_delta=None,
                 local_momentum_acceptance=None,
                 local_momentum_acceptance_scale=0.85,
                 delta_threshold=None,
                 n_macroparticles=500, n_trials=None,
                 n_target_events=2000,
                 n_trials_min=10, n_trials_max=200000,
                 n_trials_pilot=200, n_macroparticles_pilot=None,
                 max_events_per_macro=None,
                 section_assignment='centered',
                 n_sigma_cut=5.0,
                 enable_tail_veto=True,
                 seed=None, **kwargs):
        """
        Build a thermal Compton scattering study and validate the supplied
        line, optics and beam parameters.

        The study assigns to each :class:`ThermalComptonScattering` element a
        section of the ring, computes the absolute interaction rate of that
        section, and configures the elements with the local optics and beam
        parameters needed to generate weighted Monte Carlo macro-particles.

        Because the blackbody photon gas is uniform and the Thomson cross
        section is energy independent, the interaction rate per unit length is
        *constant* around the ring: the rate of a section is simply

        .. math::
            R_{\\rm sec} = N_{\\rm beam}\\,\\frac{L_{\\rm sec}}{C}\\,
                           c\\,n_\\gamma\\,\\sigma_T ,

        with :math:`N_{\\rm beam}` the number of stored particles, :math:`C`
        the circumference and :math:`n_\\gamma` the photon density.  Only the
        *distribution* of the losses depends on the optics.

        What *does* vary strongly along the ring is the local momentum
        acceptance, and hence the fraction of the scattering events that can
        actually lead to a loss.  The study therefore selects the events
        against the local momentum acceptance (as the xfields Touschek study
        does) and sizes the Monte Carlo statistics element by element, so that
        the tracked sample is neither dominated by particles that can never be
        lost, nor starved where the acceptance is large.

        After construction, call :meth:`initialise` (done automatically by
        :meth:`run`) and then :meth:`run` to obtain a
        :class:`ThermalComptonResult`.

        Parameters
        ----------
        line : xtrack.Line
            Line containing the :class:`ThermalComptonScattering` elements to
            configure.  It must have a ``particle_ref``.
        elements : str, sequence of str, or None, optional
            Thermal Compton scattering elements included in the study.  If
            ``None``, all :class:`ThermalComptonScattering` elements in the
            line are used.
        twiss : xtrack.TwissTable or None, optional
            Twiss table used for the local optics.  If ``None``, it is computed
            when :meth:`initialise` is called.
        temperature : float, optional
            Temperature of the vacuum chamber [K].  Default 300 K.
        n_particles : float, optional
            Total number of stored particles in the ring.  Mutually exclusive
            with ``bunch_intensity``/``n_bunches`` and with ``beam_current``.
        bunch_intensity : float, optional
            Number of particles per bunch.  Used together with ``n_bunches``.
        n_bunches : int, optional
            Number of stored bunches.  Default 1.
        beam_current : float, optional
            Stored beam current [A].  Converted to the number of stored
            particles with :math:`N = I\\,C/(e\\,\\beta_0 c)`.
        nemitt_x, nemitt_y : float, optional
            Normalized emittances.  Mutually exclusive with ``gemitt_x`` and
            ``gemitt_y``.
        gemitt_x, gemitt_y : float, optional
            Geometric emittances.  Mutually exclusive with ``nemitt_x`` and
            ``nemitt_y``.
        sigma_z : float
            RMS bunch length [m].
        sigma_delta : float
            RMS relative momentum spread.
        local_momentum_acceptance : xtrack.Table or None
            Local momentum acceptance (LMA) of the lattice, as returned by
            ``line.get_local_momentum_acceptance(...)``: a table with ``s``,
            ``delta_neg`` and ``delta_pos`` columns.  It is interpolated at
            each scattering element and used to select the events: only the
            scattered leptons ending up outside the local acceptance are
            generated and tracked, since they are the only ones that can be
            lost.  Mutually exclusive with ``delta_threshold``.
        local_momentum_acceptance_scale : float, optional
            Safety factor applied to ``delta_neg`` and ``delta_pos`` (default
            0.85, as in the xfields Touschek study).  It must be smaller than
            one so that the generated sample also covers the particles sitting
            just inside the nominal acceptance, which may still be lost over
            many turns.
        delta_threshold : float, optional
            Fallback for a flat, s-independent acceptance window
            ``[-delta_threshold, +delta_threshold]``, for lines for which no
            LMA is available.  Mutually exclusive with
            ``local_momentum_acceptance``.
        n_macroparticles : int, optional
            Number of macro-leptons drawn from the local beam distribution at
            each scattering element.
        n_trials : int or None, optional
            Number of scattering trials per macro-lepton.  If ``None``
            (default) it is determined automatically and *individually for each
            element* by a cheap pilot run, so that every element generates
            approximately ``n_target_events`` particles whatever its local
            momentum acceptance.  This equalises the Monte Carlo statistics
            around the ring and keeps the tracking cost under control.
        n_target_events : int, optional
            Target number of generated (trackable) particles per scattering
            element, used by the automatic sizing.
        n_trials_min, n_trials_max : int, optional
            Bounds applied to the automatically determined number of trials.
        n_trials_pilot : int, optional
            Number of trials per macro-lepton used in the pilot run.
        n_macroparticles_pilot : int or None, optional
            Number of macro-leptons used in the pilot run.  Defaults to
            ``min(n_macroparticles, 200)``.
        max_events_per_macro : int or None, optional
            Capacity of the per-macro-particle output slice.  If ``None``
            (default) it is derived from the pilot run with a wide Poisson
            margin, so that no event can be truncated.
        section_assignment : {'centered', 'preceding'}, optional
            How the ring is partitioned among the scattering elements.
            ``'centered'`` (default) assigns to each element the section
            extending to the mid-points towards its neighbours;
            ``'preceding'`` assigns the section between the previous element
            and this one (the convention of the xfields Touschek study).
            In both cases the sections tile the full circumference.
        n_sigma_cut : float or None, optional
            Truncation of the Gaussian beam distribution used to draw the
            macro-leptons.
        enable_tail_veto : bool, optional
            Enable the exact early veto of the trials that cannot push a lepton
            out of the local acceptance.  Only affects speed, not the result.
        seed : int or None, optional
            Seed of the random-number generators.  If ``None``, a seed is drawn
            with :mod:`numpy.random` when particles are generated.
        **kwargs
            Additional keyword arguments forwarded to ``line.twiss()`` when a
            Twiss table is computed internally.

        Returns
        -------
        None

        References
        ----------
        .. [1] H. Burkhardt, "Monte Carlo simulation of scattering of beam
           particles and thermal photons", CERN SL/Note 93-73 (OP) (1993).
        .. [2] V. Telnov, Nucl. Instrum. Meth. A **260** (1987) 304.
        .. [3] A. Natochii, BNL-229489-2026-TECH, EIC-ADD-TN-159 (2026), and
           https://github.com/eic/thermal-compton-mc
        """
        # Input validation
        if line is None:
            raise ValueError("`line` is required.")
        if getattr(line, "particle_ref", None) is None:
            raise ValueError("`line` must have a `particle_ref`.")
        if sigma_z is None:
            raise ValueError("`sigma_z` is required.")
        if sigma_delta is None:
            raise ValueError("`sigma_delta` is required.")
        if section_assignment not in ('centered', 'preceding'):
            raise ValueError(
                "`section_assignment` must be 'centered' or 'preceding'.")

        self.line = line
        self.particle_ref = line.particle_ref
        self.twiss = twiss
        self.temperature = float(temperature)
        self.sigma_z = float(sigma_z)
        self.sigma_delta = float(sigma_delta)
        self.n_macroparticles = int(n_macroparticles)
        self.n_trials = None if n_trials is None else int(n_trials)
        self.n_target_events = int(n_target_events)
        self.n_trials_min = int(n_trials_min)
        self.n_trials_max = int(n_trials_max)
        self.n_trials_pilot = int(n_trials_pilot)
        self.n_macroparticles_pilot = (min(int(n_macroparticles), 200)
                                       if n_macroparticles_pilot is None
                                       else int(n_macroparticles_pilot))
        self.max_events_per_macro = max_events_per_macro
        self.section_assignment = section_assignment
        self.n_sigma_cut = n_sigma_cut
        self.enable_tail_veto = bool(enable_tail_veto)
        self.seed = seed
        self.kwargs = kwargs

        if self.n_trials is not None and self.n_trials < 1:
            raise ValueError("`n_trials` must be at least 1.")
        if self.n_macroparticles < 1:
            raise ValueError("`n_macroparticles` must be at least 1.")

        # Local momentum acceptance
        if (local_momentum_acceptance is None) == (delta_threshold is None):
            raise ValueError(
                "Provide exactly one of `local_momentum_acceptance` (an "
                "`xt.Table` from `line.get_local_momentum_acceptance()`) or "
                "`delta_threshold` (a flat acceptance window).")
        if not 0 < local_momentum_acceptance_scale <= 1:
            raise ValueError(
                "`local_momentum_acceptance_scale` must be in (0, 1].")
        self.local_momentum_acceptance_scale = float(
            local_momentum_acceptance_scale)
        self.delta_threshold = (None if delta_threshold is None
                                else float(delta_threshold))
        if self.delta_threshold is not None and self.delta_threshold <= 0:
            raise ValueError("`delta_threshold` must be positive.")
        self.local_momentum_acceptance = local_momentum_acceptance
        if local_momentum_acceptance is not None:
            lma = local_momentum_acceptance
            required = {"s", "delta_neg", "delta_pos"}
            missing = required - set(getattr(lma, "_col_names", []))
            if missing:
                raise ValueError("`local_momentum_acceptance` is missing the "
                                 f"columns {sorted(missing)}.")
            # The table of the user is never modified: the scaled acceptance is
            # stored separately.
            self._lma_s = np.asarray(lma.s, dtype=float)
            self._lma_neg = (self.local_momentum_acceptance_scale
                             * np.asarray(lma.delta_neg, dtype=float))
            self._lma_pos = (self.local_momentum_acceptance_scale
                             * np.asarray(lma.delta_pos, dtype=float))
            for name, vals in (('delta_neg', self._lma_neg),
                               ('delta_pos', self._lma_pos)):
                if not np.all(np.isfinite(vals)):
                    raise ValueError(f"`{name}` contains non-finite values.")
            if np.any(self._lma_neg >= 0) or np.any(self._lma_pos <= 0):
                raise ValueError(
                    "`delta_neg` must be negative and `delta_pos` positive "
                    "everywhere in `local_momentum_acceptance`.")
            order = np.argsort(self._lma_s)
            self._lma_s = self._lma_s[order]
            self._lma_neg = self._lma_neg[order]
            self._lma_pos = self._lma_pos[order]
        else:
            self._lma_s = None
            self._lma_neg = None
            self._lma_pos = None

        mass0 = float(line.particle_ref.mass0)
        if abs(mass0 - xt.ELECTRON_MASS_EV)/xt.ELECTRON_MASS_EV > 1e-6:
            warnings.warn(
                "ThermalComptonStudy assumes a lepton beam: the Thomson and "
                "Klein-Nishina cross sections used here are the electron "
                "ones.", UserWarning, stacklevel=2)

        # Elements
        tab = line.get_table()
        elements = self._resolve_elements(line, tab, elements)
        self.elements = elements

        # Beam parameters
        self.circumference = float(line.get_length())
        beta0 = float(line.particle_ref.beta0[0])
        given = [n_particles is not None,
                 bunch_intensity is not None,
                 beam_current is not None]
        if sum(given) != 1:
            raise ValueError(
                "Provide exactly one of `n_particles`, `bunch_intensity` "
                "(with `n_bunches`) or `beam_current`.")
        if n_particles is not None:
            self.n_particles = float(n_particles)
        elif bunch_intensity is not None:
            self.n_particles = float(bunch_intensity)*int(n_bunches)
        else:
            self.n_particles = (float(beam_current)*self.circumference
                                / (QELEM_SI*beta0*C_LIGHT))
        if self.n_particles <= 0:
            raise ValueError("The number of stored particles must be positive.")
        self.beam_current = (self.n_particles*QELEM_SI*beta0*C_LIGHT
                             / self.circumference)

        nemitt_given = nemitt_x is not None and nemitt_y is not None
        gemitt_given = gemitt_x is not None and gemitt_y is not None
        if nemitt_given and gemitt_given:
            raise ValueError(
                "Provide either normalized emittances (nemitt_x, nemitt_y) OR "
                "geometric emittances (gemitt_x, gemitt_y), not both.")
        if not (nemitt_given or gemitt_given):
            raise ValueError(
                "You must provide either both normalized emittances "
                "(nemitt_x, nemitt_y) OR both geometric emittances "
                "(gemitt_x, gemitt_y).")
        if nemitt_given:
            gamma0 = float(line.particle_ref.gamma0[0])
            self.gemitt_x = nemitt_x/(beta0*gamma0)
            self.gemitt_y = nemitt_y/(beta0*gamma0)
        else:
            self.gemitt_x = float(gemitt_x)
            self.gemitt_y = float(gemitt_y)

        # Photon gas
        self.photon_density = blackbody_photon_density(self.temperature)
        self.mean_photon_energy = blackbody_mean_photon_energy(self.temperature)

        self._initialised = False
        self._event_probability = {}

    ############################################################
    # Helpers
    ############################################################
    @staticmethod
    def _resolve_elements(line, tab, elements):
        try:
            has = "ThermalComptonScattering" in set(np.unique(tab.element_type))
        except Exception:
            has = "ThermalComptonScattering" in set(
                getattr(tab, "element_type", []))
        if not has:
            raise ValueError(
                "The line does not contain any ThermalComptonScattering "
                "element. Please insert them before initialising the "
                "ThermalComptonStudy.")

        if elements is None:
            elements = [nn for nn in tab.name[:-1]
                        if isinstance(line[nn], ThermalComptonScattering)]
        elif isinstance(elements, str):
            elements = [elements]
        else:
            elements = list(elements)

        if len(elements) == 0:
            raise ValueError(
                "No ThermalComptonScattering elements selected for this study.")

        for nn in elements:
            if nn not in line.element_names:
                raise ValueError(f"Element '{nn}' is not present in the line.")
            if not isinstance(line[nn], ThermalComptonScattering):
                raise TypeError(
                    f"Element '{nn}' is not a ThermalComptonScattering "
                    f"(got {type(line[nn]).__name__}).")
        return elements

    def _section_lengths(self, s_elements):
        """
        Partition the ring among the scattering elements.

        Parameters
        ----------
        s_elements : array_like
            Longitudinal positions of the (sorted) scattering elements [m].

        Returns
        -------
        lengths : numpy.ndarray
            Length of the section represented by each element [m].  The lengths
            sum to the circumference.
        """
        s = np.asarray(s_elements, dtype=float)
        circ = self.circumference
        n = s.size
        if n == 1:
            return np.array([circ])
        # Gaps between consecutive elements, with periodic wrap-around
        gaps = np.diff(np.concatenate([s, [s[0] + circ]]))  # gap[i] = s[i+1]-s[i]
        if np.any(gaps <= 0):
            warnings.warn(
                "Some thermal Compton scattering elements are at the same "
                "longitudinal position: the corresponding sections have zero "
                "length.", UserWarning, stacklevel=2)
        if self.section_assignment == 'preceding':
            # element i represents (s[i-1], s[i]]
            lengths = np.roll(gaps, 1)
        else:
            # element i represents the mid-points towards both neighbours
            lengths = 0.5*(gaps + np.roll(gaps, 1))
        return lengths

    def momentum_acceptance_at(self, s):
        """
        Local momentum acceptance used by the study at a given position.

        The table supplied at construction is interpolated linearly and scaled
        by ``local_momentum_acceptance_scale``.  When a flat
        ``delta_threshold`` was given instead, the same window is returned
        everywhere.

        Parameters
        ----------
        s : float or array_like
            Longitudinal position(s) [m].

        Returns
        -------
        delta_neg, delta_pos : float or numpy.ndarray
            Scaled negative and positive momentum acceptance.
        """
        if self._lma_s is None:
            thr = self.delta_threshold
            return -thr*np.ones_like(np.asarray(s, dtype=float)), \
                thr*np.ones_like(np.asarray(s, dtype=float))
        return (np.interp(s, self._lma_s, self._lma_neg),
                np.interp(s, self._lma_s, self._lma_pos))

    ############################################################
    # Configuration
    ############################################################
    def initialise(self):
        """
        Configure all the thermal Compton scattering elements of the study.

        For each element this method:

        1. assigns the section of the ring it represents,
        2. computes the absolute trial rate of that section,
        3. interpolates the local momentum acceptance, which defines the
           events worth generating and tracking,
        4. stores the local optics, closed orbit and beam parameters on the
           element, so that :meth:`ThermalComptonScattering.scatter` can draw
           the local beam distribution and weight the generated
           macro-particles correctly,
        5. unless ``n_trials`` was given explicitly, runs a cheap pilot to
           measure the local probability that a scattering event leaves the
           acceptance, and sizes ``n_trials`` and the output capacity of each
           element accordingly (see :meth:`size_statistics`).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        line = self.line
        if self.twiss is None:
            tw_kwargs = dict(self.kwargs)
            tw_kwargs.setdefault('method', '6d')
            self.twiss = line.twiss(**tw_kwargs)
        twiss = self.twiss

        tab = line.get_table()
        s_all = np.array([float(tab['s', nn]) for nn in self.elements])
        order = np.argsort(s_all)
        if not np.array_equal(order, np.arange(order.size)):
            self.elements = [self.elements[ii] for ii in order]
            s_all = s_all[order]

        lengths = self._section_lengths(s_all)

        # Rate per unit length is uniform: R = N * (L/C) * c * n_gamma * sigma_T
        rate_per_length = (self.n_particles/self.circumference
                           * C_LIGHT*self.photon_density*THOMSON_CROSS_SECTION)

        if self._lma_s is not None:
            if (s_all.min() < self._lma_s.min() - 1e-9
                    or s_all.max() > self._lma_s.max() + 1e-9):
                warnings.warn(
                    "Some scattering elements lie outside the s-range of "
                    "`local_momentum_acceptance`: the acceptance is clamped "
                    "to the closest tabulated value there.",
                    UserWarning, stacklevel=2)
        delta_neg_all, delta_pos_all = self.momentum_acceptance_at(s_all)

        for nn, ss, ll, dneg, dpos in zip(self.elements, s_all, lengths,
                                          np.atleast_1d(delta_neg_all),
                                          np.atleast_1d(delta_pos_all)):
            elem = line[nn]
            elem.element_name = nn
            elem._configure(
                s=float(ss),
                particle_ref=self.particle_ref,
                element_index=line.element_names.index(nn),
                temperature=self.temperature,
                section_length=float(ll),
                section_rate=float(ll*rate_per_length),
                alfx=twiss['alfx', nn], betx=twiss['betx', nn],
                alfy=twiss['alfy', nn], bety=twiss['bety', nn],
                dx=twiss['dx', nn], dpx=twiss['dpx', nn],
                dy=twiss['dy', nn], dpy=twiss['dpy', nn],
                x_co=twiss['x', nn], px_co=twiss['px', nn],
                y_co=twiss['y', nn], py_co=twiss['py', nn],
                zeta_co=twiss['zeta', nn], delta_co=twiss['delta', nn],
                gemitt_x=self.gemitt_x, gemitt_y=self.gemitt_y,
                sigma_z=self.sigma_z, sigma_delta=self.sigma_delta,
                n_macroparticles=self.n_macroparticles,
                n_trials=(self.n_trials if self.n_trials is not None
                          else self.n_trials_pilot),
                delta_neg=float(dneg),
                delta_pos=float(dpos),
                max_events_per_macro=self.max_events_per_macro,
                enable_tail_veto=self.enable_tail_veto,
                n_sigma_cut=self.n_sigma_cut,
            )
            if (self.n_sigma_cut is not None
                    and elem.n_sigma_cut_delta < self.n_sigma_cut - 1e-9):
                warnings.warn(
                    f"Longitudinal sampling cutoff reduced at element '{nn}' "
                    f"(s = {ss:.2f} m): {elem.n_sigma_cut_delta:.2f} sigma "
                    f"instead of {self.n_sigma_cut:.2f}, so that no "
                    f"macro-lepton is drawn outside the local momentum "
                    f"acceptance ({dneg*100:.3f}%, {dpos*100:.3f}%).",
                    UserWarning, stacklevel=2)

        self._initialised = True

        if self.n_trials is None:
            self.size_statistics()

    def size_statistics(self, seed=None):
        """
        Size the Monte Carlo statistics element by element.

        The probability that a single scattering trial produces a lepton
        outside the local momentum acceptance varies strongly along the ring
        (it depends on the local acceptance, which can change by an order of
        magnitude between a dispersion-free straight and an arc).  A fixed
        number of trials would therefore give very uneven statistics, generate
        far too many particles where the acceptance is small, and far too few
        where it is large.

        This method runs a cheap pilot generation at every element, measures
        the per-trial probability :math:`p` of producing a trackable event,
        and sets

        .. math::
            n_{\\rm trials} = \\frac{n_{\\rm target}}
                                    {n_{\\rm macro}\\, p} ,

        clipped to ``[n_trials_min, n_trials_max]``, so that each element
        generates about ``n_target_events`` particles.  The output capacity of
        each element is set to a wide Poisson upper bound on the resulting
        multiplicity, which makes truncation impossible.

        The absolute normalisation is unaffected: the section rate is shared
        among ``n_macroparticles * n_trials`` trials, so a different number of
        trials only redistributes the same total weight over a different number
        of macro-particles.

        Parameters
        ----------
        seed : int or None, optional
            Seed of the pilot run.  Defaults to the seed of the study (offset
            so that the pilot and the production samples are independent).

        Returns
        -------
        None
        """
        if not self._initialised:
            self.initialise()
            return   # initialise() calls this method at the end

        if seed is None:
            seed = self.seed if self.seed is not None else 12345
        seed = int(seed) + 987654

        n_pilot = max(1, min(self.n_macroparticles_pilot,
                             self.n_macroparticles))

        for ii, nn in enumerate(self.elements):
            elem = self.line[nn]
            n_trials_pilot = self.n_trials_pilot
            p = 0.0
            # Escalate the pilot if the process is very rare locally.
            for _ in range(4):
                elem._configure(n_trials=n_trials_pilot,
                                max_events_per_macro=None)
                part = elem.scatter(n_macroparticles=n_pilot,
                                    seed=seed + ii)
                n_events = int(np.sum(part.state > -1e5))
                if n_events > 0:
                    p = n_events/float(n_pilot*n_trials_pilot)
                    break
                n_trials_pilot *= 8

            if p == 0.0:
                warnings.warn(
                    f"No thermal Compton event could reach the local momentum "
                    f"acceptance at element '{nn}' in "
                    f"{n_pilot*n_trials_pilot} pilot trials: this section "
                    f"contributes (almost) nothing to the losses. The maximum "
                    f"momentum deviation reachable is about "
                    f"{4*float(self.particle_ref.gamma0[0])**2*self.mean_photon_energy/float(self.particle_ref.p0c[0]):.2e}, "
                    f"to be compared with the local acceptance "
                    f"({elem.delta_neg*100:.3f}%, {elem.delta_pos*100:.3f}%).",
                    UserWarning, stacklevel=2)
                n_trials = self.n_trials_min
                lam = 1.0
            else:
                n_trials = int(np.ceil(self.n_target_events
                                       / (self.n_macroparticles*p)))
                n_trials = int(np.clip(n_trials, self.n_trials_min,
                                       self.n_trials_max))
                lam = n_trials*p        # expected events per macro-particle

            capacity = self.max_events_per_macro
            if capacity is None:
                # Poisson upper bound with a very wide margin (~8 sigma + 16)
                capacity = int(min(n_trials,
                                   max(8, np.ceil(lam + 8*np.sqrt(lam) + 16))))
            self._event_probability[nn] = p
            elem._configure(n_trials=n_trials,
                            n_macroparticles=self.n_macroparticles,
                            max_events_per_macro=int(capacity))
            # The pilot diagnostics must not leak into the results
            elem.rate_scattering = 0.0
            elem.rate_tail = 0.0
            elem.energy_loss_rate = 0.0
            elem.n_dropped = 0

    @property
    def event_probability(self):
        """
        Per-trial probability of generating a trackable event, per element.

        Measured by the pilot run of :meth:`size_statistics`.  It is the
        fraction of thermal Compton scatterings that push a lepton outside the
        local momentum acceptance, i.e. the local "loss-candidate" fraction.

        Returns
        -------
        event_probability : dict
            Mapping ``{element_name: probability}``.
        """
        return dict(self._event_probability)

    ############################################################
    # Analytic cross-checks
    ############################################################
    def analytic_rate(self, klein_nishina=True):
        """
        Analytic total thermal Compton interaction rate of the stored beam.

        The rate per particle in an isotropic photon gas is
        :math:`c\\,n_\\gamma\\,\\sigma` (the Moeller flux factor averages to
        one), so the ring total is :math:`N\\,c\\,n_\\gamma\\,\\sigma`.

        Parameters
        ----------
        klein_nishina : bool, optional
            If ``True`` (default), an approximate Klein-Nishina correction
            :math:`\\sigma_{KN}/\\sigma_T \\simeq 1 - 2a` is applied with
            :math:`a = \\gamma\\langle k\\rangle/(m_ec^2)` a representative
            rest-frame photon energy.  If ``False``, the Thomson cross section
            is used.

        Returns
        -------
        rate : float
            Total interaction rate [1/s].
        """
        sigma = THOMSON_CROSS_SECTION
        if klein_nishina:
            gamma0 = float(self.particle_ref.gamma0[0])
            a = gamma0*self.mean_photon_energy/float(self.particle_ref.mass0)
            sigma = sigma*max(0.0, 1.0 - 2.0*a)
        return self.n_particles*C_LIGHT*self.photon_density*sigma

    def analytic_energy_loss_rate(self):
        """
        Analytic inverse-Compton energy loss rate of the stored beam.

        In the Thomson regime the power radiated by one lepton is
        :math:`P = (4/3)\\,\\sigma_T\\,c\\,\\gamma^2\\beta^2\\,U_\\gamma` with
        :math:`U_\\gamma = n_\\gamma\\langle k\\rangle` the photon energy
        density.  This is a useful benchmark for the Monte Carlo
        ``energy_loss_rate`` (they agree to within the few-percent
        Klein-Nishina correction).

        Parameters
        ----------
        None

        Returns
        -------
        energy_loss_rate : float
            Energy loss rate of the whole beam [eV/s].
        """
        gamma0 = float(self.particle_ref.gamma0[0])
        beta0 = float(self.particle_ref.beta0[0])
        u_gamma = self.photon_density*self.mean_photon_energy  # [eV/m^3]
        return (self.n_particles*4./3.*THOMSON_CROSS_SECTION*C_LIGHT
                * gamma0**2*beta0**2*u_gamma)

    def minimum_lifetime(self):
        """
        Absolute lower bound on the thermal Compton lifetime.

        Obtained by assuming that *every* Compton scattering leads to a loss:
        :math:`\\tau_{\\min} = 1/(c\\,n_\\gamma\\,\\sigma_T)`, about 25 h at
        300 K, independently of the beam energy and current.  Any physical
        lifetime must be much larger than this.

        Parameters
        ----------
        None

        Returns
        -------
        lifetime : float
            Minimum lifetime [s].
        """
        return 1.0/(C_LIGHT*self.photon_density*THOMSON_CROSS_SECTION)

    ############################################################
    # Generation and tracking
    ############################################################
    def generate_particles(self):
        """
        Generate the thermal-Compton-scattered particles at every element.

        Parameters
        ----------
        None

        Returns
        -------
        particles_by_element : dict
            Mapping ``{element_name: xtrack.Particles}``.
        """
        if not self._initialised:
            self.initialise()
        seed = self.seed
        if seed is None:
            seed = int(np.random.randint(1, 2**31 - 1))
        return {nn: self.line[nn].scatter(seed=seed + ii)
                for ii, nn in enumerate(self.elements)}

    def run(self, *, track=False, n_turns=None, generate_particles=None,
            keep_particles=False, with_progress=False):
        """
        Run the configured thermal Compton rate or loss study.

        Parameters
        ----------
        track : bool, optional
            If ``True``, track the generated particles and compute the loss
            rate from the particles lost during tracking.  If ``False``, only
            the interaction rate is reported.
        n_turns : int or None, optional
            Number of turns to track.  Required when ``track`` is ``True``.
        generate_particles : bool or None, optional
            If ``True``, generate the weighted scattered particles even when
            tracking is disabled.  If ``None``, particles are generated only
            when needed.
        keep_particles : bool, optional
            If ``True``, store the generated particle samples in the returned
            :class:`ThermalComptonResult`.
        with_progress : bool, optional
            Forwarded to :meth:`xtrack.Line.track`.

        Returns
        -------
        result : ThermalComptonResult
            Study result with the interaction rate, the optional
            tracking-derived loss rate and lifetime, and optionally the
            generated and lost particle samples.
        """
        if keep_particles and generate_particles is False:
            raise ValueError("`keep_particles=True` is incompatible with "
                             "`generate_particles=False`.")
        if track:
            if generate_particles is False:
                raise ValueError("`generate_particles=False` is incompatible "
                                 "with `track=True`.")
            if n_turns is None:
                raise ValueError("`n_turns` is required when `track=True`.")
            generate_particles = True
        elif generate_particles is None:
            generate_particles = keep_particles

        if not self._initialised:
            self.initialise()

        particles_by_element = {}
        merged_particles = None
        lost_particles = None

        seed = self.seed
        if seed is None:
            seed = int(np.random.randint(1, 2**31 - 1))

        if generate_particles or track:
            for ii, nn in enumerate(self.elements):
                particles = self.line[nn].scatter(seed=seed + ii)
                if track:
                    self.line.track(particles,
                                    ele_start=nn, ele_stop=nn,
                                    num_turns=n_turns,
                                    with_progress=with_progress)
                particles_by_element[nn] = particles
            merged_particles = xt.Particles.merge(
                list(particles_by_element.values()))

        if generate_particles or track:
            rate_scattering = float(sum(self.line[nn].rate_scattering
                                        for nn in self.elements))
        else:
            # Nothing was generated: report the analytic interaction rate
            rate_scattering = float(self.analytic_rate())
        rate_tail = float(sum(self.line[nn].rate_tail
                              for nn in self.elements))
        energy_loss_rate = float(sum(self.line[nn].energy_loss_rate
                                     for nn in self.elements))

        rate_tracking = None
        lifetime_tracking = None
        if track:
            lost_particles = merged_particles.filter(merged_particles.state == 0)
            rate_tracking = float(np.sum(lost_particles.weight))
            lifetime_tracking = (np.inf if rate_tracking == 0
                                 else float(self.n_particles/rate_tracking))
            self._check_acceptance_window(particles_by_element)

        lifetime_scattering = (np.inf if rate_scattering == 0
                               else float(self.n_particles/rate_scattering))

        local_rates = self.local_rates(
            particles_by_element=(particles_by_element
                                  if particles_by_element else None),
            include_tracking=track)

        if not keep_particles:
            particles_by_element = None
            merged_particles = None
            lost_particles = None

        return ThermalComptonResult(
            element_names=list(self.elements),
            local_rates=local_rates,
            rate_scattering=rate_scattering,
            lifetime_scattering=lifetime_scattering,
            rate_tail=rate_tail,
            rate_tracking=rate_tracking,
            lifetime_tracking=lifetime_tracking,
            energy_loss_rate=energy_loss_rate,
            n_particles=self.n_particles,
            particles_by_element=particles_by_element,
            particles=merged_particles,
            lost_particles=lost_particles,
            tracked=track,
        )

    def _check_acceptance_window(self, particles_by_element,
                                 upper=0.995, lower=0.02):
        """
        Sanity-check the momentum acceptance window against the tracking.

        Only the leptons falling outside the *scaled* local momentum acceptance
        are generated, so the loss rate is complete only if the particles just
        *inside* that window would indeed have survived.  Two symptoms are
        checked:

        * if essentially every generated particle is lost, the window is too
          tight and losses are being missed below it: reduce
          ``local_momentum_acceptance_scale`` (or the LMA itself);
        * if almost none is lost, the acceptance is far more pessimistic than
          the tracking, and the computing time is being wasted on particles
          that survive (or ``n_turns`` is too small).

        Parameters
        ----------
        particles_by_element : dict
            Tracked particles for each element.
        upper, lower : float, optional
            Bounds on the lost weight fraction that trigger the warnings.

        Returns
        -------
        None
        """
        total = 0.0
        lost = 0.0
        for part in particles_by_element.values():
            alive_slots = part.state > -1e5
            total += float(np.sum(part.weight[alive_slots]))
            lost += float(np.sum(part.weight[part.state == 0]))
        if total <= 0:
            return
        frac = lost/total
        self.lost_fraction = frac
        if frac > upper:
            warnings.warn(
                f"{100*frac:.2f}% of the generated weight is lost during "
                f"tracking. The selection window is probably too tight: "
                f"particles just inside the local momentum acceptance would "
                f"also be lost but were never generated, so the loss rate is "
                f"underestimated. Reduce "
                f"`local_momentum_acceptance_scale` (currently "
                f"{self.local_momentum_acceptance_scale}).",
                UserWarning, stacklevel=3)
        elif frac < lower:
            warnings.warn(
                f"Only {100*frac:.2f}% of the generated weight is lost during "
                f"tracking: the local momentum acceptance is much more "
                f"pessimistic than the tracking (or `n_turns` is too small), "
                f"and most of the computing time is spent on particles that "
                f"survive.", UserWarning, stacklevel=3)

    def local_rates(self, *, particles_by_element=None,
                    include_tracking=False):
        """
        Return an :class:`xtrack.Table` with per-element diagnostics.

        Parameters
        ----------
        particles_by_element : dict or None, optional
            Mapping from element name to the particles generated at that
            element.  If provided, Monte Carlo particle counts and weight sums
            are included.
        include_tracking : bool, optional
            If ``True``, include the loss-count and lost-weight columns
            computed from the tracked particle states.

        Returns
        -------
        table : xtrack.Table
            Per-element diagnostics table.
        """
        data = {"name": [], "s": [], "section_length": [], "section_rate": [],
                "delta_neg": [], "delta_pos": [], "n_trials": [],
                "event_probability": [],
                "rate_scattering": [], "rate_tail": [], "energy_loss_rate": []}
        include_particles = particles_by_element is not None
        if include_particles:
            data.update({"num_particles": [], "sum_weight": []})
        if include_tracking:
            data.update({"num_lost_particles": [], "sum_lost_weight": []})

        for nn in self.elements:
            elem = self.line[nn]
            data["name"].append(nn)
            data["s"].append(float(getattr(elem, "s", np.nan)))
            data["section_length"].append(float(elem.section_length))
            data["section_rate"].append(float(elem.section_rate))
            data["delta_neg"].append(float(elem.delta_neg))
            data["delta_pos"].append(float(elem.delta_pos))
            data["n_trials"].append(int(elem.n_trials))
            data["event_probability"].append(
                float(self._event_probability.get(nn, np.nan)))
            data["rate_scattering"].append(float(elem.rate_scattering))
            data["rate_tail"].append(float(elem.rate_tail))
            data["energy_loss_rate"].append(float(elem.energy_loss_rate))

            particles = (particles_by_element.get(nn)
                         if particles_by_element is not None else None)
            if include_particles:
                if particles is None:
                    data["num_particles"].append(0)
                    data["sum_weight"].append(np.nan)
                else:
                    # `particles.x` spans the full capacity: count the
                    # allocated slots only (unallocated ones carry
                    # state = -999999999).
                    allocated = particles.state > -999999999
                    data["num_particles"].append(int(np.sum(allocated)))
                    data["sum_weight"].append(
                        float(np.sum(particles.weight[allocated])))
            if include_tracking:
                if particles is None:
                    data["num_lost_particles"].append(0)
                    data["sum_lost_weight"].append(np.nan)
                else:
                    lost = particles.state == 0
                    data["num_lost_particles"].append(int(np.sum(lost)))
                    data["sum_lost_weight"].append(
                        float(np.sum(particles.weight[lost])))

        for kk, vv in data.items():
            data[kk] = np.array(vv)
        return xt.Table(data)