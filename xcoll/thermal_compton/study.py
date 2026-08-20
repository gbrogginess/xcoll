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
        Rate carried by the generated (above threshold) macro-particles [1/s].
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
                 n_macroparticles=1000, n_trials=100,
                 delta_threshold=1e-3,
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
        n_macroparticles : int, optional
            Number of macro-leptons drawn from the local beam distribution at
            each scattering element.
        n_trials : int, optional
            Number of scattering trials per macro-lepton.
        delta_threshold : float, optional
            Thinning threshold on the change of the relative momentum
            deviation.  Must be well below the ring momentum acceptance: it
            controls the computational cost, not the physics.
        max_events_per_macro : int or None, optional
            Capacity of the per-macro-particle output slice.
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
            Enable the exact early veto of trials that cannot reach the
            threshold.  Only affects speed, not the result.
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
        self.n_trials = int(n_trials)
        self.delta_threshold = float(delta_threshold)
        self.max_events_per_macro = max_events_per_macro
        self.section_assignment = section_assignment
        self.n_sigma_cut = n_sigma_cut
        self.enable_tail_veto = bool(enable_tail_veto)
        self.seed = seed
        self.kwargs = kwargs

        if self.n_trials < 1:
            raise ValueError("`n_trials` must be at least 1.")
        if self.n_macroparticles < 1:
            raise ValueError("`n_macroparticles` must be at least 1.")
        if self.delta_threshold < 0:
            raise ValueError("`delta_threshold` must be non-negative.")

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

    ############################################################
    # Configuration
    ############################################################
    def initialise(self):
        """
        Configure all the thermal Compton scattering elements of the study.

        For each element this method:

        1. assigns the section of the ring it represents,
        2. computes the absolute trial rate of that section,
        3. stores the local optics, closed orbit and beam parameters on the
           element, so that :meth:`ThermalComptonScattering.scatter` can draw
           the local beam distribution and weight the generated
           macro-particles correctly.

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

        for nn, ss, ll in zip(self.elements, s_all, lengths):
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
                n_trials=self.n_trials,
                delta_threshold=self.delta_threshold,
                max_events_per_macro=self.max_events_per_macro,
                enable_tail_veto=self.enable_tail_veto,
                n_sigma_cut=self.n_sigma_cut,
            )

        self._initialised = True

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
        delta_generated = {}

        seed = self.seed
        if seed is None:
            seed = int(np.random.randint(1, 2**31 - 1))

        if generate_particles or track:
            for ii, nn in enumerate(self.elements):
                particles = self.line[nn].scatter(seed=seed + ii)
                # Indexed by particle_id: tracking reorganises the array but
                # preserves the particle ids.
                delta_generated[nn] = particles.delta.copy()
                if track:
                    self.line.track(particles,
                                    ele_start=nn, ele_stop=nn,
                                    num_turns=n_turns,
                                    with_progress=with_progress)
                particles_by_element[nn] = particles
            merged_particles = xt.Particles.merge(
                list(particles_by_element.values()))

        rate_scattering = float(sum(self.line[nn].rate_scattering
                                    for nn in self.elements))
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
            self._check_threshold(particles_by_element, delta_generated)

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

    def _check_threshold(self, particles_by_element, delta_generated,
                         margin=1.5):
        """
        Warn if the thinning threshold is too close to the momentum acceptance.

        The loss rate is only complete if *every* particle able to be lost has
        been retained by the ``delta_threshold`` cut.  This is checked a
        posteriori by looking at the smallest generated ``|delta|`` among the
        lost particles: if losses occur just above the threshold, particles
        below it would have been lost too and the loss rate is underestimated.

        Parameters
        ----------
        particles_by_element : dict
            Tracked particles for each element.
        delta_generated : dict
            Relative momentum deviations at generation for each element.
        margin : float, optional
            Factor by which the smallest lost ``|delta|`` should exceed the
            threshold.

        Returns
        -------
        None
        """
        thr = self.delta_threshold
        if thr <= 0:
            return
        min_lost = np.inf
        for nn, part in particles_by_element.items():
            lost = part.state == 0
            if not np.any(lost):
                continue
            pid = np.asarray(part.particle_id[lost], dtype=np.int64)
            min_lost = min(min_lost,
                           float(np.min(np.abs(delta_generated[nn][pid]))))
        if np.isfinite(min_lost) and min_lost < margin*thr:
            warnings.warn(
                f"Particles are lost with |delta| as small as {min_lost:.2e} "
                f"at generation, close to the thinning threshold "
                f"{thr:.2e}. The loss rate is likely underestimated: reduce "
                f"`delta_threshold`.", UserWarning, stacklevel=3)

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
                    data["num_particles"].append(int(len(particles.x)))
                    data["sum_weight"].append(float(np.sum(particles.weight)))
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