# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from warnings import warn

import numpy as np
from scipy import constants as sc

import xtrack as xt
from xtrack.particles import pdg

from ..headers.particle_states import LOST_ON_BEAMGAS
from .study import (BeamGasStudy, BeamGasResult, PDG_ID_PROTON,
                    _allocated_mask, _lost_mask, _cutoff_scan_table)
from .proton_cross_sections import (
    PROTON_PROCESSES, ELASTIC_VARIANTS, ALL_PROTON_PROCESSES, MILLIBARN,
    ProtonNucleusCrossSections, ProtonAbsorptionCalculator,
    ProtonElasticCalculator, ProtonQuasiElasticCalculator,
    ProtonDiffractionCalculator, ProtonKnockOnCalculator)


def resolve_proton_processes(process):
    """
    Validate the beam-gas processes requested for a proton beam.

    Parameters
    ----------
    process : str or sequence of str
        A process name, a sequence of them, or ``'all'`` for
        ``('absorption', 'elastic', 'quasi_elastic', 'diffractive',
        'knock_on')``.

    Returns
    -------
    processes : tuple of str
        Validated process names, without duplicates, in the order given.
    """
    if isinstance(process, str):
        processes = (ALL_PROTON_PROCESSES if process.lower() == 'all'
                     else (process,))
    else:
        processes = tuple(process)
    if len(processes) == 0:
        raise ValueError("At least one beam-gas process must be requested.")

    resolved = []
    for pp in processes:
        if not isinstance(pp, str):
            raise ValueError(f"Invalid beam-gas process {pp!r}.")
        pp = pp.lower()
        if pp in ('brems', 'bremsstrahlung', 'bremss'):
            raise ValueError(
                "Bremsstrahlung ('brems') is only available for electron and "
                "positron beams. For proton beams choose among "
                f"{PROTON_PROCESSES} or 'all'.")
        if pp not in PROTON_PROCESSES:
            raise ValueError(
                f"Unknown beam-gas process {pp!r} for a proton beam. Choose "
                f"among {PROTON_PROCESSES} or 'all'.")
        if pp not in resolved:
            resolved.append(pp)

    elastic = [pp for pp in resolved if pp in ELASTIC_VARIANTS]
    if len(elastic) > 1:
        raise ValueError(
            f"The elastic variants {ELASTIC_VARIANTS} describe the same "
            f"channel and cannot be combined (got {elastic}).")
    return tuple(resolved)


class ProtonBeamGasStudy(BeamGasStudy):
    """
    Monte Carlo study of proton-residual-gas interactions in a line.

    Returned by :class:`xcoll.BeamGasStudy` when the reference particle of
    the line is a proton. See :meth:`__init__` for the parameters.
    """

    def __init__(self, line=None, gas_density=None, process='all',
                 elements=None, twiss=None,
                 nemitt_x=None, nemitt_y=None,
                 gemitt_x=None, gemitt_y=None,
                 sigma_z=None, sigma_delta=None,
                 bunch_intensity=1.0,
                 n_scattering_events=None,
                 coulomb_theta=(1e-7, 50e-3),
                 interference=True,
                 sd_scale=1.0,
                 knock_on_cut=1e-6,
                 seed=None, brems_energy_cut=None, **kwargs):
        """
        Build a proton beam-gas study.

        The study generates, at each :class:`xcoll.BeamGasScattering`
        element, a weighted sample of protons that have undergone one
        interaction with the residual gas, stratified by process, and
        tracks them to obtain the beam-gas loss rate, lifetime and loss
        locations. The physics models are described in
        :mod:`xcoll.beamgas.proton_cross_sections`.

        Parameters
        ----------
        line : xtrack.Line
            Line containing the :class:`xcoll.BeamGasScattering` elements.
            Its reference particle must be a proton (PDG id 2212) above
            91 GeV of kinetic energy.
        gas_density : xtrack.Table
            Residual-gas density profile: a column ``s`` and one column per
            chemical element holding the *atomic* density [atoms/m^3]
            (e.g. ``H = 2 n_H2 + 4 n_CH4``), as for electron beams.
        process : str or sequence of str, optional
            Processes to simulate, among ``'absorption'``, ``'elastic'``,
            ``'nuclear_elastic'``, ``'coulomb'``, ``'quasi_elastic'``,
            ``'diffractive'`` and ``'knock_on'``, or ``'all'`` (default) for
            absorption, elastic (hadronic + Coulomb + interference),
            quasi-elastic, diffractive and knock-on.
        elements, twiss, nemitt_x, nemitt_y, gemitt_x, gemitt_y, sigma_z, \
        sigma_delta, bunch_intensity, seed, **kwargs
            As for :class:`xcoll.BeamGasStudy`.
        n_scattering_events : int or dict
            Number of Monte Carlo events generated per scattering element
            and per process, or a mapping ``{process: number}`` covering all
            the processes of the study.
        coulomb_theta : tuple of float, optional
            Window of the polar scattering angle generated for the elastic
            channel [rad]. The lower limit cuts the Coulomb part (the
            hadronic part below it is negligible); the elastic scattering
            above the upper limit is not generated and is reported as
            ``BeamGasResult.rate_above_theta_max``. Default ``(1e-7, 50e-3)``.
        interference : bool, optional
            Include the Coulomb-nuclear interference in the elastic channel.
            Default ``True``.
        sd_scale : float, optional
            Scale factor of the single-diffraction cross section, see
            :class:`xcoll.beamgas.ProtonDiffractionCalculator`. The
            absorption cross section is reduced accordingly, so that the
            inelastic cross section is unchanged. Default 1.
        knock_on_cut : float, optional
            Minimum energy transferred to a knock-on electron, as a fraction
            of the proton energy. Default 1e-6.

        Returns
        -------
        None
        """
        if brems_energy_cut is not None:
            raise ValueError("`brems_energy_cut` only applies to electron and "
                             "positron beams.")
        self.interference = bool(interference)
        self.sd_scale = float(sd_scale)
        self.knock_on_cut = float(knock_on_cut)
        self._n_events_spec = n_scattering_events

        n_base = n_scattering_events
        if isinstance(n_scattering_events, dict):
            if any(int(vv) < 0 for vv in n_scattering_events.values()):
                raise ValueError("`n_scattering_events` must be "
                                 "non-negative.")
            n_base = max((int(vv) for vv in n_scattering_events.values()),
                         default=0)

        super().__init__(line=line, gas_density=gas_density,
                         process=process, elements=elements, twiss=twiss,
                         nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                         gemitt_x=gemitt_x, gemitt_y=gemitt_y,
                         sigma_z=sigma_z, sigma_delta=sigma_delta,
                         bunch_intensity=bunch_intensity,
                         n_scattering_events=n_base,
                         coulomb_theta=coulomb_theta, seed=seed, **kwargs)

        if isinstance(n_scattering_events, dict):
            spec = {kk.lower(): int(vv)
                    for kk, vv in n_scattering_events.items()}
            missing = [pp for pp in self.process if pp not in spec]
            if missing:
                raise ValueError(f"`n_scattering_events` misses the "
                                 f"processes {missing}.")
            self.n_events = {pp: spec[pp] for pp in self.process}
        else:
            self.n_events = {pp: self.n_scattering_events
                             for pp in self.process}

    def __repr__(self):
        return (f"<ProtonBeamGasStudy process={'+'.join(self.process)}, "
                f"{len(self.elements)} elements, "
                f"gas={'+'.join(self.gas_species)}>")

    # ######################################################## #
    # Hooks of BeamGasStudy
    # ######################################################## #
    def _validate_beam(self, particle_ref, pdg_id):
        """
        Check that the beam is made of protons.

        Parameters
        ----------
        particle_ref : xtrack.Particles
            Reference particle of the line.
        pdg_id : int
            PDG id of the reference particle.

        Returns
        -------
        q0 : float
            Charge of the proton, +1.
        """
        if pdg_id != PDG_ID_PROTON:
            raise ValueError(
                f"ProtonBeamGasStudy needs a proton beam, but the reference "
                f"particle is a {pdg.get_name_from_pdg_id(pdg_id)} "
                f"(PDG id {pdg_id}).")
        if not np.isclose(float(particle_ref.q0), 1.0):
            raise ValueError(
                f"The reference particle is a proton, which must have "
                f"q0=+1, but it has q0={float(particle_ref.q0):+g}.")
        return 1.0

    def _resolve_study_process(self, process):
        """
        Validate the processes of the study.

        Parameters
        ----------
        process : str or sequence of str
            See :func:`resolve_proton_processes`.

        Returns
        -------
        processes : tuple of str
            Validated process names.
        """
        return resolve_proton_processes(process)

    def _build_calculators(self):
        """
        Build the model of each process and gas species.

        Parameters
        ----------
        None

        Returns
        -------
        calculators : dict
            Mapping ``{process: {element_symbol: calculator}}``.
        """
        cross_sections = {kk: ProtonNucleusCrossSections(Z, self.p0c)
                          for kk, Z in self.atomic_numbers.items()}
        theta = self.coulomb_theta

        def build(process, Z, xs):
            if process == 'absorption':
                return ProtonAbsorptionCalculator(
                    Z, self.p0c, sd_scale=self.sd_scale, cross_sections=xs)
            if process == 'elastic':
                return ProtonElasticCalculator(
                    Z, self.p0c, theta_lim=theta,
                    interference=self.interference, cross_sections=xs)
            if process == 'coulomb':
                return ProtonElasticCalculator(
                    Z, self.p0c, theta_lim=theta, nuclear=False,
                    cross_sections=xs)
            if process == 'nuclear_elastic':
                return ProtonElasticCalculator(
                    Z, self.p0c, theta_lim=(0.0, theta[1]), coulomb=False,
                    cross_sections=xs)
            if process == 'quasi_elastic':
                return ProtonQuasiElasticCalculator(Z, self.p0c,
                                                    cross_sections=xs)
            if process == 'diffractive':
                return ProtonDiffractionCalculator(
                    Z, self.p0c, sd_scale=self.sd_scale, cross_sections=xs)
            if process == 'knock_on':
                return ProtonKnockOnCalculator(
                    Z, self.p0c, cut=self.knock_on_cut, cross_sections=xs)
            raise ValueError(process)

        return {pp: {kk: build(pp, Z, cross_sections[kk])
                     for kk, Z in self.atomic_numbers.items()}
                for pp in self.process}

    def _compute_xsecs(self):
        """
        Compute the generated cross section of each process and species.

        Sets :attr:`process_xsecs` to ``{process: {species: xsec}}`` and
        returns the total over the processes of each species, which is what
        :meth:`initialise_beamgas` needs for the interaction rate.

        Parameters
        ----------
        None

        Returns
        -------
        xsecs : dict
            Mapping ``{element_symbol: total cross section [m^2]}``.
        """
        self.process_xsecs = {
            pp: {kk: cc.compute_xsec() for kk, cc in calcs.items()}
            for pp, calcs in self.calculators.items()}
        return {kk: sum(self.process_xsecs[pp][kk] for pp in self.process)
                for kk in self.gas_species}

    # ######################################################## #
    # Initialisation
    # ######################################################## #
    def initialise_beamgas(self, element=None, verbose=True):
        """
        Compute and configure the beam-gas interaction rates in the lattice.

        As :meth:`xcoll.BeamGasStudy.initialise_beamgas`, and additionally
        configures each element with the interaction rate of every process
        of the represented section, with the number of events to generate
        per process, and with the RF bucket, at whose separatrix the
        longitudinal distribution of the generated protons is truncated.

        Parameters
        ----------
        element : str or None, optional
            If ``None`` (default), all the scattering elements of the study
            are initialised; otherwise only the named one.
        verbose : bool, optional
            If ``True`` (default), print one line per initialised element.

        Returns
        -------
        None
        """
        super().initialise_beamgas(element=element, verbose=verbose)
        line = self.line
        elements = self.elements if element is None else [element]
        f_rev = self._f_rev
        bucket = self._rf_bucket()
        if bucket is not None:
            bucket = (*bucket, float(self.particle_ref.beta0[0]))
        for nn in elements:
            elem = line[nn]
            integrated = self._integrated_atomic_densities(
                float(elem.s) - float(elem.ds), float(elem.s))
            rates = {pp: float(self.bunch_intensity*f_rev*sum(
                         integrated[kk]*self.process_xsecs[pp][kk]
                         for kk in self.gas_species))
                     for pp in self.process}
            elem._configure(_process_xsecs=self.process_xsecs,
                            _process_rates=rates,
                            _n_events=dict(self.n_events),
                            _rf_bucket=bucket)

    # ######################################################## #
    # RF bucket
    # ######################################################## #
    def _rf_bucket(self):
        """
        Height of the RF bucket and RF frequency, from the line cavities.

        The bucket is approximated as stationary (the synchrotron-radiation
        energy loss of protons is negligible against the RF voltage), with
        a single RF frequency.

        Parameters
        ----------
        None

        Returns
        -------
        bucket : tuple of float or None
            ``(delta_max, f_rf)``: bucket half-height and RF frequency [Hz],
            or ``None`` if the line has no active cavity.
        """
        line = self.line
        tab = line.get_table()
        voltage = 0.0
        frequencies = []
        for nn in tab.rows[tab.element_type == 'Cavity'].name:
            cav = line[nn]
            if cav.voltage > 0 and cav.frequency > 0:
                voltage += float(cav.voltage)
                frequencies.append(float(cav.frequency))
        if voltage <= 0:
            return None
        f_rf = frequencies[0]
        if not np.allclose(frequencies, f_rf, rtol=1e-6):
            warn("The line has cavities at different frequencies; the RF "
                 "bucket is estimated with the first one.", stacklevel=3)
        eta = abs(float(self.twiss.slip_factor))
        beta0 = float(self.particle_ref.beta0[0])
        energy0 = float(self.particle_ref.energy0[0])
        harmonic = f_rf*self.line.get_length()/(beta0*sc.c)
        delta_max = np.sqrt(2*voltage/(np.pi*harmonic*eta*beta0**2*energy0))
        return float(delta_max), f_rf

    def _out_of_bucket(self, particles, elem, bucket):
        """
        Mask of the surviving particles outside the RF bucket.

        Uses the invariant of the stationary bucket,
        ``((delta - delta_s)/delta_max)^2 + sin^2(phi/2) > 1`` with
        ``phi = 2 pi f_rf (zeta - zeta_s)/(beta0 c)``.

        Parameters
        ----------
        particles : xtrack.Particles
            Tracked particles, sitting at ``elem``.
        elem : xcoll.BeamGasScattering
            Element where the particles are.
        bucket : tuple of float
            Output of :meth:`_rf_bucket`.

        Returns
        -------
        mask : ndarray of bool
            ``True`` for the surviving particles outside the bucket.
        """
        delta_max, f_rf = bucket
        beta0 = float(self.particle_ref.beta0[0])
        phi = 2*np.pi*f_rf*(particles.zeta - elem.zeta_co)/(beta0*sc.c)
        invariant = (((particles.delta - elem.delta_co)/delta_max)**2
                     + np.sin(0.5*phi)**2)
        return (particles.state > 0) & (invariant > 1.0)

    def _drift_out_of_bucket(self, particles_by_element, masks,
                             drift_per_turn, max_drift_turns,
                             drift_turns_per_step, with_progress):
        """
        Track the out-of-bucket survivors with an accelerated energy loss.

        Protons outside the RF bucket are not restored by the RF and slowly
        lose energy to synchrotron radiation until they reach the momentum
        acceptance of the machine. Instead of tracking the ~1e6 turns this
        takes at LHC top energy, their relative momentum is lowered by
        ``drift_per_turn`` per turn (applied every ``drift_turns_per_step``
        turns), which preserves where they are lost as long as the decrement
        stays small compared with the momentum resolution of the limiting
        aperture.

        The survivors of all the scattering elements first complete their
        turn up to the end of the line, and are then tracked together from
        its start. The results are copied back into ``particles_by_element``.

        Parameters
        ----------
        particles_by_element : dict
            Mapping ``{element_name: xtrack.Particles}``, modified in place.
        masks : dict
            Mapping ``{element_name: mask}`` of the out-of-bucket survivors.
        drift_per_turn : float
            Relative momentum lost per turn.
        max_drift_turns : int
            Maximum number of turns.
        drift_turns_per_step : int
            Turns tracked between two momentum decrements.
        with_progress : bool or int
            Forwarded to :meth:`xtrack.Line.track`.

        Returns
        -------
        None
        """
        subsets = []
        index = []
        next_tag = 0
        for nn, mask in masks.items():
            if not mask.any():
                continue
            sub = particles_by_element[nn].filter(mask)
            allocated = sub.particle_id >= 0
            # Unique ids, so that the merge below keeps them
            original_ids = sub.particle_id[allocated].copy()
            tags = np.arange(next_tag, next_tag + original_ids.size)
            sub.particle_id[allocated] = tags
            next_tag += original_ids.size
            # Complete the turn up to the end of the line
            self.line.track(sub, ele_start=nn, num_turns=1,
                            with_progress=with_progress)
            subsets.append(sub)
            index.append((nn, original_ids, tags))
        if not subsets:
            return

        merged = xt.Particles.merge(subsets)
        turns = 0
        while turns < max_drift_turns and np.any(merged.state > 0):
            n_step = min(drift_turns_per_step, max_drift_turns - turns)
            alive = merged.state > 0
            new_delta = merged.delta.copy()
            new_delta[alive] -= drift_per_turn*n_step
            merged.update_delta(new_delta)
            self.line.track(merged, num_turns=n_step,
                            with_progress=with_progress)
            turns += n_step

        # Copy the final coordinates and states back
        merged_ids = merged.particle_id
        merged_order = np.argsort(merged_ids)
        for nn, original_ids, tags in index:
            particles = particles_by_element[nn]
            pos_merged = merged_order[np.searchsorted(
                merged_ids, tags, sorter=merged_order)]
            ids = particles.particle_id
            order = np.argsort(ids)
            pos = order[np.searchsorted(ids, original_ids, sorter=order)]
            with particles._bypass_linked_vars():
                for _, name in xt.Particles.per_particle_vars:
                    if name in ('particle_id', 'parent_particle_id'):
                        continue
                    getattr(particles, name)[pos] = \
                        getattr(merged, name)[pos_merged]

    # ######################################################## #
    # Running
    # ######################################################## #
    def run(self, *, track=False, n_turns=None, generate_particles=None,
            keep_particles=False, with_progress=False,
            out_of_bucket='report', drift_per_turn=1e-6,
            max_drift_turns=5000, drift_turns_per_step=10):
        """
        Run the configured proton beam-gas rate or loss study.

        As :meth:`xcoll.BeamGasStudy.run`, with the following additions.

        * Absorbed protons are lost at their interaction point (state
          ``LOST_ON_BEAMGAS``) and count as losses.
        * ``BeamGasResult.process_rates`` gives the interaction and loss
          rates of each process.
        * ``BeamGasResult.cutoff_scan`` is a mapping ``{process: table}`` for
          the processes generated above a lower cut (the elastic variants,
          in angle, and ``'knock_on'``, in energy transfer).
        * Protons that survive the ``n_turns`` outside the RF bucket are
          handled according to ``out_of_bucket``.

        Parameters
        ----------
        track, n_turns, generate_particles, keep_particles, with_progress
            As for :meth:`xcoll.BeamGasStudy.run`.
        out_of_bucket : {'report', 'drift'}, optional
            ``'report'`` (default): the rate of the protons that survive
            outside the RF bucket is given in
            ``BeamGasResult.rate_out_of_bucket`` and not counted as a loss.
            ``'drift'``: they are tracked further with an accelerated energy
            loss (see ``drift_per_turn``) until they are lost, which mimics
            the synchrotron-radiation energy loss that drives them to the
            momentum cleaning at top energy. Not meaningful at injection,
            where they coast.
        drift_per_turn : float, optional
            Relative momentum lost per turn in the ``'drift'`` mode.
            Default 1e-6.
        max_drift_turns : int, optional
            Maximum number of turns of the ``'drift'`` mode. Default 5000.
        drift_turns_per_step : int, optional
            Turns tracked between two momentum decrements in the ``'drift'``
            mode. Default 10.

        Returns
        -------
        result : BeamGasResult
            Study result.
        """
        if out_of_bucket not in ('report', 'drift'):
            raise ValueError("`out_of_bucket` must be 'report' or 'drift'.")
        if keep_particles and generate_particles is False:
            raise ValueError(
                "`keep_particles=True` is incompatible with "
                "`generate_particles=False`.")
        if track:
            if generate_particles is False:
                raise ValueError(
                    "`generate_particles=False` is incompatible with "
                    "`track=True`.")
            if n_turns is None:
                raise ValueError("`n_turns` is required when `track=True`.")
            generate_particles = True
        elif generate_particles is None:
            generate_particles = keep_particles

        rate_scattering = float(sum(self.line[nn].interaction_rate
                                    for nn in self.elements))
        bucket = self._rf_bucket() if track else None

        particles_by_element = {}
        merged_particles = None
        lost_particles = None
        rate_out_of_bucket = None
        if generate_particles:
            for nn in self.elements:
                particles = self.line[nn].scatter(rng=self.rng)
                if track:
                    self.line.track(particles, ele_start=nn, ele_stop=nn,
                                    num_turns=n_turns,
                                    with_progress=with_progress)
                particles_by_element[nn] = particles
            if bucket is not None:
                masks = {nn: self._out_of_bucket(pp, self.line[nn], bucket)
                         for nn, pp in particles_by_element.items()}
                if out_of_bucket == 'drift':
                    self._drift_out_of_bucket(
                        particles_by_element, masks, drift_per_turn,
                        max_drift_turns, drift_turns_per_step, with_progress)
                    masks = {nn: self._out_of_bucket(pp, self.line[nn],
                                                     bucket)
                             for nn, pp in particles_by_element.items()}
                rate_out_of_bucket = float(sum(
                    np.sum(particles_by_element[nn].weight[mm])
                    for nn, mm in masks.items()))
            merged_particles = xt.Particles.merge(
                list(particles_by_element.values()))

        rate_tracking = None
        lifetime_tracking = None
        rate_tracking_error = None
        lifetime_tracking_error = None
        cutoff_scan = None
        rate_above_theta_max = None
        events = None
        if generate_particles:
            events = self._event_losses(particles_by_element)
        if track:
            lost = merged_particles.filter(_lost_mask(merged_particles))
            rate_tracking = float(np.sum(lost.weight))
            lifetime_tracking = (
                np.inf if rate_tracking == 0
                else float(self.bunch_intensity/rate_tracking))
            lost_particles = lost

            rate_tracking_error = self._rate_error(events)
            lifetime_tracking_error = (
                np.nan if rate_tracking == 0
                else lifetime_tracking*rate_tracking_error/rate_tracking)
            cutoff_scan = self._cutoff_scans(events)
            if any(pp in ELASTIC_VARIANTS for pp in self.process):
                rate_above_theta_max = self._rate_above_theta_max()
            self._warn_on_proton_truncation(events, rate_tracking,
                                            rate_above_theta_max)
            if (out_of_bucket == 'report' and rate_out_of_bucket
                    and rate_out_of_bucket > 0.01*rate_tracking):
                warn(f"Protons outside the RF bucket survive the tracking "
                     f"with a rate of {rate_out_of_bucket:.3g} 1/s "
                     f"({100*rate_out_of_bucket/rate_tracking:.2g}% of the "
                     f"loss rate). They are not counted as losses; at top "
                     f"energy synchrotron radiation eventually brings them "
                     f"to the momentum cleaning. Use out_of_bucket='drift' "
                     f"to find where they are lost.", stacklevel=2)

        lifetime_scattering = (
            np.inf if rate_scattering == 0
            else float(self.bunch_intensity/rate_scattering))

        local_rates = self.local_rates(
            particles_by_element=(
                particles_by_element if generate_particles else None),
            include_tracking=track)
        process_rates = self._process_rates_table(events, track)

        interaction_log = None
        if keep_particles:
            interaction_log = self.interaction_log()
        else:
            particles_by_element = None
            merged_particles = None
            lost_particles = None

        return BeamGasResult(
            element_names=list(self.elements),
            gas_density=self.gas_density,
            local_rates=local_rates,
            rate_scattering=rate_scattering,
            lifetime_scattering=lifetime_scattering,
            rate_tracking=rate_tracking,
            lifetime_tracking=lifetime_tracking,
            tracked=track,
            rate_tracking_error=rate_tracking_error,
            lifetime_tracking_error=lifetime_tracking_error,
            cutoff_scan=cutoff_scan,
            rate_above_theta_max=rate_above_theta_max,
            particles_by_element=particles_by_element,
            particles=merged_particles,
            lost_particles=lost_particles,
            interaction_log=interaction_log,
            process_rates=process_rates,
            rate_out_of_bucket=rate_out_of_bucket,
        )

    # ######################################################## #
    # Diagnostics
    # ######################################################## #
    def _event_losses(self, particles_by_element):
        """
        Join the interaction log of every element with the tracked states.

        Parameters
        ----------
        particles_by_element : dict
            Mapping ``{element_name: xtrack.Particles}``.

        Returns
        -------
        events : dict of ndarray
            Per-event ``process``, ``theta``, ``energy_loss``, ``weight``,
            ``lost`` and ``i_stratum`` (index of the element and process).
        """
        n_proc = len(self.process)
        out = {kk: [] for kk in ('process', 'theta', 'energy_loss', 'weight',
                                 'lost', 'i_stratum')}
        for ii, nn in enumerate(self.elements):
            log = self.line[nn].scatter_log
            particles = particles_by_element[nn]
            allocated = _allocated_mask(particles)
            ids = particles.particle_id[allocated]
            order = np.argsort(ids)
            idx = order[np.searchsorted(ids, log['particle_id'],
                                        sorter=order)]
            i_proc = np.array([self.process.index(pp)
                               for pp in log['process']], dtype=int)
            out['process'].append(np.asarray(log['process']))
            out['theta'].append(np.asarray(log['theta'], dtype=float))
            out['energy_loss'].append(np.asarray(log['energy_loss'],
                                                 dtype=float))
            out['weight'].append(np.asarray(log['weight'], dtype=float))
            out['lost'].append(particles.state[allocated][idx] <= 0)
            out['i_stratum'].append(ii*n_proc + i_proc)
        return {kk: np.concatenate(vv) for kk, vv in out.items()}

    def _rate_error(self, events):
        """Monte Carlo standard error of the tracked loss rate [1/s], with
        each element and process an independent stratum."""
        n_strata = len(self.elements)*len(self.process)
        table = _cutoff_scan_table(
            np.ones_like(events['weight']), events['weight'],
            events['lost'], events['i_stratum'], n_strata=n_strata,
            window=(1.0, 10.0), bins_per_decade=1,
            bunch_intensity=self.bunch_intensity)
        return float(table.rate_tracking_error[0])

    def _generation_windows(self):
        """Lower-cut windows of the processes that have one: the elastic
        variants in polar angle [rad], knock-on in energy transfer [eV]."""
        windows = {}
        for pp in self.process:
            if pp in ('elastic', 'coulomb'):
                windows[pp] = ('theta', self.coulomb_theta)
            elif pp == 'knock_on':
                calc = next(iter(self.calculators[pp].values()))
                windows[pp] = ('energy_loss',
                               tuple(ee*1e9 for ee in calc.energy_lim))
        return windows

    def _cutoff_scans(self, events):
        """Loss rate as a function of the lower generation cut, for each
        process generated above a cut."""
        n_strata = len(self.elements)*len(self.process)
        scans = {}
        for pp, (column, window) in self._generation_windows().items():
            mask = events['process'] == pp
            if not mask.any():
                continue
            scans[pp] = _cutoff_scan_table(
                events[column][mask], events['weight'][mask],
                events['lost'][mask], events['i_stratum'][mask],
                n_strata=n_strata, window=window,
                bins_per_decade=self._CUTOFF_SCAN_BINS_PER_DECADE,
                bunch_intensity=self.bunch_intensity)
        return scans

    def _rate_above_theta_max(self):
        """
        Elastic interaction rate above ``coulomb_theta[1]`` [1/s], which is
        not generated.
        """
        theta_max = self.coulomb_theta[1]
        if theta_max >= np.pi:
            return 0.0
        variant = next(pp for pp in self.process if pp in ELASTIC_VARIANTS)
        xsecs = {}
        for kk, Z in self.atomic_numbers.items():
            calc = self.calculators[variant][kk]
            if 2*calc.p**2*2*np.sin(0.5*theta_max)**2 >= calc.chips.t_max:
                xsecs[kk] = 0.0
                continue
            xsecs[kk] = ProtonElasticCalculator(
                Z, self.p0c, theta_lim=(theta_max, np.pi),
                coulomb=calc.coulomb, nuclear=calc.nuclear,
                interference=calc.interference,
                cross_sections=calc.cross_sections).compute_xsec()
        rate = 0.0
        for nn in self.elements:
            elem = self.line[nn]
            integrated = self._integrated_atomic_densities(
                float(elem.s) - float(elem.ds), float(elem.s))
            rate += sum(integrated[kk]*xsecs[kk] for kk in self.gas_species)
        return float(self.bunch_intensity*self._f_rev*rate)

    def _warn_on_proton_truncation(self, events, rate_tracking,
                                   rate_above_theta_max):
        """
        Warn when a generation window visibly truncates the loss rate.

        Parameters
        ----------
        events : dict of ndarray
            Output of :meth:`_event_losses`.
        rate_tracking : float
            Tracked loss rate [1/s].
        rate_above_theta_max : float or None
            Elastic interaction rate above ``coulomb_theta[1]`` [1/s].

        Returns
        -------
        None
        """
        if rate_tracking <= 0:
            return
        contribution = events['weight']*events['lost']
        names = {'elastic': '`coulomb_theta[0]`',
                 'coulomb': '`coulomb_theta[0]`',
                 'knock_on': '`knock_on_cut`'}
        for pp, (column, window) in self._generation_windows().items():
            mask = events['process'] == pp
            near = mask & (events[column] < 2*window[0])
            fraction = float(np.sum(contribution[near]))/rate_tracking
            if fraction > self._LOWER_CUT_WARN_FRACTION:
                warn(f"{100*fraction:.2g}% of the tracked loss rate comes "
                     f"from '{pp}' events generated within a factor 2 of the "
                     f"lower cut {names[pp]}. The cut lies inside the range "
                     f"that causes losses, so the losses from below it are "
                     f"missing: lower {names[pp]}, and check that "
                     f"`BeamGasResult.cutoff_scan['{pp}'].lifetime_tracking` "
                     f"is flat in its first rows.", stacklevel=3)
        if (rate_above_theta_max is not None and rate_above_theta_max
                > self._UPPER_CUT_WARN_FRACTION*rate_tracking):
            warn(f"The elastic interaction rate above `coulomb_theta[1]`="
                 f"{self.coulomb_theta[1]:.3g} rad, which is not generated, "
                 f"amounts to "
                 f"{100*rate_above_theta_max/rate_tracking:.2g}% of the "
                 f"tracked loss rate. Raise `coulomb_theta[1]`, or add "
                 f"`rate_above_theta_max` to the loss rate.", stacklevel=3)

    def _process_rates_table(self, events, tracked):
        """
        Interaction and loss rates of each process.

        Parameters
        ----------
        events : dict of ndarray or None
            Output of :meth:`_event_losses`, if particles were generated.
        tracked : bool
            Whether the particles were tracked.

        Returns
        -------
        table : xtrack.Table
            One row per process with ``process``, ``xsec_mb`` (cross
            section per atom averaged over the gas composition of the study,
            weighted by the integrated density), ``interaction_rate``
            [1/s] and, if particles were generated, ``num_events`` and
            ``sum_weight`` [1/s]; if tracked, ``rate_tracking`` [1/s],
            ``rate_tracking_error`` [1/s] and ``loss_fraction``.
        """
        rates = {pp: sum(self.line[nn]._process_rates.get(pp, 0.0)
                         for nn in self.elements) for pp in self.process}
        s_table = np.asarray(self.gas_density['s'], dtype=float)
        column_density = {kk: float(np.trapezoid(
            np.asarray(self.gas_density[kk], dtype=float), s_table))
            for kk in self.gas_species}
        total_column = sum(column_density.values())
        data = {'process': np.array(self.process, dtype=object),
                'xsec_mb': np.array([
                    sum(column_density[kk]*self.process_xsecs[pp][kk]
                        for kk in self.gas_species)/total_column/MILLIBARN
                    if total_column > 0 else np.nan
                    for pp in self.process]),
                'interaction_rate': np.array([rates[pp]
                                              for pp in self.process])}
        if events is not None:
            data['num_events'] = np.array(
                [int(np.sum(events['process'] == pp)) for pp in self.process])
            data['sum_weight'] = np.array(
                [float(np.sum(events['weight'][events['process'] == pp]))
                 for pp in self.process])
        if tracked:
            n_strata = len(self.elements)*len(self.process)
            lost_rate, lost_error = [], []
            for pp in self.process:
                mask = events['process'] == pp
                table = _cutoff_scan_table(
                    np.ones(int(mask.sum())), events['weight'][mask],
                    events['lost'][mask], events['i_stratum'][mask],
                    n_strata=n_strata, window=(1.0, 10.0),
                    bins_per_decade=1, bunch_intensity=self.bunch_intensity)
                lost_rate.append(float(table.rate_tracking[0])
                                 if mask.any() else 0.0)
                lost_error.append(float(table.rate_tracking_error[0])
                                  if mask.any() else 0.0)
            data['rate_tracking'] = np.array(lost_rate)
            data['rate_tracking_error'] = np.array(lost_error)
            with np.errstate(invalid='ignore', divide='ignore'):
                data['loss_fraction'] = (data['rate_tracking']
                                         / data['interaction_rate'])
        return xt.Table(data, index='process')

    def local_rates(self, *, particles_by_element=None,
                    include_tracking=False):
        """
        Return an ``xt.Table`` with per-scattering-element diagnostics.

        As :meth:`xcoll.BeamGasStudy.local_rates`, with an additional column
        ``interaction_rate_<process>`` [1/s] per process.

        Parameters
        ----------
        particles_by_element : dict or None, optional
            Mapping from element name to the particles generated there.
        include_tracking : bool, optional
            Include the loss columns.

        Returns
        -------
        table : xtrack.Table
            Per-element diagnostics table.
        """
        table = super().local_rates(particles_by_element=particles_by_element,
                                    include_tracking=include_tracking)
        data = {cc: table[cc] for cc in table._col_names}
        for pp in self.process:
            data[f'interaction_rate_{pp}'] = np.array(
                [self.line[nn]._process_rates.get(pp, 0.0)
                 for nn in self.elements])
        return xt.Table(data)

    def interaction_log(self):
        """
        Return an ``xt.Table`` with the record of the generated interactions.

        Columns: ``name``, ``s``, ``particle_id``, ``gas``, ``process``,
        ``theta`` [rad], ``t`` [GeV^2], ``energy_loss`` [eV], ``mass_x2``
        [GeV^2], ``coulomb_fraction``, ``weight`` [1/s] and ``absorbed``.
        ``particle_id`` refers to the particles of the element named in the
        same row (see :meth:`xcoll.BeamGasStudy.interaction_log`).

        Parameters
        ----------
        None

        Returns
        -------
        table : xtrack.Table or None
            One row per generated interaction, or ``None`` if no particles
            have been generated yet.
        """
        logs = [(nn, self.line[nn].scatter_log) for nn in self.elements
                if self.line[nn].scatter_log is not None]
        if len(logs) == 0:
            return None
        columns = ('particle_id', 'gas', 'process', 'theta', 't',
                   'energy_loss', 'mass_x2', 'coulomb_fraction', 'weight',
                   'absorbed')
        data = {'name': [], 's': []}
        data.update({cc: [] for cc in columns})
        for nn, log in logs:
            n = len(log['particle_id'])
            data['name'].append(np.full(n, nn))
            data['s'].append(np.full(n, float(self.line[nn].s)))
            for cc in columns:
                data[cc].append(np.asarray(log[cc]))
        return xt.Table({kk: np.concatenate(vv) for kk, vv in data.items()})

    def absorption_events(self, result):
        """
        Source table of the absorbed protons, e.g. for a shower simulation.

        Parameters
        ----------
        result : BeamGasResult
            Result of :meth:`run` with ``keep_particles=True``.

        Returns
        -------
        table : xtrack.Table
            One row per absorbed macro-proton with the element ``name`` and
            ``s`` [m] of the interaction, the coordinates ``x``, ``px``,
            ``y``, ``py``, ``zeta`` and ``delta`` just before it, the total
            momentum ``pc`` [eV], the gas species ``gas`` with its ``Z`` and
            ``A``, and the ``weight`` [1/s] (the rate of such interactions).
        """
        if result.particles_by_element is None:
            raise ValueError("The result has no particles; run the study "
                             "with keep_particles=True.")
        cols = ('x', 'px', 'y', 'py', 'zeta', 'delta', 'weight')
        data = {'name': [], 's': [], 'gas': [], 'Z': [], 'A': [], 'pc': []}
        data.update({cc: [] for cc in cols})
        for nn, particles in result.particles_by_element.items():
            log = self.line[nn].scatter_log
            mask = particles.state == LOST_ON_BEAMGAS
            ids = particles.particle_id[mask]
            log_ids = np.asarray(log['particle_id'])
            order = np.argsort(log_ids)
            pos = order[np.searchsorted(log_ids, ids, sorter=order)]
            gas = np.asarray(log['gas'])[pos]
            n = int(mask.sum())
            data['name'].append(np.full(n, nn, dtype=object))
            data['s'].append(np.asarray(particles.s[mask]))
            data['gas'].append(gas)
            data['Z'].append(np.array(
                [self.atomic_numbers[gg] for gg in gas], dtype=int))
            data['A'].append(np.array(
                [self.calculators[self.process[0]][gg].A for gg in gas],
                dtype=int))
            data['pc'].append(np.asarray(particles.p0c[mask]
                                         * (1 + particles.delta[mask])))
            for cc in cols:
                data[cc].append(np.asarray(getattr(particles, cc)[mask]))
        return xt.Table({kk: np.concatenate(vv) for kk, vv in data.items()})
