# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from dataclasses import dataclass

import numpy as np

import xtrack as xt

# The particle bookkeeping is the same as for beam-gas scattering: weighted
# primaries that interacted once, plus possible secondaries from collimators
from ..beamgas.analysis import (_cutoff_scan, _event_losses, _lost,
                                _sample_sums, _tracked)


def _cut_scale_scan(cut_scale, weight, lost, i_sample, bunch_intensity,
                    bins_per_decade):
    """
    Tabulate the loss rate as a function of the energy-loss cut.

    The events of all the markers are binned in the ratio of their energy
    loss to the cut of their marker, so that markers with different cuts
    (e.g. derived from the local momentum acceptance) share one scan: a row
    gives the loss rate of a simulation in which all the cuts are multiplied
    by ``cut_scale``.

    Parameters
    ----------
    cut_scale : ndarray
        Energy loss of each event divided by the cut of its marker (>= 1).
    weight, lost, i_sample : ndarray
        Weight, loss flag and sample index of each event.
    bunch_intensity : float
        Number of particles in the bunch.
    bins_per_decade : int
        Resolution of the scan.

    Returns
    -------
    table : xtrack.Table
        See :attr:`ThermalPhotonResult.cutoff_scan`.
    """
    hi = max(float(np.max(cut_scale)) * (1.0 + 1e-12), 1.0 + 1e-9)
    scan = _cutoff_scan((1.0, hi), cut_scale, weight, lost, i_sample,
                        bunch_intensity, bins_per_decade)
    return xt.Table({
        'cut_scale': np.asarray(scan['cut']),
        'num_events': np.asarray(scan['num_events']),
        'num_lost': np.asarray(scan['num_lost']),
        'loss_probability': np.asarray(scan['loss_probability']),
        'rate_tracking': np.asarray(scan['rate_tracking']),
        'rate_tracking_error': np.asarray(scan['rate_tracking_error']),
        'lifetime_tracking': np.asarray(scan['lifetime_tracking']),
    }, index='cut_scale')


@dataclass
class ThermalPhotonResult:
    """
    Result of a thermal-photon scattering simulation.

    It is returned by :meth:`xcoll.ThermalPhotonGenerator.analyse` and
    :meth:`xcoll.ThermalPhotonGenerator.run`. The result separates quantities
    inferred from the cross sections (the interaction rates configured on the
    markers) from quantities obtained from the particles (the generated and
    the lost weight).

    Parameters
    ----------
    element_names : list
        Names of the thermal-photon markers included in the result.
    local_rates : xtrack.Table
        Per-marker diagnostics table. It always contains ``name``, ``s``,
        ``ds``, ``temperature``, ``energy_loss_cut``, ``mean_free_path``,
        ``local_rate``, ``interaction_rate`` and ``total_rate``. When the
        particles of each marker are known, it also contains
        ``num_particles`` and ``sum_weight`` and, once tracked,
        ``num_lost_particles`` and ``sum_lost_weight``.
    rate_scattering : float
        Total interaction rate represented by the result [1/s]. Only the
        scatterings that are actually generated are counted, i.e. those with
        a relative energy loss above ``energy_loss_cut``.
    lifetime_scattering : float
        Beam lifetime that would result if every generated interaction led to
        a loss [s]. It is a lower limit on the thermal-photon lifetime.
    rate_total : float
        Rate of all the scatterings on thermal photons, whatever the energy
        loss [1/s]. Most of them leave the particle inside the momentum
        acceptance.
    lifetime_total : float
        Beam lifetime that would result if every scattering on a thermal
        photon led to a loss [s]: the absolute lower limit, ``tau_min`` of
        Burkhardt's note.
    rate_tracking : float or None
        Loss rate: weighted rate of the scattered particles lost in tracking
        [1/s]. ``None`` if the particles were not tracked.
    lifetime_tracking : float or None
        Thermal-photon lifetime, inferred from the loss rate
        ``rate_tracking`` [s]. ``None`` if the particles were not tracked.
    tracked : bool
        Whether the particles were tracked.
    rate_generated : float or None
        Total weight of the scattered particles [1/s]. ``None`` without
        particles.
    rate_tracking_error : float or None
        Monte Carlo standard error of ``rate_tracking`` [1/s]. The scattered
        particles are independent events, so it follows from the spread of
        their contributions within each sample. ``None`` unless the particles
        of each marker are known.
    lifetime_tracking_error : float or None
        Monte Carlo standard error of ``lifetime_tracking`` [s], propagated
        from ``rate_tracking_error`` to first order.
    cutoff_scan : xtrack.Table or None
        Loss rate as a function of the energy-loss cut, obtained from the
        single tracked sample: all the events carry the same weight, so the
        events above a higher cut are an unbiased sample of a simulation
        generated with that cut. One row per log-spaced ``cut_scale``, the
        factor by which the cut of every marker is multiplied (from 1, the
        configured cuts, upwards), with columns ``cut_scale``,
        ``num_events`` and ``num_lost`` (in ``[cut_scale, next cut_scale)``),
        ``loss_probability`` (weighted fraction of those events that is
        lost), and ``rate_tracking``, ``rate_tracking_error`` and
        ``lifetime_tracking`` that a simulation with the cuts multiplied by
        ``cut_scale`` would give. The configured cut is low enough if the
        events just above it are not lost, i.e. if ``loss_probability``
        vanishes and ``lifetime_tracking`` is flat in the first rows.
        ``None`` unless the records of the interactions are given.
    particles_by_element : dict or None
        Mapping ``{element_name: xtrack.Particles}`` with the particles of
        each marker, when kept.
    particles : xtrack.Particles or None
        Merged particles, when kept.
    lost_particles : xtrack.Particles or None
        Subset of ``particles`` lost in tracking, when kept and if any is
        lost. It is a copy taken right after tracking.
    interaction_log : xtrack.Table or None
        Record of the interactions, with columns ``name``, ``replica``,
        ``s``, ``particle_id``, ``energy_loss``, ``photon_energy``,
        ``theta``, ``incident_photon_energy`` and ``weight``.
        ``particle_id`` refers to the sample of the marker and replica of the
        same row. ``None`` unless the records are given.

    Attributes
    ----------
    element_names, local_rates, rate_scattering, lifetime_scattering,
    rate_total, lifetime_total, rate_tracking, lifetime_tracking, tracked,
    rate_generated, rate_tracking_error, lifetime_tracking_error,
    cutoff_scan, particles_by_element, particles, lost_particles,
    interaction_log
        See above.
    """
    element_names: list
    local_rates: xt.Table
    rate_scattering: float
    lifetime_scattering: float
    rate_total: float
    lifetime_total: float
    rate_tracking: float | None
    lifetime_tracking: float | None
    tracked: bool
    rate_generated: float | None = None
    rate_tracking_error: float | None = None
    lifetime_tracking_error: float | None = None
    cutoff_scan: xt.Table | None = None
    particles_by_element: dict | None = None
    particles: xt.Particles | None = None
    lost_particles: xt.Particles | None = None
    interaction_log: xt.Table | None = None
