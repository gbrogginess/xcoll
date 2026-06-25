# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
import xtrack as xt
import numpy as np
import pandas as pd
from collections import Counter

class BeamGasScattering(xt.BeamElement):
    isthick = False
    iscollective = True

    def __init__(self,
                 name=None,
                 manager=None,
                 ds=-1.0,
                 atomic_densities={},
                 **kwargs):
        
        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return
        
        super().__init__(**kwargs)

    def _configure(self, **kwargs):
        config_allowed = {'name', 'manager', 'ds', 'atomic_densities'}

        unknown = set(kwargs) - config_allowed
        if unknown:
            bad = ", ".join(sorted(unknown))
            raise KeyError(f"Unsupported configure() keys: {bad}")
        
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)

        # Derived attributes
        xsec = self.manager.brems_xsec or self.manager.coulomb_xsec
        mfp = {
            kk: 1 / (self.atomic_densities[kk] * xsec[kk]) if self.atomic_densities[kk] > 0
                else 1 / xsec[kk]  # if density is zero, assume n = 1 at/m^3 to avoid division by zero
            for kk in self.atomic_densities
        }
        mfp_tot = 1 / sum(1 / m for m in mfp.values())
        self.mfp_step = self.ds / mfp_tot

        suffix = '_brems' if self.manager.brems is not None else '_coulomb'
        self.interactions = [nn + suffix for nn in self.atomic_densities.keys()]
        self.interaction_probability = [mfp_tot / mfp[nn] for nn in self.atomic_densities.keys()]

        self._log_row_idx = np.where(self.manager.interactions_log['name'] == self.name)[0][0]


    def track(self, particles):
        if self.manager.scattering_enabled:
            if not self.manager._particles_initialised:
                raise ValueError('Particles not initialised for beam-gas tracking. Call BeamGasManager.initialise_particles(particles) first.')

            # Active primary particles and their updated path length to next interaction
            mask_active_primaries = (particles.state > 0) & (particles.particle_id == particles.parent_particle_id)
            pp = particles.filter(mask_active_primaries)
            new = pp.weight - self.mfp_step

            # Candidate interacting: those whose path length has been exhausted
            mask_candidate_interacting = new < 0
            candidate_interacting_particle_ids = pp.particle_id[mask_candidate_interacting]

            # Exclude particles that already interacted
            mask_already_interacted = np.isin(
                candidate_interacting_particle_ids,
                np.fromiter(self.manager.interacted_particle_ids, dtype=np.int64)
            )
            interacting_particle_ids = candidate_interacting_particle_ids[~mask_already_interacted]
            self.manager.interacted_particle_ids.update(interacting_particle_ids)

            # Interacting particles: sample a new path length to next interaction
            mask_interacting = np.isin(particles.particle_id, interacting_particle_ids)
            particles.weight[mask_interacting] = -np.log(self.manager.rng.random(len(interacting_particle_ids)))

            # Non-interacting active primaries: update their remaining path length
            mask_noninteracting_in_pp = ~mask_candidate_interacting
            pp_noninteracting_ids = pp.particle_id[mask_noninteracting_in_pp]
            mask_noninteracting = np.isin(particles.particle_id, pp_noninteracting_ids)
            particles.weight[mask_noninteracting] = new[mask_noninteracting_in_pp]

            n_interactions = sum(mask_interacting)
            if n_interactions == 0:
                # If no interactions occur here, nothing more to do
                return
            
            interactions = self.manager.rng.choice(self.interactions,
                                                   p=self.interaction_probability,
                                                   size=n_interactions)

            gas_counter = Counter(ii.split('_')[0] for ii in interactions)
            interacting_particles = particles.filter(mask_interacting)

            px, py, delta = [], [], []
            beamgas_weight = []
            npp = 0
            for gas, ngas in gas_counter.items():
                mask = np.zeros(n_interactions, dtype=bool)
                mask[npp:npp+ngas] = True
                pp = interacting_particles.filter(mask)
                pp_ids = interacting_particle_ids[mask]

                if self.manager.brems is not None:
                    _px, _py, _delta = self.manager.brems[gas].sample_deflections(pp, ngas)
                    _bgw = [1.0] * ngas # brems unweighted
                elif self.manager.coulomb is not None:
                    _px, _py, _bgw = self.manager.coulomb[gas].sample_deflections(pp, ngas)
                    _delta = pp.delta.tolist()

                px.extend(_px); py.extend(_py); delta.extend(_delta); beamgas_weight.extend(_bgw)

                # Update interactions log with the correct gas species for these particles
                row_idx = self._log_row_idx
                current_ids = self.manager.interactions_log.at[row_idx, 'particle_id']
                current_interactions = self.manager.interactions_log.at[row_idx, 'interaction']

                if not isinstance(current_ids, list):
                    self.manager.interactions_log.at[row_idx, 'particle_id'] = pp_ids.tolist()
                    self.manager.interactions_log.at[row_idx, 'interaction'] = [gas] * ngas
                else:
                    self.manager.interactions_log.at[row_idx, 'particle_id'] = current_ids + pp_ids.tolist()
                    self.manager.interactions_log.at[row_idx, 'interaction'] = current_interactions + [gas] * ngas

                npp += ngas

            # Update particle object
            particles.px[mask_interacting] = px
            particles.py[mask_interacting] = py
            # Update the `delta` value of the particles object. `ptau` and `rvv` and
            # `rpp` are updated accordingly.
            # Ref: https://github.com/xsuite/xtrack/blob/f45c5720c246f34cd3db593829d83f3b0c61c3b8/xtrack/particles/particles.py#L1130
            delta_temp = particles.delta.copy()
            delta_temp[mask_interacting] = delta
            particles.update_delta(delta_temp)

            self.manager.beamgas_weight.update(
                dict(zip(interacting_particle_ids.tolist(), beamgas_weight)))
            # self.manager._nint = getattr(self.manager, '_nint', 0) + int(n_interactions)
        else:
            return