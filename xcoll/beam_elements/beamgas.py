# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
import xtrack as xt
import xpart as xp
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
                 element_index=-1,
                 integrated_scattering_rate=0.0,
                 **kwargs):
        
        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return
        
        super().__init__(**kwargs)

    def _configure(self, **kwargs):
        config_allowed = {'name', 's', 'element_index', 'manager', 'atomic_densities', 'integrated_scattering_rate'}

        unknown = set(kwargs) - config_allowed
        if unknown:
            bad = ", ".join(sorted(unknown))
            raise KeyError(f"Unsupported configure() keys: {bad}")
        
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)

        # # Derived attributes
        # xsec = self.manager.brems_xsec or self.manager.coulomb_xsec
        # mfp = {
        #     kk: 1 / (self.atomic_densities[kk] * xsec[kk]) if self.atomic_densities[kk] > 0
        #         else 1 / xsec[kk]  # if density is zero, assume n = 1 at/m^3 to avoid division by zero
        #     for kk in self.atomic_densities
        # }
        # mfp_tot = 1 / sum(1 / m for m in mfp.values())
        # self.mfp_step = self.ds / mfp_tot

        # suffix = '_brems' if self.manager.brems is not None else '_coulomb'
        # self.interactions = [nn + suffix for nn in self.atomic_densities.keys()]
        # self.interaction_probability = [mfp_tot / mfp[nn] for nn in self.atomic_densities.keys()]

        # self._log_row_idx = np.where(self.manager.interactions_log['name'] == self.name)[0][0]


    def scatter(self, num_particles, **kwargs):
        line = self.manager.line
        twiss = self.manager.twiss
        name = self.name

        nemitt_x = self.manager.nemitt_x
        nemitt_y = self.manager.nemitt_y
        sigma_z = self.manager.sigma_z

        if 'capacity' in kwargs.keys():
            capacity = kwargs['capacity']
        else:
            capacity = 1*num_particles

        # Extract W_matrix and particle_on_co from the already-computed twiss
        tw_init = twiss.get_twiss_init(at_element=name)

        # if scattering == 'on':
        #     self.scattering.disable()

        # Prepared matched Gaussian bunch
        x_norm, px_norm = xp.generate_2D_gaussian(num_particles)
        y_norm, py_norm = xp.generate_2D_gaussian(num_particles)

        # Helper for longitudinal distribution generation
        def _has_longitudinal_focusing(line):
            if line.iscollective:
                line = line._get_non_collective_line()

            for ee in line.elements:
                cls = ee.__class__.__name__

                if cls == 'Cavity':
                    if ee.voltage != 0:
                        return True

                elif cls == 'LineSegmentMap':
                    if ee.longitudinal_mode in ('nonlinear',
                                                'linear_fixed_qs',
                                                'linear_fixed_rf'):
                        return True

            return False

        # The longitudinal closed orbit need to be manually supplied
        zeta_co = twiss['zeta', name]
        delta_co = twiss['delta', name]

        if _has_longitudinal_focusing(line):
            zeta, delta = xp.generate_longitudinal_coordinates(
                line=line,
                num_particles=num_particles,
                distribution='gaussian',
                sigma_z=sigma_z
            )
        else:
            zeta = np.zeros(num_particles)
            delta = np.zeros(num_particles)

        particles = line.build_particles(
            _context=self._context,
            _capacity=capacity,
            x_norm=x_norm, px_norm=px_norm,
            y_norm=y_norm, py_norm=py_norm,
            zeta=zeta + zeta_co,
            delta=delta + delta_co,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            W_matrix=tw_init.W_matrix,
            particle_on_co=tw_init.particle_on_co
        )
        particles.at_element[:] = self.element_index
        particles.s[:] = self.s
        particles.start_tracking_at_element = -1

        # ------------------------------------------------------------------
        # Uniform-theta Monte Carlo scattering
        # ------------------------------------------------------------------
        if self.manager.process != 'coulomb':
            raise NotImplementedError(
                "scatter() with uniform-theta sampling is implemented for "
                "Coulomb only. Bremsstrahlung uses its own energy sampling.")

        rng     = self.manager.rng
        species = list(self.atomic_densities.keys())

        # Sampling window in theta: a Monte Carlo choice, common to all species
        # (the physical difference between species lives in sigma and dsigma/dtheta).
        theta_min, theta_max = self.manager.coulomb[species[0]].theta_lim
        theta_min = max(theta_min, 1e-9)   # avoid the dsigma/dtheta -> 0/0 at exactly 0

        theta = rng.uniform(theta_min, theta_max, num_particles)   # <-- UNIFORM
        phi   = rng.uniform(0.0, 2.0 * np.pi, num_particles)

        # --- Apply the deflection (exact 3D rotation; depends only on theta) ---
        px0 = np.asarray(particles.px[:num_particles])
        py0 = np.asarray(particles.py[:num_particles])
        d0  = np.asarray(particles.delta[:num_particles])

        pz0    = np.sqrt((1.0 + d0)**2 - px0**2 - py0**2)
        PP     = np.column_stack((px0, py0, pz0))
        norms  = np.linalg.norm(PP, axis=1, keepdims=True)   # = 1 + delta
        PP_HAT = PP / norms

        UU  = np.zeros_like(PP_HAT)
        tol = 1e-12
        mask = (PP_HAT[:, 0]**2 + PP_HAT[:, 1]**2) < tol**2
        UU[~mask] = np.stack([-PP_HAT[~mask, 1], PP_HAT[~mask, 0],
                              np.zeros(np.count_nonzero(~mask))], axis=1)
        UU[mask]  = np.array([1.0, 0.0, 0.0])
        UU /= np.linalg.norm(UU, axis=1, keepdims=True)
        VV  = np.cross(PP_HAT, UU)

        st = np.sin(theta)[:, None]; ct = np.cos(theta)[:, None]
        cp = np.cos(phi)[:, None];   sp = np.sin(phi)[:, None]
        scattered_dir = st * cp * UU + st * sp * VV + ct * PP_HAT
        PP_OUT = scattered_dir * norms

        particles.px[:num_particles] = PP_OUT[:, 0]
        particles.py[:num_particles] = PP_OUT[:, 1]
        # delta untouched: Coulomb scattering is elastic; the rotation preserves |p|.

        # --- Per-particle weight [Hz]: document Eq. (scattering_rate_weight) ---
        #   r_i = sum_s n_s * dsigma_s/dtheta(theta_i)        (angular shape)
        #   R_i = r_i / sum_k r_k * integrated_scattering_rate
        # _compute_dxsec already returns dsigma/dtheta (carries 2*pi*sin(theta)).
        shape = np.zeros(num_particles)
        for sp_name in species:
            shape += self.atomic_densities[sp_name] \
                     * self.manager.coulomb[sp_name]._compute_dxsec(theta)

        particles.weight[:num_particles] = \
            shape / shape.sum() * self.integrated_scattering_rate

        return particles
    

    def track(self, particles):
        return


    # def track(self, particles):
    #     if self.manager.scattering_enabled:
    #         if not self.manager._particles_initialised:
    #             raise ValueError('Particles not initialised for beam-gas tracking. Call BeamGasManager.initialise_particles(particles) first.')

    #         # Active primary particles and their updated path length to next interaction
    #         mask_active_primaries = (particles.state > 0) & (particles.particle_id == particles.parent_particle_id)
    #         pp = particles.filter(mask_active_primaries)
    #         new = pp.weight - self.mfp_step

    #         # Candidate interacting: those whose path length has been exhausted
    #         mask_candidate_interacting = new < 0
    #         candidate_interacting_particle_ids = pp.particle_id[mask_candidate_interacting]

    #         # Exclude particles that already interacted
    #         mask_already_interacted = np.isin(candidate_interacting_particle_ids, self.manager.interacted_particle_ids)
    #         interacting_particle_ids = candidate_interacting_particle_ids[~mask_already_interacted]
    #         self.manager.interacted_particle_ids.update(interacting_particle_ids)

    #         # Interacting particles: sample a new path length to next interaction
    #         mask_interacting = np.isin(particles.particle_id, interacting_particle_ids)
    #         particles.weight[mask_interacting] = -np.log(self.manager.rng.random(len(interacting_particle_ids)))

    #         # Non-interacting active primaries: update their remaining path length
    #         mask_noninteracting_in_pp = ~mask_candidate_interacting
    #         pp_noninteracting_ids = pp.particle_id[mask_noninteracting_in_pp]
    #         mask_noninteracting = np.isin(particles.particle_id, pp_noninteracting_ids)
    #         particles.weight[mask_noninteracting] = new[mask_noninteracting_in_pp]

    #         n_interactions = sum(mask_interacting)
    #         if n_interactions == 0:
    #             # If no interactions occur here, nothing more to do
    #             return
            
    #         interactions = self.manager.rng.choice(self.interactions,
    #                                                p=self.interaction_probability,
    #                                                size=n_interactions)

    #         gas_counter = Counter(ii.split('_')[0] for ii in interactions)
    #         interacting_particles = particles.filter(mask_interacting)

    #         px, py, delta = [], [], []
    #         npp = 0
    #         for gas, ngas in gas_counter.items():
    #             mask = np.zeros(n_interactions, dtype=bool)
    #             mask[npp:npp+ngas] = True
    #             pp = interacting_particles.filter(mask)
    #             pp_ids = interacting_particle_ids[mask]

    #             if self.manager.brems is not None:
    #                 # Bremsstrahlung
    #                 _px, _py, _delta = self.manager.brems[gas].sample_deflections(pp, ngas)
    #             elif self.manager.coulomb is not None:
    #                 # Coulomb scattering
    #                 _px, _py = self.manager.coulomb[gas].sample_deflections(pp, ngas)
    #                 _delta = pp.delta.tolist()

    #             px.extend(_px); py.extend(_py); delta.extend(_delta)

    #             # Update interactions log with the correct gas species for these particles
    #             row_idx = self._log_row_idx
    #             current_ids = self.manager.interactions_log.at[row_idx, 'particle_id']
    #             current_interactions = self.manager.interactions_log.at[row_idx, 'interaction']

    #             if not isinstance(current_ids, list):
    #                 self.manager.interactions_log.at[row_idx, 'particle_id'] = pp_ids.tolist()
    #                 self.manager.interactions_log.at[row_idx, 'interaction'] = [gas] * ngas
    #             else:
    #                 self.manager.interactions_log.at[row_idx, 'particle_id'] = current_ids + pp_ids.tolist()
    #                 self.manager.interactions_log.at[row_idx, 'interaction'] = current_interactions + [gas] * ngas

    #             npp += ngas

    #         # Update particle object
    #         particles.px[mask_interacting] = px
    #         particles.py[mask_interacting] = py
    #         # Update the `delta` value of the particles object. `ptau` and `rvv` and
    #         # `rpp` are updated accordingly.
    #         # Ref: https://github.com/xsuite/xtrack/blob/f45c5720c246f34cd3db593829d83f3b0c61c3b8/xtrack/particles/particles.py#L1130
    #         delta_temp = particles.delta.copy()
    #         delta_temp[mask_interacting] = delta
    #         particles.update_delta(delta_temp)
    #     else:
    #         return