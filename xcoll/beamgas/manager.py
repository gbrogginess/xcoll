# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xtrack as xt
import xcoll as xc

import numpy as np
import pandas as pd
import periodictable as pt

from scipy.constants import physical_constants

############################################################
# Constants
############################################################
ELECTRON_MASS_EV = xt.ELECTRON_MASS_EV
EV_TO_MEV = 1E-6
MEV_TO_EV = 1E6

C_LIGHT = physical_constants['speed of light in vacuum'][0]
ALPHA = physical_constants['fine-structure constant'][0]
HBAR = physical_constants['Planck constant over 2 pi in eV s'][0]
CLASSICAL_ELECTRON_RADIUS = physical_constants['classical electron radius'][0]
BOHR_RADIUS = physical_constants['Bohr radius'][0]
ELECTRON_REDUCED_COMPTON_WAVELENGTH = physical_constants['reduced Compton wavelength'][0]
ATOMIC_MASS_CONSTANT_EV = physical_constants['atomic mass constant energy equivalent in MeV'][0] * MEV_TO_EV

C_TF = 1/2 * (3*np.pi/4)**(2/3) # Thomas-Fermi constant

############################################################
# Helpers
############################################################
def _energy_from_momentum(p):
    return np.sqrt(p**2 + ELECTRON_MASS_EV**2)

def _atomic_number_from_symbol(element_symbol):
    element = getattr(pt, element_symbol)
    return element.number


class ElementData:
    def __init__(self, Z):
        self.Z = Z
        self.A = Z*2 if Z != 1 else 1
        self.mass = pt.elements[Z].mass * ATOMIC_MASS_CONSTANT_EV
        self.nuclear_radius = 1.27e-15 * self.A**0.27
        self.f_el, self.f_inel = self._compute_radiation_logarithms()
        self.f_c = self._compute_Coulomb_factor()
        self.f_Z_factor_1, self.f_Z_factor_2, self.f_Z = self._compute_Z_factors()
        self.gamma_factor, self.epsilon_factor = self._compute_gamma_epsilon_factors()


    def _compute_Coulomb_factor(self):
        K1, K2, K3, K4 = 0.0083, 0.20206, 0.0020, 0.0369
        a_Z = ALPHA * self.Z

        return (K1 * a_Z**4 + K2 + 1 / (1 + a_Z**2)) * a_Z**2 - (K3 * a_Z**4 + K4) * a_Z**4


    def _compute_radiation_logarithms(self):
        # Compute elastic and inelastic radiation logarithms

        # f_el and f_inel for low-Z elements (where Thomas-Fermi model is not accurate)
        # Computed using the Dirac-Fock atomic model
        F_EL_LOWZ = [0.0, 5.3104, 4.7935, 4.7402, 4.7112, 4.6694, 4.6134, 4.5520]
        F_INEL_LOWZ = [0.0, 5.9173, 5.6125, 5.5377, 5.4728, 5.4174, 5.3688, 5.3236]

        if self.Z < 5:
            f_el = F_EL_LOWZ[self.Z]
            f_inel = F_INEL_LOWZ[self.Z]
        else:
            f_el = np.log(184.15) - np.log(self.Z) / 3
            f_inel = np.log(1194) - 2 * np.log(self.Z) / 3

        return f_el, f_inel
    

    def _compute_Z_factors(self):
        f_Z_factor_1 = (self.f_el - self.f_c) + self.f_inel/self.Z
        f_Z_factor_2 = (1 + 1/self.Z) / 12

        f_Z = np.log(self.Z)/3 + self.f_c

        return f_Z_factor_1, f_Z_factor_2, f_Z


    def _compute_gamma_epsilon_factors(self):
        gamma_factor = 100 * ELECTRON_MASS_EV * EV_TO_MEV / np.cbrt(self.Z)
        epsilon_factor = 100 * ELECTRON_MASS_EV * EV_TO_MEV / (np.cbrt(self.Z) ** 2)

        return gamma_factor, epsilon_factor


class CoulombScatteringCalculator:
    def __init__(self, Z, q0, p0c, theta_lim=(1e-7, 50e-3)):
        self.Z = Z
        self.q0 = q0
        self.p0c = p0c
        self.theta_lim = theta_lim
        self.element_data = ElementData(Z)
        self.ekin = _energy_from_momentum(p0c) - ELECTRON_MASS_EV

        etot = self.ekin + ELECTRON_MASS_EV
        self.gamma = etot / ELECTRON_MASS_EV
        self.beta = np.sqrt(1.0 - 1.0 / (self.gamma**2))

        # Thomas–Fermi radius
        self.a_TF = C_TF * BOHR_RADIUS * self.Z**(-1/3)

        # small guard factor (Geant4-style) for rejection step
        self._gmax = 1.0 + 2e-4 * (self.Z**2)

        num = 2*self.element_data.mass * self.ekin * (self.ekin + 2*ELECTRON_MASS_EV)
        den = ELECTRON_MASS_EV**2 + self.element_data.mass**2 + 2*self.element_data.mass*etot
        self.tmax = num / den


    def _screening_As(self):
        # Screening parameter (Moliere, 1947)
        return (HBAR * C_LIGHT / (2.0 * self.p0c * self.a_TF))**2 * \
               (1.13 + 3.76 * (ALPHA * self.Z / self.beta)**2)


    def _mf_ratio(self, theta):
        # McKinley and Fesbach Mott-to-Rutherford ratio
        s = np.sin(theta * 0.5)

        return 1.0 - (self.beta**2) * (s**2) - self.q0 * self.Z * ALPHA * self.beta * np.pi * s * (1.0 - s)
    

    def _exp_form_factor(self, theta):
        # From G4ScreeningMottCrossSection::FormFactor2ExpHof
        s2 = np.sin(theta * 0.5) ** 2
        t = self.tmax * s2
        q2 = (t * (t + 2.0 * self.element_data.mass)) * (HBAR*C_LIGHT)**-2
        xN = self.element_data.nuclear_radius**2 * q2
        den = 1.0 + xN / 12.0
        FN = 1.0 / den

        return FN*FN


    def _compute_dxsec(self, theta):
        # Wenztel-Mott differential cross section (no nuclear form factor)
        # Rutherford prefactor (LAB, infinite target mass)
        s_half = np.sin(theta * 0.5)
        dxsec_rutherford = (self.Z**2) * CLASSICAL_ELECTRON_RADIUS**2 / 4.0 \
                           * (self.beta**-4) * (self.gamma**-2) * (s_half**-4)

        # McKinley and Fesbach Mott-to-Rutherford ratio
        R_McF = self._mf_ratio(theta)

        # Screening parameter (Moliere, 1947)
        As = self._screening_As()
        screening_factor = (s_half**2) / (As + s_half**2)

        # Nuclear form factor 
        F2 = self._exp_form_factor(theta)

        dxsec = dxsec_rutherford * R_McF * (screening_factor**2) * F2

        return dxsec * 2.0*np.pi*np.sin(theta)


    def _sample_theta(self, n):
        # screening parameter (constant over θ for fixed beam/target)
        As = self._screening_As()

        # limits in z = 1 - cosθ
        z1 = 1.0 - np.cos(self.theta_lim[0])
        z2 = 1.0 - np.cos(self.theta_lim[1])

        def sample_z(nleft):
            u = np.random.random(nleft)
            a_lo = 1.0 / (2.0 * As + z1)
            a_hi = 1.0 / (2.0 * As + z2)
            inv = a_lo - u * (a_lo - a_hi)  # = 1 / (2 As + z)
            return (1.0 / inv) - 2.0 * As

        out = []
        while len(out) < n:
            z = sample_z(n - len(out))
            theta = np.arccos(1.0 - z)
            # rejection on MF only
            acc = np.random.random(theta.size) < (self._mf_ratio(theta) / self._gmax)
            if np.any(acc):
                out.extend(theta[acc])

        return np.array(out[:n])


    def sample_deflections(self, particles, n):
        phi = np.random.uniform(0.0, 2.0*np.pi, n)
        theta = self._sample_theta(n)

        sintheta = np.sin(theta); costheta = np.cos(theta)
        sinphi = np.sin(phi);     cosphi = np.cos(phi)

        pz = np.sqrt((1.0 + particles.delta)**2 - particles.px**2 - particles.py**2)
        PP = np.column_stack((particles.px, particles.py, pz))
        norms = np.linalg.norm(PP, axis=1, keepdims=True)
        PP_HAT = PP / norms

        UU_HAT = np.zeros_like(PP_HAT)
        tol = 1e-12
        mask = (PP_HAT[:, 0]**2 + PP_HAT[:, 1]**2) < tol**2

        UU_HAT[~mask] = np.stack([-PP_HAT[~mask, 1], PP_HAT[~mask, 0], np.zeros_like(PP_HAT[~mask, 0])], axis=1)
        UU_HAT[mask] = np.array([1.0, 0.0, 0.0])
        UU_HAT /= np.linalg.norm(UU_HAT, axis=1, keepdims=True)

        VV_HAT = np.cross(PP_HAT, UU_HAT)

        scattered_dir = (
            sintheta[:, None] * cosphi[:, None] * UU_HAT +
            sintheta[:, None] * sinphi[:, None] * VV_HAT +
            costheta[:, None] * PP_HAT
        )
        PP_OUT = scattered_dir * norms
        return PP_OUT[:, 0].tolist(), PP_OUT[:, 1].tolist()


    def compute_xsec(self):
        from scipy.integrate import quad
        return quad(self._compute_dxsec, self.theta_lim[0], self.theta_lim[1])[0]


class BremsstrahlungCalculator:
    def __init__(self, Z, p0c, energy_cut=10e3):
        self.Z = Z
        self.p0c = p0c
        self.energy_cut = energy_cut # Default energy cut is 10 keV
        self.element_data = ElementData(Z)
        self.ekin = _energy_from_momentum(p0c) - ELECTRON_MASS_EV


    def _compute_dxsec(self, gamma_energy):
        etot = self.ekin + ELECTRON_MASS_EV
        y = gamma_energy / etot
        dum0 = (1 - y) + 0.75 * y**2
        dum1 = y / (etot - gamma_energy)
        gamma = dum1 * self.element_data.gamma_factor
        epsilon = dum1 * self.element_data.epsilon_factor

        phi1, phi1m2, psi1, psi1m2 = self._compute_screening_functions(gamma, epsilon)

        if self.Z < 5:
            dxsec = dum0*self.element_data.f_Z_factor_1 + (1-y)*self.element_data.f_Z_factor_2
        else:
            dxsec = dum0*((0.25*phi1 - self.element_data.f_Z) + \
                          (0.25*psi1 - 2*np.log(self.Z)/3) / self.Z) + \
                          (0.125*(1 - y)*(phi1m2 + psi1m2/self.Z))
            
        return dxsec


    def _sample_gamma_energies(self, n):
        # Set the density correction factor to 0
        # Back-of-the-envelope-calculation shows that it is not relevant for the purpose of high-energy e+/e- on low-density gas
        f_density_corr = 0
        max_energy = self.ekin

        func_max = self.element_data.f_Z_factor_1 + self.element_data.f_Z_factor_2

        # Define the min and the max of the transformed variable
        xmin = np.log(self.energy_cut**2 + f_density_corr)
        xrange = np.log(max_energy**2 + f_density_corr) - xmin

        gamma_energies = []
        while len(gamma_energies) < n:
            # Generate two random numbers between 0 and 1
            rndm = np.random.rand(2)
            gamma_energy = np.sqrt(max(np.exp(xmin + rndm[0] * xrange) - f_density_corr, 0))
            func_val = self._compute_dxsec(gamma_energy)
            # Check if the generated gamma energy meets the acceptance condition
            if func_val >= func_max * rndm[1]:
                gamma_energies.append(gamma_energy)

        return np.array(gamma_energies)
    

    def sample_deltas(self, n):
        gamma_energies = self._sample_gamma_energies(n)
        delta = ((self.p0c*np.ones(n) - gamma_energies) - self.p0c) / self.p0c

        return delta
        

    def _sample_costheta(self, n):
        # From G4ModfiedTsai.cc
        u_max = 2 * (1 + self.ekin / ELECTRON_MASS_EV)
        a1 = 1.6
        a2 = a1 / 3.0
        border = 0.25

        costheta = []
        while len(costheta) < n:
            uu = -np.log(np.random.rand() * np.random.rand())
            u = uu * a1 if np.random.rand() < border else uu * a2
            if u <= u_max:
                cos_theta = 1.0 - 2.0 * u * u / (u_max * u_max)
                costheta.append(cos_theta)
        
        return np.array(costheta)
    
    @staticmethod
    def _rotate_to_direction(vectors, directions):
        # Rodrigues rotation
        z = np.array([0, 0, 1])
        rotated = []
        for vec, target in zip(vectors, directions):
            v = np.cross(z, target)
            s = np.linalg.norm(v)
            c = np.dot(z, target)
            if s == 0:
                rotated.append(vec * c)
            else:
                vx = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
                rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
                rotated.append(rot @ vec)

        return np.array(rotated)
    

    def sample_deflections(self, particles, n):
        # Step 1: sample gamma energies
        gamma_energies = self._sample_gamma_energies(n)

        # Step 2: sample angles using Tsai model
        costheta = self._sample_costheta(n)
        sintheta = np.sqrt(1 - costheta**2)
        phi = np.random.uniform(0, 2*np.pi, n)

        # Step 3: build gamma directions (in local z frame)
        gammadirs_local = np.column_stack((
            sintheta * np.cos(phi),
            sintheta * np.sin(phi),
            costheta
        ))

        # Step 4: rotate gamma directions into particle frame
        # Get current momentum directions from particles
        pz = np.sqrt((1 + particles.delta)**2 - particles.px**2 - particles.py**2)
        PP = np.column_stack((particles.px, particles.py, pz))
        norms = np.linalg.norm(PP, axis=1, keepdims=True) # This is equivalent to 1 + particles.delta
        PP_HAT = PP / norms

        gammadirs_rotated = self._rotate_to_direction(gammadirs_local, PP_HAT)

        # Step 5: compute gamma momenta
        PP_GAMMAS = gamma_energies[:, None] * gammadirs_rotated

        # Step 6: compute final momentum (conservation)
        PP_OUT = (PP * self.p0c - PP_GAMMAS) / self.p0c

        delta = np.linalg.norm(PP_OUT, axis=1) - 1

        return PP_OUT[:, 0].tolist(), PP_OUT[:, 1].tolist(), delta.tolist()

    
    def _compute_screening_functions(self, gamma, epsilon):
        phi1 = 16.863 - 2 * np.log(1 + 0.311877 * gamma ** 2) + 2.4 * np.exp(-0.9 * gamma) + 1.6 * np.exp(-1.5*gamma)
        phi1m2 = 2 / (3 + 19.5 * gamma + 18 * gamma ** 2)
        psi1 = 24.34 - 2 * np.log(1 + 13.111641 * epsilon ** 2) + 2.8 * np.exp(-8 * epsilon) + 1.2 * np.exp(-29.2 * epsilon)
        psi1m2 = 2 / (3 + 120 * epsilon + 1200 * epsilon ** 2)

        return phi1, phi1m2, psi1, psi1m2


    def compute_xsec(self):
        etot = _energy_from_momentum(self.p0c)
        alpha_min = np.log(self.energy_cut / etot)
        alpha_max = np.log(self.ekin / self.energy_cut)
        n_sub = max(int(0.45 * alpha_max), 0) + 4
        delta = alpha_max / n_sub

        # abscissas and weights of an 8 point Gauss-Legendre quadrature
        # for numerical integration on [0,1]
        gXGL = np.array([1.98550718e-02, 1.01666761e-01, 2.37233795e-01, 4.08282679e-01,
                        5.91717321e-01, 7.62766205e-01, 8.98333239e-01, 9.80144928e-01])
        
        gWGL = np.array([5.06142681e-02, 1.11190517e-01, 1.56853323e-01, 1.81341892e-01,
                        1.81341892e-01, 1.56853323e-01, 1.11190517e-01, 5.06142681e-02])

        # Set minimum value of the first sub-interval
        alpha_i = alpha_min

        xsec = 0
        for _ in range(n_sub):
            for igl in range(8):
                # Compute the emitted photon energy k
                k = np.exp(alpha_i + gXGL[igl] * delta) * etot
                # Compute the DCS value at k
                dcs = self._compute_dxsec(k)
                xsec += gWGL[igl] * dcs
            # Update sub-interval minimum value
            alpha_i += delta

        # Apply corrections due to variable transformation
        xsec *= delta

        return 16 * ALPHA * (CLASSICAL_ELECTRON_RADIUS**2) * (self.Z**2) / 3 * xsec


class BeamGasManager():
    interacted_particle_ids = set()
    interactions_log = pd.DataFrame(columns=['name', 's', 'particle_id', 'interaction'])

    def __init__(
            self,
            line,
            gas_density,
            process,
            particle_ref=None,
            brems_energy_cut=10e3,
            coulomb_theta=(1e-7, 50e-3),
            interaction_length_is_nturns=None):
    
        self.rng = np.random.default_rng()

        # TODO: validation of density_df

        self.line = line

        # Initialise the interactions log
        tab = line.get_table()
        tt_beamgas = tab.rows[tab.element_type == 'BeamGasScattering']
        BeamGasManager.interactions_log = pd.DataFrame({
            'name': tt_beamgas.name,
            's': tt_beamgas.s,
            'particle_id': [None] * len(tt_beamgas.name),
            'interaction': [None] * len(tt_beamgas.name)
        }).reset_index(drop=True)

        # Check that interaction_length_is_nturns is an integere if not None
        if interaction_length_is_nturns is not None:
            if not isinstance(interaction_length_is_nturns, int):
                raise ValueError(f'interaction_length_is_nturns must be an integer number or None. Got {interaction_length_is_nturns} instead.')
            
        self.gas_density = gas_density

        if particle_ref is None:
            if line.particle_ref is None:
                raise ValueError('If particle_ref is not provided, line must have a reference particle.')
            particle_ref = line.particle_ref
            self.q0 = particle_ref.q0
            self.p0c = particle_ref.p0c[0]

        self.interaction_length_is_nturns = interaction_length_is_nturns

        self.atomic_species = {
            element: _atomic_number_from_symbol(element)
            for element in list(gas_density.cols)[2:]
        }

        # Check that process is either 'brems' or 'coulomb'
        if process not in ['brems', 'coulomb']:
            raise ValueError(f'process must be either "brems" (bremsstrahlung) or "Coulomb" (Coulomb scattering). Got {process} instead.')

        self.brems = None
        self.brems_xsec = None
        self.coulomb = None
        self.coulomb_xsec = None

        # Move brems and coulomb calclulator to C-kernel?
        if process == 'brems':
            self.brems = {
                kk: BremsstrahlungCalculator(self.atomic_species[kk], self.p0c, energy_cut=brems_energy_cut)
                for kk in self.atomic_species
            }
            self.brems_xsec = {
                kk: self.brems[kk].compute_xsec()
                for kk in self.atomic_species
            }
            if interaction_length_is_nturns is not None:
                # Cross-section biasing: scale xsec so average interaction length == interaction_length_is_nturns * circumference
                avg_mfp = [
                    1 / (np.mean(gas_density[kk]) * self.brems_xsec[kk])
                    for kk in self.atomic_species
                ]
                avg_mfp_tot = 1 / sum(1 / mfp for mfp in avg_mfp)

                circumference = line.get_length()
                biasing_factor = avg_mfp_tot / (self.interaction_length_is_nturns * circumference)

                for kk in self.atomic_species:
                    self.brems_xsec[kk] *= biasing_factor
                print(f'\nBremsstrahlung cross section biased by a factor {int(biasing_factor)}\n')

        if process == 'coulomb':
            self.coulomb = {
                kk: CoulombScatteringCalculator(self.atomic_species[kk], self.q0, self.p0c, theta_lim=coulomb_theta)
                for kk in self.atomic_species
            }
            self.coulomb_xsec = {
                kk: self.coulomb[kk].compute_xsec()
                for kk in self.atomic_species
            }
            if interaction_length_is_nturns is not None:
                # Cross-section biasing: scale xsec so average interaction length == interaction_length_is_nturns * circumference
                avg_mfp = [
                    1 / (np.mean(gas_density[kk]) * self.coulomb_xsec[kk])
                    for kk in self.atomic_species
                ]
                avg_mfp_tot = 1 / sum(1 / mfp for mfp in avg_mfp)

                circumference = line.get_length()
                biasing_factor = avg_mfp_tot / (self.interaction_length_is_nturns * circumference)

                for kk in self.atomic_species:
                    self.coulomb_xsec[kk] *= biasing_factor
                print(f'\nCoulomb scattering cross section biased by a factor {int(biasing_factor)}\n')

        self.scattering_enabled = False
        self._particles_initialised = False

    def enable_scattering(self):
        self.scattering_enabled = True

    def disable_scattering(self):
        self.scattering_enabled = False

    def initialise_beamgas(self):
        line = self.line
        tab = line.get_table()

        tt_beamgas = tab.rows[tab.element_type == 'BeamGasScattering']
        ds_beamgas = np.diff(np.concatenate([[0], tt_beamgas.s]))

        gas_density = self.gas_density
        atomic_species = self.atomic_species

        # Helper to config all fields to a single BeamGas
        def _config(nn, ds):
            try:
                s = tab.rows[nn].s[0]
            except Exception:
                s = line.get_s_position(nn)

            atomic_densities = {}
            for aa in atomic_species.keys():
                n_at = np.interp(s, gas_density.s, gas_density[aa])
                atomic_densities[aa] = float(n_at)

            elem = line[nn] # xc.BeamGasScattering
            # element_index = line.element_names.index(nn)

            elem._configure(
                name=nn,
                manager=self,
                ds=ds,
                atomic_densities=atomic_densities,
            )

        # for nn in tab.name[:-1]: # Avoid the last tab.name which is _end_point
        #     if isinstance(line[nn], xc.BeamGasScattering):
        #         print(f'Initialising BeamGasScattering for {nn}')
        #         _config(nn)

        for nn, ds in zip(tt_beamgas.name, ds_beamgas):
            print(f'Initialising BeamGasScattering for {nn}')
            _config(nn, ds)

        # BeamGas calculator as xo.HybridClass ??
        # dxsec
            
    def initialise_particles(self, particles):
        n_part = particles._num_active_particles
        particles.weight[:n_part] = -np.log(self.rng.random(n_part))
        self._particles_initialised = True