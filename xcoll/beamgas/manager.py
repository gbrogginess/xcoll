import pandas as pd
import periodictable as pt

def _atomic_number_from_symbol(element_symbol):
    element = getattr(pt, element_symbol)
    return element.number

class BeamGasManager():
    log_interacted_part_ids = []
    df_interactions_log = pd.DataFrame(columns=['name', 's', 'particle_id', 'interaction'])

    def __init__(
            self,
            density_df,
            # q0, p0c, # Could be put in particle_ref
            particle_ref,
            process,
            brems_energy_cut=10e3,
            coulomb_theta=(1e-7, 50e-3),
            interaction_length_is_nturns=None):
    
        # Validation of density_df

        # Check that process is either 'brems' or 'coulomb'
        if process not in ['brems', 'coulomb']:
            raise ValueError(f'process must be either "brems" (bremsstrahlung) or "Coulomb" (Coulomb scattering). Got {process} instead.')
        
        brems = True if process == 'brems' else False
        coulomb = True if process == 'coulomb' else False

        # Check that interaction_length_is_nturns is an integere if not None
        if interaction_length_is_nturns is not None:
            if not isinstance(interaction_length_is_nturns, int):
                raise ValueError(f'interaction_length_is_nturns must be an integer number or None. Got {interaction_length_is_nturns} instead.')
            
        self.density_df = density_df
        self.q0 = particle_ref.q0
        self.p0c = particle_ref.p0c
        self.interaction_length_is_nturns = interaction_length_is_nturns

        self.atomic_species = {
            element: _atomic_number_from_symbol(element)
            for element in density_df.columns[1:]
        }

        # Will move brems and coulomb calclulator to C-kernel
        # self.brems = None 
        # self.coulomb = None
    
    def initialise_particles(self, particles):
        active_part_ids = particles.particle_id[particles.state > 0]
        n_active = len(active_part_ids)
        particles.weight[:n_active] = self.rng.random(n_active)