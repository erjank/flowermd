from hoomd_polymers.sim.simulation import Simulation

class EntanglementSystem:
    def __init__(
            self,
            gsd_file
    ):
        pass
    
    # INPUT: GSD file or snapshot

    # OUTPUT:
    # Snapshot with particles renamed by chain
    # Forcefield with bond FENE, no angles or dihedrals
    # Forcefield with pair interactions of inter-chain only

    # Create hoomd objs without angles, dihedrals
    # Create FENE bond object
    # Rename particle types with chain ID?
    # Set pair interactions of types with same ID to zero?
    # Distinction between types not important anymore?
    # Just need particle types/names by chain ID only




class Entanglement(Simulation):
    def __init__(
            self,
            initial_state,
            forcefield,
            r_cut,
            head_indices,
            tail_indices,
            seed=42,
            gsd_write_freq=1e4,
            gsd_file_name="ppa.gsd",
            log_write_freq=1e3,
            log_file_name="ppa_data.txt"
    ):
        super(Entanglement, self).__init__(
                initial_state=initial_state,
                forcefield=forcefield,
                r_cut=r_cut,
                seed=seed,
                gsd_write_freq=gsd_write_freq,
                gsd_file_name=gsd_file_name,
                log_write_freq=log_write_freq,
                log_File_name=log_file_name
        )
        self.head_indices = head_indices
        self.tail_indices = tail_indices

        # Set FENE bond potential

        # Set particle filters?
        # Fix head and tails

