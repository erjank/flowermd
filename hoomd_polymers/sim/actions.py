import hoomd
import numpy as np


class StdOutLogger(hoomd.custom.Action):
    def __init__(self, n_steps, sim):
        self.n_steps = int(n_steps)
        self.sim = sim
        self.starting_step = sim.timestep
    
    def act(self, timestep):
        if timestep != 0:
            tps = np.round(self.sim.tps,2)
            current_step = self.sim.timestep - self.starting_step
            eta = np.round((self.n_steps - current_step)/(60*tps), 1)
            print(f"Step {current_step} of {self.n_steps}; TPS: {tps}; ETA: {eta} minutes")


class PullParticles(hoomd.custom.Action):
    def __init__(self, shift_by, axis, neg_filter, pos_filter):
        self.shift_by = shift_by
        self.axis = axis
        self.neg_filter = neg_filter
        self.pos_filter = pos_filter

    def act(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            neg_filter = snap.particles.rtag[self.neg_filter.tags]
            pos_filter = snap.particles.rtag[self.pos_filter.tags]
            snap.particles.position[neg_filter] -= (self.shift_by*self.axis)
            snap.particles.position[pos_filter] += (self.shift_by*self.axis)


class UpdateWalls(hoomd.custom.Action):
    def __init__(self, sim):
        self.sim = sim

    def act(self, timestep):
        self.update_walls()

    def update_walls(self):
        for wall_axis in self.sim._wall_forces:
            wall_force = self.sim._wall_forces[wall_axis][0]
            wall_kwargs = self.sim._wall_forces[wall_axis][1]
            self.sim.remove_force(wall_force)
            self.sim.add_walls(wall_axis, **wall_kwargs)


class ScaleEpsilon(hoomd.custom.Action):
    def __init__(self, sim, scale_factor):
        self.scale_factor = scale_factor
        self.sim = sim

    def act(self, timestep):
        self.sim.adjust_epsilon(shift_by=self.scale_factor)


class ScaleSigma(hoomd.custom.Action):
    def __init__(self, sim, scale_factor):
        self.scale_factor = scale_factor
        self.sim = sim

    def act(self, timestep):
        self.sim.adjust_sigma(shift_by=self.scale_factor)
        lj_forces = self.sim._lj_force()


class BreakBonds(hoomd.custom.Action):
    def __init__(
            self,
            sim,
            break_bond_type,
            condition_func,
            batch_size
    ):
        self.break_bond_type = break_bond_type
        self.form_type = bond_form_type
        self.condition = condition_func
        self.sim = sim
        self.broken_bond_particles = []
        with self._state.cpu_local_snapshot as snap:
            self.break_bond_indices = np.where(
                    snap.bonds.typeid == snap.bonds.types[self.break_bond_type]
            )[0]

        def break_bonds(self, bond_indices):
            with self._state.cpu_local_snapshot as snap:
                snap.bonds.N -= len(bond_indices) 
                snap.bonds.group = np.delete(
                        snap.bonds.group, bond_indices, axis=0
                )
                snap.bonds.typeid = np.delete(
                        snap.bonds.typeid, bond_indices, axis=0
                )

        def get_break_bond_indices(self):
            with self._state.cpu_local_snapshot as snap:
                break_bond_indices = np.where(
                        snap.bonds.typeid == snap.bonds.types[self.break_bond_type]
                )[0]
                return break_bond_indices

        def act(self, timestep):
            break_bond_indices = self.get_break_bond_indices()
            np.random.shuffle(break_bond_indices)
            self.break_bonds(bond_indices=break_bond_indices[batch_size-1:])
