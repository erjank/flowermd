import hoomd
import numpy as np

from hoomd_polymers.sim.simulation import Simulation
from hoomd_polymers.sim.actions import PullParticles, UpdateWalls


class ShearForce(Simulation):
    def __init__(
            self,
            initial_state,
            forcefield,
            shear_axis,
            interface_axis,
            shear_force=1,
            fix_ratio=0.20,
            r_cut=2.5,
            dt=0.0001,
            device=hoomd.device.auto_select(),
            seed=42,
            restart=None,
            gsd_write_freq=1e4,
            gsd_file_name="shear.gsd",
            log_write_freq=1e3,
            log_file_name="sim_data.txt"
    ):
        super(ShearForce, self).__init__(
                initial_state=initial_state,
                forcefield=forcefield,
                r_cut=r_cut,
                dt=dt,
                device=device,
                seed=seed,
                gsd_write_freq=gsd_write_freq,
                gsd_file_name=gsd_file_name,
                log_write_freq=log_write_freq,
        )
        self.shear_axis = np.asarray(shear_axis)
        self.interface_axis = np.asarray(interface_axis)
        self.shear_force = shear_force
        self.fix_ratio = fix_ratio
        self._shear_axis_index = np.where(self.shear_axis != 0)[0]
        self._interface_axis_index = np.where(self.interface_axis != 0)[0]

        snapshot = self.state.get_snapshot()
        interface_positions = snapshot.particles.position[
                :,self._interface_axis_index
        ]
        interface_neg_tags = np.where(interface_positions < 0)[0]
        interface_pos_tags = np.where(interface_positions > 0)[0]

        shift_up_tags = interface_neg_tags
        shift_down_tags = interface_pos_tags

        # Create hoomd filters
        self.shift_up = hoomd.filter.Tags(shift_up_tags.astype(np.uint32))
        self.shift_down = hoomd.filter.Tags(shift_down_tags.astype(np.uint32))
        self.add_walls(
                wall_axis=self.interface_axis,
                sigma=1.0,
                epsilon=1.0,
                r_cut=1.5
        )
        up_force = hoomd.md.force.Constant(filter=self.shift_up)
        down_force = hoomd.md.force.Constant(filter=self.shift_down)
        up_force.constant_force[self.state.particle_types] = (
                self.shear_axis * self.shear_force 
        )
        down_force.constant_force[self.state.particle_types] = (
                self.shear_axis * -self.shear_force 
        )
        self.add_force(up_force)
        self.add_force(down_force)

    def run_shear(self, kT, tau_kT, n_steps):
        self.run_NVT(
                n_steps = n_steps+1,
                kT=kT,
                tau_kT=tau_kT
        )


class Shear(Simulation):
    def __init__(
            self,
            initial_state,
            forcefield,
            shear_axis,
            interface_axis,
            fix_ratio=0.20,
            r_cut=2.5,
            dt=0.0001,
            device=hoomd.device.auto_select(),
            seed=42,
            restart=None,
            gsd_write_freq=1e4,
            gsd_file_name="shear.gsd",
            log_write_freq=1e3,
            log_file_name="sim_data.txt"
    ):
        super(Shear, self).__init__(
                initial_state=initial_state,
                forcefield=forcefield,
                r_cut=r_cut,
                dt=dt,
                device=device,
                seed=seed,
                gsd_write_freq=gsd_write_freq,
                gsd_file_name=gsd_file_name,
                log_write_freq=log_write_freq,
        )
        self.shear_axis = np.asarray(shear_axis)
        self.interface_axis = np.asarray(interface_axis)
        self.fix_ratio = fix_ratio
        self._shear_axis_index = np.where(self.shear_axis != 0)[0]
        self._interface_axis_index = np.where(self.interface_axis != 0)[0]

        self.initial_box = self.box_lengths_reduced
        self.initial_length = self.initial_box[self._shear_axis_index]
        self.fix_length = self.initial_length * fix_ratio

        # Set up walls of fixed particles:
        snapshot = self.state.get_snapshot()
        shear_positions = snapshot.particles.position[:,self._shear_axis_index]
        interface_positions = snapshot.particles.position[
                :,self._interface_axis_index
        ]
        box_max = self.initial_length / 2
        box_min = -box_max

        # Set tag groups for filters and particle shifts
        shear_neg_tags = np.where(shear_positions<(box_min+self.fix_length))[0]
        shear_pos_tags = np.where(shear_positions>(box_max-self.fix_length))[0]
        all_shear_tags = np.union1d(shear_neg_tags, shear_pos_tags)
        interface_neg_tags = np.where(interface_positions < 0)[0]
        interface_pos_tags = np.where(interface_positions > 0)[0]

        shift_up_tags = np.intersect1d(interface_neg_tags, all_shear_tags)
        shift_down_tags = np.intersect1d(interface_pos_tags, all_shear_tags)

        # Create hoomd filters
        self.fix_shift_up = hoomd.filter.Tags(shift_up_tags.astype(np.uint32))
        self.fix_shift_down = hoomd.filter.Tags(
                shift_down_tags.astype(np.uint32)
        )
        all_fixed = hoomd.filter.Union(self.fix_shift_up, self.fix_shift_down)
        # Set the group of particles to be integrated over
        self.integrate_group = hoomd.filter.SetDifference(
                hoomd.filter.All(), all_fixed
        )
        self.add_walls(
                wall_axis=self.interface_axis,
                sigma=1.0,
                epsilon=1.0,
                r_cut=2.5
        )

    def run_shear(self, strain, kT, n_steps, period):
        current_length = self.box_lengths_reduced[self._shear_axis_index]
        final_length = current_length * (1+strain)
        final_box = np.copy(self.box_lengths_reduced)
        final_box[self._shear_axis_index] = final_length
        shift_by = (final_length - current_length) / (n_steps//period)
        resize_trigger = hoomd.trigger.Periodic(period)
        box_ramp = hoomd.variant.Ramp(
                A=0, B=1, t_start=self.timestep, t_ramp=int(n_steps)
        )
        box_resizer = hoomd.update.BoxResize(
                box1=self.box_lengths_reduced,
                box2=final_box,
                variant=box_ramp,
                trigger=resize_trigger,
                filter=hoomd.filter.Null()
        )
        particle_puller = PullParticles(
            shift_by=shift_by/2,
            axis=self.shear_axis,
            neg_filter=self.fix_shift_down,
            pos_filter=self.fix_shift_up
        )
        particle_updater = hoomd.update.CustomUpdater(
                trigger=resize_trigger, action=particle_puller
        )
        wall_update = UpdateWalls(sim=self)
        wall_updater = hoomd.update.CustomUpdater(
                trigger=resize_trigger, action=wall_update
        )
        self.operations.updaters.append(box_resizer)
        self.operations.updaters.append(particle_updater)
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.NVE,
            method_kwargs={"filter": self.integrate_group}
        )
        self.run(n_steps + 1)


class Tensile(Simulation):
    def __init__(
            self,
            initial_state,
            forcefield,
            tensile_axis,  # Tensile axis == Interface Axis
            fix_ratio=0.20,
            r_cut=2.5,
            dt=0.0001,
            device=hoomd.device.auto_select(),
            seed=42,
            restart=None,
            gsd_write_freq=1e4,
            gsd_file_name="trajectory.gsd",
            log_write_freq=1e3,
            log_file_name="sim_data.txt"
    ):
        super(Tensile, self).__init__(
                initial_state=initial_state,
                forcefield=forcefield,
                r_cut=r_cut,
                dt=dt,
                device=device,
                seed=seed,
                gsd_write_freq=gsd_write_freq,
                gsd_file_name=gsd_file_name,
                log_write_freq=log_write_freq,
        )
        self.tensile_axis=tensile_axis.lower()
        self.fix_ratio = fix_ratio
        axis_array_dict = {
                "x": np.array([1,0,0]),
                "y": np.array([0,1,0]),
                "z": np.array([0,0,1])
        }
        axis_dict = {"x": 0, "y": 1, "z": 2}
        self._axis_index = axis_dict[self.tensile_axis]
        self._axis_array = axis_array_dict[self.tensile_axis]
        self.initial_box = self.box_lengths_reduced
        self.initial_length = self.initial_box[self._axis_index]
        self.fix_length = self.initial_length * fix_ratio
        # Set up walls of fixed particles:
        snapshot = self.state.get_snapshot()
        positions = snapshot.particles.position[:,self._axis_index]
        box_max = self.initial_length / 2
        box_min = -box_max
        left_tags = np.where(positions < (box_min + self.fix_length))[0]
        right_tags = np.where(positions > (box_max - self.fix_length))[0]
        self.fix_left = hoomd.filter.Tags(left_tags.astype(np.uint32))
        self.fix_right = hoomd.filter.Tags(right_tags.astype(np.uint32))
        all_fixed = hoomd.filter.Union(self.fix_left, self.fix_right)
        # Set the group of particles to be integrated over
        self.integrate_group = hoomd.filter.SetDifference(
                hoomd.filter.All(), all_fixed
        )

    @property
    def strain(self):
        delta_L = self.box_lengths_reduced[self._axis_index]-self.initial_length
        return delta_L / self.initial_length

    def run_tensile(self, strain, kT, n_steps, period):
        current_length = self.box_lengths_reduced[self._axis_index]
        final_length = current_length * (1+strain)
        final_box = np.copy(self.box_lengths_reduced)
        final_box[self._axis_index] = final_length
        shift_by = (final_length - current_length) / (n_steps//period)
        resize_trigger = hoomd.trigger.Periodic(period)
        box_ramp = hoomd.variant.Ramp(
                A=0, B=1, t_start=self.timestep, t_ramp=int(n_steps)
        )
        box_resizer = hoomd.update.BoxResize(
                box1=self.box_lengths_reduced,
                box2=final_box,
                variant=box_ramp,
                trigger=resize_trigger,
                filter=hoomd.filter.Null()
        )
        particle_puller = PullParticles(
            shift_by=shift_by/2,
            axis=self._axis_array,
            neg_filter=self.fix_left,
            pos_filter=self.fix_right
        )
        particle_updater = hoomd.update.CustomUpdater(
                trigger=resize_trigger, action=particle_puller
        )
        self.operations.updaters.append(box_resizer)
        self.operations.updaters.append(particle_updater)
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.NVE,
            method_kwargs={"filter": self.integrate_group}
        )
        self.run(n_steps + 1)
