from base_test import BaseTest
import os
import pytest

import gsd.hoomd
import hoomd
import numpy as np

from hoomd_polymers.sim import Simulation
from hoomd_polymers.modules.welding import Interface, SlabSimulation, WeldSimulation


class TestWelding(BaseTest):
    def test_interface(self, polyethylene_system):
        sim = Simulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield
        )
        sim.add_walls(wall_axis=(1,0,0), sigma=1, epsilon=1, r_cut=2)
        sim.run_update_volume(
                n_steps=1000,
                period=10,
                kT=2.0,
                tau_kt=0.01,
                final_box_lengths=sim.box_lengths/2,
        )
        sim.save_restart_gsd()
        interface = Interface(gsd_file="restart.gsd", interface_axis=(1,0,0), gap=0.1)
        interface_snap = interface.hoomd_snapshot
        with gsd.hoomd.open("restart.gsd", "rb") as traj:
            slab_snap = traj[0]

        assert interface_snap.particles.N == slab_snap.particles.N * 2
        assert interface_snap.bonds.N == slab_snap.bonds.N * 2
        assert interface_snap.bonds.M == slab_snap.bonds.M
        assert interface_snap.angles.N == slab_snap.angles.N * 2
        assert interface_snap.angles.M == slab_snap.angles.M
        assert interface_snap.dihedrals.N == slab_snap.dihedrals.N * 2
        assert interface_snap.dihedrals.M == slab_snap.dihedrals.M
        assert interface_snap.pairs.N == slab_snap.pairs.N * 2
        assert interface_snap.pairs.M == slab_snap.pairs.M

        if os.path.isfile("restart.gsd"):
            os.remove("restart.gsd")

    def test_slab_sim_xaxis(self, polyethylene_system):
        sim = SlabSimulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield,
                interface_axis=(1,0,0)
        )
        assert isinstance(sim._forcefield[-1], hoomd.md.external.wall.LJ)
        wall_axis = np.array(sim._forcefield[-1].walls[0].origin)
        wall_axis /= np.max(wall_axis)
        assert np.array_equal(wall_axis, sim.interface_axis)
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=500)

    def test_slab_sim_yaxis(self, polyethylene_system):
        sim = SlabSimulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield,
                interface_axis=(0,1,0)
        )
        assert isinstance(sim._forcefield[-1], hoomd.md.external.wall.LJ)
        wall_axis = np.array(sim._forcefield[-1].walls[0].origin)
        wall_axis /= np.max(wall_axis)
        assert np.array_equal(wall_axis, sim.interface_axis)
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=500)

    def test_slab_sim_zaxis(self, polyethylene_system):
        sim = SlabSimulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield,
                interface_axis=(0,0,1)
        )
        assert isinstance(sim._forcefield[-1], hoomd.md.external.wall.LJ)
        wall_axis = np.array(sim._forcefield[-1].walls[0].origin)
        wall_axis /= np.max(wall_axis)
        assert np.array_equal(wall_axis, sim.interface_axis)
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=500)

    def test_weld_sim(self):
        pass
