import os
import pytest
import random
import hoomd

from hoomd_polymers.library.systems import * 
from hoomd_polymers.library.polymers import *
from hoomd_polymers.forcefields import *
from base_test import BaseTest


class TestSystems(BaseTest):
    def test_pack(self):
        pps_mols = PPS(n_mols=5, lengths=5)
        system = Pack(molecules=[pps_mols.molecules], density=1.0)

    def test_lattice(self):
        pps_mols = PPS(n_mols=5, lengths=5)
        system = Lattice(
                molecules=[pps_mols.molecules], x=1, y=1, n=4, density=1.0
        )

    def test_set_target_box(self):
        pass

    def test_hoomd_ff(self):
        pass

    def test_hoomd_snap(self):
        pekk = PEKK_para(n_mols=5, lengths=5)
        system = Pack(molecules=[pekk.molecules], density=1.0)
        system.apply_forcefield(forcefield=GAFF())
        assert system.hoomd_snapshot.particles.N == system.system.n_particles

    def test_mass(self):
        pps_mols = PPS(n_mols=20, lengths=1)
        system = Pack(molecules=[pps_mols.molecules], density=1.0)
        assert np.allclose(system.mass, ((12.011*6) + (1.008*6) + 32.06)*20, atol=1e-4) 

    def test_box(self):
        pass

    def test_density(self):
        pass

    def test_ref_distance(self):
        pe_mols = PolyEthylene(n_mols=5, lengths=5)
        system = Pack(molecules=[pe_mols.molecules], density=1.0)
        system.apply_forcefield(forcefield=GAFF())
        assert np.allclose(system.reference_distance.value, 3.39966951, atol=1e-3)
        reduced_box = system.hoomd_snapshot.configuration.box[0:3]
        calc_box = reduced_box * system.reference_distance.to("nm").value
        assert np.allclose(calc_box[0], system.box.Lx, atol=1e-2)
        assert np.allclose(calc_box[1], system.box.Ly, atol=1e-2)
        assert np.allclose(calc_box[2], system.box.Lz, atol=1e-2)

    def test_ref_mass(self):
        pe_mols = PolyEthylene(n_mols=5, lengths=5)
        system = Pack(molecules=[pe_mols.molecules], density=1.0)
        system.apply_forcefield(forcefield=GAFF())
        total_red_mass = sum(system.hoomd_snapshot.particles.mass)
        assert np.allclose(
                system.mass,
                total_red_mass * system.reference_mass.to("amu").value,
                atol=1e-1
        )

    def test_ref_energy(self):
        pe_mols = PolyEthylene(n_mols=5, lengths=5)
        system = Pack(molecules=[pe_mols.molecules], density=1.0)
        system.apply_forcefield(forcefield=GAFF())
        assert np.allclose(system.reference_energy.value, 0.1094, atol=1e-3)


    def test_apply_forcefield(self):
        pe_mols = PolyEthylene(n_mols=5, lengths=5)
        system = Pack(molecules=[pe_mols.molecules], density=1.0)
        system.apply_forcefield(forcefield=GAFF())
        assert isinstance(system.hoomd_snapshot, hoomd.snapshot.Snapshot)
        assert isinstance(system.hoomd_forcefield, list)

    def test_remove_hydrogens(self):
        pe_mols = PolyEthylene(n_mols=5, lengths=5)
        system = Pack(molecules=[pe_mols.molecules], density=1.0)
        system.apply_forcefield(forcefield=OPLS_AA(), remove_hydrogens=True)
        assert system.hoomd_snapshot.particles.N == 5*5*2
        assert np.allclose(
                system.mass,
                sum([a.mass for a in system.typed_system.atoms]), atol=1e-1
        )

    def test_remove_charges(self):
        pe_mols = PolyEthylene(n_mols=5, lengths=5)
        system = Pack(molecules=[pe_mols.molecules], density=1.0)
        system.apply_forcefield(forcefield=OPLS_AA(), remove_charges=True)
        assert sum(a.charge for a in system.typed_system.atoms) == 0
        assert sum(system.hoomd_snapshot.particles.charge) == 0

    def test_make_charge_neutral(self):
        pps_mols = PPS(n_mols=5, lengths=5)
        system = Pack(molecules=[pps_mols.molecules], density=1.0)
        system.apply_forcefield(forcefield=OPLS_AA_PPS(), make_charge_neutral=True)
        assert np.allclose(0, sum([a.charge for a in system.typed_system.atoms]), atol=1e-5)

    def test_scale_parameters(self):
        pass
