import os
import pytest
import random

import mbuild as mb
import numpy as np
import gsd.hoomd
from hoomd_polymers.library import *
#from polybinder.library import ASSETS_DIR
from base_test import BaseTest


class TestMolecules(BaseTest):
    def test_pps(self):
        chain = PPS(n_mols=1, lengths=5)
        monomer = mb.load(chain.smiles, smiles=True)
        assert chain.molecules[0].n_particles == (monomer.n_particles*5)-8

    def test_polyethylene(self):
        chain = PolyEthylene(n_mols=1, lengths=5)
        monomer = mb.load(chain.smiles, smiles=True)
        assert chain.molecules[0].n_particles == (monomer.n_particles*5)-8

    def test_pekk_meta(self):
        chain = PEKK_meta(n_mols=1, lengths=5)
        monomer = mb.load(chain.smiles, smiles=True)
        assert chain.molecules[0].n_particles == (monomer.n_particles*5)-8

    def test_pekk_para(self):
        chain = PEKK_para(n_mols=1, lengths=5)
        monomer = mb.load(chain.smiles, smiles=True)
        assert chain.molecules[0].n_particles == (monomer.n_particles*5)-8

    @pytest.mark.skip()
    def test_peek(self):
        chain = PEEK(n_mols=1, lengths=5)
        monomer = mb.load(chain.smiles, smiles=True)
        assert chain.molecules[0].n_particles == (monomer.n_particles*5)-8

    def test_lj_chain(self):
        cg_chain = LJChain(
                n_mols=1,
                lengths=3,
                bead_sequence=["A"],
                bead_mass={"A": 100},
                bond_lengths={"A-A": 1.5}
        )
        assert cg_chain.molecules[0].n_particles == 3
        assert cg_chain.molecules[0].mass == 300

    def test_lj_chain_sequence(self):
        cg_chain = LJChain(
                n_mols=1,
                lengths=3,
                bead_sequence=["A", "B"],
                bead_mass={"A": 100, "B": 150},
                bond_lengths={"A-A": 1.5, "A-B": 1.0}
        )
        assert cg_chain.molecules[0].n_particles == 6
        assert cg_chain.molecules[0].mass == 300 + 450

    def test_lj_chain_sequence_bonds(self):
        cg_chain = LJChain(
                n_mols=1,
                lengths=3,
                bead_sequence=["A", "B"],
                bead_mass={"A": 100, "B": 150},
                bond_lengths={"A-A": 1.5, "A-B": 1.0}
        )

        cg_chain_rev = LJChain(
                n_mols=1,
                lengths=3,
                bead_sequence=["A", "B"],
                bead_mass={"A": 100, "B": 150},
                bond_lengths={"A-A": 1.5, "B-A": 1.0}
        )

    def test_lj_chain_sequence_bad_bonds(self):
        with pytest.raises(ValueError):
            cg_chain = LJChain(
                    n_mols=1,
                    lengths=3,
                    bead_sequence=["A", "B"],
                    bead_mass={"A": 100, "B": 150},
                    bond_lengths={"A-A": 1.5}
            )

    def test_lj_chain_sequence_bad_mass(self):
        with pytest.raises(ValueError):
            cg_chain = LJChain(
                    n_mols=1,
                    lengths=3,
                    bead_sequence=["A", "B"],
                    bead_mass={"A": 100},
                    bond_lengths={"A-A": 1.5}
            )

    def test_copolymer(self):
        pass
