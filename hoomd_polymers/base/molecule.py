import os
import random

import mbuild as mb
from mbuild.coordinate_transform import z_axis_transform
from mbuild.lib.recipes import Polymer as mbPolymer
import numpy as np

from hoomd_polymers.utils import check_return_iterable


class Molecule:
    def __init__(self, n_mols, smiles=None, file=None, description=None):
        self.n_mols = check_return_iterable(n_mols)
        self.smiles = smiles
        self.file = file
        self.description = description
        self._molecules = []
        self._cg_molecules = []
        self._mapping = None
        self._generate()

    @property
    def molecules(self):
        """List of all instances of the molecule"""
        if self._cg_molecules:
            return self._cg_molecules
        return self._molecules

    @property
    def mapping(self):
        """Dictionary of particle index to bead mapping"""
        return self._mapping

    @mapping.setter
    def mapping(self, mapping_array):
        self._mapping = mapping_array

    def coarse_grain(self, beads=None, mapping=None):
        from grits import CG_Compound
        for comp in self.molecules:
            cg_comp = CG_Compound(comp, beads=beads, mapping=mapping)
            self._cg_molecules.append(cg_comp)

    def _load(self):
        if self.file and isinstance(self.file, str): # Loading from file takes precedent over SMILES
            return mb.load(self.file)
        elif self.smiles and isinstance(self.smiles, str):
            return mb.load(self.smiles, smiles=True)
        else:
            raise ValueError(
                    "Unable to load from ",
                    f"File: {self.file}",
                    f"SMILES: {self.smiles}"
            )

    def _generate(self):
        pass


class Polymer(Molecule):
    def __init__(
            self,
            lengths,
            n_mols,
            smiles,
            file,
            description,
            bond_indices,
            bond_length,
            bond_orientation
    ):
        self.lengths = check_return_iterable(lengths)
        self.bond_indices = bond_indices
        self.bond_length = bond_length
        self.bond_orientation = bond_orientation
        super(Polymer, self).__init__(
                n_mols=n_mols,
                smiles=smiles,
                file=file,
                description=description
        )

    @property
    def monomer(self):
        return self._load()

    def _build(self, length):
        chain = mbPolymer()
        chain.add_monomer(
                self.monomer,
                indices=self.bond_indices,
                separation=self.bond_length,
                orientation=self.bond_orientation
        )
        chain.build(n=length, sequence="A")
        return chain

    def _generate(self):
        for idx, length in enumerate(self.lengths):
            for i in range(self.n_mols[idx]):
                mol = self._build(length=length)
                self._molecules.append(mol)


class CoPolymer(Molecule):
    """Builds a polymer consisting of two monomer types.

    Parameters
    ----------
    monomer_A : hoomd_polymers.molecules.Polymer; required
        Class of the A-type monomer
    monomer_B : hoomd_polymers.molecules.Polymer: required
        Class of the B-type monomer
    length : int; required
        The total number of monomers in the molecule
    sequence : str; optional; default None
        Manually define the sequence of 'A' and 'B' monomers.
        Leave as None if generating random sequences.
        Example: sequence = "AABAABAAB"
    random_sequence : bool; optional; default True
        Creates a random 'A' 'B' sequence as a function
        of the AB_ratio. Set to False when manually
        defining sequence
    AB_ratio : float; optional; default 0.50
        The relative weight of A to B monomer types.
        Used when generating random sequences.
    seed : int; optional; default 24
        Set the seed used when generating random sequences
    """
    def __init__(
            self,
            monomer_A,
            monomer_B,
            lengths,
            n_mols,
            sequence=None,
            random_sequence=True,
            AB_ratio=0.50,
            seed=24
    ):
        self.lengths = lengths
        self.monomer_A = monomer_A(lengths=[1], n_mols=[1])
        self.monomer_B = monomer_B(lengths=[1], n_mols=[1])
        self.n_mols = n_mols
        self.sequence = sequence
        self.random_sequence = random_sequence
        self.AB_ratio = AB_ratio
        self.seed = seed
        self._A_count = 0
        self._B_count = 0
        self.smiles = {"A": self.monomer_A.smiles, "B": self.monomer_B.smiles}
        self.description = {
            "A": self.monomer_A.description, "B": self.monomer_B.description
        }
        self.file = {"A": self.monomer_A.file, "B": self.monomer_B.file}
        random.seed(self.seed)
        super(CoPolymer, self).__init__(
                n_mols=n_mols,
                smiles=self.smiles,
                file=self.file,
                description=self.description
        )

    @property
    def A_ratio(self):
        return self._A_count / (self._A_count + self._B_count)

    @property
    def B_ratio(self):
        return self._B_count / (self._A_count + self._B_count)

    def _build(self, length, sequence):
        chain = mbPolymer()
        chain.add_monomer(
                self.monomer_A.monomer,
                indices=self.monomer_A.bond_indices,
                orientation=self.monomer_A.bond_orientation,
                separation=self.monomer_A.bond_length
        )
        chain.add_monomer(
                self.monomer_B.monomer,
                indices=self.monomer_B.bond_indices,
                orientation=self.monomer_B.bond_orientation,
                separation=self.monomer_B.bond_length
        )
        chain.build(n=length, sequence=sequence)
        return chain

    def _generate(self):
        for idx, length in enumerate(self.lengths):
            for i in range(self.n_mols[idx]):
                if self.random_sequence:
                    sequence = random.choices(
                            ["A", "B"],
                            [self.AB_ratio, 1-self.AB_ratio],
                            k=length
                    )
                    self._A_count += sequence.count("A")
                    self._B_count += sequence.count("B")
                    _length = 1
                else:
                    sequence = self.sequence
                    _length = length
                mol = self._build(length=_length, sequence=sequence)
                self._molecules.append(mol)
