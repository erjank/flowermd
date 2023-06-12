import os
import random

import mbuild as mb
from mbuild.coordinate_transform import z_axis_transform
from mbuild.lib.recipes import Polymer as mbPolymer
import numpy as np

from hoomd_polymers.library import MON_DIR
from hoomd_polymers import Molecule, Polymer
from hoomd_polymers.utils import check_return_iterable


class PolyEthylene(Polymer):
    """Creates a Poly(ethylene) chain.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, lengths, n_mols):
        smiles = "CC"
        file = None
        description = "Poly(ethylene)"
        bond_indices = [2, 6]
        bond_length = 0.145
        bond_orientation = [None, None]
        super(PolyEthylene, self).__init__(
                lengths=lengths,
                n_mols=n_mols,
                smiles=smiles,
                file=file,
                description=description,
                bond_indices=bond_indices,
                bond_length=bond_length,
                bond_orientation=bond_orientation
        )


class PPS(Polymer):
    """Creates a Poly(phenylene-sulfide) (PPS) chain.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, lengths, n_mols):
        smiles = "c1ccc(S)cc1"
        file = None
        description = "Poly(phenylene-sulfide)"
        bond_indices = [7, 10]
        bond_length = 0.176
        bond_orientation = [[0, 0, 1], [0, 0, -1]]
        super(PPS, self).__init__(
                lengths=lengths,
                n_mols=n_mols,
                smiles=smiles,
                file=file,
                description=description,
                bond_indices=bond_indices,
                bond_length=bond_length,
                bond_orientation=bond_orientation
        )
    def _load(self):
        monomer = mb.load(self.smiles, smiles=True)
        # Need to align monomer along zx plane due to orientation of S-H bond
        z_axis_transform(
                monomer,
                point_on_z_axis=monomer[7],
                point_on_zx_plane=monomer[4]
        )
        return monomer


class PEEK(Polymer):
    def __init__(self, length):
        super(PEEK, self).__init__()


class PEKK_para(Polymer):
    """Creates a Poly(ether-ketone-ketone) (PEKK) chain.
    The bonding positions of consecutive ketone groups
    takes place on the para site of the phenyl ring.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, lengths, n_mols):
        smiles = "c1ccc(Oc2ccc(C(=O)c3ccc(C(=O))cc3)cc2)cc1"
        file = os.path.join(MON_DIR, "pekk_para.mol2")
        description = ("Poly(ether-ketone-ketone) with para bonding "
                            "configuration between consecutive "
                            "ketone linkage groups")
        bond_indices = [35, 36]
        bond_length = 0.148
        bond_orientation = [[0, 0, -1], [0, 0, 1]]
        super(PEKK_para, self).__init__(
                lengths=lengths,
                n_mols=n_mols,
                smiles=smiles,
                file=file,
                description=description,
                bond_indices=bond_indices,
                bond_length=bond_length,
                bond_orientation=bond_orientation
        )


class PEKK_meta(Polymer):
    """Creates a Poly(ether-ketone-ketone) (PEKK) chain.
    The bonding positions of consecutive ketone groups
    takes place on the meta site of the phenyl ring.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, lengths, n_mols):
        smiles = "c1cc(Oc2ccc(C(=O)c3cc(C(=O))ccc3)cc2)ccc1"
        file = os.path.join(MON_DIR, "pekk_meta.mol2")
        description = ("Poly(ether-ketone-ketone) with meta bonding "
                            "configuration between consectuvie "
                            "ketone linkage groups")
        bond_indices = [35, 36]
        bond_length = 0.148
        bond_orientation = [[0, 0, -1], [0, 0, 1]]
        super(PEKK_meta, self).__init__(
                lengths=lengths,
                n_mols=n_mols,
                smiles=smiles,
                file=file,
                description=description,
                bond_indices=bond_indices,
                bond_length=bond_length,
                bond_orientation=bond_orientation
        )


class LJChain(Molecule):
    """Creates a coarse-grained bead-spring polymer chain.

    Parameters
    ----------
    length : int; required
        The number of times to repeat bead_sequence in a single chain.
    bead_sequence : list; optional; default ["A"]
        The sequence of bead types in the chain.
    bond_length : dict; optional; default {"A-A": 1.0}
        The bond length between connected beads (units: nm)
    bead_mass : dict; optional; default {"A": 1.0}
        The mass of the bead types
    """
    def __init__(
            self,
            lengths,
            n_mols,
            bead_sequence=["A"],
            bead_mass={"A": 1.0},
            bond_lengths={"A-A": 1.0},
    ):
        self.description = "Simple bead-spring polymer"
        self.lengths = check_return_iterable(lengths)
        self.n_mols = check_return_iterable(n_mols)
        self.bead_sequence = bead_sequence
        self.bead_mass = bead_mass
        self.bond_lengths = bond_lengths
        super(LJChain, self).__init__(n_mols=n_mols)

    def _build(self, length):
        chain = mb.Compound()
        last_bead = None
        for i in range(length):
            for idx, bead_type in enumerate(self.bead_sequence):
                mass = self.bead_mass.get(bead_type, None)
                if not mass:
                    raise ValueError(
                            f"The bead mass for {bead_type} was not given "
                            "in the bead_mass dict."
                    )
                next_bead = mb.Compound(mass=mass, name=bead_type, charge=0)
                chain.add(next_bead)
                if last_bead:
                    bead_pair = "-".join([last_bead.name, next_bead.name])
                    bond_length = self.bond_lengths.get(bead_pair, None)
                    if not bond_length:
                        bead_pair_rev = "-".join([next_bead.name, last_bead.name])
                        bond_length = self.bond_lengths.get(bead_pair_rev, None)
                        if not bond_length:
                            raise ValueError(
                                    "The bond length for pair "
                                    f"{bead_pair} or {bead_pair_rev} "
                                    "is not found in the bond_lengths dict."
                            )
                    new_pos = last_bead.xyz[0] + (0, 0, bond_length)
                    next_bead.translate_to(new_pos)
                    chain.add_bond([next_bead, last_bead])
                last_bead = next_bead
        return chain

    def _generate(self):
        for idx, length in enumerate(self.lengths):
            for i in range(self.n_mols[idx]):
                mol = self._build(length=length)
                self._molecules.append(mol)
