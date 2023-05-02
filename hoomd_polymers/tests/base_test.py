import os

import pytest

from hoomd_polymers.library import *
from hoomd_polymers.forcefields import *


class BaseTest:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    @pytest.fixture()
    def polyethylene_system(self):
        pe = PolyEthylene(n_mols=5, lengths=5)
        system = Pack(molecules=[pe.molecules], density=0.5)
        system.apply_forcefield(forcefield=GAFF(), remove_hydrogens=False)
        return system

    @pytest.fixture()
    def polyethylene_system_ua(self):
        pe = PolyEthylene(n_mols=5, lengths=5)
        system = Pack(molecules=[pe.molecules], density=0.5)
        system.apply_forcefield(forcefield=GAFF(), remove_hydrogens=True)
        return system

    @pytest.fixture()
    def pps_system_aa_charges(self):
        pps = PPS(n_mols=5, lengths=5)
        system = Pack(molecules=[pps.molecules], density=0.5)
        system.apply_forcefield(
                forcefield=OPLS_AA_PPS(),
                make_charge_neutral=True,
                remove_hydrogens=False
        )
        return system

    @pytest.fixture()
    def pps_system_ua_charges(self):
        pps = PPS(n_mols=5, lengths=5)
        system = Pack(molecules=[pps.molecules], density=0.5)
        system.apply_forcefield(
                forcefield=OPLS_AA_PPS(),
                make_charge_neutral=True,
                remove_hydrogens=True
        )
        return system
