"""Base forcefield classes."""

import forcefield_utilities as ffutils
import foyer


class BaseXMLForcefield(foyer.Forcefield):
    """Base XML forcefield class."""

    def __init__(self, forcefield_files=None, name=None):
        super(BaseXMLForcefield, self).__init__(
            forcefield_files=forcefield_files, name=name
        )
        self.gmso_ff = (
            ffutils.FoyerFFs().load(forcefield_files or name).to_gmso_ff()
        )


class BaseHOOMDForcefield:
    """Base HOOMD forcefield class."""

    def __init__(self, hoomd_forces):
        self.hoomd_forces = hoomd_forces
        if hoomd_forces is None:
            raise NotImplementedError(
                "`hoomd_forces` must be defined in the subclass."
            )
        if not isinstance(hoomd_forces, list):
            raise TypeError("`hoomd_forces` must be a list.")
