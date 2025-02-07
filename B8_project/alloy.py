"""
Alloy
=====

This module contains code related to generating and representing alloys.
"""

from dataclasses import dataclass
import numpy as np

from B8_project.crystal import UnitCell


@dataclass
class SuperCell:
    """
    Super cell
    ==========

    A class to represent a super cell. A super cell is a conventional unit cell made
    from a collection of smaller, identical unit cells. This class can only represent
    super cells which are cuboids.

    Attributes
    ----------
    material : str
        The name of the crystal.
    unit_cell : UnitCell
        The unit cell which is to be copied to produce the super cell.
    side_lengths : np.ndarray
        The side lengths of the super cell, specified in terms of the lattice constants
        of the unit cell.

    Methods
    -------
    new_super_cell
        Generates a new super cell.
    lattice_vectors
        Returns the position vector of each unit cell in the super cell.
    to_unit_cell
        Converts a super cell into an instance of `UnitCell`, i.e. a list of atoms and
        atomic positions inside the super cell.
    """

    material: str
    unit_cell: UnitCell
    side_lengths: np.ndarray

    def __post_init__(self):
        if not isinstance(self.side_lengths, np.ndarray):
            raise TypeError("side_lengths must be a numpy array.")
        if self.side_lengths.dtype != int:
            raise TypeError("side_lengths must be an array of integers.")
        if self.side_lengths.shape != (3,):
            raise ValueError("side_lengths must be of length 3.")

    @classmethod
    def new_super_cell(
        cls,
        unit_cell: UnitCell,
        side_lengths: tuple[int, int, int],
        material: str = "",
    ):
        """
        New super cell
        ==============

        Creates a new super cell, given a material, a unit cell and the side lengths of
        the super cell.

        Parameters
        ----------
        unit_cell : UnitCell
            Represents the unit cell of the crystal that is to be converted to a super
            cell. The `UnitCell` object stores a list of atoms and atomic positions in
            the unit cell of the crystal.
        side_lengths : tuple[int, int, int]
            The desired side lengths in the x, y and z directions of the super cell,
            specified in terms of the lattice constants of the unit cell.
        material : str
            The name of the material. The default value is the same name as the original
            crystal.
        """
        if material == "":
            material = unit_cell.material

        # Error handling for if any of the side lengths are less than or equal to 0.
        if side_lengths[0] <= 0 or side_lengths[1] <= 0 or side_lengths[2] <= 0:
            raise ValueError("All entries in side_lengths must be greater than 0.")

        return cls(material, unit_cell, np.array(side_lengths))

    def lattice_vectors(self):
        """
        Get lattice vectors
        ===================

        Returns the position vector of each unit cell in the super cell, in terms of
        the lattice constants.
        """
        x_length, y_length, z_length = self.side_lengths

        # Error handling for if any of the lengths are 0 or negative.
        if x_length <= 0 or y_length <= 0 or z_length <= 0:
            raise ValueError("All entries in side_lengths must be greater than 0.")

        # lattice_vectors stores the position vectors of the corners of the unit cells
        # comprising the super cell.
        lattice_vectors = np.array(
            [
                [x, y, z]
                for x in range(x_length)
                for y in range(y_length)
                for z in range(z_length)
            ]
        )
        return lattice_vectors

    def to_unit_cell(self) -> UnitCell:
        """
        Super cell to unit cell
        =======================

        Converts a super cell into an instance of `UnitCell`, i.e. a list of atoms in
        the super cell and their positions.
        """
        unit_cell = self.unit_cell
        lattice_vectors = self.lattice_vectors()

        lattice_constants = self.side_lengths * unit_cell.lattice_constants

        atomic_numbers = unit_cell.atoms["atomic_numbers"]
        atomic_positions = unit_cell.atoms["positions"]

        # For each unit cell copy, duplicates all of the atoms and shifts their
        # positions.
        atomic_numbers = np.repeat(atomic_numbers, len(lattice_vectors))
        atomic_positions = atomic_positions[:, np.newaxis, :] + lattice_vectors

        # Define a custom datatype to represent atoms.
        dtype = np.dtype([("atomic_numbers", "i4"), ("positions", "3f8")])

        # Create a structured NumPy array to store the atoms.
        atoms = np.empty(len(atomic_numbers), dtype=dtype)
        atoms["atomic_numbers"] = atomic_numbers
        atoms["positions"] = atomic_positions.reshape(-1, 3)

        return UnitCell(self.material, lattice_constants, atoms)
