"""
Super cell
==========

TODO: Add documentation.
"""

from dataclasses import dataclass

from B8_project.archive.crystal import UnitCell


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
        - unit_cell (UnitCell): The unit cell which is to be copied to produce the
        super cell.
        - side_lengths (tuple[int, int, int]): The side lengths of the super cell,
        specified in terms of the lattice constants of the unit cell.

    Methods
    -------
    TODO: Add methods.
    """

    unit_cell: UnitCell
    side_lengths: tuple[int, int, int]

    @classmethod
    def new_super_cell(cls, unit_cell: UnitCell, side_lengths: tuple[int, int, int]):
        """
        New super cell
        ==============

        Creates a new super cell, given a unit cell and the side lengths of the super
        cell.
        """
        # Error handling for if any of the side lengths are less than or equal to 0.
        if side_lengths[0] <= 0 or side_lengths[1] <= 0 or side_lengths[2] <= 0:
            raise ValueError("All entries in side_lengths must be greater than 0.")

        return cls(unit_cell, side_lengths)

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
        lattice_vectors = [
            (x, y, z)
            for x in range(x_length)
            for y in range(y_length)
            for z in range(z_length)
        ]

        return lattice_vectors

    def to_unit_cell(self) -> UnitCell:
        """
        Super cell to unit cell
        =======================

        Creates an instance of `UnitCell` given a super cell.
        """
        unit_cell = self.unit_cell
        lattice_vectors = self.lattice_vectors()

        lattice_constants = (
            self.side_lengths[0] * unit_cell.lattice_constants[0],
            self.side_lengths[1] * unit_cell.lattice_constants[1],
            self.side_lengths[2] * unit_cell.lattice_constants[2],
        )

        x_side_length, y_side_length, z_side_length = self.side_lengths

        atoms = []

        for lattice_vector in lattice_vectors:
            for atom in unit_cell.atoms:
                atoms.append(
                    atom.shift_position(lattice_vector).scale_position(
                        (1 / x_side_length, 1 / y_side_length, 1 / z_side_length)
                    )
                )

        return UnitCell(unit_cell.material, lattice_constants, atoms)
