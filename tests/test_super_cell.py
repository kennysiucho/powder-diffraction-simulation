"""
This module contains unit tests for the super_cell.py module.
"""

from B8_project.crystal import UnitCell
from B8_project import file_reading
from B8_project.alloy import SuperCell


class TestSuperCell:
    """
    Unit tests for the `SuperCell` class.
    """

    @staticmethod
    def test_super_cell_initialization_normal_operation():
        """
        A unit test that tests the initialization of a `SuperCell` instance. This unit
        test tests initialization with normal attributes.
        """
        basis = file_reading.read_basis("tests/data/CsCl_basis.csv")
        lattice = file_reading.read_lattice("tests/data/CsCl_lattice.csv")
        unit_cell = UnitCell.new_unit_cell(basis, lattice)

        side_lengths = (2, 2, 2)
        super_cell = SuperCell(unit_cell, side_lengths)

        assert super_cell.unit_cell == unit_cell
        assert super_cell.side_lengths == side_lengths

    @staticmethod
    def test_new_super_cell_normal_operation():
        """
        A unit test that tests the new_super_cell method of the SuperCell class. This
        unit test tests the normal operation of the new_super_cell method.
        """
        basis = file_reading.read_basis("tests/data/CsCl_basis.csv")
        lattice = file_reading.read_lattice("tests/data/CsCl_lattice.csv")
        unit_cell = UnitCell.new_unit_cell(basis, lattice)

        super_cell = SuperCell.new_super_cell(unit_cell, (2, 2, 2))

        assert super_cell.unit_cell == unit_cell
        assert super_cell.side_lengths == (2, 2, 2)

    @staticmethod
    def test_get_lattice_vectors_normal_operation():
        """
        A unit test that tests the get_lattice_vectors method of the SuperCell class.
        This unit test tests the normal operation of the get_lattice_vectors method.
        """
        basis = file_reading.read_basis("tests/data/CsCl_basis.csv")
        lattice = file_reading.read_lattice("tests/data/CsCl_lattice.csv")
        unit_cell = UnitCell.new_unit_cell(basis, lattice)

        super_cell = SuperCell.new_super_cell(unit_cell, (1, 1, 1))
        assert super_cell.lattice_vectors() == [(0, 0, 0)]

        super_cell = SuperCell.new_super_cell(unit_cell, (2, 2, 2))
        assert (
            super_cell.lattice_vectors().sort()
            == [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),
                (1, 1, 1),
            ].sort()
        )

    @staticmethod
    def test_to_unit_cell_normal_operation():
        """
        A unit test that tests the to_unit_cell method of the SuperCell class. This
        unit test tests the normal operation of the to_unit_cell method.
        """
        basis = file_reading.read_basis("tests/data/CsCl_basis.csv")
        lattice = file_reading.read_lattice("tests/data/CsCl_lattice.csv")
        unit_cell = UnitCell.new_unit_cell(basis, lattice)

        super_cell = SuperCell.new_super_cell(unit_cell, (1, 1, 1))
        assert super_cell.to_unit_cell() == unit_cell

        basis = file_reading.read_basis("tests/data/Na_basis.csv")
        lattice = file_reading.read_lattice("tests/data/Na_lattice.csv")
        unit_cell = UnitCell.new_unit_cell(basis, lattice)

        super_cell = SuperCell.new_super_cell(unit_cell, (2, 2, 2))

        assert super_cell.to_unit_cell().material == "Na"

        assert super_cell.to_unit_cell().lattice_constants == (
            2 * unit_cell.lattice_constants[0],
            2 * unit_cell.lattice_constants[1],
            2 * unit_cell.lattice_constants[2],
        )

        atomic_positions = [atom.position for atom in super_cell.to_unit_cell().atoms]
        assert (
            atomic_positions.sort()
            == [
                (0, 0, 0),
                (0.25, 0.25, 0.25),
                (0.5, 0, 0),
                (0.75, 0.25, 0.25),
                (0, 0.5, 0),
                (0.25, 0.75, 0.25),
                (0, 0, 0.5),
                (0.25, 0.25, 0.75),
                (0.5, 0.5, 0),
                (0.75, 0.75, 0.25),
                (0.5, 0, 0.5),
                (0.75, 0.25, 0.75),
                (0, 0.5, 0.5),
                (0.25, 0.75, 0.75),
                (0.5, 0.5, 0.5),
                (0.75, 0.75, 0.75),
            ].sort()
        )
