"""
This module contains tests for the alloy module.
TODO: add tests for the add_disorder function.
"""

import numpy as np

from B8_project import alloy, file_reading, crystal


class TestSuperCell:
    """
    This class contains tests for the SuperCell class.
    """

    @staticmethod
    def test_get_lattice_vectors_normal_operation():
        """
        A unit test that tests the get_lattice_vectors method of the SuperCell class.
        This unit test tests the normal operation of the get_lattice_vectors method.
        """
        # pylint: disable=protected-access
        lattice_vectors = alloy.SuperCell._get_lattice_vectors((1, 1, 1))
        assert np.array_equal(lattice_vectors, np.array([[0, 0, 0]]))

        lattice_vectors = alloy.SuperCell._get_lattice_vectors((2, 2, 2))
        expected_lattice_vectors = np.array(
            [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),
                (1, 1, 1),
            ]
        )

        assert np.array_equal(
            lattice_vectors,
            expected_lattice_vectors[np.lexsort(expected_lattice_vectors.T)],
        )
        # pylint: enable=protected-access

    @staticmethod
    def test_new_super_cell_normal_operation():
        """
        A unit test that tests the new_super_cell method of the SuperCell class. This
        unit test tests the normal operation of the new_super_cell method.
        """
        # 1*1*1 CsCl super cell.
        basis = file_reading.read_basis("tests/data/CsCl_basis.csv")
        lattice = file_reading.read_lattice("tests/data/CsCl_lattice.csv")
        unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

        super_cell = alloy.SuperCell.new_super_cell(unit_cell, (1, 1, 1))

        assert super_cell.material == unit_cell.material
        assert np.array_equal(unit_cell.lattice_constants, super_cell.lattice_constants)
        assert np.array_equal(unit_cell.atoms, super_cell.atoms)

        # 2*2*2 Na super cell.
        basis = file_reading.read_basis("tests/data/Na_basis.csv")
        lattice = file_reading.read_lattice("tests/data/Na_lattice.csv")
        unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

        super_cell = alloy.SuperCell.new_super_cell(unit_cell, (2, 2, 2))

        assert unit_cell.material == super_cell.material
        assert np.array_equal(
            2 * unit_cell.lattice_constants, super_cell.lattice_constants
        )

        expected_atomic_positions = np.array(
            [
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
            ]
        )

        assert np.array_equal(
            super_cell.atoms["positions"][np.lexsort(super_cell.atoms["positions"].T)],
            expected_atomic_positions[np.lexsort(expected_atomic_positions.T)],
        )
