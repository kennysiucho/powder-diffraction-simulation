"""
This module contains unit tests for the crystal.py module.
TODO: add test for base-centred lattice type once base-centred logic has been 
implemented.
TODO: add tests for the ReciprocalSpace class.
"""

import numpy as np
from B8_project import file_reading, crystal


class TestUnitCell:
    """
    Unit tests for the `UnitCell` class.
    """

    @staticmethod
    def test_unit_cell_initialization_normal_operation():
        """
        A unit test that tests the initialization of a `UnitCell` instance. This unit
        test tests initialization with normal attributes.
        """
        atomic_numbers = np.array([11, 17])
        atomic_positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])

        # Define a custom datatype to represent atoms.
        dtype = np.dtype([("atomic_numbers", "i4"), ("positions", "3f8")])

        # Create a structured NumPy array to store the atoms.
        atoms = np.empty(len(atomic_numbers), dtype=dtype)
        atoms["atomic_numbers"] = atomic_numbers
        atoms["positions"] = atomic_positions

        unit_cell = crystal.UnitCell("NaCl", np.array([1.0, 1.0, 1.0]), atoms)
        assert unit_cell.material == "NaCl"
        assert np.array_equal(unit_cell.lattice_constants, np.array([1.0, 1.0, 1.0]))
        assert np.array_equal(unit_cell.atoms, atoms)

    @staticmethod
    def test_validate_parameters_normal_operation():
        """
        A unit test for the validate_parameters function. This unit test tests normal
        operation of the function.
        """
        CsCl_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/CsCl_basis.csv"
        )
        CsCl_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/CsCl_lattice.csv"
        )
        Cu_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/Cu_basis.csv"
        )
        Cu_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/Cu_lattice.csv"
        )
        Na_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/Na_basis.csv"
        )
        Na_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/Na_lattice.csv"
        )
        NaCl_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/NaCl_basis.csv"
        )
        NaCl_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/NaCl_lattice.csv"
        )

        # pylint: disable=W0212
        assert (
            crystal.UnitCell._validate_crystal_parameters(CsCl_basis, CsCl_lattice)
            is None
        )
        assert (
            crystal.UnitCell._validate_crystal_parameters(Cu_basis, Cu_lattice) is None
        )
        assert (
            crystal.UnitCell._validate_crystal_parameters(Na_basis, Na_lattice) is None
        )
        assert (
            crystal.UnitCell._validate_crystal_parameters(NaCl_basis, NaCl_lattice)
            is None
        )
        # pylint: enable=W0212

    @staticmethod
    def test_new_unit_cell_normal_operation():
        """
        A unit test for the crystal_parameters_to_unit_cell function. This unit test
        tests normal operation of the function.
        """
        # CsCl unit cell.
        CsCl_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/CsCl_basis.csv"
        )
        CsCl_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/CsCl_lattice.csv"
        )
        unit_cell = crystal.UnitCell.new_unit_cell(CsCl_basis, CsCl_lattice)
        assert unit_cell is not None
        assert unit_cell.material == "CsCl"
        assert np.array_equal(
            unit_cell.lattice_constants, np.array([0.4119, 0.4119, 0.4119])
        )
        assert np.array_equal(unit_cell.atoms["atomic_numbers"], np.array([55, 17]))
        assert np.array_equal(
            unit_cell.atoms["positions"], np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        )

        # Cu unit cell.
        Cu_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/Cu_basis.csv"
        )
        Cu_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/Cu_lattice.csv"
        )
        unit_cell = crystal.UnitCell.new_unit_cell(Cu_basis, Cu_lattice)
        assert unit_cell is not None
        assert unit_cell.material == "Cu"
        assert np.array_equal(
            unit_cell.lattice_constants, np.array([0.3615, 0.3615, 0.3615])
        )
        assert np.array_equal(
            unit_cell.atoms["atomic_numbers"], np.array([29, 29, 29, 29])
        )
        assert np.array_equal(
            unit_cell.atoms["positions"],
            np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]),
        )

        # Na unit cell.
        Na_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/Na_basis.csv"
        )
        Na_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/Na_lattice.csv"
        )
        unit_cell = crystal.UnitCell.new_unit_cell(Na_basis, Na_lattice)
        assert unit_cell is not None
        assert unit_cell.material == "Na"
        assert np.array_equal(
            unit_cell.lattice_constants, np.array([0.4287, 0.4287, 0.4287])
        )
        assert np.array_equal(unit_cell.atoms["atomic_numbers"], np.array([11, 11]))
        assert np.array_equal(
            unit_cell.atoms["positions"], np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        )

        # NaCl unit cell.
        NaCl_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/NaCl_basis.csv"
        )
        NaCl_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/NaCl_lattice.csv"
        )
        unit_cell = crystal.UnitCell.new_unit_cell(NaCl_basis, NaCl_lattice)
        assert unit_cell is not None
        assert unit_cell.material == "NaCl"
        assert np.array_equal(
            unit_cell.lattice_constants, np.array([0.5640, 0.5640, 0.5640])
        )
        assert np.array_equal(
            unit_cell.atoms["atomic_numbers"],
            np.array([11, 11, 11, 11, 17, 17, 17, 17]),
        )
        assert np.array_equal(
            unit_cell.atoms["positions"],
            np.array(
                [
                    [0, 0, 0],
                    [0.5, 0.5, 0],
                    [0.5, 0, 0.5],
                    [0, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [1.0, 1.0, 0.5],
                    [1.0, 0.5, 1.0],
                    [0.5, 1.0, 1.0],
                ]
            ),
        )


class TestReciprocalSpace:
    """
    Unit tests for the `ReciprocalSpace` class.
    """

    @staticmethod
    def test_get_reciprocal_lattice_vectors_normal_operation():
        """
        A unit test for the get_reciprocal_lattice_vectors function. This unit test
        tests normal operation of the function.
        """
        reciprocal_lattice_vectors = (
            crystal.ReciprocalSpace.get_reciprocal_lattice_vectors(
                1, 2 * np.pi + 0.001, np.array([1.0, 1.0, 1.0])
            )
        )

        expected_miller_indices = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        assert np.array_equal(
            reciprocal_lattice_vectors["miller_indices"][
                np.lexsort(reciprocal_lattice_vectors["miller_indices"].T)
            ],
            expected_miller_indices[np.lexsort(expected_miller_indices.T)],
        )

        expected_magnitudes = np.full(6, 2 * np.pi)
        assert np.array_equal(
            reciprocal_lattice_vectors["magnitudes"], expected_magnitudes
        )

        expected_components = expected_miller_indices * np.full(3, 2 * np.pi)
        assert np.array_equal(
            reciprocal_lattice_vectors["components"][
                np.lexsort(reciprocal_lattice_vectors["components"].T)
            ],
            expected_components[np.lexsort(expected_components.T)],
        )

    @staticmethod
    def test_rlv_magnitudes_from_deflection_angles_normal_operation():
        """
        A unit test for the rlv_magnitudes_from_deflection_angles function. This unit
        test tests normal operation of the function.
        """
        deflection_angles = np.array([45, 90])
        rlv_magnitudes = crystal.ReciprocalSpace.rlv_magnitudes_from_deflection_angles(
            deflection_angles, 0.1
        )

        assert np.all(np.isclose(rlv_magnitudes, np.array([48.089, 88.858]), rtol=1e-4))

    @staticmethod
    def test_deflection_angles_from_rlv_magnitudes_normal_operation():
        """
        A unit test for the deflection_angles_from_rlv_magnitudes function. This unit
        test tests normal operation of the function.
        """
        rlv_magnitudes = np.array([1, 2, 3, 4, 5])

        deflection_angles = (
            crystal.ReciprocalSpace.deflection_angles_from_rlv_magnitudes(
                rlv_magnitudes, 0.1
            )
        )

        expected_rlv_magnitudes = (
            crystal.ReciprocalSpace.rlv_magnitudes_from_deflection_angles(
                deflection_angles, 0.1
            )
        )

        assert np.all(np.isclose(rlv_magnitudes, expected_rlv_magnitudes, rtol=1e-6))
