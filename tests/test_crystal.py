"""
This module contains unit tests for the crystal.py module.
TODO: add test for base-centred lattice type once base-centred logic has been 
implemented.
"""

import numpy as np
import numpy.testing as nptest
import pytest
from copy import deepcopy
from B8_project.crystal import (
    Atom,
    UnitCell,
    ReciprocalLatticeVector,
    UnitCellVarieties,
    ReplacementProbability,
)
from B8_project import file_reading
from B8_project import utils


class TestAtom:
    """
    Unit tests for the `Atom` class.
    """

    @staticmethod
    def test_atom_initialization_normal_operation():
        """
        A unit test that tests the initialization of an `Atom` instance. This unit test
        tests initialization with normal attributes.
        """
        atom = Atom(11, (0, 0, 0))
        assert atom.atomic_number == 11
        assert atom.position == (0, 0, 0)

    @staticmethod
    def test_atom_shift_position_normal_operation():
        """
        A unit test that tests the shift_position method of the Atom class. This unit
        test verifies the normal operation of the shift_position method.
        """
        atom = Atom(11, (0, 0, 0))
        assert atom.shift_position((0, 0, 0)) == atom
        assert atom.shift_position((0.5, 0.5, 0.5)) == Atom(11, (0.5, 0.5, 0.5))

        atom = Atom(17, (0.5, 0.25, 0.25))
        assert atom.shift_position((0.1, 0.2, 0.3)) == Atom(17, (0.6, 0.45, 0.55))

    @staticmethod
    def test_atom_scale_position_normal_operation():
        """
        A unit test that tests the scale_position method of the Atom class. This unit
        test verifies the normal operation of the scale_position method.
        """
        atom = Atom(1, (0.5, 0.5, 0.5))
        assert atom.scale_position((2, 2, 2)) == Atom(1, (1, 1, 1))
        assert atom.scale_position((0.5, 0.5, 0.5)) == Atom(1, (0.25, 0.25, 0.25))


class TestUnitCell:
    """
    Unit tests for the `UnitCell` class.
    """

    @staticmethod
    def test_unit_cell_initialization_normal_operation():
        """
        A unit test that tests the initialization of a `UnitCell` instance. This unit test
        tests initialization with normal attributes.
        """
        atoms = [Atom(11, (0, 0, 0)), Atom(17, (0.5, 0.5, 0.5))]
        unit_cell = UnitCell("NaCl", (1, 1, 1), atoms)
        assert unit_cell.material == "NaCl"
        assert unit_cell.lattice_constants == (1, 1, 1)
        assert unit_cell.atoms == atoms

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
        assert UnitCell._validate_crystal_parameters(CsCl_basis, CsCl_lattice) is None
        assert UnitCell._validate_crystal_parameters(Cu_basis, Cu_lattice) is None
        assert UnitCell._validate_crystal_parameters(Na_basis, Na_lattice) is None
        assert UnitCell._validate_crystal_parameters(NaCl_basis, NaCl_lattice) is None
        # pylint: enable=W0212

    @staticmethod
    def test_new_unit_cell_normal_operation():
        """
        A unit test for the crystal_parameters_to_unit_cell function. This unit test tests
        normal operation of the function.
        """
        CsCl_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/CsCl_basis.csv"
        )
        CsCl_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/CsCl_lattice.csv"
        )
        unit_cell = UnitCell.new_unit_cell(CsCl_basis, CsCl_lattice)
        assert unit_cell is not None
        assert unit_cell.material == "CsCl"
        assert unit_cell.lattice_constants == (0.4119, 0.4119, 0.4119)
        assert sorted(unit_cell.atoms, key=str) == sorted(
            [Atom(55, (0, 0, 0)), Atom(17, (0.5, 0.5, 0.5))], key=str
        )

        Cu_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/Cu_basis.csv"
        )
        Cu_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/Cu_lattice.csv"
        )
        unit_cell = UnitCell.new_unit_cell(Cu_basis, Cu_lattice)
        assert unit_cell is not None
        assert unit_cell.material == "Cu"
        assert unit_cell.lattice_constants == (0.3615, 0.3615, 0.3615)
        assert sorted(unit_cell.atoms, key=str) == sorted(
            [
                Atom(29, (0, 0, 0)),
                Atom(29, (0.5, 0.5, 0)),
                Atom(29, (0.5, 0, 0.5)),
                Atom(29, (0, 0.5, 0.5)),
            ],
            key=str,
        )

        Na_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/Na_basis.csv"
        )
        Na_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/Na_lattice.csv"
        )
        unit_cell = UnitCell.new_unit_cell(Na_basis, Na_lattice)
        assert unit_cell is not None
        assert unit_cell.material == "Na"
        assert unit_cell.lattice_constants == (0.4287, 0.4287, 0.4287)
        assert sorted(unit_cell.atoms, key=str) == sorted(
            [Atom(11, (0, 0, 0)), Atom(11, (0.5, 0.5, 0.5))], key=str
        )

        NaCl_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/NaCl_basis.csv"
        )
        NaCl_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/NaCl_lattice.csv"
        )
        unit_cell = UnitCell.new_unit_cell(NaCl_basis, NaCl_lattice)
        assert unit_cell is not None
        assert unit_cell.material == "NaCl"
        assert unit_cell.lattice_constants == (0.5640, 0.5640, 0.5640)
        assert sorted(unit_cell.atoms, key=str) == sorted(
            [
                Atom(11, (0, 0, 0)),
                Atom(11, (0.5, 0.5, 0)),
                Atom(11, (0.5, 0, 0.5)),
                Atom(11, (0, 0.5, 0.5)),
                Atom(17, (0.5, 0.5, 0.5)),
                Atom(17, (1.0, 1.0, 0.5)),
                Atom(17, (1.0, 0.5, 1.0)),
                Atom(17, (0.5, 1.0, 1.0)),
            ],
            key=str,
        )

@pytest.fixture(name="gaas_unit_cell")
def fixture_gaas_unit_cell() -> UnitCell:
    """
    Returns a GaAs unit cell.
    """
    gaas_basis = file_reading.read_basis("tests/data/GaAs_basis.csv")
    gaas_lattice = file_reading.read_lattice("tests/data/GaAs_lattice.csv")
    unit_cell = UnitCell.new_unit_cell(gaas_basis, gaas_lattice)
    yield unit_cell

@pytest.fixture(name="ingaas_replacement_prob")
def fixture_ingaas_replacement_prob() -> ReplacementProbability:
    """
    Returns instance of ReplacementProbability, specifying to replace Ga with in with
    a 0.2 probability.
    """
    yield ReplacementProbability(31, 49, 0.2)

@pytest.fixture(name="uc_vars")
def fixture_uc_vars(gaas_unit_cell, ingaas_replacement_prob) -> UnitCellVarieties:
    """
    Returns instance of UnitCellVarieties
    """
    yield UnitCellVarieties(gaas_unit_cell, ingaas_replacement_prob)

class TestUnitCellVarieties:
    """
    Unit tests for the `UnitCellVarieties` class.
    """

    @staticmethod
    def test_generate_all_unit_cells_correct_atom_combinations(uc_vars,
                                                               gaas_unit_cell):
        """
        Tests that all combinations of unit cells of an alloy are generated
        """
        expected_ucs = [deepcopy(gaas_unit_cell) for _ in range(16)]
        ga_indices = [i for i in range(len(gaas_unit_cell.atoms)) if
                      gaas_unit_cell.atoms[i].atomic_number == 31]
        assert len(ga_indices) == 4

        # 1 Ga atom replaced with In
        expected_ucs[1].atoms[ga_indices[0]].atomic_number = 49
        expected_ucs[2].atoms[ga_indices[1]].atomic_number = 49
        expected_ucs[3].atoms[ga_indices[2]].atomic_number = 49
        expected_ucs[4].atoms[ga_indices[3]].atomic_number = 49

        # 2 Ga atoms replaced with In
        expected_ucs[5].atoms[ga_indices[0]].atomic_number = 49
        expected_ucs[5].atoms[ga_indices[1]].atomic_number = 49
        expected_ucs[6].atoms[ga_indices[0]].atomic_number = 49
        expected_ucs[6].atoms[ga_indices[2]].atomic_number = 49
        expected_ucs[7].atoms[ga_indices[0]].atomic_number = 49
        expected_ucs[7].atoms[ga_indices[3]].atomic_number = 49
        expected_ucs[8].atoms[ga_indices[1]].atomic_number = 49
        expected_ucs[8].atoms[ga_indices[2]].atomic_number = 49
        expected_ucs[9].atoms[ga_indices[1]].atomic_number = 49
        expected_ucs[9].atoms[ga_indices[3]].atomic_number = 49
        expected_ucs[10].atoms[ga_indices[2]].atomic_number = 49
        expected_ucs[10].atoms[ga_indices[3]].atomic_number = 49

        # 3 Ga atoms replaced with In
        expected_ucs[11].atoms[ga_indices[0]].atomic_number = 49
        expected_ucs[11].atoms[ga_indices[1]].atomic_number = 49
        expected_ucs[11].atoms[ga_indices[2]].atomic_number = 49
        expected_ucs[12].atoms[ga_indices[0]].atomic_number = 49
        expected_ucs[12].atoms[ga_indices[1]].atomic_number = 49
        expected_ucs[12].atoms[ga_indices[3]].atomic_number = 49
        expected_ucs[13].atoms[ga_indices[0]].atomic_number = 49
        expected_ucs[13].atoms[ga_indices[2]].atomic_number = 49
        expected_ucs[13].atoms[ga_indices[3]].atomic_number = 49
        expected_ucs[14].atoms[ga_indices[1]].atomic_number = 49
        expected_ucs[14].atoms[ga_indices[2]].atomic_number = 49
        expected_ucs[14].atoms[ga_indices[3]].atomic_number = 49

        # 4 Ga atoms replaced with In
        expected_ucs[15].atoms[ga_indices[0]].atomic_number = 49
        expected_ucs[15].atoms[ga_indices[1]].atomic_number = 49
        expected_ucs[15].atoms[ga_indices[2]].atomic_number = 49
        expected_ucs[15].atoms[ga_indices[3]].atomic_number = 49

        assert len(uc_vars.unit_cell_varieties) == 16

        # Check whether uc_vars contains all elements of expected_ucs
        for uc in uc_vars.unit_cell_varieties:
            try:
                expected_ucs.remove(uc)
            except ValueError as e:
                raise AssertionError(f"Unexpected unit cell found: {uc}") from e
        assert len(expected_ucs) == 0

    @staticmethod
    def test_generate_all_unit_cells_does_not_alter_atom_positions(uc_vars):
        """
        Check that all the unit cells have the same atomic positions in the same order.
        """
        expected_positions = np.array(uc_vars.unit_cell_varieties[0].positions())
        for uc in uc_vars.unit_cell_varieties:
            pos = np.array(uc.positions())
            nptest.assert_allclose(pos, expected_positions)

    @staticmethod
    def test_generate_all_unit_cells_unique_references(uc_vars):
        """
        Check that uc_vars contains unique UnitCell objects
        """
        for i in range(len(uc_vars.unit_cell_varieties)):
            for j in range(i + 1, len(uc_vars.unit_cell_varieties)):
                assert uc_vars.unit_cell_varieties[i] is not \
                       uc_vars.unit_cell_varieties[j]

    @staticmethod
    def test_calculate_probabilities_correct(uc_vars):
        """
        Tests that the correct set of probabilities is generated
        """
        probs = uc_vars.probabilities
        probs.sort()
        # Expected probability distribution for 4 atoms with replacement probability
        # of 0.2
        expected_probs = [0.0016,
                          0.0064, 0.0064, 0.0064, 0.0064,
                          0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256,
                          0.1024, 0.1024, 0.1024, 0.1024,
                          0.4096]

        nptest.assert_allclose(probs, expected_probs)

    @staticmethod
    def test_calculate_probabilities_sum_to_one(uc_vars):
        """
        Tests that the probability distribution for unit cell varieties sum to one.
        """
        nptest.assert_allclose(np.sum(uc_vars.probabilities), 1.0)

    @staticmethod
    def test_atomic_number_lists(uc_vars):
        """
        Tests that the lists of atomic numbers are correct and have the correct
        probability.
        """
        atomic_numbers, probs = uc_vars.atomic_number_lists()
        actual = {(tuple(atomic_numbers[i]), probs[i]) for i in range(len(
            atomic_numbers))}
        expected = {((31, 31, 31, 31, 33, 33, 33, 33), 0.4096),
                    ((49, 31, 31, 31, 33, 33, 33, 33), 0.1024),
                    ((31, 49, 31, 31, 33, 33, 33, 33), 0.1024),
                    ((31, 31, 49, 31, 33, 33, 33, 33), 0.1024),
                    ((31, 31, 31, 49, 33, 33, 33, 33), 0.1024),
                    ((49, 49, 31, 31, 33, 33, 33, 33), 0.0256),
                    ((49, 31, 49, 31, 33, 33, 33, 33), 0.0256),
                    ((49, 31, 31, 49, 33, 33, 33, 33), 0.0256),
                    ((31, 49, 49, 31, 33, 33, 33, 33), 0.0256),
                    ((31, 49, 31, 49, 33, 33, 33, 33), 0.0256),
                    ((31, 31, 49, 49, 33, 33, 33, 33), 0.0256),
                    ((49, 49, 49, 31, 33, 33, 33, 33), 0.0064),
                    ((49, 49, 31, 49, 33, 33, 33, 33), 0.0064),
                    ((49, 31, 49, 49, 33, 33, 33, 33), 0.0064),
                    ((31, 49, 49, 49, 33, 33, 33, 33), 0.0064),
                    ((49, 49, 49, 49, 33, 33, 33, 33), 0.0016)}

        assert len(actual) == len(expected)

        # Convert to sorted list to compare using numpy
        expected_sorted = sorted(expected, key=lambda x: (x[0], x[1]))
        actual_sorted = sorted(actual, key=lambda x: (x[0], x[1]))

        for (expected_tuple, expected_float), (actual_tuple, actual_float) in zip(
                expected_sorted, actual_sorted):
            # Compare the tuple elements directly
            assert expected_tuple == actual_tuple
            # Compare floats
            np.testing.assert_allclose(expected_float, actual_float)

class TestReciprocalLatticeVector:
    """
    Unit tests for the `ReciprocalLatticeVector` class.
    """

    @staticmethod
    def test_reciprocal_lattice_vector_initialization_normal_operation():
        """
        A unit test that tests the initialization of a `ReciprocalLatticeVector` instance.
        This unit test tests initialization with normal attributes.
        """
        CsCl_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/CsCl_basis.csv"
        )
        CsCl_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/CsCl_lattice.csv"
        )
        unit_cell = UnitCell.new_unit_cell(CsCl_basis, CsCl_lattice)
        assert unit_cell is not None

        reciprocal_lattice_vector = ReciprocalLatticeVector(
            (1, 2, 3), unit_cell.lattice_constants
        )
        assert reciprocal_lattice_vector.miller_indices == (1, 2, 3)
        assert reciprocal_lattice_vector.lattice_constants == (0.4119, 0.4119, 0.4119)

    @staticmethod
    def test_components_normal_operation():
        """
        A unit test for the components function. This unit test tests normal operation
        of the function.
        """
        CsCl_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/CsCl_basis.csv"
        )
        CsCl_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/CsCl_lattice.csv"
        )
        unit_cell = UnitCell.new_unit_cell(CsCl_basis, CsCl_lattice)
        assert unit_cell is not None

        reciprocal_lattice_vector = ReciprocalLatticeVector(
            (1, 2, 3), unit_cell.lattice_constants
        )
        components = reciprocal_lattice_vector.components()

        a, b, c = reciprocal_lattice_vector.lattice_constants
        expected_components = (2 * np.pi / a, 4 * np.pi / b, 6 * np.pi / c)

        assert np.isclose(components[0], expected_components[0], rtol=1e-6)
        assert np.isclose(components[1], expected_components[1], rtol=1e-6)
        assert np.isclose(components[2], expected_components[2], rtol=1e-6)

    @staticmethod
    def test_magnitude_normal_operation():
        """
        A unit test for the magnitude function. This unit test tests normal operation
        of the function.
        """
        CsCl_basis = file_reading.read_basis(  # pylint: disable=C0103
            "tests/data/CsCl_basis.csv"
        )
        CsCl_lattice = file_reading.read_lattice(  # pylint: disable=C0103
            "tests/data/CsCl_lattice.csv"
        )
        unit_cell = UnitCell.new_unit_cell(CsCl_basis, CsCl_lattice)
        assert unit_cell is not None

        reciprocal_lattice_vector = ReciprocalLatticeVector(
            (1, 2, 3), unit_cell.lattice_constants
        )

        a, b, c = reciprocal_lattice_vector.lattice_constants
        expected_components = (2 * np.pi / a, 4 * np.pi / b, 6 * np.pi / c)

        assert np.isclose(
            reciprocal_lattice_vector.magnitude(),
            np.sqrt(utils.dot_product_tuples(expected_components, expected_components)),
            rtol=1e-6,
        )

    @staticmethod
    def test_get_reciprocal_lattice_vectors_normal_operation():
        """
        A unit test for the get_reciprocal_lattice_vectors function. This unit test tests
        normal operation of the function.
        """
        basis = file_reading.read_basis("tests/data/test_basis.csv")
        lattice = file_reading.read_lattice("tests/data/test_lattice.csv")
        unit_cell = UnitCell.new_unit_cell(basis, lattice)

        reciprocal_lattice_vectors = (
            ReciprocalLatticeVector.get_reciprocal_lattice_vectors(
                1, 2 * np.pi + 0.001, unit_cell
            )
        )

        expected_reciprocal_lattice_vectors = [
            ReciprocalLatticeVector((-1, 0, 0), unit_cell.lattice_constants),
            ReciprocalLatticeVector((0, -1, 0), unit_cell.lattice_constants),
            ReciprocalLatticeVector((0, 0, -1), unit_cell.lattice_constants),
            ReciprocalLatticeVector((1, 0, 0), unit_cell.lattice_constants),
            ReciprocalLatticeVector((0, 1, 0), unit_cell.lattice_constants),
            ReciprocalLatticeVector((0, 0, 1), unit_cell.lattice_constants),
        ]

        assert sorted(reciprocal_lattice_vectors, key=str) == sorted(
            expected_reciprocal_lattice_vectors, key=str
        )
