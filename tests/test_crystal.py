"""
This module contains unit tests for the crystal.py module.
TODO: add test for base-centred lattice type once base-centred logic has been 
implemented.
TODO: add tests for Diffraction class.

TODO: add tests for error handling for crystal module.
"""

import math
from B8_project.crystal import (
    Atom,
    UnitCell,
    ReciprocalLatticeVector,
    NeutronFormFactor,
    XRayFormFactor,
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
        A unit test that tests the shift_position method of the Atom class. This unit test
        verifies the normal operation of the shift_position method.
        """
        atom = Atom(11, (0, 0, 0))
        assert atom.shift_position((0, 0, 0)) == atom
        assert atom.shift_position((0.5, 0.5, 0.5)) == Atom(11, (0.5, 0.5, 0.5))

        atom = Atom(17, (0.5, 0.25, 0.25))
        assert atom.shift_position((0.1, 0.2, 0.3)) == Atom(17, (0.6, 0.45, 0.55))


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
        CsCl_basis = file_reading.read_basis("tests/data/CsCl_basis.csv")
        CsCl_lattice = file_reading.read_lattice("tests/data/CsCl_lattice.csv")
        Cu_basis = file_reading.read_basis("tests/data/Cu_basis.csv")
        Cu_lattice = file_reading.read_lattice("tests/data/Cu_lattice.csv")
        Na_basis = file_reading.read_basis("tests/data/Na_basis.csv")
        Na_lattice = file_reading.read_lattice("tests/data/Na_lattice.csv")
        NaCl_basis = file_reading.read_basis("tests/data/NaCl_basis.csv")
        NaCl_lattice = file_reading.read_lattice("tests/data/NaCl_lattice.csv")

        assert UnitCell.validate_crystal_parameters(CsCl_basis, CsCl_lattice) is None
        assert UnitCell.validate_crystal_parameters(Cu_basis, Cu_lattice) is None
        assert UnitCell.validate_crystal_parameters(Na_basis, Na_lattice) is None
        assert UnitCell.validate_crystal_parameters(NaCl_basis, NaCl_lattice) is None

    @staticmethod
    def test_get_unit_cell_normal_operation():
        """
        A unit test for the crystal_parameters_to_unit_cell function. This unit test tests
        normal operation of the function.
        """
        CsCl_basis = file_reading.read_basis("tests/data/CsCl_basis.csv")
        CsCl_lattice = file_reading.read_lattice("tests/data/CsCl_lattice.csv")
        unit_cell = UnitCell.get_unit_cell(CsCl_basis, CsCl_lattice)
        assert unit_cell is not None
        assert unit_cell.material == "CsCl"
        assert unit_cell.lattice_constants == (0.4119, 0.4119, 0.4119)
        assert sorted(unit_cell.atoms, key=str) == sorted(
            [Atom(55, (0, 0, 0)), Atom(17, (0.5, 0.5, 0.5))], key=str
        )

        Cu_basis = file_reading.read_basis("tests/data/Cu_basis.csv")
        Cu_lattice = file_reading.read_lattice("tests/data/Cu_lattice.csv")
        unit_cell = UnitCell.get_unit_cell(Cu_basis, Cu_lattice)
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

        Na_basis = file_reading.read_basis("tests/data/Na_basis.csv")
        Na_lattice = file_reading.read_lattice("tests/data/Na_lattice.csv")
        unit_cell = UnitCell.get_unit_cell(Na_basis, Na_lattice)
        assert unit_cell is not None
        assert unit_cell.material == "Na"
        assert unit_cell.lattice_constants == (0.4287, 0.4287, 0.4287)
        assert sorted(unit_cell.atoms, key=str) == sorted(
            [Atom(11, (0, 0, 0)), Atom(11, (0.5, 0.5, 0.5))], key=str
        )

        NaCl_basis = file_reading.read_basis("tests/data/NaCl_basis.csv")
        NaCl_lattice = file_reading.read_lattice("tests/data/NaCl_lattice.csv")
        unit_cell = UnitCell.get_unit_cell(NaCl_basis, NaCl_lattice)
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
        CsCl_basis = file_reading.read_basis("tests/data/CsCl_basis.csv")
        CsCl_lattice = file_reading.read_lattice("tests/data/CsCl_lattice.csv")
        unit_cell = UnitCell.get_unit_cell(CsCl_basis, CsCl_lattice)
        assert unit_cell is not None

        reciprocal_lattice_vector = ReciprocalLatticeVector(
            (1, 2, 3), unit_cell.lattice_constants
        )
        assert reciprocal_lattice_vector.miller_indices == (1, 2, 3)
        assert reciprocal_lattice_vector.lattice_constants == (0.4119, 0.4119, 0.4119)

    @staticmethod
    def test_get_components_normal_operation():
        """
        A unit test for the get_components function. This unit test tests normal operation
        of the function.
        """
        CsCl_basis = file_reading.read_basis("tests/data/CsCl_basis.csv")
        CsCl_lattice = file_reading.read_lattice("tests/data/CsCl_lattice.csv")
        unit_cell = UnitCell.get_unit_cell(CsCl_basis, CsCl_lattice)
        assert unit_cell is not None

        reciprocal_lattice_vector = ReciprocalLatticeVector(
            (1, 2, 3), unit_cell.lattice_constants
        )
        components = reciprocal_lattice_vector.get_components()

        a, b, c = reciprocal_lattice_vector.lattice_constants
        expected_components = (2 * math.pi / a, 4 * math.pi / b, 6 * math.pi / c)

        assert math.isclose(components[0], expected_components[0], rel_tol=1e-6)
        assert math.isclose(components[1], expected_components[1], rel_tol=1e-6)
        assert math.isclose(components[2], expected_components[2], rel_tol=1e-6)

    @staticmethod
    def test_get_magnitude_normal_operation():
        """
        A unit test for the get_magnitude function. This unit test tests normal operation of the function.
        """
        CsCl_basis = file_reading.read_basis("tests/data/CsCl_basis.csv")
        CsCl_lattice = file_reading.read_lattice("tests/data/CsCl_lattice.csv")
        unit_cell = UnitCell.get_unit_cell(CsCl_basis, CsCl_lattice)
        assert unit_cell is not None

        reciprocal_lattice_vector = ReciprocalLatticeVector(
            (1, 2, 3), unit_cell.lattice_constants
        )

        a, b, c = reciprocal_lattice_vector.lattice_constants
        expected_components = (2 * math.pi / a, 4 * math.pi / b, 6 * math.pi / c)

        assert math.isclose(
            reciprocal_lattice_vector.get_magnitude(),
            math.sqrt(
                utils.dot_product_tuples(expected_components, expected_components)
            ),
            rel_tol=1e-6,
        )

    @staticmethod
    def test_get_reciprocal_lattice_vectors_normal_operation():
        """
        A unit test for the get_reciprocal_lattice_vectors function. This unit test tests
        normal operation of the function.
        """
        basis = file_reading.read_basis("tests/data/test_basis.csv")
        lattice = file_reading.read_lattice("tests/data/test_lattice.csv")
        unit_cell = UnitCell.get_unit_cell(basis, lattice)

        reciprocal_lattice_vectors = (
            ReciprocalLatticeVector.get_reciprocal_lattice_vectors(
                2 * math.pi + 0.001, unit_cell
            )
        )

        expected_reciprocal_lattice_vectors = [
            ReciprocalLatticeVector((-1, 0, 0), unit_cell.lattice_constants),
            ReciprocalLatticeVector((0, -1, 0), unit_cell.lattice_constants),
            ReciprocalLatticeVector((0, 0, -1), unit_cell.lattice_constants),
            ReciprocalLatticeVector((0, 0, 0), unit_cell.lattice_constants),
            ReciprocalLatticeVector((1, 0, 0), unit_cell.lattice_constants),
            ReciprocalLatticeVector((0, 1, 0), unit_cell.lattice_constants),
            ReciprocalLatticeVector((0, 0, 1), unit_cell.lattice_constants),
        ]

        assert sorted(reciprocal_lattice_vectors, key=str) == sorted(
            expected_reciprocal_lattice_vectors, key=str
        )


class TestNeutronFormFactor:
    """
    Unit tests for the `NeutronFormFactor` class.
    """

    @staticmethod
    def test_neutron_form_factor_initialization_normal_operation():
        """
        A unit test that tests the initialization of a `NeutronFormFactor` instance.
        This unit test tests initialization with normal attributes.
        """
        neutron_form_factor = NeutronFormFactor(1)
        assert neutron_form_factor.neutron_scattering_length == 1

    @staticmethod
    def test_get_form_factor_normal_operation():
        """
        A unit test for the get_form_factor function. This unit test tests normal
        operation of the function.
        """
        neutron_form_factor = NeutronFormFactor(1)
        assert (
            neutron_form_factor.get_form_factor(
                ReciprocalLatticeVector((0, 0, 0), (1, 1, 1))
            )
            == 1
        )


class TestXRayFormFactor:
    """
    Unit tests for the `XRayFormFactor` class.
    """

    @staticmethod
    def test_xray_form_factor_initialization_normal_operation():
        """
        A unit test that tests the initialization of a `XRayFormFactor` instance.
        This unit test tests initialization with normal attributes.
        """
        xray_form_factor = XRayFormFactor(1, 2, 3, 4, 5, 6, 7, 8, 9)

        assert xray_form_factor.a1 == 1
        assert xray_form_factor.b1 == 2
        assert xray_form_factor.a2 == 3
        assert xray_form_factor.b2 == 4
        assert xray_form_factor.a3 == 5
        assert xray_form_factor.b3 == 6
        assert xray_form_factor.a4 == 7
        assert xray_form_factor.b4 == 8
        assert xray_form_factor.c == 9

    @staticmethod
    def test_get_form_factor():
        """
        A unit test for the get_form_factor function. This unit test tests normal
        operation of the function.
        """
        x_ray_form_factor = XRayFormFactor(1, 2, 3, 4, 5, 6, 7, 8, 9)
        reciprocal_lattice_vector = ReciprocalLatticeVector((1, 0, 0), (1, 1, 1))

        assert math.isclose(
            reciprocal_lattice_vector.get_magnitude(), 2 * math.pi, rel_tol=1e-6
        )

        result = x_ray_form_factor.get_form_factor(reciprocal_lattice_vector)

        a = [1, 3, 5, 7]
        b = [2, 4, 6, 8]
        c = 9

        expected_result = 0

        for i in range(4):
            expected_result += a[i] * math.exp(-b[i] / 4)

        expected_result += c

        assert math.isclose(result, expected_result, rel_tol=1e-6)


class TestDiffraction:
    """
    Unit tests for the `Diffraction` class.
    """
