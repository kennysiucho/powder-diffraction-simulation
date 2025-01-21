"""
This module contains unit tests for the crystal_lattice.py module.
TODO: add tests for ReciprocalLatticeVector class.
TODO: add tests for XRayFormFactor class.

TODO: finish tests for UnitCell class.
"""

from B8_project.crystal_lattice import Atom, UnitCell
from B8_project import file_reading


def test_atom_initialization_normal_operation():
    """
    A unit test that tests the initialization of an `Atom` instance. This unit test
    tests initialization with normal attributes.
    """
    atom = Atom(11, (0, 0, 0))
    assert atom.atomic_number == 11
    assert atom.position == (0, 0, 0)


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


# def test_validate_parameters_normal_operation():
#     """
#     A unit test for the validate_parameters function. This unit test tests normal
#     operation of the function.
#     """
#     filename_1 = "tests/parameters/test_lattice.csv"
#     filename_2 = "tests/parameters/test_basis.csv"

#     lattice = file_reading.get_lattice_from_csv(filename_1)
#     basis = file_reading.get_basis_from_csv(filename_2)

#     assert UnitCell.validate_crystal_parameters(lattice, basis) is None


# def test_crystal_parameters_to_unit_cell_normal_operation():
#     """
#     A unit test for the crystal_parameters_to_unit_cell function. This unit test tests
#     normal operation of the function.
#     """
#     filename_1 = "tests/parameters/test_lattice.csv"
#     filename_2 = "tests/parameters/test_basis.csv"

#     lattice = file_reading.get_lattice_from_csv(filename_1)
#     basis = file_reading.get_basis_from_csv(filename_2)

#     unit_cell = UnitCell.crystal_parameters_to_unit_cell(lattice, basis)
#     assert unit_cell is not None

#     assert unit_cell.material == "NaCl"
#     assert unit_cell.lattice_constants == (1, 1, 1)
#     # Add assertion for atoms in the unit cell.
