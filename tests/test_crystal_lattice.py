"""
This module contains unit tests for the crystal_lattice.py module.
TODO: add tests for functions in crystal_lattice.py.
"""

from B8_project.crystal_lattice import Atom


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


# def test_validate_parameters_normal_operation():
#     """
#     A unit test for the validate_parameters function. This unit test tests normal
#     operation of the function.
#     """
#     filename_1 = "tests/parameters/test_lattice.csv"
#     filename_2 = "tests/parameters/test_basis.csv"

#     lattice = get_lattice_from_csv(filename_1)
#     basis = get_basis_from_csv(filename_2)

#     assert validate_parameters(lattice, basis) is None
