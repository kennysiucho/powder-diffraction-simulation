"""
This module contains tests for the file_reading.py module.
TODO: add tests for error handling for get_lattice_from_csv.
TODO: add tests for error handling for get_basis_from_csv.
TODO: add tests for error handling for validate_parameters.
TODO: add tests for get_neutron_scattering_lengths_from_csv.
TODO: add tests for get_x_ray_form_factors_from_csv.
"""

from B8_project.file_reading import (
    get_lattice_from_csv,
    get_basis_from_csv,
    validate_parameters,
)


def test_get_lattice_from_csv_normal_operation():
    """
    A unit test for the get_lattice_from_csv function. This unit tests normal operation
    of the function.
    """
    filename = "tests/parameters/test_lattice.csv"
    material, lattice_type, lattice_constants = get_lattice_from_csv(filename)
    assert material == "NaCl"
    assert lattice_type == 3
    assert lattice_constants == (1, 1, 1)


def test_get_basis_from_csv_normal_operation():
    """
    A unit test for the get_basis_from_csv function. This unit test tests normal
    operation of the function.
    """
    filename = "tests/parameters/test_basis.csv"
    atomic_numbers, atomic_positions = get_basis_from_csv(filename)
    assert atomic_numbers == [11, 17]
    assert atomic_positions == [(0, 0, 0), (0.5, 0.5, 0.5)]


def test_validate_parameters_normal_operation():
    """
    A unit test for the validate_parameters function. This unit test tests normal
    operation of the function.
    """
    filename_1 = "tests/parameters/test_lattice.csv"
    filename_2 = "tests/parameters/test_basis.csv"

    lattice = get_lattice_from_csv(filename_1)
    basis = get_basis_from_csv(filename_2)

    assert validate_parameters(lattice, basis) is None
