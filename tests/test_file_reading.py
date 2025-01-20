"""
This module contains unit tests for the file_reading.py module.
TODO: add tests for error handling for get_lattice_from_csv.
TODO: add tests for error handling for get_basis_from_csv.
TODO: add tests for error handling for validate_parameters.
TODO: add tests for error handling for get_neutron_scattering_lengths_from_csv.
TODO: add tests for error handling for get_x_ray_form_factors_from_csv.
"""

from B8_project.file_reading import (
    get_lattice_from_csv,
    get_basis_from_csv,
    validate_parameters,
    get_neutron_scattering_lengths_from_csv,
    get_x_ray_form_factors_from_csv,
)

from B8_project.crystal_lattice import XRayFormFactor


def test_get_lattice_from_csv_normal_operation():
    """
    A unit test for the get_lattice_from_csv function. This unit test tests normal
    operation of the function.
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


def test_get_neutron_scattering_lengths_from_csv():
    """
    A unit test for the get_neutron_scattering_lengths_from_csv function. This unit
    test tests normal operation of the function.
    """
    filename = "tests/parameters/test_neutron_scattering_lengths.csv"

    neutron_scattering_lengths = get_neutron_scattering_lengths_from_csv(filename)

    assert neutron_scattering_lengths[11] == 3.63
    assert neutron_scattering_lengths[17] == 9.5792


def test_get_x_ray_form_factors_from_csv():
    """
    A unit test for the get_x_ray_form_factors_from_csv function. This unit test tests
    normal operation of the function.
    """
    filename = "tests/parameters/test_xray_form_factors.csv"

    x_ray_form_factors = get_x_ray_form_factors_from_csv(filename)

    assert x_ray_form_factors[11] == XRayFormFactor(
        4.7626, 3.285, 3.1736, 8.8422, 1.2674, 0.3136, 1.1128, 129.424, 0.676
    )
    assert x_ray_form_factors[17] == XRayFormFactor(
        11.4604, 0.0104, 7.1964, 1.1662, 6.2556, 18.5194, 1.6455, 47.7784, -9.5574
    )
