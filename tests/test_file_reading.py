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
    get_neutron_scattering_lengths_from_csv,
    get_x_ray_form_factors_from_csv,
)

from B8_project.crystal import XRayFormFactor


def test_get_lattice_from_csv_normal_operation():
    """
    A unit test for the get_lattice_from_csv function. This unit test tests normal
    operation of the function.
    """
    CsCl_lattice = "tests/parameters/CsCl_lattice.csv"
    material, lattice_type, lattice_constants = get_lattice_from_csv(CsCl_lattice)
    assert material == "CsCl"
    assert lattice_type == 1
    assert lattice_constants == (0.4119, 0.4119, 0.4119)

    Cu_lattice = "tests/parameters/Cu_lattice.csv"
    material, lattice_type, lattice_constants = get_lattice_from_csv(Cu_lattice)
    assert material == "Cu"
    assert lattice_type == 3
    assert lattice_constants == (0.3615, 0.3615, 0.3615)

    Na_lattice = "tests/parameters/Na_lattice.csv"
    material, lattice_type, lattice_constants = get_lattice_from_csv(Na_lattice)
    assert material == "Na"
    assert lattice_type == 2
    assert lattice_constants == (0.4287, 0.4287, 0.4287)

    NaCl_lattice = "tests/parameters/NaCl_lattice.csv"
    material, lattice_type, lattice_constants = get_lattice_from_csv(NaCl_lattice)
    assert material == "NaCl"
    assert lattice_type == 3
    assert lattice_constants == (0.5640, 0.5640, 0.5640)


def test_get_basis_from_csv_normal_operation():
    """
    A unit test for the get_basis_from_csv function. This unit test tests normal
    operation of the function.
    """
    CsCl_basis = "tests/parameters/CsCl_basis.csv"
    atomic_numbers, atomic_positions = get_basis_from_csv(CsCl_basis)
    assert atomic_numbers == [55, 17]
    assert atomic_positions == [(0, 0, 0), (0.5, 0.5, 0.5)]

    Cu_basis = "tests/parameters/Cu_basis.csv"
    atomic_numbers, atomic_positions = get_basis_from_csv(Cu_basis)
    assert atomic_numbers == [29]
    assert atomic_positions == [(0, 0, 0)]

    Na_basis = "tests/parameters/Na_basis.csv"
    atomic_numbers, atomic_positions = get_basis_from_csv(Na_basis)
    assert atomic_numbers == [11]
    assert atomic_positions == [(0, 0, 0)]

    NaCl_basis = "tests/parameters/NaCl_basis.csv"
    atomic_numbers, atomic_positions = get_basis_from_csv(NaCl_basis)
    assert atomic_numbers == [11, 17]
    assert atomic_positions == [(0, 0, 0), (0.5, 0.5, 0.5)]


def test_get_neutron_scattering_lengths_from_csv_normal_operation():
    """
    A unit test for the get_neutron_scattering_lengths_from_csv function. This unit
    test tests normal operation of the function.
    """
    filename = "tests/parameters/neutron_scattering_lengths.csv"

    neutron_scattering_lengths = get_neutron_scattering_lengths_from_csv(filename)

    assert neutron_scattering_lengths[11] == 3.63
    assert neutron_scattering_lengths[17] == 9.5792
    assert neutron_scattering_lengths[29] == 7.718
    assert neutron_scattering_lengths[55] == 5.42


def test_get_x_ray_form_factors_from_csv_normal_operation():
    """
    A unit test for the get_x_ray_form_factors_from_csv function. This unit test tests
    normal operation of the function.
    """
    filename = "tests/parameters/xray_form_factors.csv"

    x_ray_form_factors = get_x_ray_form_factors_from_csv(filename)

    assert x_ray_form_factors[11] == XRayFormFactor(
        4.7626, 3.285, 3.1736, 8.8422, 1.2674, 0.3136, 1.1128, 129.424, 0.676
    )
    assert x_ray_form_factors[17] == XRayFormFactor(
        11.4604, 0.0104, 7.1964, 1.1662, 6.2556, 18.5194, 1.6455, 47.7784, -9.5574
    )
