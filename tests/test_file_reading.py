"""
This module contains unit tests for the file_reading.py module.
TODO: add tests for error handling for get_lattice_from_csv.
TODO: add tests for error handling for get_basis_from_csv.
TODO: add tests for error handling for validate_parameters.
TODO: add tests for error handling for get_neutron_scattering_lengths_from_csv.
TODO: add tests for error handling for get_x_ray_form_factors_from_csv.
"""

from B8_project.file_reading import (
    read_lattice,
    read_basis,
    read_neutron_scattering_lengths,
    read_xray_form_factors,
)

from B8_project.crystal import NeutronFormFactor, XRayFormFactor


def test_read_lattice_normal_operation():
    """
    A unit test for the read_lattice function. This unit test tests normal operation of
    the function.
    """
    CsCl_lattice = "tests/data/CsCl_lattice.csv"
    material, lattice_type, lattice_constants = read_lattice(CsCl_lattice)
    assert material == "CsCl"
    assert lattice_type == 1
    assert lattice_constants == (0.4119, 0.4119, 0.4119)

    Cu_lattice = "tests/data/Cu_lattice.csv"
    material, lattice_type, lattice_constants = read_lattice(Cu_lattice)
    assert material == "Cu"
    assert lattice_type == 3
    assert lattice_constants == (0.3615, 0.3615, 0.3615)

    Na_lattice = "tests/data/Na_lattice.csv"
    material, lattice_type, lattice_constants = read_lattice(Na_lattice)
    assert material == "Na"
    assert lattice_type == 2
    assert lattice_constants == (0.4287, 0.4287, 0.4287)

    NaCl_lattice = "tests/data/NaCl_lattice.csv"
    material, lattice_type, lattice_constants = read_lattice(NaCl_lattice)
    assert material == "NaCl"
    assert lattice_type == 3
    assert lattice_constants == (0.5640, 0.5640, 0.5640)


def test_read_basis_normal_operation():
    """
    A unit test for the read_basis function. This unit test tests normal operation of
    the function.
    """
    CsCl_basis = "tests/data/CsCl_basis.csv"
    atomic_numbers, atomic_positions = read_basis(CsCl_basis)
    assert atomic_numbers == [55, 17]
    assert atomic_positions == [(0, 0, 0), (0.5, 0.5, 0.5)]

    Cu_basis = "tests/data/Cu_basis.csv"
    atomic_numbers, atomic_positions = read_basis(Cu_basis)
    assert atomic_numbers == [29]
    assert atomic_positions == [(0, 0, 0)]

    Na_basis = "tests/data/Na_basis.csv"
    atomic_numbers, atomic_positions = read_basis(Na_basis)
    assert atomic_numbers == [11]
    assert atomic_positions == [(0, 0, 0)]

    NaCl_basis = "tests/data/NaCl_basis.csv"
    atomic_numbers, atomic_positions = read_basis(NaCl_basis)
    assert atomic_numbers == [11, 17]
    assert atomic_positions == [(0, 0, 0), (0.5, 0.5, 0.5)]


def test_read_neutron_scattering_lengths_normal_operation():
    """
    A unit test for the read_neutron_scattering_lengths function. This unit test tests
    normal operation of the function.
    """
    filename = "tests/data/neutron_scattering_lengths.csv"

    neutron_scattering_lengths = read_neutron_scattering_lengths(filename)

    assert neutron_scattering_lengths[11] == NeutronFormFactor(3.63)
    assert neutron_scattering_lengths[17] == NeutronFormFactor(9.5792)
    assert neutron_scattering_lengths[29] == NeutronFormFactor(7.718)
    assert neutron_scattering_lengths[55] == NeutronFormFactor(5.42)


def test_read_x_ray_form_factors_normal_operation():
    """
    A unit test for the read_xray_form_factors function. This unit test tests normal
    operation of the function.
    """
    filename = "tests/data/xray_form_factors.csv"

    x_ray_form_factors = read_xray_form_factors(filename)

    assert x_ray_form_factors[11] == XRayFormFactor(
        4.7626, 3.285, 3.1736, 8.8422, 1.2674, 0.3136, 1.1128, 129.424, 0.676
    )
    assert x_ray_form_factors[17] == XRayFormFactor(
        11.4604, 0.0104, 7.1964, 1.1662, 6.2556, 18.5194, 1.6455, 47.7784, -9.5574
    )
