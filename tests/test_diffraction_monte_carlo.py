"""
This module contains unit tests for the diffraction_monte_carlo.py module.
"""

import pytest
import numpy as np
import numpy.testing as nptest

from B8_project.file_reading import read_lattice, read_basis, \
    read_neutron_scattering_lengths
from B8_project.crystal import UnitCell
from B8_project.diffraction_monte_carlo import NeutronDiffractionMonteCarlo


@pytest.fixture(name="nd_monte_carlo")
def fixture_nd_monte_carlo():
    """
    Returns instance of `NeutronDiffractionMonteCarlo`, containing data for NaCl and
    wavelength of 0.123nm
    """
    nacl_lattice = read_lattice("tests/data/NaCl_lattice.csv")
    nacl_basis = read_basis("tests/data/NaCl_basis.csv")
    unit_cell = UnitCell.new_unit_cell(nacl_basis, nacl_lattice)
    nd = NeutronDiffractionMonteCarlo(unit_cell, 0.123)

    yield nd

@pytest.fixture(name="pro2_nd_form_factors")
def fixture_pro2_nd_form_factors():
    """
    Returns a dictionary of neutron form factors for PrO2
    """
    all_nd_form_factors = read_neutron_scattering_lengths(
        "data/neutron_scattering_lengths.csv")
    nd_form_factors = {
        11: all_nd_form_factors[11],
        17: all_nd_form_factors[17]
    }
    print(nd_form_factors)
    yield nd_form_factors

random_unit_vectors_1 = np.array(
    [
        [-0.40044041, -0.24037489, 0.88423266],
        [0.12157585, -0.98785375, -0.09676926],
        [0.22134359, -0.89937977, -0.3769921],
        [0.14452089, -0.57825636, -0.80295286],
        [0.27381633, 0.78604012, 0.55422518],
        [0.49786666, 0.86040003, -0.10881442],
        [-0.25020022, 0.0722504, -0.96549455],
        [0.22147444, -0.97182729, 0.08062746],
        [0.85740459, -0.34541421, 0.38150543],
        [-0.22353788, 0.00877532, 0.97465574],
    ]
)
random_unit_vectors_2 = np.array(
    [
        [0.55110704, -0.01638334, -0.83427371],
        [0.34531261, -0.28696691, -0.89353746],
        [-0.06524188, -0.89451913, -0.44224316],
        [0.97776524, 0.18597073, -0.09690211],
        [0.25050435, -0.52246139, 0.81503476],
        [0.69015485, -0.33681997, 0.64049871],
        [-0.35305807, -0.00383988, -0.93559353],
        [0.77693291, -0.58287346, -0.23797853],
        [0.46626091, 0.85543081, 0.2254748],
        [-0.06581907, 0.22353627, -0.97247076],
    ]
)

def test_monte_carlo_calculate_diffraction_pattern(nd_monte_carlo, mocker):
    """
    A unit test for the Monte Carlo calculate_diffraction_pattern function. This unit
    test tests normal operation of the function.
    """
    # Mocks the `random_uniform_unit_vectors` method to return the same random
    # vectors for consistent result.
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_unit_vectors_1, random_unit_vectors_2],
    )

    # Run one batch of 10 trials without any filtering based on angle or intensity
    two_thetas, intensities = nd_monte_carlo.calculate_diffraction_pattern(
        target_accepted_trials=10,
        trials_per_batch=10,
        unit_cells_in_crystal=(8, 8, 8),
        min_angle_deg=0,
        max_angle_deg=180,
        angle_bins=10
    )

    expected_two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.,
                                    180.])
    expected_intensities = np.array([0.000000e+00, 8.349908e-05, 0.000000e+00,
                                     2.231075e-04, 7.179199e-06, 6.286620e-06,
                                     0.000000e+00, 0.000000e+00, 1.000000e+00,
                                     2.489788e-05])

    nptest.assert_allclose(two_thetas, expected_two_thetas, rtol=1e-6)
    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6)

def test_monte_carlo_calculate_diffraction_pattern_ideal_crystal(
        nd_monte_carlo, pro2_nd_form_factors, mocker):
    """
    A unit test for the Monte Carlo calculate_diffraction_pattern_ideal_crystal
    function. This unit test tests normal operation of the function.
    """
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_unit_vectors_1, random_unit_vectors_2],
    )

    two_thetas, intensities = (
        nd_monte_carlo.calculate_diffraction_pattern_ideal_crystal(
            pro2_nd_form_factors,
            target_accepted_trials=10,
            trials_per_batch=10,
            unit_cells_in_crystal=(8, 8, 8),
            min_angle_deg=0,
            max_angle_deg=180,
            angle_bins=10
        ))

    expected_two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.,
                                    180.])
    expected_intensities = np.array([0.000000e+00, 8.349908e-05, 0.000000e+00,
                                     2.231075e-04, 7.179199e-06, 6.286620e-06,
                                     0.000000e+00, 0.000000e+00, 1.000000e+00,
                                     2.489788e-05])

    nptest.assert_allclose(two_thetas, expected_two_thetas, rtol=1e-6)
    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6)
