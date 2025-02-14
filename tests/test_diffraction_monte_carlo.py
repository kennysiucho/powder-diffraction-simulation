"""
This module contains unit tests for the diffraction_monte_carlo.py module.
"""

import pytest
import numpy as np
import numpy.testing as nptest

from B8_project.file_reading import read_lattice, read_basis, \
    read_neutron_scattering_lengths
from B8_project.crystal import UnitCell
from B8_project.diffraction_monte_carlo import DiffractionMonteCarlo
from B8_project import utils


@pytest.fixture(name="diffraction_monte_carlo")
def fixture_diffraction_monte_carlo():
    """
    Returns instance of `DiffractionMonteCarlo`, containing data for NaCl and
    wavelength of 0.123nm
    """
    nacl_lattice = read_lattice("tests/data/NaCl_lattice.csv")
    nacl_basis = read_basis("tests/data/NaCl_basis.csv")
    unit_cell = UnitCell.new_unit_cell(nacl_basis, nacl_lattice)
    nd = DiffractionMonteCarlo(unit_cell, 0.123)

    yield nd

@pytest.fixture(name="nacl_nd_form_factors")
def fixture_nacl_nd_form_factors():
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

def test_unit_cell_positions(diffraction_monte_carlo):
    """
    Tests output of _unit_cell_positions. Order does not matter.
    """
    a = diffraction_monte_carlo.unit_cell.lattice_constants
    unit_cell_pos = diffraction_monte_carlo._unit_cell_positions((1, 2, 3)) # pylint: disable=protected-access
    expected = np.array([
        [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2]
    ]) * a

    assert utils.have_same_elements(expected, unit_cell_pos) is True

def test_atoms_and_pos_in_uc(diffraction_monte_carlo):
    """
    Tests output of _atoms_and_pos_in_uc. Order does not matter.
    """
    atoms_in_uc, atom_pos_in_uc = diffraction_monte_carlo._atoms_and_pos_in_uc() # pylint: disable=protected-access
    # Convert to non-NumPy data types
    atoms_in_uc = [int(x) for x in atoms_in_uc]
    atom_pos_in_uc = atom_pos_in_uc.tolist()
    res = list(zip(atoms_in_uc, atom_pos_in_uc))

    expected_atoms = [11, 11, 11, 11, 17, 17, 17, 17]
    expected_pos = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
                    [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0.5]]
    expected_pos = (np.array(expected_pos) *
                    np.array(diffraction_monte_carlo.unit_cell.lattice_constants
                             )).tolist()
    expected_res = list(zip(expected_atoms, expected_pos))

    assert utils.have_same_elements(res, expected_res, close=True) is True

def test_monte_carlo_calculate_diffraction_pattern(
        diffraction_monte_carlo, nacl_nd_form_factors, mocker):
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
    two_thetas, intensities = diffraction_monte_carlo.calculate_diffraction_pattern(
        nacl_nd_form_factors,
        target_accepted_trials=10,
        trials_per_batch=10,
        unit_cell_reps=(8, 8, 8),
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
        diffraction_monte_carlo, nacl_nd_form_factors, mocker):
    """
    A unit test for the Monte Carlo calculate_diffraction_pattern_ideal_crystal
    function. This unit test tests normal operation of the function.
    """
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_unit_vectors_1, random_unit_vectors_2],
    )

    two_thetas, intensities = (
        diffraction_monte_carlo.calculate_diffraction_pattern_ideal_crystal(
            nacl_nd_form_factors,
            target_accepted_trials=10,
            trials_per_batch=10,
            unit_cell_reps=(8, 8, 8),
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
