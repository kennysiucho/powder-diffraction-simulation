"""
This module contains unit tests for the diffraction_monte_carlo.py module.
"""

import numpy.testing as nptest
import matplotlib.pyplot as plt
import scipy
from B8_project.diffraction_monte_carlo import WeightingFunction
from B8_project import utils
from tests.conftest import *

def test_get_gaussians_at_peaks():
    """
    Test if get_gaussians_at_peaks gives expected weighting function.
    """
    pdf = WeightingFunction.get_gaussians_at_peaks([22, 33], 0.1, 3)
    def expected_pdf(x):
        return (0.1 + np.exp(-0.5 * ((x - 22) / 3) ** 2) +
                np.exp(-0.5 * ((x - 33) / 3) ** 2))
    x_axis = np.linspace(0, 180, 200)
    nptest.assert_allclose(pdf(x_axis), expected_pdf(x_axis))

def test_unit_cell_positions(mc_ideal_nacl):
    """
    Tests output of _unit_cell_positions. Order does not matter.
    """
    a = mc_ideal_nacl._unit_cell.lattice_constants
    unit_cell_pos = mc_ideal_nacl._unit_cell_positions((1, 2, 3)) # pylint: disable=protected-access
    expected = np.array([
        [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2]
    ]) * a

    assert utils.have_same_elements(expected, unit_cell_pos) is True

def test_atoms_and_pos_in_uc(mc_ideal_nacl):
    """
    Tests output of _atoms_and_pos_in_uc. Order does not matter.
    """
    atoms_in_uc, atom_pos_in_uc = mc_ideal_nacl._atoms_and_pos_in_uc() # pylint: disable=protected-access
    # Convert to non-NumPy data types
    atoms_in_uc = [int(x) for x in atoms_in_uc]
    atom_pos_in_uc = atom_pos_in_uc.tolist()
    res = list(zip(atoms_in_uc, atom_pos_in_uc))

    expected_atoms = [11, 11, 11, 11, 17, 17, 17, 17]
    expected_pos = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
                    [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0.5]]
    expected_pos = (np.array(expected_pos) *
                    np.array(mc_ideal_nacl._unit_cell.lattice_constants
                             )).tolist()
    expected_res = list(zip(expected_atoms, expected_pos))

    assert utils.have_same_elements(res, expected_res, close=True) is True

def theoretical_inverse_cdf_natural_distribution(y: [float, np.ndarray],
                                                 min_x: float,
                                                 max_x: float):
    """
    The inverse CDF of the natural distribution for the scattering angle.
    i.e. PDF = sin(two_theta)
    """
    return np.degrees(
        np.arccos(np.cos(np.radians(min_x)) -
                  y * (np.cos(np.radians(min_x)) - np.cos(np.radians(max_x))))
    )

def test_get_scattering_vecs_and_angles_angle_range(mc_ideal_nacl):
    """
    Tests that the scattering angles are within desired angle range.
    """
    min_angle_deg = 20
    max_angle_deg = 60
    mc_ideal_nacl.set_angle_range(min_angle_deg=min_angle_deg,
                                                 max_angle_deg=max_angle_deg)

    _, angles = mc_ideal_nacl._get_scattering_vecs_and_angles(1000) # pylint: disable=protected-access

    assert np.all((angles >= min_angle_deg) & (angles <= max_angle_deg))

def test_scattering_vec_magnitude_distribution(mc_ideal_nacl):
    """
    Visual test to check distribution of magnitude of scattering vectors, normalized
    for initial/scatter k vectors of length 1.
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Skipped test: visual tests are off.")
    vecs, _ = mc_ideal_nacl._get_scattering_vecs_and_angles(500000) # pylint: disable=protected-access
    mags = np.linalg.norm(vecs, axis=1) / mc_ideal_nacl.k()
    plt.hist(mags, bins=100, density=True)
    x_axis = np.linspace(0, 2, 200)
    plt.plot(x_axis, x_axis / 2, "--",
             label="Theoretical")
    plt.xlabel("Magnitude of scattering vector / k")
    plt.ylabel("Normalized Frequency")
    plt.title("Distribution of magnitudes of scattering vectors. Should range from "
              "0->2 \nwith linearly increasing frequency")
    plt.legend()
    plt.show()

def test_scattering_angle_distribution(mc_ideal_nacl):
    """
    Visual test to check distribution of magnitude of scattering vectors
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Skipped test: visual tests are off.")
    _, angles = mc_ideal_nacl._get_scattering_vecs_and_angles(500000) # pylint: disable=protected-access

    plt.hist(angles, bins=100, density=True)
    x_axis = np.linspace(0, 180, 200)
    plt.plot(x_axis, WeightingFunction.natural_distribution(x_axis), "--",
             label="Theoretical")
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("Normalized frequency")
    plt.title("Distribution of scattering angles - should be sin(x) shaped")
    plt.legend()
    plt.show()

def test_scattering_angles_calculation(mc_ideal_nacl, mocker):
    """
    Test that the method calculates the scattering angle correctly.
    """
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
                     np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]])]
    )
    _, angles = mc_ideal_nacl._get_scattering_vecs_and_angles(3) # pylint: disable=protected-access
    expected_angles = np.array([0., 90., 180.])
    nptest.assert_allclose(angles, expected_angles)

def test_inverse_cdf_natural_distribution(mc_ideal_nacl):
    """
    Test whether the computed inverse CDF for the natural distribution of scattering
    angles matches the theoretical inverse CDF.
    """
    min_angle = 25
    max_angle = 135
    mc_ideal_nacl.set_angle_range(min_angle, max_angle)

    inputs = np.linspace(0, 1, 200)
    outputs = mc_ideal_nacl._inverse_cdf(inputs) # pylint: disable=protected-access
    expected_outputs = theoretical_inverse_cdf_natural_distribution(
        inputs, min_angle, max_angle)

    assert np.all((outputs >= min_angle) & (outputs <= max_angle))
    nptest.assert_allclose(outputs, expected_outputs)

def test_weighted_sampling_magnitudes_natural_distribution(mc_ideal_nacl):
    """
    Visual test for verifying if scattering magnitude follows expected linear
    distribution, for arbitrary angle range.
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Skipped test: visual tests are off.")
    min_angle = 20
    max_angle = 70
    mc_ideal_nacl.set_angle_range(min_angle, max_angle)
    vecs, _ = mc_ideal_nacl._get_scattering_vecs_and_angles_weighted( # pylint: disable=protected-access
        500000)
    mags = np.linalg.norm(vecs, axis=1) / mc_ideal_nacl.k()
    plt.hist(mags, bins=100, density=True)
    plt.xlabel("Magnitude of scattering vector / k")
    plt.xlim(0, 2)
    plt.ylabel("Normalized frequency")
    plt.title("Distribution of magnitudes of scattering vectors. Should be linearly"
              "increasing")
    plt.show()

def test_weighted_sampling_angles_natural_distribution(mc_ideal_nacl):
    """
    Visual test for verifying if scattering angle follows sin(two_theta) distribution,
    for arbitrary angle range.
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Skipped test: visual tests are off.")
    min_angle = 20
    max_angle = 120
    mc_ideal_nacl.set_angle_range(min_angle, max_angle)
    _, angles = mc_ideal_nacl._get_scattering_vecs_and_angles_weighted( # pylint: disable=protected-access
        500000)
    plt.hist(angles, bins=100, density=True)
    plt.xlabel("Scattering angle (deg)")
    plt.xlim(0, 180)
    plt.ylabel("Normalized frequency")
    plt.title("Distribution of scattering angles - should be sin(x) shaped")
    plt.show()

def test_weighted_sampling_angles_gaussians(mc_ideal_nacl):
    """
    Visual test for verifying if scattering angle follows the specified distribution
    (sum of Gaussians) arbitrary angle range.
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Skipped test: visual tests are off.")
    min_angle = 20
    max_angle = 70
    mc_ideal_nacl.set_angle_range(min_angle, max_angle)
    pdf = WeightingFunction.get_gaussians_at_peaks([22, 26, 36, 44, 46, 54], 0.1, 1)
    mc_ideal_nacl.set_pdf(pdf)
    _, angles = (mc_ideal_nacl._get_scattering_vecs_and_angles_weighted( # pylint: disable=protected-access
        500000))
    plt.hist(angles, bins=100, density=True)
    x_axis = np.linspace(min_angle, max_angle, 300)
    norm, _ = scipy.integrate.quad(pdf, min_angle, max_angle)
    plt.plot(x_axis, pdf(x_axis) / norm, "--", label="Target")
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("Normalized frequency")
    plt.title("Distribution of scattering angles - should follow target distribution")
    plt.legend()
    plt.show()

def test_neighborhood_spectrum_random_occupation_matches_arbitrary_crystal(
        mc_ideal_gaas, ingaas_nd_form_factors, mocker
):
    """
    Test diffraction spectrum for random occupation crystal matches that of arbitrary
    crystal.
    """
    pytest.skip("Skipped test: spectrum iterative has not been implemented")
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_unit_vectors_1, random_unit_vectors_2,
                     random_unit_vectors_1, random_unit_vectors_2],
    )
    mock_rng = mocker.Mock()
    mock_rng.choice.return_value = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    mocker.patch("numpy.random.default_rng", return_value=mock_rng)
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)

    two_thetas_rand, intensities_rand = (
        mc_ideal_gaas.calculate_neighborhood_diffraction_pattern_random_occupation(
            31, 49, 0.1,
            ingaas_nd_form_factors,
            angle_bins=9,
            brute_force_uc_reps=(2, 2, 2),
            neighbor_uc_reps=(2, 2, 2),
            brute_force_trials=10,
            num_top=5,
            resample_cnt=5,
            weighted=False,
            sigma=0.05,
            plot_diagnostics=False
        ))
    two_thetas_arb, intensities_arb = (
        mc_ideal_gaas.calculate_neighborhood_diffraction_pattern(
            two_rep_ingaas_one_indium_list,
            ingaas_nd_form_factors,
            angle_bins=9,
            brute_force_trials=10,
            num_top=5,
            resample_cnt=5,
            weighted=False,
            sigma=0.05,
            plot_diagnostics=False
        ))
    nptest.assert_allclose(two_thetas_rand, two_thetas_arb, rtol=1e-6, atol=1e-8)
    nptest.assert_allclose(intensities_rand, intensities_arb, rtol=1e-6, atol=1e-8)
