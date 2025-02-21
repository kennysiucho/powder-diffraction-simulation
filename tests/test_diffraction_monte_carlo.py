"""
This module contains unit tests for the diffraction_monte_carlo.py module.
"""

import pytest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from B8_project.file_reading import read_lattice, read_basis, \
    read_neutron_scattering_lengths
from B8_project.crystal import UnitCell, Atom
from B8_project.diffraction_monte_carlo import DiffractionMonteCarlo, WeightingFunction
from B8_project import utils

RUN_VISUAL_TESTS = False

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

GAAS_A = 0.565315
scattering_vecs_gaas = np.array([
    [4 * np.pi / GAAS_A, 0., 0.], # (200) reciprocal lattice vector for NaCl
    [0., 4 * np.pi / GAAS_A, 0.], # (020)
    # (002), pointed in the wrong direction
    [4 * np.pi / GAAS_A / np.sqrt(2), 4 * np.pi / GAAS_A / np.sqrt(2), 0.],
    [6 * np.pi / GAAS_A, -4 * np.pi / GAAS_A, 2 * np.pi / GAAS_A] # (3,-2,1)
])
scattering_angles_gaas = np.array([
    25.1336118,
    25.1336118,
    25.1336118,
    48.039411
])

@pytest.fixture(name="diffraction_monte_carlo_nacl")
def fixture_diffraction_monte_carlo_nacl():
    """
    Returns instance of `DiffractionMonteCarlo`, containing data for NaCl and
    wavelength of 0.123nm
    """
    nacl_lattice = read_lattice("tests/data/NaCl_lattice.csv")
    nacl_basis = read_basis("tests/data/NaCl_basis.csv")
    unit_cell = UnitCell.new_unit_cell(nacl_basis, nacl_lattice)
    nd = DiffractionMonteCarlo(unit_cell, 0.123,
                               min_angle_deg=0.0, max_angle_deg=180.0)
    yield nd

@pytest.fixture(name="diffraction_monte_carlo_gaas")
def fixture_diffraction_monte_carlo_gaas():
    """
    Returns instance of `DiffractionMonteCarlo`, containing data for GaAs and
    wavelength of 0.123nm
    """
    lattice = read_lattice("tests/data/GaAs_lattice.csv")
    basis = read_basis("tests/data/GaAs_basis.csv")
    unit_cell = UnitCell.new_unit_cell(basis, lattice)
    nd = DiffractionMonteCarlo(unit_cell, 0.123,
                               min_angle_deg=0.0, max_angle_deg=180.0)
    yield nd

@pytest.fixture(name="nacl_nd_form_factors")
def fixture_nacl_nd_form_factors():
    """
    Returns a dictionary of neutron form factors for Nacl
    """
    all_nd_form_factors = read_neutron_scattering_lengths(
        "data/neutron_scattering_lengths.csv")
    nd_form_factors = {
        11: all_nd_form_factors[11],
        17: all_nd_form_factors[17]
    }
    yield nd_form_factors


@pytest.fixture(name="ingaas_nd_form_factors")
def fixture_ingaas_nd_form_factors():
    """
    Returns a dictionary of neutron form factors for InGaAs
    """
    all_nd_form_factors = read_neutron_scattering_lengths(
        "data/neutron_scattering_lengths.csv")
    nd_form_factors = {
        31: all_nd_form_factors[31],
        33: all_nd_form_factors[33],
        49: all_nd_form_factors[49]
    }
    yield nd_form_factors

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

def test_unit_cell_positions(diffraction_monte_carlo_nacl):
    """
    Tests output of _unit_cell_positions. Order does not matter.
    """
    a = diffraction_monte_carlo_nacl.unit_cell.lattice_constants
    unit_cell_pos = diffraction_monte_carlo_nacl._unit_cell_positions((1, 2, 3)) # pylint: disable=protected-access
    expected = np.array([
        [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2]
    ]) * a

    assert utils.have_same_elements(expected, unit_cell_pos) is True

def test_atoms_and_pos_in_uc(diffraction_monte_carlo_nacl):
    """
    Tests output of _atoms_and_pos_in_uc. Order does not matter.
    """
    atoms_in_uc, atom_pos_in_uc = diffraction_monte_carlo_nacl._atoms_and_pos_in_uc() # pylint: disable=protected-access
    # Convert to non-NumPy data types
    atoms_in_uc = [int(x) for x in atoms_in_uc]
    atom_pos_in_uc = atom_pos_in_uc.tolist()
    res = list(zip(atoms_in_uc, atom_pos_in_uc))

    expected_atoms = [11, 11, 11, 11, 17, 17, 17, 17]
    expected_pos = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
                    [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0.5]]
    expected_pos = (np.array(expected_pos) *
                    np.array(diffraction_monte_carlo_nacl.unit_cell.lattice_constants
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

def test_get_scattering_vecs_and_angles_angle_range(diffraction_monte_carlo_nacl):
    """
    Tests that the scattering angles are within desired angle range.
    """
    min_angle_deg = 20
    max_angle_deg = 60
    diffraction_monte_carlo_nacl.set_angle_range(min_angle_deg=min_angle_deg,
                                                 max_angle_deg=max_angle_deg)

    _, angles = diffraction_monte_carlo_nacl._get_scattering_vecs_and_angles(1000) # pylint: disable=protected-access

    assert np.all((angles >= min_angle_deg) & (angles <= max_angle_deg))

def test_scattering_vec_magnitude_distribution(diffraction_monte_carlo_nacl):
    """
    Visual test to check distribution of magnitude of scattering vectors, normalized
    for initial/scatter k vectors of length 1.
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Skipped test: visual tests are off.")
    vecs, _ = diffraction_monte_carlo_nacl._get_scattering_vecs_and_angles(500000) # pylint: disable=protected-access
    mags = np.linalg.norm(vecs, axis=1) / diffraction_monte_carlo_nacl.k()
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

def test_scattering_angle_distribution(diffraction_monte_carlo_nacl):
    """
    Visual test to check distribution of magnitude of scattering vectors
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Skipped test: visual tests are off.")
    _, angles = diffraction_monte_carlo_nacl._get_scattering_vecs_and_angles(500000) # pylint: disable=protected-access

    plt.hist(angles, bins=100, density=True)
    x_axis = np.linspace(0, 180, 200)
    plt.plot(x_axis, WeightingFunction.natural_distribution(x_axis), "--",
             label="Theoretical")
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("Normalized frequency")
    plt.title("Distribution of scattering angles - should be sin(x) shaped")
    plt.legend()
    plt.show()

def test_scattering_angles_calculation(diffraction_monte_carlo_nacl, mocker):
    """
    Test that the method calculates the scattering angle correctly.
    """
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
                     np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]])]
    )
    _, angles = diffraction_monte_carlo_nacl._get_scattering_vecs_and_angles(3) # pylint: disable=protected-access
    expected_angles = np.array([0., 90., 180.])
    nptest.assert_allclose(angles, expected_angles)

def test_inverse_cdf_natural_distribution(diffraction_monte_carlo_nacl):
    """
    Test whether the computed inverse CDF for the natural distribution of scattering
    angles matches the theoretical inverse CDF.
    """
    min_angle = 25
    max_angle = 135
    diffraction_monte_carlo_nacl.set_angle_range(min_angle, max_angle)

    inputs = np.linspace(0, 1, 200)
    outputs = diffraction_monte_carlo_nacl._inverse_cdf(inputs) # pylint: disable=protected-access
    expected_outputs = theoretical_inverse_cdf_natural_distribution(
        inputs, min_angle, max_angle)

    assert np.all((outputs >= min_angle) & (outputs <= max_angle))
    nptest.assert_allclose(outputs, expected_outputs)

def test_weighted_sampling_magnitudes_natural_distribution(diffraction_monte_carlo_nacl):
    """
    Visual test for verifying if scattering magnitude follows expected linear
    distribution, for arbitrary angle range.
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Skipped test: visual tests are off.")
    min_angle = 20
    max_angle = 70
    diffraction_monte_carlo_nacl.set_angle_range(min_angle, max_angle)
    vecs, _ = diffraction_monte_carlo_nacl._get_scattering_vecs_and_angles_weighted( # pylint: disable=protected-access
        500000)
    mags = np.linalg.norm(vecs, axis=1) / diffraction_monte_carlo_nacl.k()
    plt.hist(mags, bins=100, density=True)
    plt.xlabel("Magnitude of scattering vector / k")
    plt.xlim(0, 2)
    plt.ylabel("Normalized frequency")
    plt.title("Distribution of magnitudes of scattering vectors. Should be linearly"
              "increasing")
    plt.show()

def test_weighted_sampling_angles_natural_distribution(diffraction_monte_carlo_nacl):
    """
    Visual test for verifying if scattering angle follows sin(two_theta) distribution,
    for arbitrary angle range.
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Skipped test: visual tests are off.")
    min_angle = 20
    max_angle = 120
    diffraction_monte_carlo_nacl.set_angle_range(min_angle, max_angle)
    _, angles = diffraction_monte_carlo_nacl._get_scattering_vecs_and_angles_weighted( # pylint: disable=protected-access
        500000)
    plt.hist(angles, bins=100, density=True)
    plt.xlabel("Scattering angle (deg)")
    plt.xlim(0, 180)
    plt.ylabel("Normalized frequency")
    plt.title("Distribution of scattering angles - should be sin(x) shaped")
    plt.show()

def test_diffraction_spectrum_known_vecs(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors, mocker):
    """
    Check calculation of diffraction spectrum of crystal against manual calculation
    using hand-picked reciprocal lattice vectors.
    """
    mocker.patch(
        "B8_project.diffraction_monte_carlo.DiffractionMonteCarlo"
        "._get_scattering_vecs_and_angles",
        side_effect=[(scattering_vecs_gaas, scattering_angles_gaas)]
    )

    # (2x1x1) unit cell reps
    unit_cell_pos = np.array([[0, 0, 0], [GAAS_A, 0, 0]])
    atoms = []
    for uc_pos in unit_cell_pos:
        for atom in diffraction_monte_carlo_gaas.unit_cell.atoms:
            atoms.append(Atom(atom.atomic_number, uc_pos + np.array(atom.position) *
                              diffraction_monte_carlo_gaas.unit_cell.lattice_constants))

    two_thetas, intensities = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern(
            atoms,
            ingaas_nd_form_factors,
            target_accepted_trials=4,
            trials_per_batch=4,
            angle_bins=9
        ))

    expected_two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])
    expected_structure_factors = np.array([5.664, 5.664,
                                           1.16874088 - 6.47419018j, 0.])
    expected_intensities = np.zeros(expected_two_thetas.shape)
    expected_intensities[1] = np.sum(np.abs(expected_structure_factors)[0:2]**2)
    expected_intensities /= np.max(expected_intensities)
    # expected_intensities is just [0, 1, 0, 0...]

    nptest.assert_allclose(two_thetas, expected_two_thetas, rtol=1e-6, atol=1e-8)
    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6, atol=1e-8)

def test_monte_carlo_calculate_diffraction_pattern(
        diffraction_monte_carlo_nacl, nacl_nd_form_factors, mocker):
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

    unit_cell_pos = diffraction_monte_carlo_nacl._unit_cell_positions((8, 8, 8))
    atoms = []
    for uc_pos in unit_cell_pos:
        for atom in diffraction_monte_carlo_nacl.unit_cell.atoms:
            atoms.append(Atom(atom.atomic_number, uc_pos + np.array(atom.position) *
                              diffraction_monte_carlo_nacl.unit_cell.lattice_constants))

    # Run one batch of 10 trials without any filtering based on angle or intensity
    two_thetas, intensities = diffraction_monte_carlo_nacl.calculate_diffraction_pattern(
        atoms,
        nacl_nd_form_factors,
        target_accepted_trials=10,
        trials_per_batch=10,
        angle_bins=9
    )

    expected_two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])
    expected_intensities = np.array([3.118890e-04, 0.000000e+00, 1.179717e-03,
                                     3.824213e-05, 7.736132e-06, 0.000000e+00,
                                     0.000000e+00, 1.000000e+00, 1.699957e-05])

    nptest.assert_allclose(two_thetas, expected_two_thetas, rtol=1e-6)
    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6)

def test_monte_carlo_calculate_diffraction_pattern_ideal_crystal(
        diffraction_monte_carlo_nacl, nacl_nd_form_factors, mocker):
    """
    A unit test for the Monte Carlo calculate_diffraction_pattern_ideal_crystal
    function. This unit test tests normal operation of the function.
    """
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_unit_vectors_1, random_unit_vectors_2],
    )

    two_thetas, intensities = (
        diffraction_monte_carlo_nacl.calculate_diffraction_pattern_ideal_crystal(
            nacl_nd_form_factors,
            target_accepted_trials=10,
            trials_per_batch=10,
            unit_cell_reps=(8, 8, 8),
            angle_bins=9
        ))

    expected_two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])
    expected_intensities = np.array([3.118890e-04, 0.000000e+00, 1.179717e-03,
                                     3.824213e-05, 7.736132e-06, 0.000000e+00,
                                     0.000000e+00, 1.000000e+00, 1.699957e-05])

    nptest.assert_allclose(two_thetas, expected_two_thetas, rtol=1e-6)
    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6)

def test_ideal_crystal_matches_random_occupation_with_zero_concentration(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors, mocker):
    """
    Test if the spectrum for an ideal crystal equals the spectrum for a random
    occupation crystal with the substitute concentration of zero.
    """
    random_uvs1 = utils.random_uniform_unit_vectors(1000, 3)
    random_uvs2 = utils.random_uniform_unit_vectors(1000, 3)
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_uvs1, random_uvs2,
                     random_uvs1, random_uvs2],
    )

    _, intensities_ideal = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern_ideal_crystal(
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            unit_cell_reps=(8, 8, 8),
            angle_bins=10
        ))

    _, intensities_random = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern_random_occupation(
            31,
            49,
            0.0,
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            unit_cell_reps=(8, 8, 8),
            angle_bins=10
        ))

    nptest.assert_allclose(intensities_ideal, intensities_random)

def test_ideal_crystal_matches_list(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors, mocker):
    """
    Test if the spectrum calculated by the ideal case matches that of
    calculate_diffraction_pattern.
    """
    random_uvs1 = utils.random_uniform_unit_vectors(1000, 3)
    random_uvs2 = utils.random_uniform_unit_vectors(1000, 3)
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_uvs1, random_uvs2,
                     random_uvs1, random_uvs2],
    )

    unit_cell_pos = (np.vstack(np.mgrid[0:8, 0:8, 0:8]).reshape(3, -1).T
                     * diffraction_monte_carlo_gaas.unit_cell.lattice_constants)
    atoms = []
    for uc_pos in unit_cell_pos:
        for atom in diffraction_monte_carlo_gaas.unit_cell.atoms:
            atoms.append(Atom(atom.atomic_number, uc_pos + np.array(atom.position) *
                              diffraction_monte_carlo_gaas.unit_cell.lattice_constants))

    _, intensities_list = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern(
            atoms,
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            angle_bins=10
        )
    )

    _, intensities_ideal = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern_ideal_crystal(
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            unit_cell_reps=(8, 8, 8),
            angle_bins=10
        ))

    nptest.assert_allclose(intensities_list, intensities_ideal)

def test_random_occupation_matches_list(diffraction_monte_carlo_gaas,
                                        ingaas_nd_form_factors, mocker):
    """
    Test if the spectrum calculated by random occupation matches that of
    calculate_diffraction_pattern.
    """
    random_uvs1 = utils.random_uniform_unit_vectors(1000, 3)
    random_uvs2 = utils.random_uniform_unit_vectors(1000, 3)
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_uvs1, random_uvs2,
                     random_uvs1, random_uvs2],
    )
    mock_rng = mocker.Mock()
    mock_rng.choice.return_value = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    mocker.patch("numpy.random.default_rng", return_value=mock_rng)

    unit_cell_pos = (np.vstack(np.mgrid[0:2, 0:2, 0:2]).reshape(3, -1).T
                     * diffraction_monte_carlo_gaas.unit_cell.lattice_constants)
    atoms = []
    for i, uc_pos in enumerate(unit_cell_pos):
        for atom in diffraction_monte_carlo_gaas.unit_cell.atoms:
            if i == 1 and atom.position == (0, 0, 0):
                # Substitute Ga with In
                atoms.append(Atom(49, uc_pos + np.array(atom.position) *
                                  diffraction_monte_carlo_gaas.unit_cell.lattice_constants))
            else:
                atoms.append(Atom(atom.atomic_number, uc_pos + np.array(atom.position) *
                                  diffraction_monte_carlo_gaas.unit_cell.lattice_constants))

    _, intensities_list = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern(
            atoms,
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            angle_bins=10
        )
    )

    _, intensities_random = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern_random_occupation(
            31,
            49,
            0.1,
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            unit_cell_reps=(2, 2, 2),
            angle_bins=10
        )
    )

    nptest.assert_allclose(intensities_list, intensities_random)
