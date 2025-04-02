"""
This module contains unit tests for the diffraction_monte_carlo.py module.
"""

import pytest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
import scipy
from B8_project.file_reading import read_lattice, read_basis, \
    read_neutron_scattering_lengths
from B8_project.crystal import UnitCell, Atom, UnitCellVarieties, ReplacementProbability
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

normal_offsets = np.array([
    [0, 0, 0],
    [0.05, 0, 0],
    [0, -0.05, 0],
    [0, 0, 0.03],
    [0.1, 0.05, -0.1]
])
def mock_multivariate_normal(mean, cov, size):
    return mean + normal_offsets[0:size]

@pytest.fixture(name="diffraction_monte_carlo_nacl")
def fixture_diffraction_monte_carlo_nacl():
    """
    Returns instance of `DiffractionMonteCarlo`, containing data for NaCl and
    wavelength of 0.123Å
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
    wavelength of 0.123Å
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

def test_weighted_sampling_angles_gaussians(diffraction_monte_carlo_nacl):
    """
    Visual test for verifying if scattering angle follows the specified distribution
    (sum of Gaussians) arbitrary angle range.
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Skipped test: visual tests are off.")
    min_angle = 20
    max_angle = 70
    diffraction_monte_carlo_nacl.set_angle_range(min_angle, max_angle)
    pdf = WeightingFunction.get_gaussians_at_peaks([22, 26, 36, 44, 46, 54], 0.1, 1)
    diffraction_monte_carlo_nacl.set_pdf(pdf)
    _, angles = (diffraction_monte_carlo_nacl._get_scattering_vecs_and_angles_weighted( # pylint: disable=protected-access
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

@pytest.fixture(name="gaas_atom_list")
def fixture_gaas_atom_list(diffraction_monte_carlo_gaas):
    """
    Returns list of Atoms for a (2x1x1)-rep GaAs crystal
    """
    unit_cell_pos = np.array([[0, 0, 0], [GAAS_A, 0, 0]])
    atoms = []
    for uc_pos in unit_cell_pos:
        for atom in diffraction_monte_carlo_gaas.unit_cell.atoms:
            atoms.append(Atom(atom.atomic_number, uc_pos + np.array(atom.position) *
                              diffraction_monte_carlo_gaas.unit_cell.lattice_constants))
    return atoms

@pytest.fixture(name="gaas_atoms_pos_and_num")
def fixture_gaas_atoms_pos_and_num(gaas_atom_list):
    all_atom_pos = np.array([np.array(atom.position) for atom in gaas_atom_list])
    all_atoms = np.array([atom.atomic_number for atom in gaas_atom_list])
    return all_atom_pos, all_atoms

def test_compute_intensities_arbitrary_known_vecs(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors, gaas_atoms_pos_and_num
):
    """
    Check calculation of intensities given hand-packed scattering vectors for arbitrary
    crystal.
    """
    all_atom_pos, all_atoms = gaas_atoms_pos_and_num
    intensities = diffraction_monte_carlo_gaas.compute_intensities(
        scattering_vecs_gaas, all_atom_pos, all_atoms, ingaas_nd_form_factors
    )
    expected_structure_factors = np.array([5.664, 5.664,
                                           1.16874088 - 6.47419018j, 0.])
    expected_intensities = np.abs(expected_structure_factors)**2

    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6, atol=1e-8)

def test_diffraction_spectrum_known_vecs(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors, gaas_atom_list, mocker):
    """
    Check calculation of diffraction spectrum of crystal against manual calculation
    using hand-picked reciprocal lattice vectors.
    """
    mocker.patch(
        "B8_project.diffraction_monte_carlo.DiffractionMonteCarlo"
        "._get_scattering_vecs_and_angles",
        side_effect=[(scattering_vecs_gaas, scattering_angles_gaas)]
    )

    two_thetas, intensities, _, _ = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern(
            gaas_atom_list,
            ingaas_nd_form_factors,
            target_accepted_trials=4,
            trials_per_batch=4,
            angle_bins=9,
            weighted=False
        ))
    intensities /= np.max(intensities)

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

    unit_cell_pos = diffraction_monte_carlo_nacl._unit_cell_positions((8, 8, 8)) # pylint: disable=protected-access
    atoms = []
    for uc_pos in unit_cell_pos:
        for atom in diffraction_monte_carlo_nacl.unit_cell.atoms:
            atoms.append(Atom(atom.atomic_number, uc_pos + np.array(atom.position) *
                              diffraction_monte_carlo_nacl.unit_cell.lattice_constants))

    # Run one batch of 10 trials without any filtering based on angle or intensity
    two_thetas, intensities, _, _ \
        = diffraction_monte_carlo_nacl.calculate_diffraction_pattern(
        atoms,
        nacl_nd_form_factors,
        target_accepted_trials=10,
        trials_per_batch=10,
        angle_bins=9,
        weighted=False
    )

    intensities /= np.max(intensities)

    expected_two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])
    expected_normed_intensities = np.array([3.118890e-04, 0.000000e+00, 1.179717e-03,
                                     3.824213e-05, 7.736132e-06, 0.000000e+00,
                                     0.000000e+00, 1.000000e+00, 1.699957e-05])

    nptest.assert_allclose(two_thetas, expected_two_thetas, rtol=1e-6)
    nptest.assert_allclose(intensities, expected_normed_intensities, rtol=1e-6)

def test_neighborhood_intensity_arbitrary_crystal(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors, gaas_atom_list, mocker
):
    """
    Test calculation of neighborhood intensity given points to resample for arbitrary
    crystal.
    """
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)

    two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])
    intensities, counts = diffraction_monte_carlo_gaas.neighborhood_intensity(
        scattering_vecs_gaas,
        two_thetas,
        gaas_atom_list,
        ingaas_nd_form_factors,
        sigma=0.05,
        cnt_per_point=5
    )

    expected_intensities = np.array(
        [0.00000000e+00, 1.01558111e+02, 1.60892739e-03, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00])
    expected_counts = np.array([0, 15, 5, 0, 0, 0, 0, 0, 0])
    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6, atol=1e-8)
    nptest.assert_equal(counts, expected_counts)

def test_neighborhood_spectrum_arbitrary_crystal(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors, gaas_atom_list, mocker
):
    """
    Test calculation of diffraction spectrum using neighborhood sampling method for
    arbitrary crystal.
    """
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_unit_vectors_1, random_unit_vectors_2],
    )
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)

    two_thetas, intensities = diffraction_monte_carlo_gaas.calculate_neighborhood_diffraction_pattern(
        gaas_atom_list,
        ingaas_nd_form_factors,
        angle_bins=9,
        brute_force_trials=10,
        num_top=5,
        resample_cnt=5,
        weighted=False,
        sigma=0.05,
        plot_diagnostics=False
    )
    expected_two_thetas = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
    expected_intensities = np.array([
        0., 0., 0., 2561.08533298, 1745.81467154,
        0., 0., 3125.60321027, 0.
    ])
    nptest.assert_allclose(two_thetas, expected_two_thetas, rtol=1e-6, atol=1e-8)
    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6, atol=1e-8)

@pytest.fixture(name="ideal_crystal_gaas")
def ideal_crystal_gaas_fixture():
    """
    Returns unit_cell_pos, atoms_in_uc, and atom_pos_in_uc for a ideal GaAs crystal of
    (2x1x1) reps
    """
    unit_cell_pos = np.array([[0, 0, 0], [GAAS_A, 0, 0]])
    atoms_in_uc = np.array([31, 31, 31, 31, 33, 33, 33, 33])
    atom_pos_in_uc = GAAS_A * np.array([
        [0., 0., 0.],
        [0.5, 0.5, 0.],
        [0.5, 0., 0.5],
        [0., 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75]
    ])
    return unit_cell_pos, atoms_in_uc, atom_pos_in_uc

@pytest.fixture(name="random_scattering_vecs_gaas")
def fixture_random_scattering_vecs_gaas(diffraction_monte_carlo_gaas):
    """
    Returns scattering vectors computed from the pre-defined random unit vectors.
    """
    return ((random_unit_vectors_1 - random_unit_vectors_2)
            * diffraction_monte_carlo_gaas.k())

def test_compute_intensities_ideal_crystal(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors, ideal_crystal_gaas,
        random_scattering_vecs_gaas
):
    """
    Tests calculation of intensities given scattering vectors for ideal crystal.
    """
    unit_cell_pos, atoms_in_uc, atom_pos_in_uc = ideal_crystal_gaas

    intensities = diffraction_monte_carlo_gaas.compute_intensities_ideal_crystal(
        random_scattering_vecs_gaas, unit_cell_pos,
        atom_pos_in_uc, atoms_in_uc, ingaas_nd_form_factors
    )
    expected_intensities = np.array([
        2.47784222e+00, 2.21738735e+03, 6.60613922e+01, 1.76709983e+02,
        5.64541133e+02, 1.13523792e+03, 2.54167683e+01, 2.73226066e+01,
        5.06043212e+02, 2.90075981e+03
    ])
    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6, atol=1e-6)

def test_compute_intensity_ideal_crystal_matches_arbitrary_crystal(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors,
        gaas_atoms_pos_and_num, ideal_crystal_gaas
):
    """
    Tests compute intensity for ideal crystal matches that of arbitrary crystal.
    """
    vecs = ((utils.random_uniform_unit_vectors(100, 3) -
            utils.random_uniform_unit_vectors(100, 3))
            * diffraction_monte_carlo_gaas.k())
    unit_cell_pos, atoms_in_uc, atom_pos_in_uc = ideal_crystal_gaas
    all_atoms_pos, all_atoms = gaas_atoms_pos_and_num

    intensities_ideal = diffraction_monte_carlo_gaas.compute_intensities_ideal_crystal(
        vecs, unit_cell_pos, atom_pos_in_uc, atoms_in_uc, ingaas_nd_form_factors
    )
    intensities_arb = diffraction_monte_carlo_gaas.compute_intensities(
        vecs, all_atoms_pos, all_atoms, ingaas_nd_form_factors
    )
    nptest.assert_allclose(intensities_ideal, intensities_arb, rtol=1e-6, atol=1e-6)

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

    two_thetas, intensities, _, _ = (
        diffraction_monte_carlo_nacl.calculate_diffraction_pattern_ideal_crystal(
            nacl_nd_form_factors,
            target_accepted_trials=10,
            trials_per_batch=10,
            unit_cell_reps=(8, 8, 8),
            angle_bins=9,
            weighted=False
        ))

    intensities /= np.max(intensities)

    expected_two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])
    expected_normed_intensities = np.array([3.118890e-04, 0.000000e+00, 1.179717e-03,
                                     3.824213e-05, 7.736132e-06, 0.000000e+00,
                                     0.000000e+00, 1.000000e+00, 1.699957e-05])

    nptest.assert_allclose(two_thetas, expected_two_thetas, rtol=1e-6)
    nptest.assert_allclose(intensities, expected_normed_intensities, rtol=1e-6)

def test_neighborhood_intensity_ideal_crystal_matches_arbitrary_crystal(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors, gaas_atom_list, mocker
):
    """
    Test calculation of neighborhood intensity given points to resample for ideal
    crystal matches that of arbitrary crystal.
    """
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)

    vecs = ((utils.random_uniform_unit_vectors(100, 3) -
            utils.random_uniform_unit_vectors(100, 3))
            * diffraction_monte_carlo_gaas.k())

    two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])
    intensities_ideal, counts_ideal = (
        diffraction_monte_carlo_gaas.neighborhood_intensity_ideal_crystal(
        vecs, two_thetas, ingaas_nd_form_factors, (2, 1, 1),
        sigma=0.05, cnt_per_point=5
    ))
    intensities_arb, counts_arb = diffraction_monte_carlo_gaas.neighborhood_intensity(
        vecs, two_thetas, gaas_atom_list, ingaas_nd_form_factors,
        sigma=0.05, cnt_per_point=5
    )
    nptest.assert_allclose(intensities_ideal, intensities_arb, rtol=1e-6, atol=1e-8)
    nptest.assert_equal(counts_ideal, counts_arb)

def test_neighborhood_spectrum_ideal_crystal_matches_arbitrary_crystal(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors, gaas_atom_list, mocker
):
    """
    Test calculation of diffraction spectrum using neighborhood sampling method for
    ideal crystal matches that of arbitrary crystal.
    """
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_unit_vectors_1, random_unit_vectors_2,
                     random_unit_vectors_1, random_unit_vectors_2],
    )
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)

    two_thetas_ideal, intensities_ideal = (
        diffraction_monte_carlo_gaas.calculate_neighborhood_diffraction_pattern_ideal_crystal(
            ingaas_nd_form_factors,
            angle_bins=9,
            brute_force_uc_reps=(2, 1, 1),
            neighbor_uc_reps=(2, 1, 1),
            brute_force_trials=10,
            num_top=5,
            resample_cnt=5,
            weighted=False,
            sigma=0.05,
            plot_diagnostics=False
        ))
    two_thetas_arb, intensities_arb = (
        diffraction_monte_carlo_gaas.calculate_neighborhood_diffraction_pattern(
            gaas_atom_list,
            ingaas_nd_form_factors,
            angle_bins=9,
            brute_force_trials=10,
            num_top=5,
            resample_cnt=5,
            weighted=False,
            sigma=0.05,
            plot_diagnostics=False
        ))
    nptest.assert_allclose(two_thetas_ideal, two_thetas_arb, rtol=1e-6, atol=1e-8)
    nptest.assert_allclose(intensities_ideal, intensities_arb, rtol=1e-6, atol=1e-8)

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

    _, intensities_ideal, _, _ = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern_ideal_crystal(
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            unit_cell_reps=(8, 8, 8),
            angle_bins=10,
            weighted=False
        ))

    _, intensities_random, _, _ = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern_random_occupation(
            31,
            49,
            0.0,
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            unit_cell_reps=(8, 8, 8),
            angle_bins=10,
            weighted=False
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

    _, intensities_list, _, _ = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern(
            atoms,
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False
        )
    )

    _, intensities_ideal, _, _ = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern_ideal_crystal(
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            unit_cell_reps=(8, 8, 8),
            angle_bins=10,
            weighted=False
        ))

    nptest.assert_allclose(intensities_list, intensities_ideal)

@pytest.fixture(name="two_rep_uc_pos_gaas")
def fixture_two_rep_uc_pos_gaas(diffraction_monte_carlo_gaas):
    """
    Returns unit cell positions of a 2x2x2 GaAs crystal
    """
    return (np.vstack(np.mgrid[0:2, 0:2, 0:2]).reshape(3, -1).T
     * diffraction_monte_carlo_gaas.unit_cell.lattice_constants)

@pytest.fixture(name="two_rep_ingaas_one_indium_list")
def fixture_two_rep_ingaas_one_indium_list(
        diffraction_monte_carlo_gaas, two_rep_uc_pos_gaas):
    """
    Returns the list of atoms for a 2x2x2 GaAs crystal with one Ga atom substituted
    for an In atom.
    """
    atoms = []
    for i, uc_pos in enumerate(two_rep_uc_pos_gaas):
        for atom in diffraction_monte_carlo_gaas.unit_cell.atoms:
            if i == 1 and atom.position == (0, 0, 0):
                # Substitute Ga with In
                atoms.append(Atom(49, uc_pos + np.array(atom.position) *
                                  diffraction_monte_carlo_gaas.unit_cell.lattice_constants))
            else:
                atoms.append(Atom(atom.atomic_number, uc_pos + np.array(atom.position) *
                                  diffraction_monte_carlo_gaas.unit_cell.lattice_constants))
    return atoms

@pytest.fixture(name="ingaas_two_rep")
def fixture_ingaas_two_rep(diffraction_monte_carlo_gaas):
    """
    Returns atom_pos_in_uc, atomic_numbers_vars, and probs for InGaAs
    """
    _, atom_pos_in_uc = diffraction_monte_carlo_gaas._atoms_and_pos_in_uc()
    uc_vars = UnitCellVarieties(diffraction_monte_carlo_gaas.unit_cell,
                                ReplacementProbability(31, 49, 0.1))
    atomic_numbers_vars, probs = uc_vars.atomic_number_lists()
    return atom_pos_in_uc, atomic_numbers_vars, probs

def test_compute_intensities_random_occupation_matches_arbitrary_crystal(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors,
        ingaas_two_rep, two_rep_uc_pos_gaas,
        two_rep_ingaas_one_indium_list,
        random_scattering_vecs_gaas, mocker
):
    """
    Tests calculation of intensities given scattering vectors for ideal crystal.
    """
    mock_rng = mocker.Mock()
    mock_rng.choice.return_value = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    atom_pos_in_uc, atoms_in_uc_vars, probs = ingaas_two_rep
    intensities_rand = diffraction_monte_carlo_gaas.compute_intensities_random_occupation(
        random_scattering_vecs_gaas, two_rep_uc_pos_gaas, atom_pos_in_uc,
        atoms_in_uc_vars, probs, ingaas_nd_form_factors, mock_rng
    )

    all_atom_pos = [atom.position for atom in two_rep_ingaas_one_indium_list]
    all_atoms = [atom.atomic_number for atom in two_rep_ingaas_one_indium_list]
    intensities_arb = diffraction_monte_carlo_gaas.compute_intensities(
        random_scattering_vecs_gaas, all_atom_pos, all_atoms, ingaas_nd_form_factors
    )

    nptest.assert_allclose(intensities_rand, intensities_arb, rtol=1e-6, atol=1e-6)


def test_random_occupation_matches_list(
        diffraction_monte_carlo_gaas,
        two_rep_ingaas_one_indium_list,
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

    atoms = two_rep_ingaas_one_indium_list

    _, intensities_list, _, _ = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern(
            atoms,
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False
        )
    )

    _, intensities_random, _, _ = (
        diffraction_monte_carlo_gaas.calculate_diffraction_pattern_random_occupation(
            31,
            49,
            0.1,
            ingaas_nd_form_factors,
            target_accepted_trials=1000,
            trials_per_batch=1000,
            unit_cell_reps=(2, 2, 2),
            angle_bins=10,
            weighted=False
        )
    )

    nptest.assert_allclose(intensities_list, intensities_random)


def test_neighborhood_intensity_random_occupation_matches_arbitrary_crystal(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors,
        two_rep_ingaas_one_indium_list, mocker
):
    """
    Test calculation of neighborhood intensity given points to resample for random
    occupation crystal matches that of arbitrary crystal.
    """
    mock_rng = mocker.Mock()
    mock_rng.choice.return_value = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    mocker.patch("numpy.random.default_rng", return_value=mock_rng)
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)
    vecs = ((utils.random_uniform_unit_vectors(100, 3) -
             utils.random_uniform_unit_vectors(100, 3))
            * diffraction_monte_carlo_gaas.k())

    two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])
    intensities_rand, counts_rand = (
        diffraction_monte_carlo_gaas.neighborhood_intensity_random_occupation(
        31, 49, 0.1, vecs, two_thetas, ingaas_nd_form_factors, (2, 2, 2),
        sigma=0.05, cnt_per_point=5
    ))
    intensities_arb, counts_arb = diffraction_monte_carlo_gaas.neighborhood_intensity(
        vecs, two_thetas, two_rep_ingaas_one_indium_list, ingaas_nd_form_factors,
        sigma=0.05, cnt_per_point=5
    )
    nptest.assert_allclose(intensities_rand, intensities_arb, rtol=1e-6, atol=1e-8)
    nptest.assert_equal(counts_rand, counts_arb)

def test_neighborhood_spectrum_random_occupation_matches_arbitrary_crystal(
        diffraction_monte_carlo_gaas, ingaas_nd_form_factors,
        two_rep_ingaas_one_indium_list, mocker
):
    """
    Test diffraction spectrum for random occupation crystal matches that of arbitrary
    crystal.
    """
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
        diffraction_monte_carlo_gaas.calculate_neighborhood_diffraction_pattern_random_occupation(
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
        diffraction_monte_carlo_gaas.calculate_neighborhood_diffraction_pattern(
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
