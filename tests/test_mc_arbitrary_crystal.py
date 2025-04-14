"""
This module contains unit tests for the mc_arbitrary_crystal.py module.
"""

import numpy.testing as nptest
from tests.conftest import *

def test_compute_intensities_arbitrary_known_vecs(
        mc_arbitrary_gaas, ingaas_nd_form_factors
):
    """
    Check calculation of intensities given hand-packed scattering vectors for arbitrary
    crystal.
    """
    intensities = mc_arbitrary_gaas.compute_intensities(
        scattering_vecs_gaas, ingaas_nd_form_factors
    )
    expected_structure_factors = np.array([5.664, 5.664,
                                           1.16874088 - 6.47419018j, 0.])
    expected_intensities = np.abs(expected_structure_factors)**2

    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6, atol=1e-8)

def test_diffraction_spectrum_known_vecs(
        mc_arbitrary_gaas, ingaas_nd_form_factors, mocker):
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
        mc_arbitrary_gaas.spectrum_uniform(
            ingaas_nd_form_factors,
            total_trials=4,
            trials_per_batch=4,
            angle_bins=9,
            weighted=False,
            threshold=0.
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
        mc_ideal_nacl, nacl_nd_form_factors, mocker):
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

    unit_cell_pos = mc_ideal_nacl._unit_cell_positions((8, 8, 8)) # pylint: disable=protected-access
    atoms = []
    for uc_pos in unit_cell_pos:
        for atom in mc_ideal_nacl._unit_cell.atoms:
            atoms.append(Atom(atom.atomic_number, uc_pos + np.array(atom.position) *
                              mc_ideal_nacl._unit_cell.lattice_constants))
    mc_arbitrary_nacl = MCArbitraryCrystal(
        wavelength=0.123,
        atoms=atoms,
        pdf=None,
        min_angle_deg=0.,
        max_angle_deg=180.,
    )

    # Run one batch of 10 trials without any filtering based on angle or intensity
    two_thetas, intensities, _, _ \
        = mc_arbitrary_nacl.spectrum_uniform(
            nacl_nd_form_factors,
            total_trials=10,
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
        mc_arbitrary_gaas, ingaas_nd_form_factors, gaas_atom_list, mocker
):
    """
    Test calculation of neighborhood intensity given points to resample for arbitrary
    crystal.
    """
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)

    two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])
    intensities, _, counts = mc_arbitrary_gaas.spectrum_neighborhood(
        scattering_vecs_gaas,
        two_thetas,
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
        mc_arbitrary_gaas, ingaas_nd_form_factors, mocker
):
    """
    Test calculation of diffraction spectrum using neighborhood sampling method for
    arbitrary crystal.
    """
    pytest.skip("Skipped test: spectrum iterative has not been implemented")
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_unit_vectors_1, random_unit_vectors_2],
    )
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)

    two_thetas, intensities = mc_arbitrary_gaas.spectrum_iterative_refinement(
        ingaas_nd_form_factors,
        angle_bins=9,
        brute_force_trials=10,
        threshold=0.,
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
