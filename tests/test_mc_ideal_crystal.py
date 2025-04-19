"""
This module contains unit tests for the mc_ideal_crystal.py module.
"""

import numpy.testing as nptest
from B8_project import utils
from tests.conftest import *

def test_compute_intensities_ideal_crystal(
        mc_ideal_gaas, ingaas_nd_form_factors
):
    """
    Tests calculation of intensities given scattering vectors for ideal crystal.
    """
    vecs = random_scattering_vecs(mc_ideal_gaas.k())
    intensities = mc_ideal_gaas.compute_intensities(
        vecs, ingaas_nd_form_factors
    )
    expected_intensities = np.array([
        2.47784222e+00, 2.21738735e+03, 6.60613922e+01, 1.76709983e+02,
        5.64541133e+02, 1.13523792e+03, 2.54167683e+01, 2.73226066e+01,
        5.06043212e+02, 2.90075981e+03
    ])
    nptest.assert_allclose(intensities, expected_intensities, rtol=1e-6, atol=1e-6)

def test_compute_intensity_ideal_crystal_matches_arbitrary_crystal(
        mc_ideal_gaas, mc_arbitrary_gaas, ingaas_nd_form_factors
):
    """
    Tests compute intensity for ideal crystal matches that of arbitrary crystal.
    """
    vecs = ((utils.random_uniform_unit_vectors(100, 3) -
            utils.random_uniform_unit_vectors(100, 3))
            * mc_ideal_gaas.k())
    intensities_ideal = mc_ideal_gaas.compute_intensities(
        vecs, ingaas_nd_form_factors
    )
    intensities_arb = mc_arbitrary_gaas.compute_intensities(
        vecs, ingaas_nd_form_factors
    )
    nptest.assert_allclose(intensities_ideal, intensities_arb, rtol=1e-6, atol=1e-6)

def test_monte_carlo_calculate_diffraction_pattern_ideal_crystal(
        mc_ideal_nacl, nacl_nd_form_factors, mocker):
    """
    A unit test for the Monte Carlo calculate_diffraction_pattern_ideal_crystal
    function. This unit test tests normal operation of the function.
    """
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_unit_vectors_1, random_unit_vectors_2],
    )

    two_thetas, intensities, _, _ = (
        mc_ideal_nacl.spectrum_uniform(
            nacl_nd_form_factors,
            total_trials=10,
            trials_per_batch=10,
            angle_bins=9,
            weighted=False,
            threshold=0.
        ))
    intensities /= np.max(intensities)

    expected_two_thetas = np.array([10., 30., 50., 70., 90., 110., 130., 150., 170.])
    expected_normed_intensities = np.array([3.118890e-04, 0.000000e+00, 1.179717e-03,
                                     3.824213e-05, 7.736132e-06, 0.000000e+00,
                                     0.000000e+00, 1.000000e+00, 1.699957e-05])
    nptest.assert_allclose(two_thetas, expected_two_thetas, rtol=1e-6)
    nptest.assert_allclose(intensities, expected_normed_intensities, rtol=1e-6)

def test_ideal_crystal_matches_arbitrary(
        mc_ideal_gaas, ingaas_nd_form_factors, mocker):
    """
    Test if the spectrum calculated by the ideal case matches that of
    arbitrary crystal.
    """
    random_uvs1 = utils.random_uniform_unit_vectors(1000, 3)
    random_uvs2 = utils.random_uniform_unit_vectors(1000, 3)
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_uvs1, random_uvs2,
                     random_uvs1, random_uvs2],
    )

    atoms = []
    for uc_pos in mc_ideal_gaas._unit_cell_pos:
        for atom in mc_ideal_gaas._unit_cell.atoms:
            atoms.append(Atom(atom.atomic_number, uc_pos + np.array(atom.position) *
                              mc_ideal_gaas._unit_cell.lattice_constants))
    mc_arb = MCArbitraryCrystal(
        wavelength=mc_ideal_gaas.wavelength,
        atoms=atoms,
        pdf=mc_ideal_gaas._pdf,
        min_angle_deg=mc_ideal_gaas._min_angle_deg,
        max_angle_deg=mc_ideal_gaas._max_angle_deg
    )

    _, intensities_arb, _, _ = (
        mc_arb.spectrum_uniform(
            ingaas_nd_form_factors,
            total_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False,
            threshold=0.
        )
    )

    _, intensities_ideal, _, _ = (
        mc_ideal_gaas.spectrum_uniform(
            ingaas_nd_form_factors,
            total_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False,
            threshold=0.
        ))

    nptest.assert_allclose(intensities_arb, intensities_ideal)

def test_neighborhood_intensity_ideal_crystal_matches_arbitrary_crystal(
        mc_ideal_gaas, mc_arbitrary_gaas, ingaas_nd_form_factors, mocker
):
    """
    Test calculation of neighborhood intensity given points to resample for ideal
    crystal matches that of arbitrary crystal.
    """
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)
    vecs = ((utils.random_uniform_unit_vectors(100, 3) -
            utils.random_uniform_unit_vectors(100, 3))
            * mc_ideal_gaas.k())
    two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])

    intensities_ideal, _, counts_ideal = (
        mc_ideal_gaas.spectrum_neighborhood(
        vecs, two_thetas, ingaas_nd_form_factors,
        sigma=0.05, cnt_per_point=5
    ))
    intensities_arb, _, counts_arb = (
        mc_arbitrary_gaas.spectrum_neighborhood(
        vecs, two_thetas, ingaas_nd_form_factors,
        sigma=0.05, cnt_per_point=5
    ))
    nptest.assert_allclose(intensities_ideal, intensities_arb, rtol=1e-6, atol=1e-8)
    nptest.assert_equal(counts_ideal, counts_arb)

def test_neighborhood_spectrum_ideal_crystal_matches_arbitrary_crystal(
        mc_ideal_gaas, mc_arbitrary_gaas, ingaas_nd_form_factors, mocker
):
    """
    Test calculation of diffraction spectrum using neighborhood sampling method for
    ideal crystal matches that of arbitrary crystal.
    """
    pytest.skip("Skipped test: spectrum iterative has not been implemented")
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_unit_vectors_1, random_unit_vectors_2,
                     random_unit_vectors_1, random_unit_vectors_2],
    )
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)

    two_thetas_ideal, intensities_ideal = (
        mc_ideal_gaas.spectrum_iterative_refinement(
            ingaas_nd_form_factors,
            angle_bins=9,
            brute_force_uc_reps=(2, 1, 1),
            neighbor_uc_reps=(2, 1, 1),
            brute_force_trials=10,
            threshold=0.,
            resample_cnt=5,
            weighted=False,
            sigma=0.05,
            plot_diagnostics=False
        ))
    two_thetas_arb, intensities_arb = (
        mc_arbitrary_gaas.spectrum_iterative_refinement(
            ingaas_nd_form_factors,
            angle_bins=9,
            brute_force_trials=10,
            threshold=0.,
            resample_cnt=5,
            weighted=False,
            sigma=0.05,
            plot_diagnostics=False
        ))
    nptest.assert_allclose(two_thetas_ideal, two_thetas_arb, rtol=1e-6, atol=1e-8)
    nptest.assert_allclose(intensities_ideal, intensities_arb, rtol=1e-6, atol=1e-8)
