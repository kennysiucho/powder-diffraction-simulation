"""
This module contains unit tests for the mc_random_occupation.py module.
"""

import numpy.testing as nptest
from B8_project import utils
from tests.conftest import *

@pytest.fixture(name="mc_arbitrary_ingaas")
def fixture_mc_arbitrary_ingaas(mc_random_gaas):
    """
    Returns MCArbitraryCrystal for a 2x2x2 GaAs crystal with one Ga atom substituted
    for an In atom.
    """
    atoms = []
    uc_choices = [0, 1, 0, 0, 0, 0, 0, 0]
    for i, uc_pos in enumerate(mc_random_gaas._unit_cell_pos):
        for j, atom in enumerate(mc_random_gaas._atomic_numbers_vars[uc_choices[i]]):
            atoms.append(Atom(atom, uc_pos + mc_random_gaas._atom_pos_in_uc[j]))
    nd = MCArbitraryCrystal(
        wavelength=0.123,
        atoms=atoms,
        pdf=None,
        min_angle_deg=0.,
        max_angle_deg=180.,
    )
    yield nd

def test_ideal_crystal_matches_random_occupation_with_zero_concentration(
        mc_ideal_gaas, mc_random_gaas, ingaas_nd_form_factors, mocker):
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

    mc_ideal_gaas.setup_cuboid_crystal((8, 8, 8))
    _, intensities_ideal, _, _ = (
        mc_ideal_gaas.spectrum_uniform(
            ingaas_nd_form_factors,
            total_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False,
            threshold=0.
        ))

    mc_random_gaas.set_random_occupation_parameters(31, 41, 0.0)
    mc_random_gaas.setup_cuboid_crystal((8, 8, 8))
    _, intensities_random, _, _ = (
        mc_ideal_gaas.spectrum_uniform(
            ingaas_nd_form_factors,
            total_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False,
            threshold=0.
        ))

    nptest.assert_allclose(intensities_ideal, intensities_random)

def test_compute_intensities_random_occupation_matches_arbitrary_crystal(
        mc_random_gaas, mc_arbitrary_ingaas, ingaas_nd_form_factors, mocker
):
    """
    Tests calculation of intensities given scattering vectors for ideal crystal.
    """
    vecs = random_scattering_vecs(mc_random_gaas.k())
    mock_rng = mocker.Mock()
    mock_rng.choice.return_value = np.array(
        [[0, 1, 0, 0, 0, 0, 0, 0] for _ in range(len(vecs))])
    mc_random_gaas._rng = mock_rng
    intensities_rand = mc_random_gaas.compute_intensities(
        vecs, ingaas_nd_form_factors
    )
    intensities_arb = mc_arbitrary_ingaas.compute_intensities(
        vecs, ingaas_nd_form_factors
    )

    nptest.assert_allclose(intensities_rand, intensities_arb, rtol=1e-6, atol=1e-6)

def test_random_occupation_matches_arbitrary(
        mc_random_gaas, mc_arbitrary_ingaas,
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
    mock_rng.choice.return_value = np.array(
        [[0, 1, 0, 0, 0, 0, 0, 0] for _ in range(len(random_uvs1))])
    mc_random_gaas._rng = mock_rng

    _, intensities_arb, _, _ = (
        mc_arbitrary_ingaas.spectrum_uniform(
            ingaas_nd_form_factors,
            total_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False,
            threshold=0.
        )
    )

    _, intensities_random, _, _ = (
        mc_random_gaas.spectrum_uniform(
            ingaas_nd_form_factors,
            total_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False,
            threshold=0.
        )
    )

    nptest.assert_allclose(intensities_arb, intensities_random)

def test_neighborhood_intensity_random_occupation_matches_arbitrary_crystal(
        mc_random_gaas, mc_arbitrary_ingaas, ingaas_nd_form_factors, mocker
):
    """
    Test calculation of neighborhood intensity given points to resample for random
    occupation crystal matches that of arbitrary crystal.
    """
    vecs = ((utils.random_uniform_unit_vectors(100, 3) -
             utils.random_uniform_unit_vectors(100, 3))
            * mc_random_gaas.k())
    mock_rng = mocker.Mock()
    mock_rng.choice.return_value = np.array(
        [[0, 1, 0, 0, 0, 0, 0, 0] for _ in range(5)])
    mc_random_gaas._rng = mock_rng
    mocker.patch("numpy.random.default_rng", return_value=mock_rng)
    mocker.patch('numpy.random.multivariate_normal',
                 side_effect=mock_multivariate_normal)

    two_thetas = np.array([0., 20., 40., 60., 80., 100., 120., 140., 160.])
    intensities_rand, _, counts_rand = (
        mc_random_gaas.spectrum_neighborhood(
            vecs, two_thetas, ingaas_nd_form_factors,
            sigma=0.05, cnt_per_point=5
    ))
    intensities_arb, _, counts_arb = (
        mc_arbitrary_ingaas.spectrum_neighborhood(
            vecs, two_thetas, ingaas_nd_form_factors,
            sigma=0.05, cnt_per_point=5
        ))
    nptest.assert_allclose(intensities_rand, intensities_arb, rtol=1e-6, atol=1e-8)
    nptest.assert_equal(counts_rand, counts_arb)
