"""
This module contains unit tests for the mc_displacement.py module.
"""

import numpy.testing as nptest
from B8_project import utils
from B8_project.mc_displacement import MCDisplacement
from tests.conftest import *

@pytest.fixture(name="mc_displacement_ingaas")
def fixture_mc_displacement_ingaas():
    """
    Returns instance of `MCDisplacement`, containing data for InGaAs and
    wavelength of 0.123Å
    """
    gaas_lattice = read_lattice("tests/data/GaAs_lattice.csv")
    gaas_basis = read_basis("tests/data/GaAs_basis.csv")
    unit_cell = UnitCell.new_unit_cell(gaas_basis, gaas_lattice)
    nd = MCDisplacement(
        atom_from=31,
        atom_to=49,
        probability=0.1,
        wavelength=0.123,
        pdf=None,
        min_angle_deg=0.,
        max_angle_deg=180.,
        unit_cell=unit_cell,
        displace_func=lambda pos, uc: MCDisplacement.gaussian_displaced(
            pos, uc, sigma=0.05, atoms_to_displace=[31, 49]
        )
    )
    nd.setup_cuboid_crystal((2, 2, 2))
    yield nd

# Deterministic displacement
def nudge_indium_and_gallium(pos, uc):
    modified = np.empty_like(pos)
    for i in range(len(pos)):
        modified[i] = pos[i]
        if uc[i] in (31, 49):
            modified[i] += np.array([0.1, 0.1, 0.1])
    return modified

@pytest.fixture(name="mc_disp_ingaas_deterministic")
def fixture_mc_disp_ingaas_deterministic(mc_displacement_ingaas):
    mc_displacement_ingaas.setup_cuboid_crystal((2, 2, 2))
    mc_displacement_ingaas._displace_func = nudge_indium_and_gallium
    yield mc_displacement_ingaas

@pytest.fixture(name="mc_disp_ingaas_arb")
def fixture_mc_disp_ingaas_arb(mc_displacement_ingaas):
    # Build arbitrary crystal
    atoms = []
    uc_choices = [0, 1, 2, 3, 4, 0, 0, 0]
    for i, uc_pos in enumerate(mc_displacement_ingaas._unit_cell_pos):
        for j, atom in enumerate(mc_displacement_ingaas._atomic_numbers_vars[uc_choices[i]]):
            displacement = np.array([0.1, 0.1, 0.1]) if atom in (31, 49) \
                else np.array([0., 0., 0.])
            atoms.append(Atom(atom, displacement +
                              uc_pos + mc_displacement_ingaas._atom_pos_in_uc[j]))
    mc_arb = MCArbitraryCrystal(
        wavelength=0.123,
        atoms=atoms,
        pdf=None,
        min_angle_deg=0.,
        max_angle_deg=180.,
    )
    yield mc_arb

def test_displace_gaussian_normal_operation():
    """
    Test that displace_gaussian returns positions with correct shape and
    only displaces specified atom types.
    """
    np.random.seed(42)

    positions = np.zeros((6, 3))
    atoms_in_uc = np.array([31, 49, 8, 8, 31, 49])  # Only displace 31 and 49
    sigma = 0.05
    atoms_to_displace = [31, 49]

    displaced = MCDisplacement.gaussian_displaced(
        positions=positions,
        atoms_in_uc=atoms_in_uc,
        sigma=sigma,
        atoms_to_displace=atoms_to_displace
    )

    assert displaced.shape == positions.shape

    # Check displacement: only for selected atom types
    for i, atom_type in enumerate(atoms_in_uc):
        if atom_type in atoms_to_displace:
            assert not np.allclose(displaced[i],
                                   positions[i]), f"Atom {i} should be displaced"
        else:
            assert np.allclose(displaced[i],
                               positions[i]), f"Atom {i} should not be displaced"

    # Check statistical properties
    displacement_magnitudes = np.linalg.norm(displaced - positions, axis=1)
    displaced_atoms = [i for i in range(len(atoms_in_uc)) if
                       atoms_in_uc[i] in atoms_to_displace]
    mean_disp = displacement_magnitudes[displaced_atoms].mean()

    assert mean_disp > 0, "Displaced atoms should have non-zero displacement"
    assert mean_disp < 0.2, "Displacement should be reasonable for sigma=0.05"

def test_random_occupation_matches_zero_displacement(
        mc_random_gaas, mc_displacement_ingaas, ingaas_nd_form_factors, mocker
):
    """
    The spectrum produced by MCDisplacement with 0 displacement should match that of
    MCRandomOccupation if they both generate the same random crystal.
    """
    random_uvs1 = utils.random_uniform_unit_vectors(1000, 3)
    random_uvs2 = utils.random_uniform_unit_vectors(1000, 3)
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_uvs1, random_uvs2,
                     random_uvs1, random_uvs2],
    )
    mock_rng1 = mocker.Mock()
    mock_rng1.choice.return_value = np.array(
        [[0, 1, 0, 0, 0, 0, 0, 0] for _ in range(len(random_uvs1))])
    mc_random_gaas._rng = mock_rng1
    mock_rng2 = mocker.Mock()
    mock_rng2.choice.return_value = np.array(
        [0, 1, 0, 0, 0, 0, 0, 0])
    mc_displacement_ingaas._rng = mock_rng2

    mc_random_gaas.setup_cuboid_crystal((2, 2, 2))
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

    mc_displacement_ingaas.setup_cuboid_crystal((2, 2, 2))
    mc_displacement_ingaas._displace_func = lambda pos, uc: (
        MCDisplacement.gaussian_displaced(pos, uc, sigma=0., atoms_to_displace=[]))
    _, intensities_displacement, _, _ = (
        mc_displacement_ingaas.spectrum_uniform(
            ingaas_nd_form_factors,
            total_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False,
            threshold=0.
        )
    )

    nptest.assert_allclose(intensities_displacement, intensities_random)

def test_compute_intensities_displacement_matches_arbitrary(
        mc_disp_ingaas_deterministic, mc_disp_ingaas_arb, ingaas_nd_form_factors, mocker
):
    """
    Test calculation of intensities given scattering vectors.
    """
    vecs = random_scattering_vecs(mc_disp_ingaas_deterministic.k())
    # Deterministic random occupation
    mock_rng = mocker.Mock()
    mock_rng.choice.return_value = np.array([0, 1, 2, 3, 4, 0, 0, 0])
    mc_disp_ingaas_deterministic._rng = mock_rng

    intensities_disp = mc_disp_ingaas_deterministic.compute_intensities(
        vecs, ingaas_nd_form_factors
    )
    intensities_arb = mc_disp_ingaas_arb.compute_intensities(
        vecs, ingaas_nd_form_factors
    )

    nptest.assert_allclose(intensities_disp, intensities_arb, rtol=1e-6, atol=1e-6)

def test_displacement_matches_arbitrary(
    mc_disp_ingaas_deterministic, mc_disp_ingaas_arb, ingaas_nd_form_factors, mocker
):
    """
    Test if the spectrum calculated by displacement matches that of arbitrary, given
    the crystals are identical.
    """
    random_uvs1 = utils.random_uniform_unit_vectors(1000, 3)
    random_uvs2 = utils.random_uniform_unit_vectors(1000, 3)
    mocker.patch(
        "B8_project.utils.random_uniform_unit_vectors",
        side_effect=[random_uvs1, random_uvs2,
                     random_uvs1, random_uvs2],
    )
    mock_rng = mocker.Mock()
    mock_rng.choice.return_value = np.array([0, 1, 2, 3, 4, 0, 0, 0])
    mc_disp_ingaas_deterministic._rng = mock_rng

    _, intensities_arb, _, _ = (
        mc_disp_ingaas_arb.spectrum_uniform(
            ingaas_nd_form_factors,
            total_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False,
            threshold=0.
        )
    )

    _, intensities_disp, _, _ = (
        mc_disp_ingaas_deterministic.spectrum_uniform(
            ingaas_nd_form_factors,
            total_trials=1000,
            trials_per_batch=1000,
            angle_bins=10,
            weighted=False,
            threshold=0.
        )
    )

    nptest.assert_allclose(intensities_arb, intensities_disp)
