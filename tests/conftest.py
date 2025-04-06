"""
Shared data and fixtures for Monte Carlo tests.
"""
import pytest
import numpy as np

from B8_project.crystal import UnitCell, Atom
from B8_project.file_reading import read_lattice, read_basis, \
    read_neutron_scattering_lengths
from B8_project.mc_arbitrary_crystal import MCArbitraryCrystal
from B8_project.mc_ideal_crystal import MCIdealCrystal
from B8_project.mc_random_occupation import MCRandomOccupation

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
def random_scattering_vecs(k: float):
    """
    Returns scattering vectors computed from the pre-defined random unit vectors.
    """
    return (random_unit_vectors_1 - random_unit_vectors_2) * k

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

@pytest.fixture(name="nacl_nd_form_factors")
def fixture_nacl_nd_form_factors():
    """
    Returns a dictionary of neutron form factors for Nacl
    """
    all_nd_form_factors = read_neutron_scattering_lengths(
        "tests/data/neutron_scattering_lengths.csv")
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
        "tests/data/neutron_scattering_lengths.csv")
    nd_form_factors = {
        31: all_nd_form_factors[31],
        33: all_nd_form_factors[33],
        49: all_nd_form_factors[49]
    }
    yield nd_form_factors


@pytest.fixture(name="mc_ideal_nacl")
def fixture_mc_ideal_nacl():
    """
    Returns instance of `MCIdealCrystal`, containing data for NaCl and
    wavelength of 0.123Å
    """
    nacl_lattice = read_lattice("tests/data/NaCl_lattice.csv")
    nacl_basis = read_basis("tests/data/NaCl_basis.csv")
    unit_cell = UnitCell.new_unit_cell(nacl_basis, nacl_lattice)
    nd = MCIdealCrystal(
        unit_cell_reps=(8, 8, 8),
        wavelength=0.123,
        pdf=None,
        min_angle_deg=0.,
        max_angle_deg=180.,
        unit_cell=unit_cell
    )
    yield nd

@pytest.fixture(name="gaas_atom_list")
def fixture_gaas_atom_list():
    """
    Returns list of Atoms for a (2x1x1)-rep GaAs crystal
    """
    unit_cell_pos = np.array([[0, 0, 0], [GAAS_A, 0, 0]])
    lattice = read_lattice("tests/data/GaAs_lattice.csv")
    basis = read_basis("tests/data/GaAs_basis.csv")
    unit_cell: UnitCell = UnitCell.new_unit_cell(basis, lattice)
    atoms = []
    for uc_pos in unit_cell_pos:
        for atom in unit_cell.atoms:
            atoms.append(Atom(atom.atomic_number, uc_pos + np.array(atom.position) *
                         unit_cell.lattice_constants))
    return atoms

@pytest.fixture(name="mc_arbitrary_gaas")
def fixture_mc_arbitrary_gaas(gaas_atom_list):
    """
    Returns instance of `MCArbitraryCrystal`, containing data for GaAs and
    wavelength of 0.123Å
    """
    nd = MCArbitraryCrystal(
        wavelength=0.123,
        atoms=gaas_atom_list,
        pdf=None,
        min_angle_deg=0.,
        max_angle_deg=180.,
    )
    yield nd

@pytest.fixture(name="mc_ideal_gaas")
def fixture_mc_ideal_gaas():
    """
    Returns instance of `MCIdealCrystal`, containing data for GaAs and
    wavelength of 0.123Å
    """
    gaas_lattice = read_lattice("tests/data/GaAs_lattice.csv")
    gaas_basis = read_basis("tests/data/GaAs_basis.csv")
    unit_cell = UnitCell.new_unit_cell(gaas_basis, gaas_lattice)
    nd = MCIdealCrystal(
        unit_cell_reps=(2, 1, 1),
        wavelength=0.123,
        pdf=None,
        min_angle_deg=0.,
        max_angle_deg=180.,
        unit_cell=unit_cell
    )
    yield nd

@pytest.fixture(name="mc_random_gaas")
def fixture_mc_random_gaas():
    """
    Returns instance of `MCRandomOccupation`, containing data for InGaAs and
    wavelength of 0.123Å
    """
    gaas_lattice = read_lattice("tests/data/GaAs_lattice.csv")
    gaas_basis = read_basis("tests/data/GaAs_basis.csv")
    unit_cell = UnitCell.new_unit_cell(gaas_basis, gaas_lattice)
    nd = MCRandomOccupation(
        atom_from=31,
        atom_to=49,
        probability=0.1,
        unit_cell_reps=(2, 2, 2),
        wavelength=0.123,
        pdf=None,
        min_angle_deg=0.,
        max_angle_deg=180.,
        unit_cell=unit_cell
    )
    yield nd
