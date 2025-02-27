"""
This script calculates the diffraction peaks using evenly spaced scattering vectors.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from B8_project import file_reading
from B8_project.crystal import UnitCell
from B8_project.diffraction_monte_carlo import DiffractionMonteCarlo

two_thetas_file = Path("two_thetas.txt")
intensities_file = Path("intensities.txt")
if two_thetas_file.is_file() or intensities_file.is_file():
    while True:
        res = input(
                "Previous scattering intensity data found in 'two_thetas.txt' or "
                "'intensities.txt'. Plot existing data? [y/n]")
        if res.lower() == "y":
            CALCULATE_SPECTRUM = False
            break
        if res.lower() == "n":
            CALCULATE_SPECTRUM = True
            break
else:
    CALCULATE_SPECTRUM = True

LATTICE_FILE = "data/PrO2_lattice.csv"
BASIS_FILE = "data/PrO2_basis.csv"

lattice = file_reading.read_lattice(LATTICE_FILE)
basis = file_reading.read_basis(BASIS_FILE)

unit_cell = UnitCell.new_unit_cell(basis, lattice)
diff = DiffractionMonteCarlo(unit_cell,
                             1.23,
                             min_angle_deg=43,
                             max_angle_deg=48)

if CALCULATE_SPECTRUM:

    all_nd_form_factors = file_reading.read_neutron_scattering_lengths(
        "data/neutron_scattering_lengths.csv")
    nd_form_factors = {}
    for atom in diff.unit_cell.atoms:
        nd_form_factors[atom.atomic_number] = all_nd_form_factors[atom.atomic_number]
    nd_form_factors[49] = all_nd_form_factors[49]

    # all_xray_form_factors = file_reading.read_xray_form_factors(
    #     "data/x_ray_form_factors.csv")
    # xrd_form_factors = {}
    # for atom in diff.unit_cell.atoms:
    #     xrd_form_factors[atom.atomic_number] = all_xray_form_factors[atom.atomic_number]
    # xrd_form_factors[49] = all_xray_form_factors[49]

    two_thetas, intensities = (
        diff.calculate_diffraction_pattern_evenly_spaced(
            nd_form_factors,
            unit_cell_reps=(20, 20, 20),
            num_angles=50,
            points_per_angle=10000))
    np.savetxt('two_thetas.txt', two_thetas)
    np.savetxt('intensities.txt', intensities)
else:
    two_thetas = np.loadtxt("two_thetas.txt")
    intensities = np.loadtxt("intensities.txt")


plt.scatter(two_thetas, intensities, s=5)
plt.plot(two_thetas, intensities, color='k', label="Intensity")
plt.axhline(0, linestyle="--", color="grey")
plt.xlabel("Scattering angle (2θ) (deg)")
plt.ylabel("Normalized intensity")
plt.title(f"{unit_cell.material} Diffraction Spectrum")
plt.legend()
plt.grid(linestyle=":")
plt.show()
