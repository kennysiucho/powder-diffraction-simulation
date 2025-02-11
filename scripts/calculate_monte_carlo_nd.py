"""
This script calculates the diffraction peaks for PrO2 using Monte Carlo and plots the
spectrum.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from B8_project import file_reading
import B8_project.crystal as unit_cell
from B8_project.diffraction_monte_carlo import NeutronDiffractionMonteCarlo

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

if CALCULATE_SPECTRUM:
    LATTICE_FILE = "data/PrO2_lattice.csv"
    BASIS_FILE = "data/PrO2_basis.csv"

    lattice = file_reading.read_lattice(LATTICE_FILE)
    basis = file_reading.read_basis(BASIS_FILE)

    unit_cell = unit_cell.UnitCell.new_unit_cell(basis, lattice)
    nd = NeutronDiffractionMonteCarlo(unit_cell, 0.123)

    two_thetas, intensities = (
        nd.calculate_diffraction_pattern_ideal_crystal(
            target_accepted_trials=1000000,
            unit_cells_in_crystal=(10, 10, 10),
            trials_per_batch=1000,
            min_angle_deg=18,
            max_angle_deg=55,
            angle_bins=100))
    np.savetxt('two_thetas.txt', two_thetas)
    np.savetxt('intensities.txt', intensities)
else:
    two_thetas = np.loadtxt("two_thetas.txt")
    intensities = np.loadtxt("intensities.txt")

# plt.scatter(two_thetas, intensities, s=2, label="Intensity")
plt.plot(two_thetas, intensities, color='k', label="Intensity")
plt.xlabel("Scattering angle (2θ) (deg)")
plt.ylabel("Normalized intensity")
plt.title("PrO2 Neutron Diffraction Spectrum")
plt.legend()
plt.show()
