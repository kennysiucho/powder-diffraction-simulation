"""
This script calculates the diffraction peaks for PrO2 using Monte Carlo and plots the
spectrum.
"""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from B8_project import file_reading
from B8_project.crystal import UnitCell
from B8_project.mc_ideal_crystal import MCIdealCrystal

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

    unit_cell: UnitCell = UnitCell.new_unit_cell(basis, lattice)
    diff = MCIdealCrystal(1.23,
                          unit_cell,
                          (10, 10, 10),
                          min_angle_deg=42,
                          max_angle_deg=50)

    start_time = time.time()

    two_thetas, intensities, _, _ = (
        diff.calculate_diffraction_pattern_brute_force(
            diff.all_nd_form_factors,
            total_trials=4_000_000,
            trials_per_batch=1000,
            angle_bins=100,
            weighted=False))
    np.savetxt('two_thetas.txt', two_thetas)
    np.savetxt('intensities.txt', intensities)
    print(f"Total run time = {time.time() - start_time}s")
else:
    two_thetas = np.loadtxt("two_thetas.txt")
    intensities = np.loadtxt("intensities.txt")

# two_thetas = two_thetas[::2]
# intensities = (intensities[::2] + intensities[1::2]) / 2

# plt.scatter(two_thetas, intensities, s=2, label="Intensity")
plt.plot(two_thetas, intensities, color='k', label="Intensity")
# plt.plot(two_thetas, pdf(two_thetas) / np.max(pdf(two_thetas)), "--", label="PDF")
plt.ylim(bottom=0)
plt.xlabel("Scattering angle (2θ) (deg)")
plt.ylabel("Normalized intensity")
# plt.title("In_0.25Ga_0.75As Neutron Diffraction Spectrum")
plt.title("PrO2 Neutron Diffraction Spectrum")
plt.legend()
plt.grid(linestyle=":")
plt.show()
