"""
This script calculates the diffraction peaks for PrO2 using Monte Carlo and plots the
spectrum.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from B8_project import file_reading
import B8_project.crystal as unit_cell
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

if CALCULATE_SPECTRUM:
    LATTICE_FILE = "data/GaAs_lattice.csv"
    BASIS_FILE = "data/GaAs_basis.csv"

    lattice = file_reading.read_lattice(LATTICE_FILE)
    basis = file_reading.read_basis(BASIS_FILE)

    unit_cell = unit_cell.UnitCell.new_unit_cell(basis, lattice)
    nd = DiffractionMonteCarlo(unit_cell, 0.123)

    all_nd_form_factors = file_reading.read_neutron_scattering_lengths(
        "data/neutron_scattering_lengths.csv")
    nd_form_factors = {}
    for atom in nd.unit_cell.atoms:
        nd_form_factors[atom.atomic_number] = all_nd_form_factors[atom.atomic_number]
    nd_form_factors[49] = all_nd_form_factors[49]

    all_xray_form_factors = file_reading.read_xray_form_factors(
        "data/x_ray_form_factors.csv")
    xrd_form_factors = {}
    for atom in nd.unit_cell.atoms:
        xrd_form_factors[atom.atomic_number] = all_xray_form_factors[atom.atomic_number]
    xrd_form_factors[49] = all_xray_form_factors[49]

    two_thetas, intensities = (
        nd.calculate_diffraction_pattern_random_occupation(
            31,
            49,
            0.25,
            xrd_form_factors,
            target_accepted_trials=1000000,
            unit_cell_reps=(10, 10, 10),
            trials_per_batch=1000,
            min_angle_deg=18,
            max_angle_deg=57,
            angle_bins=200))
    np.savetxt('two_thetas.txt', two_thetas)
    np.savetxt('intensities.txt', intensities)
else:
    two_thetas = np.loadtxt("two_thetas.txt")
    intensities = np.loadtxt("intensities.txt")

# two_thetas = two_thetas[::2]
# intensities = (intensities[::2] + intensities[1::2]) / 2

# plt.scatter(two_thetas, intensities, s=2, label="Intensity")
plt.plot(two_thetas, intensities, color='k', label="Intensity")
plt.xlabel("Scattering angle (2θ) (deg)")
plt.ylabel("Normalized intensity")
plt.title("In_0.25Ga_0.75As Neutron Diffraction Spectrum")
plt.legend()
plt.show()
