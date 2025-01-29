"""
This module currently extracts the lattice and basis parameters from two CSV files,
creates a `UnitCell` instance with these parameters, and prints this `UnitCell`
instance.
"""

from B8_project import file_reading
import B8_project.crystal as unit_cell
from B8_project.diffraction import NeutronDiffractionMonteCarlo

import matplotlib.pyplot as plt
import numpy as np

LATTICE_FILE = "data/PrO2_lattice.csv"
BASIS_FILE = "data/PrO2_basis.csv"

lattice = file_reading.read_lattice(LATTICE_FILE)

basis = file_reading.read_basis(BASIS_FILE)

my_cell = unit_cell.UnitCell.new_unit_cell(basis, lattice)
nd = NeutronDiffractionMonteCarlo(my_cell, 0.123)

# TODO: better handling of whether to calculate or read from file
two_thetas, intensities = nd.calculate_diffraction_pattern(200)
np.savetxt('two_thetas.txt', two_thetas)
np.savetxt('intensities.txt', intensities)

two_thetas = np.loadtxt("two_thetas.txt")
intensities = np.loadtxt("intensities.txt")

indices = np.where(np.logical_and(two_thetas >= 18, two_thetas <= 57))
two_thetas = two_thetas[indices]
intensities = intensities[indices]
intensities = intensities / np.max(intensities)

num_bins = 60
min_theta = np.min(two_thetas)
max_theta = np.max(two_thetas)
bin_size = (max_theta - min_theta) / num_bins
two_theta_bins = np.linspace(min_theta + bin_size / 2, max_theta - bin_size / 2, num_bins)
intensities_binned = np.zeros(num_bins)
for i in range(two_thetas.size):
    bin_i = int((two_thetas[i] - min_theta) // bin_size)
    if bin_i == num_bins: continue
    intensities_binned[bin_i] += intensities[i]
intensities_binned /= np.max(intensities_binned)

plt.scatter(two_thetas, intensities, s=2, label="Scattering trials")
plt.plot(two_theta_bins, intensities_binned, color='k', label="Aggregated intensities")
plt.xlabel("Scattering angle (2θ) (deg)")
plt.ylabel("Normalized intensity")
plt.title("PrO2 Neutron Diffraction Spectrum")
plt.legend()
plt.show()
print(f"{my_cell}")
