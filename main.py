"""
This module currently extracts the lattice and basis parameters from two CSV files,
creates a `UnitCell` instance with these parameters, and prints this `UnitCell`
instance.
"""

from B8_project import file_reading
import B8_project.unit_cell as unit_cell
from B8_project.neutron_diffraction import NeutronDiffraction

import matplotlib.pyplot as plt
import numpy as np

LATTICE_FILE = "Parameters/lattice.csv"
BASIS_FILE = "Parameters/basis.csv"

lattice = file_reading.get_lattice_from_csv(LATTICE_FILE)

basis = file_reading.get_basis_from_csv(BASIS_FILE)

my_cell = unit_cell.UnitCell.parameters_to_unit_cell(lattice, basis)
nd = NeutronDiffraction(my_cell, 0.123)

# TODO: better handling of whether to calculate or read from file
two_thetas, intensities = nd.calculate_diffraction_pattern(1000)
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

plt.scatter(two_thetas, intensities, s=2)
plt.plot(two_theta_bins, intensities_binned, color='k')
plt.show()
print(f"{my_cell}")
