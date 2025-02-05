import matplotlib.pyplot as plt
import numpy as np

from B8_project import file_reading
import B8_project.crystal as unit_cell
from B8_project.diffraction_monte_carlo import NeutronDiffractionMonteCarlo

LATTICE_FILE = "data/PrO2_lattice.csv"
BASIS_FILE = "data/PrO2_basis.csv"

lattice = file_reading.read_lattice(LATTICE_FILE)

basis = file_reading.read_basis(BASIS_FILE)

my_cell = unit_cell.UnitCell.new_unit_cell(basis, lattice)
nd = NeutronDiffractionMonteCarlo(my_cell, 0.123)

# TODO: better handling of whether to calculate or read from file
two_thetas, intensities = (
    nd.calculate_diffraction_pattern(300000,
                                     min_angle_deg=18,
                                     max_angle_deg=57,
                                     angle_bins=200))
np.savetxt('two_thetas.txt', two_thetas)
np.savetxt('intensities.txt', intensities)

two_thetas = np.loadtxt("two_thetas.txt")
intensities = np.loadtxt("intensities.txt")

# plt.scatter(two_thetas, intensities, s=2, label="Intensity")
plt.plot(two_thetas, intensities, color='k', label="Intensity")
plt.xlabel("Scattering angle (2θ) (deg)")
plt.ylabel("Normalized intensity")
plt.title("PrO2 Neutron Diffraction Spectrum")
plt.legend()
plt.show()
print(f"{my_cell}")
