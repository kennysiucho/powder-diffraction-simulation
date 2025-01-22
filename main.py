"""
This module currently extracts the lattice and basis parameters from two CSV files, 
creates a `UnitCell` instance with these parameters, and prints this `UnitCell` 
instance.
"""

from B8_project import file_reading
import B8_project.unit_cell as unit_cell
from B8_project.neutron_diffraction import NeutronDiffraction
import matplotlib.pyplot as plt

LATTICE_FILE = "Parameters/lattice.csv"
BASIS_FILE = "Parameters/basis.csv"

lattice = file_reading.get_lattice_from_csv(LATTICE_FILE)

basis = file_reading.get_basis_from_csv(BASIS_FILE)


my_cell = unit_cell.UnitCell.parameters_to_unit_cell(lattice, basis)

neutron_diffraction = NeutronDiffraction(my_cell, 0.123)
two_thetas, intensities = neutron_diffraction.diffraction_pattern(min_angle=20, max_angle=55)

plt.plot(two_thetas, intensities)
plt.show()

print(f"{my_cell}")
