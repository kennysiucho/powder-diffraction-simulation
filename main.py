"""
This module currently extracts the lattice and basis parameters from two CSV files, 
creates a `UnitCell` instance with these parameters, and prints this `UnitCell` 
instance.
"""

from B8_project import file_reading
from B8_project import crystal_lattice

LATTICE_FILE = "Parameters/lattice.csv"
BASIS_FILE = "Parameters/basis.csv"

lattice = file_reading.get_lattice_from_csv(LATTICE_FILE)

basis = file_reading.get_basis_from_csv(BASIS_FILE)

try:
    my_cell = crystal_lattice.UnitCell.crystal_parameters_to_unit_cell(lattice, basis)
    print(f"{my_cell}")
except ValueError as exc:
    print(f"Error: {exc}")

FILENAME = "tests/parameters/test_xray_form_factors.csv"

x_ray_form_factors = file_reading.get_x_ray_form_factors_from_csv(FILENAME)

print(x_ray_form_factors[11])
print(x_ray_form_factors[17])
