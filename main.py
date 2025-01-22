"""
This module currently extracts the lattice and basis parameters from two CSV files, 
creates a `UnitCell` instance with these parameters, and prints this `UnitCell` 
instance.
"""

import math
from B8_project import file_reading
from B8_project import crystal

basis = file_reading.get_basis_from_csv("tests/parameters/test_basis.csv")
lattice = file_reading.get_lattice_from_csv("tests/parameters/test_lattice.csv")
unit_cell = crystal.UnitCell.get_unit_cell(basis, lattice)


reciprocal_lattice_vectors = (
    crystal.ReciprocalLatticeVector.get_reciprocal_lattice_vectors(
        2 * math.pi + 0.001, unit_cell
    )
)
print(f"{reciprocal_lattice_vectors}")
