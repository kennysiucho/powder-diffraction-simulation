"""
pandas is used to extract parameters from .csv files
"""

import extract_parameters

LATTICE_FILE = "Parameters/lattice.csv"
BASIS_FILE = "Parameters/basis.csv"

material, lattice_type, lattice_constants = (
    extract_parameters.extract_lattice_parameters_from_csv(LATTICE_FILE)
)

atomic_numbers, atomic_masses, atomic_positions = (
    extract_parameters.extract_basis_from_csv(BASIS_FILE)
)

print(
    f"material = {material}, lattice_type = {lattice_type}, lattice_constants = {lattice_constants}"
)

print(
    f"atomic numbers = {atomic_numbers}, atomic masses = {atomic_masses}, "
    f"atomic positions = {atomic_positions}"
)
