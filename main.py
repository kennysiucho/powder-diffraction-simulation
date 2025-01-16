import extract_parameters
import unit_cell

LATTICE_FILE = "Parameters/lattice.csv"
BASIS_FILE = "Parameters/basis.csv"

material, lattice_type, lattice_constants = extract_parameters.get_lattice_from_csv(
    LATTICE_FILE
)

atomic_numbers, atomic_masses, atomic_positions = extract_parameters.get_basis_from_csv(
    BASIS_FILE
)

print(
    f"material = {material}, lattice_type = {lattice_type}, lattice_constants = {lattice_constants}"
)

print(
    f"atomic numbers = {atomic_numbers}, atomic masses = {atomic_masses}, "
    f"atomic positions = {atomic_positions}"
)

lattice = material, lattice_type, lattice_constants
basis = atomic_numbers, atomic_masses, atomic_positions
my_cell = unit_cell.UnitCell.lattice_and_basis_to_unit_cell(lattice, basis)
print(f"{my_cell}")
