from B8_project import extract_parameters
import B8_project.unit_cell as unit_cell

LATTICE_FILE = "Parameters/lattice.csv"
BASIS_FILE = "Parameters/basis.csv"

lattice = extract_parameters.get_lattice_from_csv(LATTICE_FILE)

basis = extract_parameters.get_basis_from_csv(BASIS_FILE)

try:
    my_cell = unit_cell.UnitCell.parameters_to_unit_cell(lattice, basis)
    print(f"{my_cell}")
except ValueError as exc:
    print(f"Error: {exc}")
