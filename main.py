"""
This module uses the crystal and file_reading modules to calculate a diffraction 
pattern for a PrO2 crystal.
"""

from B8_project import file_reading
from B8_project import crystal, diffraction

# Get basis, lattice and neutron form factors from CSV files.
GaAs_basis = file_reading.read_basis("data/GaAs_basis.csv")
GaAs_lattice = file_reading.read_lattice("data/GaAs_lattice.csv")
InAs_basis = file_reading.read_basis("data/InAs_basis.csv")
InAs_lattice = file_reading.read_lattice("data/InAs_lattice.csv")
AlAs_basis = file_reading.read_basis("data/AlAs_basis.csv")
AlAs_lattice = file_reading.read_lattice("data/AlAs_lattice.csv")
AlP_basis = file_reading.read_basis("data/AlP_basis.csv")
AlP_lattice = file_reading.read_lattice("data/AlP_lattice.csv")
AlSb_basis = file_reading.read_basis("data/AlSb_basis.csv")
AlSb_lattice = file_reading.read_lattice("data/AlSb_lattice.csv")

neutron_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv"
)
x_ray_form_factors = file_reading.read_xray_form_factors("data/x_ray_form_factors.csv")

# Calculate the unit cells.
GaAs_unit_cell = crystal.UnitCell.new_unit_cell(GaAs_basis, GaAs_lattice)
InAs_unit_cell = crystal.UnitCell.new_unit_cell(InAs_basis, InAs_lattice)
AlAs_unit_cell = crystal.UnitCell.new_unit_cell(AlAs_basis, AlAs_lattice)
AlP_unit_cell = crystal.UnitCell.new_unit_cell(AlP_basis, AlP_lattice)
AlSb_unit_cell = crystal.UnitCell.new_unit_cell(AlSb_basis, AlSb_lattice)

# Make a list of the unit cells.
unit_cells = [
    AlAs_unit_cell,
    AlP_unit_cell,
    AlSb_unit_cell,
]

# Plot the XRD and ND patterns.
for unit_cell in unit_cells:
    # Pick wavelength such that peaks all occur at the same position.
    wavelength = (
        0.1 * unit_cell.lattice_constants[0] / AlAs_unit_cell.lattice_constants[0]
    )

    diffraction.plot_diffraction_pattern(
        unit_cell, x_ray_form_factors, wavelength, 10, 90, 0.1, plot=True
    )
