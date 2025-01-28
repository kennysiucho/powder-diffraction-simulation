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

# Make a list of the unit cells and desired diffraction types.
unit_cells_with_diffraction_types = [
    (AlP_unit_cell, "XRD"),
    (AlAs_unit_cell, "XRD"),
    (AlSb_unit_cell, "XRD"),
]

# Plot the XRD patterns
diffraction.plot_superimposed_diffraction_patterns(
    unit_cells_with_diffraction_types,
    neutron_form_factors,
    x_ray_form_factors,
    0.1,
    min_deflection_angle=20,
    max_deflection_angle=60,
    peak_width=0.1,
    variable_wavelength=True,
)
