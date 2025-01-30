# pylint: disable=invalid-name
"""
This script plots diffraction patterns for a range of type 3-5 semiconductors.
"""

from B8_project import file_reading
from B8_project import crystal, diffraction

# Get basis and lattice parameters from CSV files.
CRYSTAL_PATH = "data/type_3_5_semiconductors/"
AlAs_basis = file_reading.read_basis(CRYSTAL_PATH + "AlAs_basis.csv")
AlAs_lattice = file_reading.read_lattice(CRYSTAL_PATH + "AlAs_lattice.csv")
AlP_basis = file_reading.read_basis(CRYSTAL_PATH + "AlP_basis.csv")
AlP_lattice = file_reading.read_lattice(CRYSTAL_PATH + "AlP_lattice.csv")
AlSb_basis = file_reading.read_basis(CRYSTAL_PATH + "AlSb_basis.csv")
AlSb_lattice = file_reading.read_lattice(CRYSTAL_PATH + "AlSb_lattice.csv")
BAs_basis = file_reading.read_basis(CRYSTAL_PATH + "BAs_basis.csv")
BAs_lattice = file_reading.read_lattice(CRYSTAL_PATH + "BAs_lattice.csv")
BP_basis = file_reading.read_basis(CRYSTAL_PATH + "BP_basis.csv")
BP_lattice = file_reading.read_lattice(CRYSTAL_PATH + "BP_lattice.csv")
BSb_basis = file_reading.read_basis(CRYSTAL_PATH + "BSb_basis.csv")
BSb_lattice = file_reading.read_lattice(CRYSTAL_PATH + "BSb_lattice.csv")
GaAs_basis = file_reading.read_basis(CRYSTAL_PATH + "GaAs_basis.csv")
GaAs_lattice = file_reading.read_lattice(CRYSTAL_PATH + "GaAs_lattice.csv")
GaP_basis = file_reading.read_basis(CRYSTAL_PATH + "GaP_basis.csv")
GaP_lattice = file_reading.read_lattice(CRYSTAL_PATH + "GaP_lattice.csv")
GaSb_basis = file_reading.read_basis(CRYSTAL_PATH + "GaSb_basis.csv")
GaSb_lattice = file_reading.read_lattice(CRYSTAL_PATH + "GaSb_lattice.csv")
InAs_basis = file_reading.read_basis(CRYSTAL_PATH + "InAs_basis.csv")
InAs_lattice = file_reading.read_lattice(CRYSTAL_PATH + "InAs_lattice.csv")
InP_basis = file_reading.read_basis(CRYSTAL_PATH + "InP_basis.csv")
InP_lattice = file_reading.read_lattice(CRYSTAL_PATH + "InP_lattice.csv")
InSb_basis = file_reading.read_basis(CRYSTAL_PATH + "InSb_basis.csv")
InSb_lattice = file_reading.read_lattice(CRYSTAL_PATH + "InSb_lattice.csv")

# Get form factors from CSV files.
neutron_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv"
)
x_ray_form_factors = file_reading.read_xray_form_factors("data/x_ray_form_factors.csv")

# Calculate the unit cells.
BP_unit_cell = crystal.UnitCell.new_unit_cell(BP_basis, BP_lattice)
BAs_unit_cell = crystal.UnitCell.new_unit_cell(BAs_basis, BAs_lattice)
BSb_unit_cell = crystal.UnitCell.new_unit_cell(BSb_basis, BSb_lattice)
AlP_unit_cell = crystal.UnitCell.new_unit_cell(AlP_basis, AlP_lattice)
AlAs_unit_cell = crystal.UnitCell.new_unit_cell(AlAs_basis, AlAs_lattice)
AlSb_unit_cell = crystal.UnitCell.new_unit_cell(AlSb_basis, AlSb_lattice)
GaP_unit_cell = crystal.UnitCell.new_unit_cell(GaP_basis, GaP_lattice)
GaAs_unit_cell = crystal.UnitCell.new_unit_cell(GaAs_basis, GaAs_lattice)
GaSb_unit_cell = crystal.UnitCell.new_unit_cell(GaSb_basis, GaSb_lattice)
InP_unit_cell = crystal.UnitCell.new_unit_cell(InP_basis, InP_lattice)
InAs_unit_cell = crystal.UnitCell.new_unit_cell(InAs_basis, InAs_lattice)
InSb_unit_cell = crystal.UnitCell.new_unit_cell(InSb_basis, InSb_lattice)

# Make a list of the unit cells and desired diffraction types.
groups_of_unit_cells = [
    [
        (BP_unit_cell, "XRD"),
        (BAs_unit_cell, "XRD"),
        (BSb_unit_cell, "XRD"),
    ],
    [
        (AlP_unit_cell, "XRD"),
        (AlAs_unit_cell, "XRD"),
        (AlSb_unit_cell, "XRD"),
    ],
    [
        (GaP_unit_cell, "XRD"),
        (GaAs_unit_cell, "XRD"),
        (GaSb_unit_cell, "XRD"),
    ],
    [
        (InP_unit_cell, "XRD"),
        (InAs_unit_cell, "XRD"),
        (InSb_unit_cell, "XRD"),
    ],
    [
        (BP_unit_cell, "XRD"),
        (AlP_unit_cell, "XRD"),
        (GaP_unit_cell, "XRD"),
        (InP_unit_cell, "XRD"),
    ],
    [
        (BAs_unit_cell, "XRD"),
        (AlAs_unit_cell, "XRD"),
        (GaAs_unit_cell, "XRD"),
        (InAs_unit_cell, "XRD"),
    ],
    [
        (BSb_unit_cell, "XRD"),
        (AlSb_unit_cell, "XRD"),
        (GaSb_unit_cell, "XRD"),
        (InSb_unit_cell, "XRD"),
    ],
]


# Plot the XRD patterns.
for unit_cells in groups_of_unit_cells:
    diffraction.plot_superimposed_diffraction_patterns(
        unit_cells,
        neutron_form_factors,
        x_ray_form_factors,
        min_deflection_angle=20,
        max_deflection_angle=60,
        peak_width=0.2,
        variable_wavelength=True,
        file_path="results/type_3_5_semiconductors/",
    )
