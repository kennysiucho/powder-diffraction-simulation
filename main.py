"""
This module uses the crystal and file_reading modules to calculate a diffraction 
pattern for a PrO2 crystal.
"""

from B8_project import file_reading
from B8_project import crystal, diffraction

# Get basis, lattice and neutron form factors from CSV files.
basis = file_reading.read_basis("data/PrO2_basis.csv")
lattice = file_reading.read_lattice("data/PrO2_lattice.csv")
neutron_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv"
)

# Calculate the unit cell
unit_cell = crystal.UnitCell.get_unit_cell(basis, lattice)

# Plot the diffraction pattern
print(
    diffraction.plot_diffraction_pattern(
        unit_cell,
        neutron_form_factors,
        0.123,
        20,
        55,
        0.05,
    )
)
