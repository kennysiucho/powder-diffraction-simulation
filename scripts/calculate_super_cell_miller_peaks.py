"""
This script calculates the diffraction peaks for GaAs, first for the unit cell, and 
then for a 1*1*1, 2*2*2 and 3*3*3 super cell. 
It prints the deflection angle, miller indices and relative intensity of each peak for 
the cells above.
The purpose of this script is ensure that the super cell implementation is valid - the 
miller peaks for all of the cells should be identical.
"""

from B8_project import alloy, crystal, file_reading, diffraction

# Get basis and lattice parameters from CSV files.
CRYSTAL_PATH = "data/type_3_5_semiconductors/"
GaAs_basis = file_reading.read_basis(CRYSTAL_PATH + "GaAs_basis.csv")
GaAs_lattice = file_reading.read_lattice(CRYSTAL_PATH + "GaAs_lattice.csv")

# Get form factors from CSV files.
neutron_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv"
)
x_ray_form_factors = file_reading.read_xray_form_factors("data/x_ray_form_factors.csv")

# Calculate the unit cell.
GaAs_unit_cell = crystal.UnitCell.new_unit_cell(GaAs_basis, GaAs_lattice)

# Calculate the super cells and convert them to unit cells.
MAX_SIDE_LENGTH = 5

super_cells = []
for side_length in range(1, MAX_SIDE_LENGTH + 1):
    GaAs_super_cell = alloy.SuperCell.new_super_cell(
        GaAs_unit_cell, (side_length, side_length, side_length)
    )

    super_cells.append(GaAs_super_cell)

# Diffraction parameters
WAVELENGTH = 0.1
MIN_DEFLECTION_ANGLE = 20
MAX_DEFLECTION_ANGLE = 60
INTENSITY_CUTOFF = 0.001

# Calculate and print the miller peaks for the unit cell.
print("GaAs unit cell miller peaks:")
GaAs_unit_cell_hkl_peaks = diffraction.get_miller_peaks(
    GaAs_unit_cell,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    WAVELENGTH,
    MIN_DEFLECTION_ANGLE,
    MAX_DEFLECTION_ANGLE,
    INTENSITY_CUTOFF,
    print_peak_data=True,
)

# Calculate and print the miller peaks for the super cells.
for i, GaAs_super_cell in enumerate(super_cells):
    print(f"\nGaAs super cell, side length = {i+1}")
    GaAs_super_cell_hkl_peaks = diffraction.get_miller_peaks(
        GaAs_super_cell,
        "XRD",
        neutron_form_factors,
        x_ray_form_factors,
        WAVELENGTH,
        MIN_DEFLECTION_ANGLE,
        MAX_DEFLECTION_ANGLE,
        INTENSITY_CUTOFF,
        print_peak_data=True,
    )
