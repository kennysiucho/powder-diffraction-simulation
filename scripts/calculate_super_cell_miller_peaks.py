"""
This script calculates the diffraction peaks for GaAs, first for the unit cell, and 
then for a 1*1*1, 2*2*2 and 3*3*3 super cell. 
It prints the deflection angle, miller indices and relative intensity of each peak for 
the cells above.
The purpose of this script is ensure that the super cell implementation is valid - the 
miller peaks for all of the cells should be identical.
"""

from B8_project import file_reading, crystal, diffraction
import B8_project.super_cell as super_cell

# Get basis and lattice parameters from CSV files.
CRYSTAL_PATH = "data/type_3_5_semiconductors/"
GaAs_basis = file_reading.read_basis(CRYSTAL_PATH + "GaAs_basis.csv")
GaAs_lattice = file_reading.read_lattice(CRYSTAL_PATH + "GaAs_lattice.csv")

# Get form factors from CSV files.
neutron_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv"
)
x_ray_form_factors = file_reading.read_xray_form_factors("data/x_ray_form_factors.csv")

# Calculate the unit cells.
GaAs_unit_cell = crystal.UnitCell.new_unit_cell(GaAs_basis, GaAs_lattice)

# Calculate the super cells, and convert them to unit cells
GaAs_super_cell_1 = super_cell.SuperCell.new_super_cell(
    GaAs_unit_cell, (1, 1, 1)
).to_unit_cell()

GaAs_super_cell_2 = super_cell.SuperCell.new_super_cell(
    GaAs_unit_cell, (2, 2, 2)
).to_unit_cell()

GaAs_super_cell_3 = super_cell.SuperCell.new_super_cell(
    GaAs_unit_cell, (3, 3, 3)
).to_unit_cell()

# Diffraction parameters
WAVELENGTH = 0.1
MIN_DEFLECTION_ANGLE = 20
MAX_DEFLECTION_ANGLE = 60
INTENSITY_CUTOFF = 0.001

# Calculate the miller peaks for the cells
GaAs_unit_cell_hkl_peaks = diffraction.calculate_miller_peaks(
    GaAs_unit_cell,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    WAVELENGTH,
    MIN_DEFLECTION_ANGLE,
    MAX_DEFLECTION_ANGLE,
    INTENSITY_CUTOFF,
)

GaAs_super_cell_1_hkl_peaks = diffraction.calculate_miller_peaks(
    GaAs_super_cell_1,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    WAVELENGTH,
    MIN_DEFLECTION_ANGLE,
    MAX_DEFLECTION_ANGLE,
    INTENSITY_CUTOFF,
)

GaAs_super_cell_2_hkl_peaks = diffraction.calculate_miller_peaks(
    GaAs_super_cell_2,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    WAVELENGTH,
    MIN_DEFLECTION_ANGLE,
    MAX_DEFLECTION_ANGLE,
    INTENSITY_CUTOFF,
)

# GaAs_super_cell_3_hkl_peaks = diffraction.calculate_miller_peaks(
#     GaAs_super_cell_3,
#     "XRD",
#     neutron_form_factors,
#     x_ray_form_factors,
#     WAVELENGTH,
#     MIN_DEFLECTION_ANGLE,
#     MAX_DEFLECTION_ANGLE,
#     INTENSITY_CUTOFF,
# )

# Print the miller peaks for the different cells.
print("GaAs unit cell diffraction peaks.")
for i, peak in enumerate(GaAs_unit_cell_hkl_peaks):
    print(
        f"Peak {i+1}: "
        f"(h, k, l) = {peak[1]}; deflection angle = {round(peak[0], 2)}°; "
        f"relative intensity = {round(peak[2], 4)}"
    )

print("\nGaAs super cell (1*1*1) diffraction peaks.")
for i, peak in enumerate(GaAs_super_cell_1_hkl_peaks):
    print(
        f"Peak {i+1}: "
        f"(h, k, l) = {peak[1]}; deflection angle = {round(peak[0], 2)}°; "
        f"relative intensity = {round(peak[2], 4)}"
    )

print("\nGaAs super cell (2*2*2) diffraction peaks.")
for i, peak in enumerate(GaAs_super_cell_2_hkl_peaks):
    print(
        f"Peak {i+1}: "
        f"(h, k, l) = {peak[1]}; deflection angle = {round(peak[0], 2)}°; "
        f"relative intensity = {round(peak[2], 4)}"
    )

# print("\nGaAs super cell (3*3*3) diffraction peaks.")
# for i, peak in enumerate(GaAs_super_cell_3_hkl_peaks):
#     print(
#         f"Peak {i+1}: "
#         f"(h, k, l) = {peak[1]}; deflection angle = {round(peak[0], 2)}°; "
#         f"relative intensity = {round(peak[2], 4)}"
#     )
