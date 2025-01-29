from B8_project import file_reading
from B8_project import crystal, diffraction

# Get basis and lattice parameters from CSV files.
CRYSTAL_PATH = "data/type_3_5_semiconductors/"
GaAs_basis = file_reading.read_basis(CRYSTAL_PATH + "GaAs_basis.csv")
GaAs_lattice = file_reading.read_lattice(CRYSTAL_PATH + "GaAs_lattice.csv")
InAs_basis = file_reading.read_basis(CRYSTAL_PATH + "InAs_basis.csv")
InAs_lattice = file_reading.read_lattice(CRYSTAL_PATH + "InAs_lattice.csv")

# Get form factors from CSV files.
neutron_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv"
)
x_ray_form_factors = file_reading.read_xray_form_factors("data/x_ray_form_factors.csv")

# Calculate the unit cells.
GaAs_unit_cell = crystal.UnitCell.new_unit_cell(GaAs_basis, GaAs_lattice)
InAs_unit_cell = crystal.UnitCell.new_unit_cell(InAs_basis, InAs_lattice)

# Specify diffraction
GaAs_hkl_peaks = diffraction.calculate_miller_peaks(
    GaAs_unit_cell, "XRD", neutron_form_factors, x_ray_form_factors, 0.1, 10, 90
)
InAs_hkl_peaks = diffraction.calculate_miller_peaks(
    InAs_unit_cell, "XRD", neutron_form_factors, x_ray_form_factors, 0.1, 10, 90
)

print("GaAs diffraction peaks.")
for i, peak in enumerate(GaAs_hkl_peaks):
    print(
        f"Peak {i+1}: "
        f"Deflection angle = {round(peak[0], 2)}°; (h, k, l) = {peak[1]}; "
        f"relative intensity = {round(peak[2], 3)}"
    )

print("\nInAs diffraction peaks.")
for i, peak in enumerate(InAs_hkl_peaks):
    print(
        f"Peak {i+1}: "
        f"Deflection angle = {round(peak[0], 2)}°; (h, k, l) = {peak[1]}; "
        f"relative intensity = {round(peak[2], 3)}"
    )
