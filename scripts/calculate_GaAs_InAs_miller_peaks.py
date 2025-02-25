# pylint: disable=invalid-name
"""
This script calculates the diffraction peaks for GaAs and InAs, and prints the 
deflection angle, miller indices and relative intensity of each peak.
"""

from B8_project import file_reading, crystal, diffraction

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

# Specify the GaAs diffraction parameters.
WAVELENGTH = 1.541838
MIN_DEFLECTION_ANGLE = 27
MAX_DEFLECTION_ANGLE = 84
INTENSITY_CUTOFF = 0.001

# Calculate miller peaks for GaAs.
GaAs_hkl_peaks = diffraction.calculate_miller_peaks(
    GaAs_unit_cell,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    WAVELENGTH,
    MIN_DEFLECTION_ANGLE,
    MAX_DEFLECTION_ANGLE,
    INTENSITY_CUTOFF,
)

# Plot the GaAs diffraction pattern.
diffraction.plot_diffraction_pattern(
    GaAs_unit_cell,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    WAVELENGTH,
    MIN_DEFLECTION_ANGLE,
    MAX_DEFLECTION_ANGLE,
    file_path="results/type_3_5_semiconductors/",
)

# Specify the InAs diffraction parameters.
WAVELENGTH = 1.541838
MIN_DEFLECTION_ANGLE = 25
MAX_DEFLECTION_ANGLE = 84
INTENSITY_CUTOFF = 0.001

# Calculate miller peaks for InAs.
InAs_hkl_peaks = diffraction.calculate_miller_peaks(
    InAs_unit_cell,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    WAVELENGTH,
    MIN_DEFLECTION_ANGLE,
    MAX_DEFLECTION_ANGLE,
    INTENSITY_CUTOFF,
)

# Plot the InAs diffraction pattern.
diffraction.plot_diffraction_pattern(
    InAs_unit_cell,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    WAVELENGTH,
    MIN_DEFLECTION_ANGLE,
    MAX_DEFLECTION_ANGLE,
    file_path="results/type_3_5_semiconductors/",
)

# Print the miller peaks for GaAs.
print("\nGaAs diffraction peaks.")
for i, peak in enumerate(GaAs_hkl_peaks):
    print(
        f"Peak {i+1}: "
        f"(h, k, l) = {peak[1]}; deflection angle = {round(peak[0], 2)}°; "
        f"relative intensity = {round(peak[2], 4)}"
    )

# Print the miller peaks for InAs.
print("\nInAs diffraction peaks.")
for i, peak in enumerate(InAs_hkl_peaks):
    print(
        f"Peak {i+1}: "
        f"(h, k, l) = {peak[1]}; deflection angle = {round(peak[0], 2)}°; "
        f"relative intensity = {round(peak[2], 4)}"
    )
