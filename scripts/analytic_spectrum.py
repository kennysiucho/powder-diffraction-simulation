from B8_project import file_reading, crystal, diffraction

basis = file_reading.read_basis("data/GaAs_basis.csv")
lattice = file_reading.read_lattice("data/GaAs_lattice.csv")

# Get form factors from CSV files.
neutron_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv"
)
x_ray_form_factors = file_reading.read_xray_form_factors("data/x_ray_form_factors.csv")

unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

WAVELENGTH = 1.54
MIN_DEFLECTION_ANGLE = 20
MAX_DEFLECTION_ANGLE = 160
INTENSITY_CUTOFF = 0.001

# Calculate miller peaks for GaAs.
hkl_peaks = diffraction.calculate_miller_peaks(
    unit_cell,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    WAVELENGTH,
    MIN_DEFLECTION_ANGLE,
    MAX_DEFLECTION_ANGLE,
    INTENSITY_CUTOFF,
)

# Plot the diffraction pattern.
diffraction.plot_diffraction_pattern(
    unit_cell,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    WAVELENGTH,
    MIN_DEFLECTION_ANGLE,
    MAX_DEFLECTION_ANGLE,
    file_path=None,
)

# Print the miller peaks for GaAs.
print("\nDiffraction peaks.")
for i, peak in enumerate(hkl_peaks):
    print(
        f"Peak {i+1}: "
        f"(h, k, l) = {peak[1]}; deflection angle = {round(peak[0], 3)}°; "
        f"relative intensity = {round(peak[2], 5)}"
    )