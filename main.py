"""
This module uses the crystal and file_reading modules to calculate a diffraction 
pattern for a PrO2 crystal. Numpy and matplotlib are then used to plot the diffraction 
pattern.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from B8_project import file_reading
from B8_project import crystal

# Get basis, lattice and neutron form factors from CSV files.
basis = file_reading.get_basis_from_csv("parameters/PrO2_basis.csv")
lattice = file_reading.get_lattice_from_csv("parameters/PrO2_lattice.csv")
neutron_form_factors = file_reading.get_neutron_scattering_lengths_from_csv(
    "parameters/neutron_scattering_lengths.csv"
)

# Calculate the intensity peaks
unit_cell = crystal.UnitCell.get_unit_cell(basis, lattice)
intensity_peaks = crystal.Diffraction.get_intensity_peaks(
    unit_cell, neutron_form_factors, 0.123
)

# Convert angle in radians to deflection angle in degrees
intensity_peaks = [
    (angle * 360 / math.pi, relative_intensity)
    for (angle, relative_intensity) in intensity_peaks
]

# Remove deflection angles outside of a set range
min_angle = 20
max_angle = 55
intensity_peaks = [
    intensity_peak
    for intensity_peak in intensity_peaks
    if min_angle <= intensity_peak[0] <= max_angle
]

# Normalize the intensities
angles, intensities = zip(*intensity_peaks)

max_intensity = max(intensities)
intensities = [intensity / max_intensity for intensity in intensities]

normalized_intensity_peaks = list(zip(angles, intensities))

# x values
x_values = np.linspace(20, 55, 1000)


# Gaussian function
def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Initialize y values, and then sum peaks
y_values = np.zeros_like(x_values)

for angle, intensity in normalized_intensity_peaks:
    y_values += gaussian(x_values, angle, 0.1, intensity)

# Plot the intensity pattern
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, linestyle="-", color="black")
plt.xlabel("2θ (degrees)")
plt.ylabel("Normalized intensity")
plt.title("PrO2 neutron diffraction pattern")
plt.grid(True)

# Save the plot as a PDF file
plt.savefig("results/PrO2_diffraction_pattern_23_01_25.pdf", format="pdf")
