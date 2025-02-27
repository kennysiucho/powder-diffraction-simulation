"""
This script evaluates how well Monte Carlo calculation of an intensity value converges
for a particular scattering angle. This is done using the ratio of intensity of two
peaks.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from B8_project import file_reading
from B8_project.crystal import UnitCell
from B8_project.diffraction_monte_carlo import DiffractionMonteCarlo
from B8_project import utils

LATTICE_FILE = "data/PrO2_lattice.csv"
BASIS_FILE = "data/PrO2_basis.csv"

lattice = file_reading.read_lattice(LATTICE_FILE)
basis = file_reading.read_basis(BASIS_FILE)

unit_cell = UnitCell.new_unit_cell(basis, lattice)
diff = DiffractionMonteCarlo(unit_cell,
                             1.23,
                             min_angle_deg=18,
                             max_angle_deg=60)

all_nd_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv")
nd_form_factors = {}
for atom in diff.unit_cell.atoms:
    nd_form_factors[atom.atomic_number] = all_nd_form_factors[atom.atomic_number]
nd_form_factors[49] = all_nd_form_factors[49]

# all_xray_form_factors = file_reading.read_xray_form_factors(
#     "data/x_ray_form_factors.csv")
# xrd_form_factors = {}
# for atom in diff.unit_cell.atoms:
#     xrd_form_factors[atom.atomic_number] = all_xray_form_factors[atom.atomic_number]
# xrd_form_factors[49] = all_xray_form_factors[49]

# PrO2 angles: 37.5687, 44.3686, 46.4542
angle1, angle2 = 44.3686, 46.4542
unit_cell_pos = diff._unit_cell_positions((4, 4, 4))
atoms_in_uc, atom_pos_in_uc = diff._atoms_and_pos_in_uc()

vecs, _ = diff._get_uniform_scattering_vecs_and_angles_single(10000, angle1)
evenly_spaced_int1 = diff._compute_intensity(
    vecs, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc
)
vecs, _ = diff._get_uniform_scattering_vecs_and_angles_single(10000, angle2)
evenly_spaced_int2 = diff._compute_intensity(
    vecs, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc
)
ratio_evenly_spaced = evenly_spaced_int2 / evenly_spaced_int1
print(ratio_evenly_spaced)

# integral_int1, abserr1 = diff._compute_intensity_integral(
#     angle1, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc
# )
# integral_int2, abserr2 = diff._compute_intensity_integral(
#     angle2, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc
# )
# ratio_integral = integral_int2 / integral_int1
# print(integral_int1, abserr1)
# print(integral_int2, abserr2)
# print(ratio_integral)

# Define theta and phi ranges
theta_vals = np.linspace(0, np.pi, 100)   # Theta varies from 0 to pi
phi_vals = np.linspace(0, 2 * np.pi, 100)  # Phi varies from 0 to 2*pi

# Create a meshgrid for evaluation
Theta, Phi = np.meshgrid(theta_vals, phi_vals)

# Compute function values
f_vals = np.vectorize(lambda t, p: diff.intensity_over_sphere(
    t, p, angle1, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc
))(Theta, Phi)

# Plot the function as a heatmap
plt.figure(figsize=(8, 6))
plt.pcolormesh(Phi, Theta, f_vals, shading='auto', cmap='inferno')
plt.colorbar(label="Intensity")
plt.xlabel("Phi (radians)")
plt.ylabel("Theta (radians)")
plt.title("Intensity Over Sphere")
plt.show()

two_thetas = np.linspace(43, 48, 50)
intensities = np.zeros_like(two_thetas)
start_time = time.time()
for i, two_theta in enumerate(two_thetas):
    intensity, abserr = diff._compute_intensity_integral(
        two_theta, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc
    )
    print(i, intensity, abserr, time.time() - start_time)
    intensities[i] = intensity

plt.scatter(two_thetas, intensities, s=5)
plt.plot(two_thetas, intensities, color='k', label="Intensity")
plt.axhline(0, linestyle="--", color="grey")
plt.xlabel("Scattering angle (2θ) (deg)")
plt.ylabel("Normalized intensity")
plt.title(f"{unit_cell.material} Diffraction Spectrum")
plt.legend()
plt.grid(linestyle=":")
plt.show()


# monte_ratios = []
# for i in range(100):
#     magnitude = 2 * diff.k() * np.sin(np.radians(angle1) / 2)
#     vecs = utils.random_uniform_unit_vectors(100000, 3) * magnitude
#     monte_int1 = diff._compute_intensity(
#         vecs, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc
#     )
#     magnitude = 2 * diff.k() * np.sin(np.radians(angle2) / 2)
#     vecs = utils.random_uniform_unit_vectors(100000, 3) * magnitude
#     monte_int2 = diff._compute_intensity(
#         vecs, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc
#     )
#     ratio_monte = monte_int2 / monte_int1
#     monte_ratios.append(ratio_monte)
#     print(i, ratio_monte)
#
# print(monte_ratios)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.violinplot(monte_ratios, showmeans=True, widths=0.3)
# ax.axhline(ratio_evenly_spaced, label="Ratio using evenly spaced", linestyle="--",
#             color="k")
# ax.set_ylabel("Ratios")
# ax.set_xticks([])  # Remove x-axis ticks
# plt.legend()
# plt.show()


