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
unit_cell_pos = diff._unit_cell_positions((10, 10, 10))
atoms_in_uc, atom_pos_in_uc = diff._atoms_and_pos_in_uc()

# peaks = diff.get_peaks(44.4, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc)
# print(peaks)
# intensities = []
# for i, (theta, phi) in enumerate(peaks):
#     intensities.append(diff.intensity_over_sphere(theta, phi, 44.4,
#                                              nd_form_factors, unit_cell_pos,
#                                                   atom_pos_in_uc, atoms_in_uc))
# plt.plot(intensities[:100])
# plt.yscale("log")
# plt.show()

two_thetas = np.linspace(43, 48, 50)
intensities = np.zeros_like(two_thetas)
start_time = time.time()
for i, two_theta in enumerate(two_thetas):
    intensity = diff._compute_intensity_subdomain(two_theta, nd_form_factors,
                                                  unit_cell_pos, atom_pos_in_uc, atoms_in_uc)
    print(i, intensity, "time remaining", (time.time() - start_time) /
          (i + 1) * (50 - i - 1))
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


