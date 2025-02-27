"""
This script evaluates how well Monte Carlo calculation of an intensity value converges
for a particular scattering angle. This is done using the ratio of intensity of two
peaks.
"""

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

monte_ratios = []
for i in range(100):
    magnitude = 2 * diff.k() * np.sin(np.radians(angle1) / 2)
    vecs = utils.random_uniform_unit_vectors(100000, 3) * magnitude
    monte_int1 = diff._compute_intensity(
        vecs, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc
    )
    magnitude = 2 * diff.k() * np.sin(np.radians(angle2) / 2)
    vecs = utils.random_uniform_unit_vectors(100000, 3) * magnitude
    monte_int2 = diff._compute_intensity(
        vecs, nd_form_factors, unit_cell_pos, atom_pos_in_uc, atoms_in_uc
    )
    ratio_monte = monte_int2 / monte_int1
    monte_ratios.append(ratio_monte)
    print(i, ratio_monte)

print(monte_ratios)

fig, ax = plt.subplots(figsize=(6, 6))
ax.violinplot(monte_ratios, showmeans=True, widths=0.3)
ax.axhline(ratio_evenly_spaced, label="Ratio using evenly spaced", linestyle="--",
            color="k")
ax.set_ylabel("Ratios")
ax.set_xticks([])  # Remove x-axis ticks
plt.legend()
plt.show()


