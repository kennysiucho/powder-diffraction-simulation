import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from B8_project import file_reading
from B8_project.crystal import UnitCell
from B8_project.diffraction_monte_carlo import DiffractionMonteCarlo

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["cmss10"],
    "font.size": 14,
})

LATTICE_FILE = "data/GaAs_lattice.csv"
BASIS_FILE = "data/GaAs_basis.csv"

lattice = file_reading.read_lattice(LATTICE_FILE)
basis = file_reading.read_basis(BASIS_FILE)

unit_cell = UnitCell.new_unit_cell(basis, lattice)
diff = DiffractionMonteCarlo(unit_cell,
                                    0.123)

all_nd_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv")
nd_form_factors = {}
for atom in diff.unit_cell.atoms:
    nd_form_factors[atom.atomic_number] = all_nd_form_factors[atom.atomic_number]
nd_form_factors[49] = all_nd_form_factors[49]

all_xray_form_factors = file_reading.read_xray_form_factors(
    "data/x_ray_form_factors.csv")
xrd_form_factors = {}
for atom in diff.unit_cell.atoms:
    xrd_form_factors[atom.atomic_number] = all_xray_form_factors[atom.atomic_number]
xrd_form_factors[49] = all_xray_form_factors[49]

# exact peak = 35.8379, 42.2958
vecs, _ = diff._get_uniform_scattering_vecs_and_angles_single(10000, 35.8379)

# Prepare crystal
unit_cell_pos = diff._unit_cell_positions((10, 10, 10))
all_atoms = []
all_atom_pos = []
for uc_pos in unit_cell_pos:
    for atom in diff.unit_cell.atoms:
        all_atoms.append(atom.atomic_number)
        all_atom_pos.append(uc_pos + np.array(atom.position) *
                            diff.unit_cell.lattice_constants)
all_atoms = np.array(all_atoms)
all_atom_pos = np.array(all_atom_pos)

# Evaluate intensities
def evaluate_intensities(form_factors):
    dot_products = np.einsum("ik,jk", vecs, all_atom_pos)
    form_factors_evaluated = {}
    for atomic_number, form_factor in form_factors.items():
        form_factors_evaluated[atomic_number] = (
            form_factor.evaluate_form_factors(vecs))
    all_form_factors = np.array([form_factors_evaluated[z] for z in
                                 all_atoms]).T
    exps = np.exp(1j * dot_products)
    exp_terms = np.multiply(all_form_factors, exps)
    structure_factors = np.sum(exp_terms, axis=1)
    intensities = np.abs(structure_factors)**2
    return intensities

intensities = evaluate_intensities(xrd_form_factors)
# intensities /= np.max(intensities)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=15)
sc = ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2],
                c=intensities, cmap='YlOrRd', s=10,
                norm=PowerNorm(gamma=0.25, vmin=intensities.min(), vmax=intensities.max()))
ax.set_box_aspect([1,1,1])
plt.colorbar(sc, ax=ax, label=r"$I(\mathbf{Q})$", shrink=0.7, pad=0.04)

plt.tight_layout()
plt.show()

