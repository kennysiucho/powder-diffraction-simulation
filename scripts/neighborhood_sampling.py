import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from B8_project import file_reading
import B8_project.crystal as unit_cell
from B8_project.diffraction_monte_carlo import DiffractionMonteCarlo

two_thetas_file = Path("two_thetas.txt")
intensities_file = Path("intensities.txt")
if two_thetas_file.is_file() or intensities_file.is_file():
    while True:
        res = input(
                "Previous scattering intensity data found in 'two_thetas.txt' or "
                "'intensities.txt'. Plot existing data? [y/n]")
        if res.lower() == "y":
            CALCULATE_SPECTRUM = False
            break
        if res.lower() == "n":
            CALCULATE_SPECTRUM = True
            break
else:
    CALCULATE_SPECTRUM = True

if CALCULATE_SPECTRUM:
    LATTICE_FILE = "data/CsPbBr3_lattice.csv"
    BASIS_FILE = "data/CsPbBr3_basis.csv"

    lattice = file_reading.read_lattice(LATTICE_FILE)
    basis = file_reading.read_basis(BASIS_FILE)

    unit_cell = unit_cell.UnitCell.new_unit_cell(basis, lattice)

    CONC = 0.
    lat = 5.94863082 + CONC * (6.27514229 - 5.94863082)
    unit_cell.lattice_constants = (lat, lat, lat)
    print("Lattice constants:", unit_cell.lattice_constants)

    diff = DiffractionMonteCarlo(unit_cell,
                                 1.54,
                                 min_angle_deg=25,
                                 max_angle_deg=42)

    # TODO: improve UI for reading form factors
    # all_nd_form_factors = file_reading.read_neutron_scattering_lengths(
    #     "data/neutron_scattering_lengths.csv")
    #     # nd_form_factors = {}
    #     # for atom in diff.unit_cell.atoms:
    #     #     nd_form_factors[atom.atomic_number] = all_nd_form_factors[atom.atomic_number]
    # nd_form_factors[49] = all_nd_form_factors[49]

    all_xray_form_factors = file_reading.read_xray_form_factors(
        "data/x_ray_form_factors.csv")
    xrd_form_factors = {}
    for atom in diff.unit_cell.atoms:
        xrd_form_factors[atom.atomic_number] = all_xray_form_factors[atom.atomic_number]
    xrd_form_factors[53] = all_xray_form_factors[53]


    start_time = time.time()

    atom_from, atom_to, prob = 35, 53, CONC

    two_thetas, intensities = diff.calculate_neighborhood_diffraction_pattern_random_occupation(
        atom_from, atom_to, prob,
        xrd_form_factors,
        angle_bins=200,
        brute_force_uc_reps=(4, 4, 4),
        neighbor_uc_reps=(8, 8, 8),
        brute_force_trials=1_000_000,
        num_top=10000,
        resample_cnt=40,
        plot_diagnostics=True
    )

    # print("Total neighbors sampled =", np.sum(counts_neigh))
    print("Total run time =", time.time() - start_time, "s")

    np.savetxt('two_thetas.txt', two_thetas)
    np.savetxt('intensities.txt', intensities)
else:
    two_thetas = np.loadtxt("two_thetas.txt")
    intensities = np.loadtxt("intensities.txt")

# two_thetas = two_thetas[::2]
# intensities = (intensities[::2] + intensities[1::2]) / 2

plt.scatter(two_thetas, intensities, s=3)
plt.plot(two_thetas, intensities, color='k', label="Intensity")
# plt.plot(two_thetas, pdf(two_thetas) / np.max(pdf(two_thetas)), "--", label="PDF")
plt.ylim(bottom=0)
plt.xlabel("Scattering angle (2θ) (deg)")
plt.ylabel("Intensity")
# plt.title("In_0.25Ga_0.75As Neutron Diffraction Spectrum")
plt.title("CsPbBr3 XRD Diffraction Spectrum")
plt.legend()
plt.grid(linestyle=":")
plt.show()
