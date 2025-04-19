import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from B8_project import file_reading
import B8_project.crystal as unit_cell
from B8_project.diffraction_monte_carlo import RefinementIteration, UniformSettings, \
    NeighborhoodSettings, UniformPrunedSettings
from B8_project.mc_ideal_crystal import MCIdealCrystal

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
    LATTICE_FILE = "data/CsPbBr3_cubic_lattice.csv"
    BASIS_FILE = "data/CsPbBr3_cubic_basis.csv"

    lattice = file_reading.read_lattice(LATTICE_FILE)
    basis = file_reading.read_basis(BASIS_FILE)

    unit_cell = unit_cell.UnitCell.new_unit_cell(basis, lattice)

    # CONC = 0.
    # lat = 5.94863082 + CONC * (6.27514229 - 5.94863082)
    # unit_cell.lattice_constants = (lat, lat, lat)
    # print("Lattice constants:", unit_cell.lattice_constants)

    diff = MCIdealCrystal(1.54,
                          unit_cell,
                          min_angle_deg=5,
                          max_angle_deg=179)

    start_time = time.time()

    # atom_from, atom_to, prob = 35, 53, CONC

    iterations = []
    iterations.append(RefinementIteration(
        setup=lambda: diff.setup_spherical_crystal(20),
        settings=UniformSettings(
            total_trials=10_000_000,
            angle_bins=1000,
            threshold=0.003
        )
    ))
    iterations.append(RefinementIteration(
        setup=lambda: diff.setup_spherical_crystal(40),
        settings=NeighborhoodSettings(
            sigma=0.02,
            cnt_per_point=10,
            threshold=0.003
        )
    ))
    iterations.append(RefinementIteration(
        setup=lambda: diff.setup_spherical_crystal(70),
        settings=UniformPrunedSettings(
            dist=0.02,
            total_trials=2_000_000,
            trials_per_batch=5_000,
            threshold=0.005
        )
    ))

    form_factors = diff.all_xray_form_factors
    two_thetas, intensities = diff.spectrum_iterative_refinement(
        form_factors,
        iterations,
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
plt.title(f"{diff._unit_cell.material} "
          f"{'XRD' if form_factors is diff.all_xray_form_factors else 'ND'} Spectrum")
plt.legend()
plt.grid(linestyle=":")
plt.show()
