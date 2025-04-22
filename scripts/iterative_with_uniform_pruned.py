import inspect
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from B8_project import file_reading
import B8_project.crystal as unit_cell
from B8_project.diffraction_monte_carlo import RefinementIteration, UniformSettings, \
    NeighborhoodSettings, UniformPrunedSettings
from B8_project.mc_displacement import MCDisplacement
from B8_project.mc_ideal_crystal import MCIdealCrystal

spectrum_file = Path("spectrum.csv")
if spectrum_file.is_file():
    while True:
        res = input(
                "Previous spectrum data found in 'spectrum.csv'. Plot existing data? [y/n]")
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

    def setup():
        mc = MCDisplacement(1.54,
                              unit_cell,
                              35, 53, 0.0,
                              displace_func=lambda pos, uc: (
                                  MCDisplacement.gaussian_displaced(
                                      pos, uc, sigma=0.2, atoms_to_displace=[35, 53, 55, 82])
                              ),
                              min_angle_deg=10,
                              max_angle_deg=90)
        return mc

    diff = setup()

    start_time = time.time()

    # atom_from, atom_to, prob = 35, 53, CONC

    iterations = []
    iterations.append(RefinementIteration(
        setup=lambda: diff.setup_spherical_crystal(30),
        settings=UniformSettings(
            total_trials=5_000_000,
            angle_bins=1000,
            threshold=0.1,
            weighted=True
        )
    ))
    # iterations.append(RefinementIteration(
    #     setup=lambda: diff.setup_spherical_crystal(50),
    #     settings=NeighborhoodSettings(
    #         sigma=0.05,
    #         cnt_per_point=10,
    #         threshold=0.002
    #     )
    # ))
    # iterations.append(RefinementIteration(
    #     setup=lambda: diff.setup_spherical_crystal(60),
    #     settings=UniformPrunedSettings(
    #         dist=0.03,
    #         num_cells=(200, 200, 200),
    #         total_trials=200_000,
    #         trials_per_batch=5_000,
    #         threshold=0.005,
    #     )
    # ))

    form_factors = diff.all_xray_form_factors
    two_thetas, intensities = diff.spectrum_iterative_refinement(
        form_factors,
        iterations,
        plot_diagnostics=True
    )

    print("Total run time =", time.time() - start_time, "s")

    data = np.column_stack((two_thetas, intensities))
    with open("spectrum.csv", "w") as f:
        # Metadata
        material = diff._unit_cell.material
        diff_type = 'XRD' if form_factors is diff.all_xray_form_factors else 'ND'
        f.write(f"#META {material}\n")
        f.write(f"#META {diff_type}\n")
        # Settings
        f.write(f"#SETTINGS\n")
        class_params = inspect.getsource(setup)
        settings_block = "\n".join(f"# {line}" for line in class_params.splitlines())
        for it in iterations:
            settings_block += "\n"
            settings_block += "\n".join(f"# {line}" for line in it.__str__().splitlines())
        f.write(settings_block + "\n")
        np.savetxt(f, data, delimiter=",", header="two theta,intensity", comments="")
else:
    metadata = []
    with open("spectrum.csv", "r") as f:
        for line in f:
            if line.startswith("#META"):
                metadata.append(line[6:].strip())
            elif not line.startswith("#"):
                break  # First line of data reached
    material = metadata[0]
    diff_type = metadata[1]
    two_thetas, intensities = np.genfromtxt("spectrum.csv", delimiter=",", skip_header=1).T

plt.scatter(two_thetas, intensities, s=3)
plt.plot(two_thetas, intensities, color='k', label="Intensity")
plt.ylim(bottom=0)
plt.xlabel("Scattering angle (2θ) (deg)")
plt.ylabel("Intensity")
plt.title(f"{material} {diff_type} Spectrum")
plt.legend()
plt.grid(linestyle=":")
plt.show()
