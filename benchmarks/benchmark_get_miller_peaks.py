"""
This module contains code which benchmarks the get_reciprocal_lattice_vectors function.
"""

import numpy as np

from B8_project import (
    alloy,
    crystal,
    file_reading,
    utils,
    diffraction,
)

# Get a GaAs unit cell.
basis = file_reading.read_basis("data/GaAs_basis.csv")
lattice = file_reading.read_lattice("data/GaAs_lattice.csv")
GaAs_unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

# Read X-ray form factors.
x_ray_form_factors = file_reading.read_xray_form_factors("data/x_ray_form_factors.csv")
neutron_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv"
)


# Generate GaAs super cells.
GaAs_super_cells = []

MAX_SIDE_LENGTH = 5
for side_length in range(1, MAX_SIDE_LENGTH + 1):
    GaAs_super_cell = alloy.SuperCell.new_super_cell(
        GaAs_unit_cell, (side_length, side_length, side_length), "GaAs"
    )

    GaAs_super_cells.append(GaAs_super_cell)

# Run a benchmark for each super cell.
benchmark_data = []

for i, GaAs_super_cell in enumerate(GaAs_super_cells):
    lattice_constants = np.array(GaAs_super_cell.lattice_constants)

    diffraction_peaks, average_time, std_dev_time = utils.benchmark_function(
        diffraction.get_miller_peaks,
        GaAs_super_cell,
        "ND",
        neutron_form_factors,
        x_ray_form_factors,
        0.1,
        min_deflection_angle=20,
        max_deflection_angle=60,
        intensity_cutoff=1e-6,
        print_peak_data=False,
        save_to_csv=False,
        number_of_runs=5,
    )
    benchmark_data.append((i + 1, average_time, std_dev_time))

# Print the benchmark data.
print(
    "_calculate_diffraction_peaks benchmark. "
    "For this benchmark, a GaAs super cell was used. "
    "A list of all diffraction peaks for deflection angles in the range [20°, 60°] was "
    "generated."
)

for data_point in benchmark_data:
    print(
        f"Super cell side length: {data_point[0]}; "
        f"Average time: {data_point[1]:.6f}; "
        f"Standard deviation in times: {data_point[2]:.6f}. "
    )
