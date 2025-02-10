"""
This module contains code which benchmarks the get_reciprocal_lattice_vectors function.
"""

import numpy as np

from B8_project import file_reading, utils
from B8_project.archive import crystal, diffraction, super_cell

# Get a GaAs unit cell.
basis = file_reading.read_basis("data/GaAs_basis.csv")
lattice = file_reading.read_lattice("data/GaAs_lattice.csv")
GaAs_unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

# Generate GaAs super cells.
GaAs_super_cells = []

MAX_SIDE_LENGTH = 10
for side_length in range(1, MAX_SIDE_LENGTH + 1):
    GaAs_super_cell = super_cell.SuperCell.new_super_cell(
        GaAs_unit_cell, (side_length, side_length, side_length)
    ).to_unit_cell()

    GaAs_super_cells.append(GaAs_super_cell)

# Parameters for the benchmark.
min_magnitude = float(diffraction._reciprocal_lattice_vector_magnitude(20, 0.1))
max_magnitude = float(diffraction._reciprocal_lattice_vector_magnitude(60, 0.1))

# Run a benchmark for each super cell.
benchmark_data = []

for i, GaAs_super_cell in enumerate(GaAs_super_cells):
    lattice_constants = np.array(GaAs_super_cell.lattice_constants)

    reciprocal_lattice_vectors, average_time, std_dev_time = utils.benchmark_function(
        crystal.ReciprocalLatticeVector.get_reciprocal_lattice_vectors,
        min_magnitude,
        max_magnitude,
        GaAs_super_cell,
    )
    benchmark_data.append(
        (i + 1, len(reciprocal_lattice_vectors), average_time, std_dev_time)
    )

# Print the benchmark data.
print(
    "get_reciprocal_lattice_vectors (old version) benchmark. "
    "For this benchmark, a GaAs super cell was used. "
    "A list of all reciprocal lattice vectors associated with deflection angles in "
    "the range [20°, 60°] was generated."
)

for data_point in benchmark_data:
    print(
        f"Super cell side length: {data_point[0]}; "
        f"Number of reciprocal lattice vectors generated: {data_point[1]}; "
        f"Average time: {data_point[2]:.6f}; "
        f"Standard deviation in times: {data_point[3]:.6f}. "
    )
