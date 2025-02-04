"""
This module contains benchmarks for the crystal file.
"""

import numpy as np
from B8_project import file_reading, crystal, diffraction, super_cell
from benchmarks import benchmark

# Get a unit cell from .csv files.
basis = file_reading.read_basis("data/GaAs_basis.csv")
lattice = file_reading.read_lattice("data/GaAs_lattice.csv")
GaAs_unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)


## Benchmark the new_unit_cell function
unit_cell, average_time, std_dev_time = benchmark.benchmark_function(
    crystal.UnitCell.new_unit_cell, basis, lattice, number_of_runs=5
)

new_unit_cell_benchmark_data = (len(unit_cell.atoms), average_time, std_dev_time)


## Benchmark the get_reciprocal_lattice_vectors function from the
## ReciprocalLatticeVector class.
get_reciprocal_lattice_vectors_benchmark_data = []

# Specify the maximum super cell side length
MAX_SIDE_LENGTH = 3

for side_length in range(1, MAX_SIDE_LENGTH + 1):
    # Create a super cell.
    GaAs_super_cell = super_cell.SuperCell.new_super_cell(
        GaAs_unit_cell, (side_length, side_length, side_length)
    ).to_unit_cell()

    # pylint: disable=protected-access
    min_magnitude = diffraction._reciprocal_lattice_vector_magnitude(20, 0.1)
    max_magnitude = diffraction._reciprocal_lattice_vector_magnitude(90, 0.1)
    # pylint: enable=protected-access

    # Benchmark the get_reciprocal_lattice_vectors function.
    reciprocal_lattice_vectors, average_time, std_dev_time = (
        benchmark.benchmark_function(
            crystal.ReciprocalLatticeVector.get_reciprocal_lattice_vectors,
            min_magnitude,
            max_magnitude,
            GaAs_super_cell,
            number_of_runs=5,
        )
    )

    get_reciprocal_lattice_vectors_benchmark_data.append(
        (side_length, len(reciprocal_lattice_vectors), average_time, std_dev_time)
    )


## Benchmark the _get_reciprocal_lattice_vectors function from the
## ReciprocalLatticeVectorV2 class.
get_reciprocal_lattice_vectors_v2_benchmark_data = []

for side_length in range(1, MAX_SIDE_LENGTH + 1):
    # Create a super cell.
    GaAs_super_cell = super_cell.SuperCell.new_super_cell(
        GaAs_unit_cell, (side_length, side_length, side_length)
    ).to_unit_cell()

    lattice_constants = np.array(GaAs_super_cell.lattice_constants)

    # pylint: disable=protected-access
    min_magnitude = diffraction._reciprocal_lattice_vector_magnitude(20, 0.1)
    max_magnitude = diffraction._reciprocal_lattice_vector_magnitude(90, 0.1)

    # Benchmark the get_reciprocal_lattice_vectors function.
    reciprocal_lattice_vectors, average_time, std_dev_time = (
        benchmark.benchmark_function(
            crystal.ReciprocalLatticeVectorV2.get_reciprocal_lattice_vectors,
            min_magnitude,
            max_magnitude,
            lattice_constants,
            number_of_runs=5,
        )
    )
    # pylint: enable=protected-access

    get_reciprocal_lattice_vectors_v2_benchmark_data.append(
        (side_length, len(reciprocal_lattice_vectors), average_time, std_dev_time)
    )

# Print the benchmark data for new_unit_cell.
print(
    f"New unit cell benchmark data:\n"
    f"Number of atoms in unit cell: {new_unit_cell_benchmark_data[0]}; "
    f"Average time: {new_unit_cell_benchmark_data[1]:.3f}; "
    f"Standard deviation in time: {new_unit_cell_benchmark_data[2]:.3f}"
)

# Print the benchmark data for get_reciprocal_lattice_vectors.
print("\nGet reciprocal lattice vectors benchmark data:")
for (
    side_length,
    num_reciprocal_lattice_vectors,
    average_time,
    std_dev_time,
) in get_reciprocal_lattice_vectors_benchmark_data:
    print(
        f"Side Length: {side_length}; "
        f"Number of RLVs: {num_reciprocal_lattice_vectors}; "
        f"Average time: {average_time:.3f}; "
        f"Standard deviation in time: {std_dev_time:.3f}; "
    )

# Print the benchmark data for get_reciprocal_lattice_vectors_v2.
print("\nGet reciprocal lattice vectors v2 benchmark data:")
for (
    side_length,
    num_reciprocal_lattice_vectors,
    average_time,
    std_dev_time,
) in get_reciprocal_lattice_vectors_v2_benchmark_data:
    print(
        f"Side Length: {side_length}; "
        f"Number of RLVs: {num_reciprocal_lattice_vectors}; "
        f"Average time: {average_time:.3f}; "
        f"Standard deviation in time: {std_dev_time:.3f}; "
    )
