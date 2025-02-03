import time
from B8_project import file_reading, crystal, diffraction, super_cell

# Get a unit cell from .csv files.
basis = file_reading.read_basis("data/GaAs_basis.csv")
lattice = file_reading.read_lattice("data/GaAs_lattice.csv")
GaAs_unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

# Specify the maximum side length
MAX_SIDE_LENGTH = 6

benchmark_data = []
for side_length in range(1, MAX_SIDE_LENGTH + 1):
    # Create a super cell.
    GaAs_super_cell = super_cell.SuperCell.new_super_cell(
        GaAs_unit_cell, (side_length, side_length, side_length)
    ).to_unit_cell()

    # pylint: disable=protected-access
    min_magnitude = diffraction._reciprocal_lattice_vector_magnitude(20, 0.1)
    max_magnitude = diffraction._reciprocal_lattice_vector_magnitude(90, 0.1)
    # pylint: enable=protected-access

    # Start the timer
    start_time = time.time()

    reciprocal_lattice_vectors_long = (
        crystal.ReciprocalLatticeVector.get_reciprocal_lattice_vectors(
            min_magnitude, max_magnitude, GaAs_super_cell
        )
    )

    # Stop the timer
    stop_time = time.time()

    time_long_list = stop_time - start_time

    # Start the timer
    start_time = time.time()

    reciprocal_lattice_vectors_short = (
        crystal.ReciprocalLatticeVector.get_magnitudes_and_multiplicities(
            min_magnitude, max_magnitude, GaAs_super_cell
        )
    )

    # Stop the timer
    stop_time = time.time()

    time_short_list = stop_time - start_time

    # Append the data to benchmark_data
    benchmark_data.append(
        (
            side_length,
            time_long_list,
            time_short_list,
            len(reciprocal_lattice_vectors_long),
            len(reciprocal_lattice_vectors_short),
        )
    )

for (
    side_length,
    time_long_list,
    time_short_list,
    vector_count_long,
    vector_count_short,
) in benchmark_data:
    print(
        f"Side Length: {side_length}; "
        f"Initial number of RLVs: {vector_count_long}; "
        f"Time to get initial RLVS: {time_long_list:.3f}; "
        f"Final number of RLVs: {vector_count_short}; "
        f"Time to get final RLVS: {time_short_list:.3f}."
    )
