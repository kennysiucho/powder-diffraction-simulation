# TODO: add unit tests for functions in this module.

"""
pandas is used to extract parameters from .csv files.
"""

import pandas as pd

# Required columns for the lattice and basis csv files
LATTICE_REQUIRED_COLUMNS = {"material", "lattice_type", "a", "b", "c"}
BASIS_REQUIRED_COLUMNS = {"atomic_number", "atomic_mass", "x", "y", "z"}


def get_lattice_from_csv(
    filename: str,
) -> tuple[str, int, tuple[float, float, float]]:
    """
    Extract lattice parameters from a CSV file
    ==========================================

    Format of the CSV file
    ----------------------
    The CSV file must contain the following columns:
        - "material" (str): Chemical formula of the crystal (e.g. "NaCl").
        - "lattice_type" (int): Integer (1 - 9 inclusive) that represents the Bravais lattice type.
            - 1 -> Simple.
            - 2 -> Body centred.
            - 3 -> Face centred.
            - 4 -> Base centred.
        - "a", "b", "c" (float): Side lengths of the unit cell in the x, y and z
        directions respectively in Angstroms (Å).

    Parameters
    ----------
        - filename (str): The path to the CSV file containing the lattice parameters.

    Returns
    -------
        - material (str): The material name.
        - lattice_type (int): The Bravais lattice type.
        - lattice_constants (tuple[float, float, float]): The side lengths (a, b, c) of
        the unit cell in the x, y, z directions respectively.
    """
    # Read the CSV file containing the lattice parameters into a DataFrame.
    lattice_df = pd.read_csv(filename)

    # Ensure the CSV file contains the required columns.
    if not LATTICE_REQUIRED_COLUMNS.issubset(lattice_df.columns):
        raise ValueError(
            f"The CSV file must contain the following columns: {LATTICE_REQUIRED_COLUMNS}"
        )

    # Extract the material from the DataFrame.
    try:
        material = str(lattice_df.loc[0, "material"])
    except Exception as exc:
        raise ValueError("Error extracting material.") from exc

    # Extract the lattice_type from the DataFrame.
    try:
        lattice_type = int(pd.to_numeric(lattice_df.loc[0, "lattice_type"]))
    except ValueError as exc:
        raise ValueError("Error extracting lattice type.") from exc

    # Extract lattice constants from the DataFrame.
    try:
        a = float(pd.to_numeric(lattice_df.loc[0, "a"]))
        b = float(pd.to_numeric(lattice_df.loc[0, "b"]))
        c = float(pd.to_numeric(lattice_df.loc[0, "c"]))
    except ValueError as exc:
        raise ValueError("Error extracting lattice constants.") from exc

    lattice_constants = (a, b, c)

    # Return the lattice parameters.
    return material, lattice_type, lattice_constants


def get_basis_from_csv(
    filename: str,
) -> tuple[list[int], list[float], list[tuple[float, float, float]]]:
    """
    Extract basis parameters from a CSV file
    ========================================

    Format of the CSV file
    ----------------------
    Each row of the CSV file corresponds to a different atom in the unit cell. The CSV
    file must contain the following columns:
        - "atomic_number" (int): The atomic number of the atom.
        - "atomic_mass" (float): The atomic mass of the atom, in atomic mass units
        (AMU).
        - "x", "y", "z" (float): The position of the atom in the unit cell in terms of
        the lattice constants. E.g., (x, y, z) = (0.5, 0.5, 0.5) corresponds to a
        position (0.5*a, 0.5*b. 0.5*c), where a, b, c are the side lengths of the unit
        cell in the x, y, z directions respectively.

    Parameters
    ----------
        - filename (str): The path to the CSV file containing the lattice parameters.

    Returns
    -------
        - atomic_numbers(list[int]): A list of atomic numbers for each atom in the unit cell.
        - atomic_masses(list[int]): A list of atomic masses for each atom in the unit
        cell, in atomic mass units (AMU).
        - atomic_positions (list[tuple[float, float, float]]): A list of atomic
        positions for each atom in the unit cell. The position of each atom is
        specified in terms of the lattice constants as a tuple (x, y, z) of floats.
    """
    # Read the CSV file containing the lattice parameters into a DataFrame.
    basis_df = pd.read_csv(filename)

    # Ensure the CSV file contains the required columns.
    if not BASIS_REQUIRED_COLUMNS.issubset(basis_df.columns):
        raise ValueError(
            f"The CSV file must contain the following columns: {BASIS_REQUIRED_COLUMNS}"
        )

    # Extract the atomic numbers from the DataFrame.
    try:
        atomic_numbers = basis_df["atomic_number"].astype(int).tolist()
    except ValueError as exc:
        raise ValueError("Error extracting atomic numbers.") from exc

    # Extract the atomic masses from the DataFrame.
    try:
        atomic_masses = basis_df["atomic_mass"].astype(float).tolist()
    except ValueError as exc:
        raise ValueError("Error extracting atomic masses.") from exc

    # Extract the atomic positions from the DataFrame.
    try:
        x_positions = basis_df["x"].astype(float).tolist()
        y_positions = basis_df["y"].astype(float).tolist()
        z_positions = basis_df["z"].astype(float).tolist()
    except ValueError as exc:
        raise ValueError("Error extracting atomic positions.") from exc

    atomic_positions = list(zip(x_positions, y_positions, z_positions))

    # Return the basis parameters.
    return atomic_numbers, atomic_masses, atomic_positions


def validate_parameters(
    lattice: tuple[str, int, tuple[float, float, float]],
    basis: tuple[list[int], list[float], list[tuple[float, float, float]]],
):
    """
    Validate parameters
    ===================

    Processes lattice and basis parameters, and returns an error if they are invalid.

    Parameters
    ----------
    TODO: add parameters.

    Returns
    -------
    TODO: add returns
    """
    _, lattice_type, lattice_constants = lattice
    atomic_numbers, atomic_masses, atomic_positions = basis

    # Validate that the length of atomic_numbers, atomic_masses and
    # atomic_positions is the same.
    if not len(atomic_numbers) == len(atomic_masses) == len(atomic_positions):
        raise ValueError(
            """Length of atomic_numbers, atomic_masses and atomic_positions must be the same"""
        )

    # Validate lattice_type
    if lattice_type < 1 or lattice_type > 4:
        raise ValueError(
            """lattice_type should be an integer between 1 and 4 inclusive"""
        )

    (a, b, c) = lattice_constants

    # Validate that the lattice constants are non-negative and non-zero.
    if not (a > 0 and b > 0 and c > 0):
        raise ValueError(
            """Lattice constants should all be non-negative and non-zero."""
        )

    # Validate that lattice_type and lattice_constants are compatible for a cubic
    # unit cell
    if (a == b == c) and lattice_type == 4:
        raise ValueError(
            """Base centred lattice type is not permitted for a cubic lattice"""
        )

    # Validate that lattice_type and lattice_constants are compatible for a
    # tetragonal unit cell
    if (a == b and not a == c) or (a == c and not a == b) or (b == c and not a == b):
        if lattice_type == 3:
            raise ValueError(
                """Face centred lattice type is not permitted for a tetragonal lattice"""
            )
        elif lattice_type == 4:
            raise ValueError(
                """Base centred lattice type is not permitted for a tetragonal unit cell"""
            )
