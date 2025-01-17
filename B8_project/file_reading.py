"""
pandas is used to read parameters from .csv files.
TODO: add unit tests for functions in this module.
"""

import pandas as pd
from B8_project.unit_cell import XRayFormFactor


def get_lattice_from_csv(
    filename: str,
) -> tuple[str, int, tuple[float, float, float]]:
    """
    Get lattice parameters from a CSV file
    ======================================

    Format of the CSV file
    ----------------------
    The CSV file must contain the following columns:
        - "material" (str): Chemical formula of the crystal (e.g. "NaCl").
        - "lattice_type" (int): Integer (1 - 4 inclusive) that represents the Bravais lattice type.
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
    try:
        lattice_df = pd.read_csv(filename)
    except ValueError as exc:
        raise ValueError(f"Error reading '{filename}' into DataFrame") from exc

    # Expected columns of the lattice CSV file
    lattice_columns = {"material", "lattice_type", "a", "b", "c"}

    # Ensure the CSV file contains the required columns.
    if not lattice_columns.issubset(lattice_df.columns):
        raise ValueError(
            f"The {filename} must contain the following columns: {lattice_columns}"
        )

    # Read the material from the DataFrame.
    try:
        material = str(lattice_df.loc[0, "material"])
    except Exception as exc:
        raise ValueError(f"Error reading material from {filename}.") from exc

    # Read the lattice_type from the DataFrame.
    try:
        lattice_type = int(pd.to_numeric(lattice_df.loc[0, "lattice_type"]))
    except ValueError as exc:
        raise ValueError(f"Error reading lattice type from {filename}.") from exc

    # Read lattice constants from the DataFrame.
    try:
        lattice_constants = (
            float(pd.to_numeric(lattice_df.loc[0, "a"])),
            float(pd.to_numeric(lattice_df.loc[0, "b"])),
            float(pd.to_numeric(lattice_df.loc[0, "c"])),
        )
    except ValueError as exc:
        raise ValueError(f"Error reading lattice constants from {filename}.") from exc

    # Return the lattice parameters.
    return material, lattice_type, lattice_constants


def get_basis_from_csv(
    filename: str,
) -> tuple[list[int], list[tuple[float, float, float]]]:
    """
    Get basis parameters from a CSV file
    ====================================

    Format of the CSV file
    ----------------------
    Each row of the CSV file corresponds to a different atom in the unit cell. The CSV
    file must contain the following columns:
        - "atomic_number" (int): The atomic number of the atom.
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
        - atomic_positions (list[tuple[float, float, float]]): A list of atomic
        positions for each atom in the unit cell. The position of each atom is
        specified in terms of the lattice constants as a tuple (x, y, z) of floats.
    """
    # Read the CSV file containing the lattice parameters into a DataFrame.
    try:
        basis_df = pd.read_csv(filename)
    except ValueError as exc:
        raise ValueError(f"Error reading '{filename}' into DataFrame") from exc

    # Expected columns of the basis CSV file
    basis_columns = {"atomic_number", "x", "y", "z"}

    # Ensure the CSV file contains the required columns.
    if not basis_columns.issubset(basis_df.columns):
        raise ValueError(
            f"The {filename} must contain the following columns: {basis_columns}"
        )

    # Read the atomic numbers from the DataFrame.
    try:
        atomic_numbers = basis_df["atomic_number"].astype(int).tolist()
    except ValueError as exc:
        raise ValueError(f"Error reading atomic numbers from {filename}.") from exc

    # Read the atomic positions from the DataFrame.
    try:
        atomic_positions = list(
            zip(
                basis_df["x"].astype(float).tolist(),
                basis_df["y"].astype(float).tolist(),
                basis_df["z"].astype(float).tolist(),
            )
        )
    except ValueError as exc:
        raise ValueError(f"Error reading atomic positions from {filename}.") from exc

    # Check that atomic_numbers and atomic_positions have the same length
    if not len(atomic_numbers) == len(atomic_positions):
        raise ValueError(
            """atomic_numbers and atomic_positions must have the same length"""
        )

    # Return the basis parameters.
    return atomic_numbers, atomic_positions


def validate_parameters(
    lattice: tuple[str, int, tuple[float, float, float]],
    basis: tuple[list[int], list[tuple[float, float, float]]],
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

    Todos
    -----
    TODO: add validation
    """
    _, lattice_type, lattice_constants = lattice
    atomic_numbers, atomic_positions = basis

    # Validate that the length of atomic_numbers and atomic_positions is the same.
    if not len(atomic_numbers) == len(atomic_positions):
        raise ValueError(
            """Length of atomic_numbers and atomic_positions must be the same"""
        )

    # Validate that the lattice constants are non-negative and non-zero.
    (a, b, c) = lattice_constants
    if not (a > 0 and b > 0 and c > 0):
        raise ValueError(
            """Lattice constants should all be non-negative and non-zero."""
        )

    # Validate lattice_type
    if lattice_type < 1 or lattice_type > 4:
        raise ValueError(
            """lattice_type should be an integer between 1 and 4 inclusive"""
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


def get_neutron_form_factors_from_csv(filename: str) -> dict[int, float]:
    """
    Get neutron form factors from a CSV file
    ========================================

    TODO: add description.

    Format of the CSV file
    ----------------------
    TODO: add format of the CSV file.

    Parameters
    ----------
        - filename (str): The path to the CSV file containing the neutron form factors.

    Returns
    -------
    TODO: add returns.
    """
    # Read the CSV file containing the neutron form factors into a DataFrame.
    try:
        neutron_df = pd.read_csv(filename)
    except ValueError as exc:
        raise ValueError(f"Error reading '{filename}' into DataFrame") from exc

    # Expected columns of the neutron form factors CSV file
    neutron_columns = {"atomic_number", "form_factor"}

    # Ensure the CSV file contains the required columns.
    if not neutron_columns.issubset(neutron_df.columns):
        raise ValueError(
            f"The {filename} must contain the following columns: {neutron_columns}"
        )

    # Read the atomic numbers
    try:
        atomic_numbers = neutron_df["atomic_number"].astype(int).tolist()
    except ValueError as exc:
        raise ValueError(f"Error reading atomic numbers from {filename}") from exc

    # Read the neutron form factors
    try:
        neutron_form_factors = neutron_df["form_factor"].astype(float).tolist()
    except ValueError as exc:
        raise ValueError(f"Error reading form factors from {filename}") from exc

    # Validate that atomic_numbers and form_factors have the same length
    if not len(atomic_numbers) == len(neutron_form_factors):
        raise ValueError("atomic_numbers and form_factors must have the same length")

    # Create a dictionary containing the neutron form factors
    length = len(atomic_numbers)
    return {atomic_numbers[i]: neutron_form_factors[i] for i in range(length)}


def get_x_ray_form_factors_from_csv(
    filename: str,
) -> dict[int, XRayFormFactor]:
    """
    Get X-ray form factors from a CSV file
    ========================================

    TODO: add description.

    Format of the CSV file
    ----------------------
    TODO: add format of the CSV file.

    Parameters
    ----------
        - filename (str): The path to the CSV file containing the neutron form factors.

    Returns
    -------
    TODO: add returns.
    """
    # Read the CSV file containing the X-ray form factors into a DataFrame.
    try:
        xray_df = pd.read_csv(filename)
    except ValueError as exc:
        raise ValueError(f"Error reading '{filename}' into DataFrame") from exc

    # Expected columns of the X-ray form factors CSV file
    xray_columns = {
        "atomic_number",
        "a1",
        "b1",
        "a2",
        "b2",
        "a3",
        "b3",
        "a4",
        "b4",
        "c",
    }

    # Ensure the CSV file contains the required columns.
    if not xray_columns.issubset(xray_df.columns):
        raise ValueError(
            f"The {filename} must contain the following columns: {xray_columns}"
        )

    # Read the atomic numbers
    try:
        atomic_numbers = xray_df["atomic_number"].astype(int).tolist()
    except ValueError as exc:
        raise ValueError(f"Error reading atomic numbers from {filename}") from exc

    # Read the X-ray form factors
    try:
        xray_form_factors = [
            XRayFormFactor(a1, b1, a2, b2, a3, b3, a4, b4, c)
            for a1, b1, a2, b2, a3, b3, a4, b4, c in zip(
                xray_df["a1"].astype(float).tolist(),
                xray_df["b1"].astype(float).tolist(),
                xray_df["a2"].astype(float).tolist(),
                xray_df["b2"].astype(float).tolist(),
                xray_df["a3"].astype(float).tolist(),
                xray_df["b3"].astype(float).tolist(),
                xray_df["a4"].astype(float).tolist(),
                xray_df["b4"].astype(float).tolist(),
                xray_df["c"].astype(float).tolist(),
            )
        ]
    except ValueError as exc:
        raise ValueError(f"Error reading form factors from {filename}") from exc

    # Validate that atomic_numbers and form_factors have the same length
    if not len(atomic_numbers) == len(xray_form_factors):
        raise ValueError("atomic_numbers and form_factors must have the same length")

    # Create a dictionary containing the neutron form factors
    length = len(atomic_numbers)
    return {atomic_numbers[i]: xray_form_factors[i] for i in range(length)}
