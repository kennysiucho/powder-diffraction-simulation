"""
pandas is used to extract parameters from .csv files.
"""

import pandas as pd

# Required columns for the lattice and basis csv files
LATTICE_REQUIRED_COLUMNS = {"material", "lattice_type", "a", "b", "c"}
BASIS_REQUIRED_COLUMNS = {"atomic_number", "atomic_mass", "x", "y", "z"}


def extract_lattice_parameters_from_csv(filename: str):
    """
    Extracts the lattice parameters from a CSV file. The CSV file should be formatted
    with columns: "material", "lattice_type", "a", "b", and "c".

    "material" should typically be the chemical formula of the crystal, inputted as a
    string (e.g. "NaCl").

    "lattice_type" should be an integer.

    "a", "b", "c" should be floats - these parameters represent the side lengths of the
    unit cell in the x, y and z directions respectively. These parameters should be
    given in units of Angstroms (Å).

    Parameters
    ----------
    filename : str
        The path to the CSV file containing the lattice parameters.

    Returns
    -------
    tuple
        A tuple containing:
        - material (str): The material name.
        - lattice_type (int): The type of the lattice.
        - lattice_constants (tuple): The side lengths (a, b, c) of the unit cell in the
        x, y, z directions respectively.
    """
    # Read the CSV file containing the lattice parameters into a DataFrame
    lattice_df = pd.read_csv(filename)

    # Ensure the CSV file contains the required columns
    if not LATTICE_REQUIRED_COLUMNS.issubset(lattice_df.columns):
        raise ValueError(
            f"The CSV file must contain the following columns: {LATTICE_REQUIRED_COLUMNS}"
        )

    # Extract the material from the DataFrame
    try:
        material = str(lattice_df.loc[0, "material"])
    except ValueError as error:
        raise ValueError(f"Error extracting material. Error: {error}") from error

    # Extract the lattice_type from the DataFrame
    try:
        lattice_type = int(pd.to_numeric(lattice_df.loc[0, "lattice_type"]))
    except ValueError as error:
        raise ValueError(f"Error extracting lattice type. Error: {error}") from error

    # Extract lattice constants from the DataFrame
    try:
        a = float(pd.to_numeric(lattice_df.loc[0, "a"], errors="coerce"))
        b = float(pd.to_numeric(lattice_df.loc[0, "b"], errors="coerce"))
        c = float(pd.to_numeric(lattice_df.loc[0, "c"], errors="coerce"))
    except ValueError as error:
        raise ValueError(
            f"Error extracting lattice constants. Error: {error}"
        ) from error

    lattice_constants = (a, b, c)

    # Return the extracted parameters
    return material, lattice_type, lattice_constants


def extract_basis_from_csv(filename: str):
    """
    Extracts the basis from a CSV file. The CSV file should be formatted with columns:
    "atomic_number", "atomic_mass", "x", "y", "z". Each row of the CSV file corresponds
    to a different atom in the unit cell.

    "atomic_number" should be an integer equal to the atomic number of the atom.

    "atomic_mass" should be a float equal to the atomic mass of the atom in atomic mass
    units.

    "x", "y", "z" should be floats - these parameters represent the position of the
    atom in the unit cell. The position is specified in terms of the lattice constants,
    e.g. specifying (x, y, z) = (0.5, 0.5, 0.5) corresponds to a position
    (0.5*a, 0.5*b. 0.5*c), where a, b, c are the side lengths of the unit cell in the
    x, y, z directions respectively.

    Parameters
    ----------
    filename : str
        The path to the CSV file containing the lattice parameters.

    Returns
    -------
    tuple
        A tuple containing:
        - atomic_numbers
        - atomic_masses
        - atomic_positions
    """
    # Read the CSV file containing the lattice parameters into a DataFrame
    basis_df = pd.read_csv(filename)

    # Ensure the CSV file contains the required columns
    if not BASIS_REQUIRED_COLUMNS.issubset(basis_df.columns):
        raise ValueError(
            f"The CSV file must contain the following columns: {BASIS_REQUIRED_COLUMNS}"
        )

    # Extract the atomic numbers from the DataFrame
    try:
        atomic_numbers = basis_df["atomic_number"].astype(int).tolist()
    except ValueError as error:
        raise ValueError(f"Error extracting atomic numbers. Error: {error}") from error

    # Extract the atomic masses from the DataFrame
    try:
        atomic_masses = basis_df["atomic_mass"].astype(float).tolist()
    except ValueError as error:
        raise ValueError(f"Error extracting atomic masses. Error: {error}") from error

    # Extract the atomic positions from the DataFrame
    try:
        x_positions = basis_df["x"].astype(float).tolist()
        y_positions = basis_df["y"].astype(float).tolist()
        z_positions = basis_df["z"].astype(float).tolist()
    except ValueError as error:
        raise ValueError(
            f"Error extracting atomic positions. Error: {error}"
        ) from error

    atomic_positions = list(zip(x_positions, y_positions, z_positions))

    # Return the extracted basis information as a tuple
    return atomic_numbers, atomic_masses, atomic_positions
