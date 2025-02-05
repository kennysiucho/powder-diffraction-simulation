"""
Crystal
=======

This module contains a selection of classes which represent various properties of a 
crystal lattice.

Classes
-------
    - Atom: A class to represent an atom in a unit cell.
    - UnitCell: A class to represent a unit cell. This class can only represent unit 
    cells where all of the angles are 90 degrees.
"""

from dataclasses import dataclass
import numpy as np

import B8_project.utils as utils


@dataclass
class Atom:
    """
    Atom
    ====

    A class to represent an atom in a unit cell.

    Attributes
    ----------
        - atomic_number (int): The atomic number of the atom.
        - position (tuple[float, float, float]): The position of the atom in the unit
        cell, given in terms of the lattice constants.

    Methods
    -------
        - shift_position: shifts the `position` of an `Atom` instance by a specified
        amount.
        - scale_position: scales the x, y, z coordinates of an atom by a specified
        amount.
    """

    atomic_number: int
    position: tuple[float, float, float]

    def shift_position(self, shift: tuple[float, float, float]) -> "Atom":
        """
        Shift position
        ==============

        Shifts the `position` of an `Atom` instance by `shift`, and returns this new
        `Atom` instance.

        Parameters
        ----------
            - shift (tuple[float, float, float]): the amount the the `position`
            attribute is shifted by.

        Returns
        -------
            - (Atom): an `Atom` instance.
        """
        return Atom(self.atomic_number, utils.add_tuples(self.position, shift))

    def scale_position(self, scale_factor: tuple[float, float, float]) -> "Atom":
        """
        Scale position
        ==============

        Scales the x, y, z coordinates of an atom by a specified amount.
        """
        return Atom(
            self.atomic_number,
            (
                self.position[0] * scale_factor[0],
                self.position[1] * scale_factor[1],
                self.position[2] * scale_factor[2],
            ),
        )


@dataclass
class UnitCell:
    """
    Unit cell
    =========

    A class to represent a unit cell. This class can only represent unit cells where all
    of the angles are 90 degrees (i.e. cubic, tetragonal and orthorhombic cells).

    Attributes
    ----------
        - material (str): The chemical formula of the crystal, e.g. "NaCl".
        - lattice_constants (tuple[float, float, float]): The side lengths (a, b, c) of
        the unit cell in the (x, y, z) directions respectively, given in nanometers
        (nm).
        - atoms(list[Atom]): A list of the atoms in the unit cell. Each atom is
        represented by an `Atom` instance.

    Methods
    -------
        - _validate_crystal_parameters: Takes lattice and basis parameters as inputs,
        and raises an error if the parameters are invalid. This function has no returns.
        - new_unit_cell: Converts lattice and basis parameters to an instance
        of `UnitCell`.
    """

    material: str
    lattice_constants: tuple[float, float, float]
    atoms: list[Atom]

    @staticmethod
    def _validate_crystal_parameters(
        basis: tuple[list[int], list[tuple[float, float, float]]],
        lattice: tuple[str, int, tuple[float, float, float]],
    ) -> None:
        """
        Validate parameters
        ===================

        Processes lattice and basis parameters, and raises an error if they are invalid.
        This function has no return value.

        Parameters
        ----------
            - lattice (tuple[str, int, tuple[float, float, float]]): The lattice parameters,
            stored as a tuple (material, lattice_type, lattice_constants).
            - basis (tuple[list[int], list[tuple[float, float, float]]]): The basis
            parameters, stored as a tuple (atomic_numbers, atomic_positions).
        """
        _, lattice_type, lattice_constants = lattice
        atomic_numbers, atomic_positions = basis

        # Validate that the length of atomic_numbers and atomic_positions is the same.
        if not len(atomic_numbers) == len(atomic_positions):
            raise ValueError(
                "Length of atomic_numbers and atomic_positions must be the same"
            )

        # Validate that the lattice constants are non-negative and non-zero.
        (a, b, c) = lattice_constants
        if not (a > 0 and b > 0 and c > 0):
            raise ValueError(
                "Lattice constants should all be non-negative and non-zero."
            )

        # Validate lattice_type
        if lattice_type < 1 or lattice_type > 4:
            raise ValueError(
                "lattice_type should be an integer between 1 and 4 inclusive"
            )

        # Validate that lattice_type and lattice_constants are compatible for a cubic
        # unit cell
        if (a == b == c) and lattice_type == 4:
            raise ValueError(
                "Base centred lattice type is not permitted for a cubic lattice"
            )

        # Validate that lattice_type and lattice_constants are compatible for a
        # tetragonal unit cell
        if (
            (a == b and not a == c)
            or (a == c and not a == b)
            or (b == c and not a == b)
        ):
            if lattice_type == 3:
                raise ValueError(
                    "Face centred lattice type is not permitted for a tetragonal lattice"
                )
            elif lattice_type == 4:
                raise ValueError(
                    "Base centred lattice type is not permitted for a tetragonal unit cell"
                )

    @classmethod
    def new_unit_cell(
        cls,
        basis: tuple[list[int], list[tuple[float, float, float]]],
        lattice: tuple[str, int, tuple[float, float, float]],
    ):
        """
        New unit cell
        =============

        Returns an instance of `UnitCell` given the parameters of the lattice and the
        basis.

        Parameters
        ----------
            - lattice (tuple[str, int, tuple[float, float, float]]): a tuple
            (material, lattice_type, lattice_constants) that represents the lattice.
                - "material" (str): Chemical formula of the crystal (e.g. "NaCl").
                - "lattice_type" (int): Integer (1 - 4 inclusive) that represents the
                Bravais lattice type.
                    - 1 -> Simple.
                    - 2 -> Body centred.
                    - 3 -> Face centred.
                    - 4 -> Base centred.
                - "a", "b", "c" (float): Side lengths of the unit cell in the x, y and z
                directions respectively in nanometers (nm).
            - basis (tuple[list[int], list[tuple[float, float, float]]]): a tuple
            (atomic_numbers, atomic_positions) that represents the basis.
                - atomic_numbers (list[int]): The atomic number of each atom in the
                basis.
                - atomic_positions (list[tuple[float, float, float]]): The position of
                each atom in the basis.

        Returns
        -------
            - (UnitCell): An instance of `UnitCell`, which represents the unit cell of
            the crystal.
            - (None): If an error is encountered, the function returns None and raises
            an error.

        Todos
        -----
        TODO: Implement base centred lattice logic.
        TODO: modify algorithm so that the positions of the new atoms are defined modulo
        a real lattice vector.
        """
        material, lattice_type, lattice_constants = lattice
        atomic_numbers, atomic_positions = basis

        # Validate the lattice and basis parameters
        try:
            cls._validate_crystal_parameters(basis, lattice)
        except ValueError as exc:
            raise ValueError(f"Invalid parameters: {exc}") from exc

        # Convert the basis into a list of atoms.
        atoms = [
            Atom(number, position)
            for number, position in zip(atomic_numbers, atomic_positions)
        ]

        # Simple lattice
        if lattice_type == 1:
            # No modification needed - the conventional unit cell is equal to the
            # primitive unit cell.
            return cls(material, lattice_constants, atoms)

        # Body centered lattice
        elif lattice_type == 2:
            # Duplicates every atom in the unit cell two times, as the conventional
            # unit cell contains two lattice points.
            atoms = utils.duplicate_elements(atoms, 2)

            # Amount that duplicate atoms are shifted by.
            shifts = {1: (0.5, 0.5, 0.5)}

            length = len(atoms)
            for i in range(0, length):
                if not i % 2 == 0:
                    atoms[i] = atoms[i].shift_position(shifts[i % 2])

            return cls(material, lattice_constants, atoms)

        # Face centred lattice
        elif lattice_type == 3:
            # Duplicates every atom in the unit cell four times, as the conventional
            # unit cell contains four lattice points.
            atoms = utils.duplicate_elements(atoms, 4)

            # Shifts all of the duplicate atoms.
            shifts = {1: (0.5, 0.5, 0), 2: (0.5, 0, 0.5), 3: (0, 0.5, 0.5)}

            length = len(atoms)
            for i in range(0, length):
                if not i % 4 == 0:
                    atoms[i] = atoms[i].shift_position(shifts[i % 4])

            return cls(material, lattice_constants, atoms)

        # Base centred lattice
        else:
            # Implement base centred lattice logic here.

            raise ValueError(
                """Base centred lattice logic not implemented yet. Please choose a 
                different lattice type."""
            )


class ReciprocalSpace:
    """
    A class to group functions related to reciprocal space, reciprocal lattice vectors
    and scattering vectors.
    """

    @staticmethod
    def get_reciprocal_lattice_vectors(
        min_magnitude: float,
        max_magnitude: float,
        lattice_constants: np.ndarray,
    ):
        """
        Get reciprocal lattice vectors
        ==============================

        Finds all the reciprocal lattice vectors with a magnitude in between a specified
        minimum and maximum magnitude. Returns a structured NumPy array representing the
        valid reciprocal lattice vectors.

        Array format
        ------------
        The structured NumPy array has the following fields:
            - 'miller_indices': An ndarray representing the Miller indices (h, k, l).
            - 'magnitude': A float representing the magnitude of the reciprocal lattice
            vector.
            - 'components': An ndarray representing the components of the reciprocal
            lattice vector.
        """
        # Error handling.
        if not (max_magnitude > 0 and min_magnitude >= 0):
            raise ValueError(
                "max_magnitude and min_magnitude should be greater than or equal to 0."
            )
        if not max_magnitude > min_magnitude:
            raise ValueError("max_magnitude must be greater than min_magnitude.")
        if not (
            isinstance(lattice_constants, np.ndarray)
            and lattice_constants.shape == (3,)
        ):
            raise ValueError("lattice_constants must be a numpy array of length 3.")
        if not np.issubdtype(lattice_constants.dtype, np.floating):
            raise ValueError("lattice_constants must contain only floats.")

        # Upper bounds on Miller indices.
        max_hkl = np.ceil((lattice_constants * max_magnitude) / (2 * np.pi)).astype(int)

        # Generate all possible Miller indices within the bounds.
        miller_indices = (
            np.vstack(
                np.meshgrid(
                    np.arange(-max_hkl[0], max_hkl[0] + 1),
                    np.arange(-max_hkl[1], max_hkl[1] + 1),
                    np.arange(-max_hkl[2], max_hkl[2] + 1),
                    indexing="ij",
                )
            )
            .reshape(3, -1)
            .T
        )

        # Compute reciprocal lattice vector components and magnitudes.
        components = (2 * np.pi * miller_indices) / lattice_constants
        magnitudes = np.linalg.norm(components, axis=1)

        # Filter the reciprocal lattice vectors based on their magnitude.
        mask = (magnitudes >= min_magnitude) & (magnitudes <= max_magnitude)
        valid_miller_indices = miller_indices[mask]
        valid_magnitudes = magnitudes[mask]

        # Define a custom datatype to represent reciprocal lattice vectors.
        dtype = np.dtype(
            [("miller_indices", "3i4"), ("magnitudes", "f8"), ("components", "3f8")]
        )

        # Create a structured NumPy array to store the valid reciprocal lattice vectors.
        reciprocal_lattice_vectors = np.empty(
            valid_miller_indices.shape[0], dtype=dtype
        )
        reciprocal_lattice_vectors["miller_indices"] = valid_miller_indices
        reciprocal_lattice_vectors["magnitudes"] = valid_magnitudes
        reciprocal_lattice_vectors["components"] = (
            2 * np.pi * valid_miller_indices
        ) / lattice_constants

        return reciprocal_lattice_vectors

    @staticmethod
    def rlv_magnitudes_from_deflection_angles(
        deflection_angles: np.ndarray, wavelength: float
    ):
        """
        Reciprocal lattice vector magnitude from deflection angle
        =========================================================

        Calculates the magnitudes of the reciprocal lattice vectors associated with a
        range of given deflection angles.

        TODO: review documentation.
        """
        if deflection_angles.min() < 0 or deflection_angles.max() > 180:
            raise ValueError("Invalid deflection angle.")

        angles = deflection_angles * np.pi / 360
        return 4 * np.pi * np.sin(angles) / wavelength

    @staticmethod
    def deflection_angles_from_rlv_magnitudes(
        reciprocal_lattice_vector_magnitudes: np.ndarray, wavelength: float
    ):
        """
        Deflection angle from reciprocal lattice vector magnitude
        =========================================================

        Calculates the deflection angle associated with a reciprocal lattice vector of
        a given magnitude.

        TODO: review documentation.
        """
        sin_angles = (wavelength * reciprocal_lattice_vector_magnitudes) / (4 * np.pi)

        if sin_angles.max() > 1 or sin_angles.min() < 0:
            raise ValueError("Invalid reciprocal lattice vector magnitude(s)")

        return np.arcsin(sin_angles) * 360 / np.pi
