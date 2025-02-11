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
    - ReciprocalLatticeVector: A class to represent a reciprocal lattice vector.
"""

from dataclasses import dataclass
from copy import deepcopy
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

    def __eq__(self, other):
        if not isinstance(other, Atom):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (self.atomic_number == other.atomic_number and
                self.position == other.position)


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

    def __eq__(self, other):
        if not isinstance(other, UnitCell):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (self.material == other.material and
                self.lattice_constants == other.lattice_constants and
                self.atoms == other.atoms)

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


@dataclass
class ReplacementProbability:
    """
    Specifies how to generate unit cell varieties for a disordered alloy: atoms with
    atomic number `atom_from` in a unit cell should be replaced with atomic number
    `atom_to` with `probability`.

    Currently only supports one-to-one replacement, in other words only two elements
    on a lattice site.
    """
    atom_from: int
    atom_to: int
    probability: float

class UnitCellVarieties:
    def __init__(self,
                 pure_unit_cell: UnitCell,
                 replacement_prob: ReplacementProbability):
        self.pure_unit_cell = pure_unit_cell
        self.replacement_prob = replacement_prob
        self.unit_cell_varieties = []
        self.generate_all_unit_cell_varieties()

    def generate_all_unit_cell_varieties(self):
        """
        Generates the list of all possible disordered unit cells for the alloy and
        stores it in the `unit_cell_varieties` attribute.
        """
        indices_to_replace = []
        for i, atom in enumerate(self.pure_unit_cell.atoms):
            if atom.atomic_number == self.replacement_prob.atom_from:
                indices_to_replace.append(i)

        lattice_sites = len(indices_to_replace)
        total_varieties = 2**lattice_sites
        print(f"INFO: Total number of unit cell varieties = {total_varieties}")

        # Create the new Atom object
        new_atoms = [deepcopy(self.pure_unit_cell.atoms[i]) for i in indices_to_replace]
        for new_atom in new_atoms:
            new_atom.atomic_number = self.replacement_prob.atom_to

        # Iterate over all possible permutations
        for bitmask in range(total_varieties):
            new_unit_cell = deepcopy(self.pure_unit_cell)
            for i in range(lattice_sites):
                if bitmask & (1 << i):
                    new_unit_cell.atoms[indices_to_replace[i]] = deepcopy(new_atoms[i])
            self.unit_cell_varieties.append(new_unit_cell)

    def unit_cell_distribution(self):
        """
        Returns a list of all possible disordered unit cells for the alloy with their
        associated probabilities such that a random sample of unit cells results in the
        desired concentration of elements.

        Returns
        -------

        """
        pass

    def atoms_list_distribution(self):
        """
        Returns a list of all possible disordered unit cells, represented as a list
        of atomic numbers, with their associated probabilities.

        Returns
        -------

        """



@dataclass
class ReciprocalLatticeVector:
    """
    Reciprocal lattice vector
    =========================

    A class to represent a reciprocal lattice vector.

    Attributes
    ----------
        - miller_indices (tuple[float, float, float]): The miller indices (h, k, l)
        associated with a reciprocal lattice vector.
        - lattice_constants (tuple[float, float, float]): The side lengths (a, b, c) of
        the unit cell in the x, y and z directions respectively.

    Methods
    -------
        - components: Returns the components of a reciprocal lattice vector, in
        units of inverse nanometers (nm^-1). For miller indices (h, k, l) and lattice
        constants (a, b, c), the components of a reciprocal lattice vector are
        (2π/a, 2π/b, 2π/c).
        - magnitude: Returns the magnitude of a reciprocal lattice vector, in
        units of inverse nanometers (nm^-1).
        - get_reciprocal_lattice_vectors: Returns a list of `ReciprocalLatticeVectors`
        with magnitude in between a specified minimum and maximum magnitude (i.e.
        returns a list of all reciprocal lattice vectors that lie within a spherical
        shell in k-space).
    """

    miller_indices: tuple[int, int, int]
    lattice_constants: tuple[float, float, float]

    def components(self) -> tuple[float, float, float]:
        """
        Components
        ==========

        Returns the components of the reciprocal lattice vector associated with an
        instance of `ReciprocalLatticeVector`.
        """
        return (
            2 * np.pi * self.miller_indices[0] / self.lattice_constants[0],
            2 * np.pi * self.miller_indices[1] / self.lattice_constants[1],
            2 * np.pi * self.miller_indices[2] / self.lattice_constants[2],
        )

    def magnitude(self) -> float:
        """
        Magnitude
        =========

        Returns the magnitude of the reciprocal lattice vector associated with an
        instance  of `ReciprocalLatticeVector`.
        """
        return np.sqrt(utils.dot_product_tuples(self.components(), self.components()))

    @classmethod
    def get_reciprocal_lattice_vectors(
        cls, min_magnitude: float, max_magnitude: float, unit_cell: UnitCell
    ) -> list["ReciprocalLatticeVector"]:
        """
        Get reciprocal lattice vectors
        ==============================

        Returns a list of all reciprocal lattice vectors with `magnitude` in between
        `min_magnitude` and `max_magnitude`.
        """
        # Validate that max_magnitude and min_magnitude are greater than 0.
        if not (max_magnitude > 0 and min_magnitude >= 0):
            raise ValueError(
                "max_magnitude and min_magnitude should be greater than or equal to 0."
            )

        # Validate that max_magnitude is greater than min_magnitude
        if not max_magnitude > min_magnitude:
            raise ValueError("max_magnitude must be greater than min_magnitude.")

        a, b, c = unit_cell.lattice_constants

        # Upper bounds on Miller indices.
        max_h = np.ceil((a * max_magnitude) / (2 * np.pi)).astype(int)
        max_k = np.ceil((b * max_magnitude) / (2 * np.pi)).astype(int)
        max_l = np.ceil((c * max_magnitude) / (2 * np.pi)).astype(int)

        # List to store reciprocal lattice vectors.
        reciprocal_lattice_vectors = []

        # Iterate through all the Miller indices, and add all reciprocal lattice vectors
        # with magnitude greater than min_magnitude and less than max_magnitude.
        for h in range(-max_h, max_h + 1, 1):
            for k in range(-max_k, max_k + 1, 1):
                for l in range(-max_l, max_l + 1, 1):
                    # Define an instance of `ReciprocalLatticeVector` associated with
                    # the Miller indices (hkl)
                    reciprocal_lattice_vector = cls(
                        (h, k, l), unit_cell.lattice_constants
                    )

                    # If reciprocal_lattice_vector has a valid magnitude, append it to
                    # the list
                    if (
                        reciprocal_lattice_vector.magnitude() >= min_magnitude
                        and reciprocal_lattice_vector.magnitude() <= max_magnitude
                    ):
                        reciprocal_lattice_vectors.append(reciprocal_lattice_vector)

        return reciprocal_lattice_vectors

    @classmethod
    def get_magnitudes_and_multiplicities(
        cls, min_magnitude: float, max_magnitude: float, unit_cell: UnitCell
    ) -> list[tuple["ReciprocalLatticeVector", float, int]]:
        """
        Get reciprocal lattice vector magnitudes and multiplicities
        ===========================================================

        Returns a list of all reciprocal lattice vectors with a unique magnitude in
        between `min_magnitude` and `max_magnitude` and the corresponding
        magnitudes and multiplicities of these reciprocal lattice vectors.

        The format of the list is
        `list[(ReciprocalLatticeVector, magnitude, multiplicity)]`.
        """
        # Get a list of all valid reciprocal lattice vectors.
        reciprocal_lattice_vectors = (
            ReciprocalLatticeVector.get_reciprocal_lattice_vectors(
                min_magnitude, max_magnitude, unit_cell
            )
        )

        # Store the reciprocal lattice vectors, magnitudes and multiplicities in the
        # same list, and sort the list by magnitude.
        magnitudes = [x.magnitude() for x in reciprocal_lattice_vectors]
        multiplicities = [1 for x in reciprocal_lattice_vectors]
        reciprocal_lattice_vectors = list(
            zip(reciprocal_lattice_vectors, magnitudes, multiplicities)
        )
        reciprocal_lattice_vectors.sort(key=lambda x: x[1])

        # Remove any reciprocal lattice vectors with the same magnitudes and update
        # multiplicities.
        i = 0
        while i < len(reciprocal_lattice_vectors) - 1:
            reciprocal_lattice_vector = reciprocal_lattice_vectors[i][0]
            magnitude = reciprocal_lattice_vectors[i][1]

            while i < len(reciprocal_lattice_vectors) - 1 and np.isclose(
                magnitude, reciprocal_lattice_vectors[i + 1][1], rtol=1e-6
            ):
                reciprocal_lattice_vectors[i] = (
                    reciprocal_lattice_vector,
                    magnitude,
                    reciprocal_lattice_vectors[i][2]
                    + reciprocal_lattice_vectors[i + 1][2],
                )
                del reciprocal_lattice_vectors[i + 1]

            i += 1

        return reciprocal_lattice_vectors
