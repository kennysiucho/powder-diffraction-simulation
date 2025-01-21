"""
This module contains a selection of classes which represent various properties of a 
crystal lattice.

Classes:
TODO: add classes.
"""

import math
import B8_project.utils as utils


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
    """

    def __init__(
        self,
        atomic_number: int,
        position: tuple[float, float, float],
    ):
        """
        Initialize an `Atom` instance
        """
        self.atomic_number = atomic_number
        self.position = position

    def __eq__(self, other):
        if isinstance(other, Atom):
            return (
                self.atomic_number == other.atomic_number
                and self.position == other.position
            )
        return False

    def __str__(self):
        """
        Return a string representing an `Atom` instance for printing.
        """
        return f"Atomic Number: {self.atomic_number}, " f"Position: {self.position}"

    def __repr__(self):
        """
        Return a string representation of an `Atom` instance.
        """
        return self.__str__()

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
        - parameters_to_unit_cell: Converts lattice and basis parameters to an instance
        of `UnitCell`.
    """

    def __init__(
        self,
        material: str,
        lattice_constants: tuple[float, float, float],
        atoms: list[Atom],
    ):
        """
        Initialize a `UnitCell` instance
        """
        self.material = material
        self.lattice_constants = lattice_constants
        self.atoms = atoms

    def __eq__(self, other):
        if isinstance(other, UnitCell):
            return (
                self.material == other.material
                and self.lattice_constants == other.lattice_constants
                and self.atoms == other.atoms
            )
        return False

    def __str__(self):
        """
        Return a string representing a `UnitCell` instance for printing.
        """
        atoms_str = "\n".join([str(atom) for atom in self.atoms])
        return (
            f"Material: {self.material}\n"
            f"Lattice Constants: {self.lattice_constants}\n"
            f"Atoms:\n{atoms_str}"
        )

    def __repr__(self):
        """
        Return a string representation of a `UnitCell` instance.
        """
        return self.__str__()

    @staticmethod
    def validate_crystal_parameters(
        lattice: tuple[str, int, tuple[float, float, float]],
        basis: tuple[list[int], list[tuple[float, float, float]]],
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
    def crystal_parameters_to_unit_cell(
        cls,
        lattice: tuple[str, int, tuple[float, float, float]],
        basis: tuple[list[int], list[tuple[float, float, float]]],
    ):
        """
        Parameters to unit cell
        =======================

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
            cls.validate_crystal_parameters(lattice, basis)
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
        elif lattice_type == 4:
            # Implement base centred lattice logic here.

            raise ValueError(
                """Base centred lattice logic not implemented yet. Please choose a 
                different lattice type."""
            )


class ReciprocalLatticeVector:
    """
    Reciprocal lattice vector
    =========================

    A class to represent a reciprocal lattice vector.

    Attributes
    ----------
        - miller_indices (tuple[float, float, float]): The miller indices (hkl)
        associated with a reciprocal lattice vector.
        - lattice_constants (tuple[float, float, float]): The side lengths of the unit
        cell in the x, y and z directions respectively.

    Methods
    -------
    TODO: add methods.
    """

    def __init__(self, miller_indices: tuple[int, int, int], unit_cell: UnitCell):
        """
        Initialize a `ReciprocalLatticeVector` instance
        """
        self.miller_indices = miller_indices
        self.lattice_constants = unit_cell.lattice_constants

    def __eq__(self, other):
        if isinstance(other, ReciprocalLatticeVector):
            return (
                self.miller_indices == other.miller_indices
                and self.lattice_constants == other.lattice_constants
            )
        return False

    def __str__(self):
        """
        Return a string representing a `ReciprocalLatticeVector` instance for printing.
        """
        return (
            f"(h, k, l): ({self.miller_indices[0]}, {self.miller_indices[1]}, "
            f"{self.miller_indices[2]}). \n "
            f"(a, b, c): ({self.lattice_constants[0]}, {self.lattice_constants[1]}, "
            f"{self.lattice_constants[2]})"
        )

    def __repr__(self):
        """
        Return a string representation of a `ReciprocalLatticeVector` instance.
        """
        return self.__str__()

    def get_components(self) -> tuple[float, float, float]:
        """
        Get components
        ==============

        Returns the components of the reciprocal lattice vector associated with an
        instance of `ReciprocalLatticeVector`.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.

        """
        return (
            2 * math.pi * self.miller_indices[0] / self.lattice_constants[0],
            2 * math.pi * self.miller_indices[1] / self.lattice_constants[1],
            2 * math.pi * self.miller_indices[2] / self.lattice_constants[2],
        )

    def get_magnitude(self) -> float:
        """
        Get magnitude
        =============

        Returns the magnitude of the reciprocal lattice vector associated with an
        instance  of `ReciprocalLatticeVector`.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.
        """
        return math.sqrt(
            utils.dot_product_tuples(self.get_components(), self.get_components())
        )

    @classmethod
    def get_reciprocal_lattice_vectors_inside_sphere(
        cls, max_magnitude: float, unit_cell: UnitCell
    ) -> list["ReciprocalLatticeVector"]:
        """
        Get reciprocal lattice vectors
        ==============================

        Returns a list of all reciprocal lattice vectors with `magnitude` less than
        `max_magnitude`.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.
        """
        # Validate that max_magnitude is greater than 0.
        if not max_magnitude > 0:
            raise ValueError("max_magnitude should be greater than 0.")

        a, b, c = unit_cell.lattice_constants

        # Upper bounds on Miller indices.
        max_h = math.ceil((a * max_magnitude) / (2 * math.pi))
        max_k = math.ceil((b * max_magnitude) / (2 * math.pi))
        max_l = math.ceil((c * max_magnitude) / (2 * math.pi))

        # List to store reciprocal lattice vectors.
        reciprocal_lattice_vectors = []

        # Iterate through all the Miller indices, and add all reciprocal lattice vectors
        # with magnitude less than max_magnitude.
        for h in range(-max_h, max_h + 1, 1):
            for k in range(-max_k, max_k + 1, 1):
                for l in range(-max_l, max_l + 1, 1):
                    # Define an instance of `ReciprocalLatticeVector` associated with
                    # the Miller indices (hkl)
                    reciprocal_lattice_vector = cls((h, k, l), unit_cell)

                    # If reciprocal_lattice_vector has a magnitude less than
                    # max_magnitude, append it to the list
                    if reciprocal_lattice_vector.get_magnitude() <= max_magnitude:
                        reciprocal_lattice_vectors.append(reciprocal_lattice_vector)

        return reciprocal_lattice_vectors


class XRayFormFactor:
    """
    X-ray form factor
    ==================

    A class to represent the X-ray form factor of an atom.

    The X-ray form factor of an atom can be approximated by a sum of four Gaussian
    functions and a constant term.
    Each Gaussian has a height and a width, which gives us nine total parameters.
    An instance of `XRayFormFactor` stores these nine parameters, allowing the form
    factor to be calculated.

    For more information, see `this website <https://lampz.tugraz.at/~hadley/ss1/
    crystaldiffraction/atomicformfactors/formfactors.php>`_

    Attributes
    ----------
        - a1, a2, a3, a4 (float): The height of Gaussian 1, 2, 3, 4 respectively.
        - b1, b2, b3, b4 (float): Proportional to the width of Gaussian 1, 2, 3, 4
        respectively.
        - c (float): The constant term.

    Methods
    -------
        - get_xray_form_factor:
            TODO: implement this function.
    """

    def __init__(
        self,
        a1: float,
        b1: float,
        a2: float,
        b2: float,
        a3: float,
        b3: float,
        a4: float,
        b4: float,
        c: float,
    ):
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2
        self.a3 = a3
        self.b3 = b3
        self.a4 = a4
        self.b4 = b4
        self.c = c

    def __eq__(self, other):
        if isinstance(other, XRayFormFactor):
            return (
                self.a1 == other.a1
                and self.b1 == other.b1
                and self.a2 == other.a2
                and self.b2 == other.b2
                and self.a3 == other.a3
                and self.b3 == other.b3
                and self.a4 == other.a4
                and self.b4 == other.b4
                and self.c == other.c
            )
        return False

    def __str__(self):
        """
        Return a string representing an `XRayFormFactor` instance for printing.
        """
        return (
            f"a1: {self.a1}\n"
            f"b1: {self.b1}\n"
            f"a2: {self.a2}\n"
            f"b2: {self.b2}\n"
            f"a3: {self.a3}\n"
            f"b3: {self.b3}\n"
            f"a4: {self.a4}\n"
            f"b4: {self.b4}\n"
            f"c: {self.c}"
        )

    def __repr__(self):
        """
        Return a string representation of an `XRayFormFactor` instance.
        """
        return self.__str__()
