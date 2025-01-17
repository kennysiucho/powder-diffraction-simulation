"""
utils contains utility functions for manipulating tuples and arrays.
file_reading contains functions for reading and validating data extracted from CSV 
files.
TODO: add unit tests for functions in this module.
"""

import B8_project.utils as utils
import B8_project.file_reading as file_reading


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
        the unit cell in the (x, y, z) directions respectively, given in Angstroms (Å).
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

    @classmethod
    def parameters_to_unit_cell(
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
                directions respectively in Angstroms (Å).
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
        ----
        TODO: Implement base centred lattice logic.
        """
        material, lattice_type, lattice_constants = lattice
        atomic_numbers, atomic_positions = basis

        # Validate the lattice and basis parameters
        try:
            file_reading.validate_parameters(lattice, basis)
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

    For more information, see `this website <https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php>`_

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
