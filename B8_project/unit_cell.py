"""
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
    TODO: add methods.
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

    def shift_position(self, shift: tuple[float, float, float]):
        """
        Shift position
        ==============

        Shifts the `position` of an `Atom` instance by `shift`, and returns this new
        `Atom` instance.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.
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
        - lattice_constants (tuple[float, float, float]): The side lengths of the unit
        cell, given in Angstroms (Å).
        - atoms(list[Atom]): A list of the atoms in the unit cell.

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
        ==============================

        Returns an instance of `UnitCell` given the parameters of the lattice and the
        basis.

        Parameters
        ----------
        TODO: add parameters

        Returns
        -------
        TODO: add returns

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

    TODO: add description.

    Attributes
    ----------
    TODO: add attributes.

    Methods
    -------
    TODO: add methods.
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
