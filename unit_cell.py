# TODO: add unit tests for functions in this module.

import utils


class Atom:
    """
    A class to represent an atom in a unit cell.
    """

    def __init__(
        self,
        atomic_number: int,
        atomic_mass: float,
        position: tuple[float, float, float],
    ):
        """
        Initialize an `Atom` instance
        """
        self.atomic_number = atomic_number
        self.atomic_mass = atomic_mass
        self.position = position

    def __str__(self):
        """
        Return a string representation of the Atom instance for printing.
        """
        return (
            f"Atomic Number: {self.atomic_number}, "
            f"Atomic Mass: {self.atomic_mass}, "
            f"Position: {self.position}"
        )

    def __repr__(self):
        """
        Return a detailed string representation of the Atom instance.
        """
        return self.__str__()


class UnitCell:
    """
    A class to represent a unit cell. This class can only represent unit cells where all
    of the angles are 90 degrees.
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
        Return a string representation of the UnitCell instance for printing.
        """
        atoms_str = "\n".join([str(atom) for atom in self.atoms])
        return (
            f"Material: {self.material}\n"
            f"Lattice Constants: {self.lattice_constants}\n"
            f"Atoms:\n{atoms_str}"
        )

    def __repr__(self):
        """
        Return a detailed string representation of the UnitCell instance.
        """
        return self.__str__()

    @classmethod
    def lattice_and_basis_to_unit_cell(
        cls,
        lattice: tuple[str, int, tuple[float, float, float]],
        basis: tuple[list[int], list[float], list[tuple[float, float, float]]],
    ):
        """
        Lattice and basis to unit cell
        ==============================

        Returns an instance of `UnitCell` given the parameters of the lattice and the
        basis.
        """

        # TODO: improve documentation of this function.

        material, lattice_type, lattice_constants = lattice
        atomic_numbers, atomic_masses, atomic_positions = basis

        # Validate that the lattice constants are non-negative and non-zero.
        if not (
            lattice_constants[0] > 0
            and lattice_constants[1] > 0
            and lattice_constants[2] > 0
        ):
            return ValueError(
                """Lattice constants should all be non-negative and 
                              non-zero."""
            )

        # Validate that the length of atomic_numbers, atomic_masses and
        # atomic_positions is the same.
        if not len(atomic_numbers) == len(atomic_masses) == len(atomic_positions):
            return ValueError(
                """Length of atomic_numbers, atomic_masses and atomic_positions must be the same"""
            )

        # TODO: add validation that the lattice constants provided match with the
        # lattice type provided.

        # Convert the basis into a list of atoms.
        atoms = [
            Atom(number, mass, position)
            for number, mass, position in zip(
                atomic_numbers, atomic_masses, atomic_positions
            )
        ]

        # Simple lattice
        if lattice_type in {1, 4, 6}:
            # No modification needed - the conventional unit cell is equal to the
            # primitive unit cell.
            return cls(material, lattice_constants, atoms)

        # Body centered lattice
        elif lattice_type in {2, 5, 7}:
            # Duplicates every atom in the unit cell two times, as the conventional
            # unit cell contains two lattice points.
            atoms = utils.duplicate_elements(atoms, 2)

            # Shifts the position of every duplicate atom.
            shifts = {1: (0.5, 0.5, 0.5)}

            length = len(atoms)
            for i in range(0, length):
                if not i % 2 == 0:
                    atoms[i] = Atom(
                        atoms[i].atomic_number,
                        atoms[i].atomic_mass,
                        utils.add_tuples(atoms[i].position, shifts[i % 2]),
                    )

            return cls(material, lattice_constants, atoms)

        # Face centred lattice
        elif lattice_type in {3, 8}:
            # Duplicates every atom in the unit cell four times, as the conventional
            # unit cell contains four lattice points.
            atoms = utils.duplicate_elements(atoms, 4)

            # Shifts all of the duplicate atoms.
            shifts = {1: (0.5, 0.5, 0), 2: (0.5, 0, 0.5), 3: (0, 0.5, 0.5)}

            length = len(atoms)
            for i in range(0, length):
                if not i % 4 == 0:
                    atoms[i] = Atom(
                        atoms[i].atomic_number,
                        atoms[i].atomic_mass,
                        utils.add_tuples(atoms[i].position, shifts[i % 4]),
                    )

            return cls(material, lattice_constants, atoms)

        # Base centred lattice
        elif lattice_type in {9}:
            # TODO: Implement base centred lattice logic.

            return ValueError(
                """Base centred lattice logic not implemented yet. Please choose a 
                different lattice type."""
            )

        else:
            return ValueError(
                """lattice_type must be an integer between 1 and 9 inclusive."""
            )
