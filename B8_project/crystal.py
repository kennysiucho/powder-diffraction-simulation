"""
This module contains a selection of classes which represent various properties of a 
crystal lattice.

Classes:
TODO: add classes.
"""

import cmath
from dataclasses import dataclass
import math
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
        - parameters_to_unit_cell: Converts lattice and basis parameters to an instance
        of `UnitCell`.
    """

    material: str
    lattice_constants: tuple[float, float, float]
    atoms: list[Atom]

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
    def get_unit_cell(
        cls,
        basis: tuple[list[int], list[tuple[float, float, float]]],
        lattice: tuple[str, int, tuple[float, float, float]],
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
        else:
            # Implement base centred lattice logic here.

            raise ValueError(
                """Base centred lattice logic not implemented yet. Please choose a 
                different lattice type."""
            )


@dataclass
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

    miller_indices: tuple[int, int, int]
    lattice_constants: tuple[float, float, float]

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
    def get_reciprocal_lattice_vectors(
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
                    reciprocal_lattice_vector = cls(
                        (h, k, l), unit_cell.lattice_constants
                    )

                    # If reciprocal_lattice_vector has a magnitude less than
                    # max_magnitude, append it to the list
                    if reciprocal_lattice_vector.get_magnitude() <= max_magnitude:
                        reciprocal_lattice_vectors.append(reciprocal_lattice_vector)

        return reciprocal_lattice_vectors


@dataclass
class NeutronFormFactor:
    """
    Neutron form factor
    ===================

    A class to represent the neutron form factor of an atom.

    The neutron form factor is proportional to the neutron scattering length. Since we
    are only interested in relative intensities, we do not need to make a distinction
    between the neutron form factor and the neutron scattering length of an atom.

    Attributes
    ----------
        - neutron_scattering_length (float): The neutron scattering length of an atom.

    Methods
    -------
    TODO: add methods.
    """

    neutron_scattering_length: float

    def get_form_factor(self, _reciprocal_lattice_vector: ReciprocalLatticeVector):
        """
        Get neutron form factor
        =======================

        Returns the neutron scattering length of an instance of `NeutronFormFactor`.
        The neutron scattering length of an atom is proportional to the neutron form
        factor.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.
        """
        return self.neutron_scattering_length


@dataclass
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

    For more information, see this website:
    TODO: add link.

    Attributes
    ----------
        - a1, a2, a3, a4 (float): The height of Gaussian 1, 2, 3, 4 respectively.
        - b1, b2, b3, b4 (float): Proportional to the width of Gaussian 1, 2, 3, 4
        respectively.
        - c (float): The constant term.

    Methods
    -------
        - get_xray_form_factor:
            TODO: add documentation.
    """

    a1: float
    b1: float
    a2: float
    b2: float
    a3: float
    b3: float
    a4: float
    b4: float
    c: float

    def get_form_factor(self, reciprocal_lattice_vector: ReciprocalLatticeVector):
        """
        Get X-ray form factor
        =====================

        Returns the form factor associated with an instance of `XRayFormFactor`.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.
        """
        reciprocal_lattice_vector_magnitude = reciprocal_lattice_vector.get_magnitude()

        a = [self.a1, self.a2, self.a3, self.a4]
        b = [self.b1, self.b2, self.b3, self.b4]
        c = self.c

        form_factor = 0
        for i in range(4):
            form_factor += a[i] * math.exp(
                -b[i] * (reciprocal_lattice_vector_magnitude / (4 * math.pi)) ** 2
            )

        form_factor += c
        return form_factor


@dataclass
class Diffraction:
    """
    Diffraction
    ===========

    A class to calculate diffraction patterns for a given crystal.
    """

    @staticmethod
    def get_structure_factor(
        unit_cell: UnitCell,
        form_factors,
        reciprocal_lattice_vector: ReciprocalLatticeVector,
    ) -> complex:
        """
        Get structure factor
        ====================

        Returns the structure factor of a crystal evaluated at a given reciprocal lattice vector.

        An instance of `UnitCell` represents the crystal. The form factors are stored
        in a dictionary which maps atomic number to form factor.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.

        Todos
        -----
        TODO: add error handling for when form_factors is not the correct type.
        """
        structure_factor = 0 + 0j

        for atom in unit_cell.atoms:
            exponent = (2 * math.pi * 1j) * utils.dot_product_tuples(
                reciprocal_lattice_vector.miller_indices, atom.position
            )

            try:
                form_factor = form_factors[atom.atomic_number]

                structure_factor += form_factor.get_form_factor(
                    reciprocal_lattice_vector
                ) * cmath.exp(exponent)

            except KeyError as exc:
                raise KeyError(f"Error reading form factor dictionary: {exc}") from exc
        return structure_factor

    @staticmethod
    def get_structure_factors(
        unit_cell: UnitCell,
        form_factors,
        max_magnitude: float,
    ) -> list[tuple["ReciprocalLatticeVector", complex]]:
        """
        Get structure factors
        =====================

        Computes the structure factors for all reciprocal lattice vectors whose
        magnitudes are less than the specified max_magnitude.

        The function returns a list of tuples, where each tuple contains a reciprocal
        lattice vector and the corresponding structure factor.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.

        Todos
        -----
        TODO: add error handling for when form_factors is not the correct type.
        """
        # Generates a list of all reciprocal lattice vectors within a sphere of radius
        # max_magnitude in k-space.
        try:
            reciprocal_lattice_vectors = (
                ReciprocalLatticeVector.get_reciprocal_lattice_vectors(
                    max_magnitude, unit_cell
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"Error generating reciprocal lattice vectors: {exc}"
            ) from exc

        # Empty list that will store the structure factors associated with each
        # reciprocal lattice vector.
        structure_factors = []

        # Iterates through reciprocal_lattice_vectors. For each RLV, calculates the
        # structure factor and appends this to structure_factors.
        for reciprocal_lattice_vector in reciprocal_lattice_vectors:
            try:
                structure_factors.append(
                    Diffraction.get_structure_factor(
                        unit_cell, form_factors, reciprocal_lattice_vector
                    )
                )

            except Exception as exc:
                raise ValueError(f"Error computing structure factor: {exc}") from exc

        # Returns a list of tuples, where each tuple contains a reciprocal lattice
        # vector and the corresponding structure factor.
        return list(zip(reciprocal_lattice_vectors, structure_factors))

    @staticmethod
    def get_intensity_peaks(
        unit_cell: UnitCell,
        form_factors,
        wavelength: float,
    ) -> list[tuple[float, float]]:
        """
        Get diffraction peaks
        =====================

        Calculates the angles and relative intensities of intensity peaks for a given
        crystal. Returns a list of tuples, each containing the angle and relative
        intensity of a peak.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.

        Todos
        -----
        TODO: Convert angle into deflection angle in degrees.
        TODO: add error handling for when form_factors is not the correct type.
        """
        # Calculate maximum magnitude of RLV for scattering to still occur.
        max_magnitude = ((4 * math.pi) / wavelength) - 1e-10

        # Calculate list of RLVs and corresponding structure factors.
        structure_factors = Diffraction.get_structure_factors(
            unit_cell, form_factors, max_magnitude
        )

        # A list of tuples (angle, intensity).
        intensity_peaks = []

        # Iterates through neutron_structure_factors and populates intensity_peaks.
        for reciprocal_lattice_vector, structure_factor in structure_factors:
            # Calculate sin of the diffraction angle.
            sin_angle = (
                wavelength * reciprocal_lattice_vector.get_magnitude() / (4 * math.pi)
            )

            if sin_angle >= 1 or sin_angle <= -1:
                continue

            angle = math.asin(sin_angle)
            intensity = abs(structure_factor) ** 2

            intensity_peaks.append((angle, intensity))

        # Sort the intensity peaks by angle, and separate intensity_peaks into two
        # lists.
        intensity_peaks.sort(key=lambda x: x[0])
        angles, intensities = zip(*intensity_peaks)

        # Iterate over angles. Any angles which are the same are merged, and the
        # intensities are summed.
        merged_intensity_peaks = []

        i = 0
        length = len(angles)
        while i < length:
            angle = angles[i]
            intensity = intensities[i]

            i += 1
            while i < length and math.isclose(angle, angles[i], rel_tol=1e-10):
                intensity += intensities[i]
                i += 1

            merged_intensity_peaks.append((angle, intensity))

        # Find the maximum intensity
        angles, intensities = zip(*merged_intensity_peaks)
        max_intensity = max(intensities)

        # Divide all intensities by max_intensity to get relative intensities
        relative_intensities = [x / max_intensity for x in intensities]

        return list(zip(angles, relative_intensities))
