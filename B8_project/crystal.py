"""
This module contains a selection of classes which represent various properties of a 
crystal lattice.

Classes:
TODO: add classes.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Mapping, runtime_checkable
import numpy as np
import matplotlib.pyplot as plt
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
        - validate_crystal_parameters: Takes lattice and basis parameters as inputs,
        and raises an error if the parameters are invalid. This function has no returns.
        - get_unit_cell: Converts lattice and basis parameters to an instance
        of `UnitCell`.
    """

    material: str
    lattice_constants: tuple[float, float, float]
    atoms: list[Atom]

    @staticmethod
    def validate_crystal_parameters(
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
    def get_unit_cell(
        cls,
        basis: tuple[list[int], list[tuple[float, float, float]]],
        lattice: tuple[str, int, tuple[float, float, float]],
    ):
        """
        Get unit cell
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
            cls.validate_crystal_parameters(basis, lattice)
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
        - miller_indices (tuple[float, float, float]): The miller indices (h, k, l)
        associated with a reciprocal lattice vector.
        - lattice_constants (tuple[float, float, float]): The side lengths (a, b, c) of
        the unit cell in the x, y and z directions respectively.

    Methods
    -------
        - get_components: Returns the components of a reciprocal lattice vector, in
        units of inverse nanometers (nm^-1). For miller indices (h, k, l) and lattice
        constants (a, b, c), the components of a reciprocal lattice vector are
        (2π/a, 2π/b, 2π/c).
        - get_magnitude: Returns the magnitude of a reciprocal lattice vector, in
        units of inverse nanometers (nm^-1).
        - get_reciprocal_lattice_vectors: Returns a list of `ReciprocalLatticeVectors`
        with magnitude in between a specified minimum and maximum magnitude (i.e.
        returns a list of all reciprocal lattice vectors that lie within a spherical
        shell in k-space).
    """

    miller_indices: tuple[int, int, int]
    lattice_constants: tuple[float, float, float]

    def get_components(self) -> tuple[float, float, float]:
        """
        Get components
        ==============

        Returns the components of the reciprocal lattice vector associated with an
        instance of `ReciprocalLatticeVector`.
        """
        return (
            2 * np.pi * self.miller_indices[0] / self.lattice_constants[0],
            2 * np.pi * self.miller_indices[1] / self.lattice_constants[1],
            2 * np.pi * self.miller_indices[2] / self.lattice_constants[2],
        )

    def get_magnitude(self) -> float:
        """
        Get magnitude
        =============

        Returns the magnitude of the reciprocal lattice vector associated with an
        instance  of `ReciprocalLatticeVector`.
        """
        return np.sqrt(
            utils.dot_product_tuples(self.get_components(), self.get_components())
        )

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
                        reciprocal_lattice_vector.get_magnitude() >= min_magnitude
                        and reciprocal_lattice_vector.get_magnitude() <= max_magnitude
                    ):
                        reciprocal_lattice_vectors.append(reciprocal_lattice_vector)

        return reciprocal_lattice_vectors


@runtime_checkable
class FormFactorProtocol(Protocol):
    """
    Form factor protocol
    ====================

    This protocol defines the interface for any class that represents a form factor.
    Form factor classes must implement the `get_form_factor` method.
    """

    def get_form_factor(
        self, reciprocal_lattice_vector: ReciprocalLatticeVector
    ) -> float:
        """
        Get form factor
        ===============

        Calculates the form factor given a reciprocal lattice vector. The way the form
        factor is calculated varies depending on the class that implements the form
        factor interface.
        """
        ...  # pylint: disable=W2301


@dataclass
class NeutronFormFactor:
    """
    Neutron form factor
    ===================

    A class to represent the neutron form factor of an atom.

    The neutron form factor is proportional to the neutron scattering length. Since we
    are only interested in relative intensities, we do not make a distinction
    between the neutron form factor and the neutron scattering length of an atom.

    Attributes
    ----------
        - neutron_scattering_length (float): The neutron scattering length of an atom.

    Methods
    -------
    TODO: add methods.
    """

    neutron_scattering_length: float

    def get_form_factor(
        self,
        reciprocal_lattice_vector: ReciprocalLatticeVector,  # pylint: disable=W0613
    ) -> float:
        """
        Get neutron form factor
        =======================

        Returns the neutron scattering length of an instance of `NeutronFormFactor`.
        The neutron scattering length of an atom is proportional to the neutron form
        factor.
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

    def get_form_factor(
        self, reciprocal_lattice_vector: ReciprocalLatticeVector
    ) -> float:
        """
        Get X-ray form factor
        =====================

        Returns the form factor associated with an instance of `XRayFormFactor`.
        """
        reciprocal_lattice_vector_magnitude = reciprocal_lattice_vector.get_magnitude()

        a = [self.a1, self.a2, self.a3, self.a4]
        b = [self.b1, self.b2, self.b3, self.b4]
        c = self.c

        form_factor = 0
        for i in range(4):
            form_factor += a[i] * np.exp(
                -b[i] * (reciprocal_lattice_vector_magnitude / (4 * np.pi)) ** 2
            )

        form_factor += c
        return form_factor


class Diffraction:
    """
    Diffraction
    ===========

    A class to calculate diffraction patterns for a given crystal.
    """

    @staticmethod
    def get_structure_factor(
        unit_cell: UnitCell,
        form_factors: Mapping[int, FormFactorProtocol],
        reciprocal_lattice_vector: ReciprocalLatticeVector,
    ) -> complex:
        """
        Get structure factor
        ====================

        Returns the structure factor of a crystal evaluated at a given reciprocal lattice vector.

        An instance of `UnitCell` represents the crystal. The form factors are stored
        in a `Mapping` which maps atomic number to form factor.
        """
        structure_factor = 0 + 0j

        for atom in unit_cell.atoms:
            exponent = (2 * np.pi * 1j) * utils.dot_product_tuples(
                reciprocal_lattice_vector.miller_indices, atom.position
            )

            try:
                form_factor = form_factors[atom.atomic_number]

                structure_factor += form_factor.get_form_factor(
                    reciprocal_lattice_vector
                ) * np.exp(exponent)

            except KeyError as exc:
                raise KeyError(f"Error reading form factor Mapping: {exc}") from exc
        return structure_factor

    @staticmethod
    def get_structure_factors(
        unit_cell: UnitCell,
        form_factors: Mapping[int, FormFactorProtocol],
        min_magnitude: float,
        max_magnitude: float,
    ) -> list[tuple["ReciprocalLatticeVector", complex]]:
        """
        Get structure factors
        =====================

        Computes the structure factors for all reciprocal lattice vectors whose
        magnitudes are greater than a specified minimum magnitude and less than a
        specified maximum magnitude.

        The function returns a list of tuples, where each tuple contains a reciprocal
        lattice vector and the corresponding structure factor.

        The form factors are stored in a `Mapping` which maps atomic number to form
        factor.
        """
        # Generates a list of all reciprocal lattice vectors with valid magnitudes.
        try:
            reciprocal_lattice_vectors = (
                ReciprocalLatticeVector.get_reciprocal_lattice_vectors(
                    min_magnitude, max_magnitude, unit_cell
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
    def get_reciprocal_lattice_vector_magnitude(
        deflection_angle: float, wavelength: float
    ):
        """
        Get reciprocal lattice vector magnitude
        =======================================

        Calculates the magnitude of the reciprocal lattice vector(s) associated with a
        given deflection angle.
        """
        if deflection_angle < 0 or deflection_angle > 180:
            raise ValueError("Invalid deflection angle.")

        angle = deflection_angle * np.pi / 360
        return 4 * np.pi * np.sin(angle) / wavelength

    @staticmethod
    def get_deflection_angle(
        reciprocal_lattice_vector_magnitude: float, wavelength: float
    ):
        """
        Get deflection angle
        ====================

        Calculates the deflection angle associated with a reciprocal lattice vector of
        a given magnitude
        """
        sin_angle = (wavelength * reciprocal_lattice_vector_magnitude) / (4 * np.pi)

        if sin_angle > 1 or sin_angle < 0:
            raise ValueError("Invalid reciprocal lattice vector magnitude")

        return np.arcsin(sin_angle) * 360 / np.pi

    @staticmethod
    def get_diffraction_peaks(
        unit_cell: UnitCell,
        form_factors: Mapping[int, FormFactorProtocol],
        wavelength: float,
        min_deflection_angle: float,
        max_deflection_angle: float,
    ) -> list[tuple[float, float]]:
        """
        Get diffraction peaks
        =====================

        Calculates the angles and relative intensities of diffraction peaks for a given
        crystal. Returns a list of tuples, each containing the deflection angle (2θ)
        and relative intensity of a peak.

        Example use case
        ----------------
        TODO: add example use case.
        """
        # Validate min_deflection_angle and max_angle are both greater than 0
        if not (min_deflection_angle >= 0 and max_deflection_angle > 0):
            raise ValueError(
                """min_deflection_angle and max_deflection_angle should be greater than
                or equal to 0."""
            )

        # Validate that max_deflection_angle is larger than min_deflection_angle
        if not max_deflection_angle > min_deflection_angle:
            raise ValueError(
                "max_deflection_angle should be larger than min_deflection_angle"
            )

        # Calculate the minimum and maximum RLV magnitudes
        try:
            min_magnitude = Diffraction.get_reciprocal_lattice_vector_magnitude(
                min_deflection_angle, wavelength
            )
            max_magnitude = Diffraction.get_reciprocal_lattice_vector_magnitude(
                max_deflection_angle, wavelength
            )
        except ValueError as exc:
            raise ValueError(
                f"Error calculating RLV max and min magnitudes: {exc}"
            ) from exc

        # Calculate list of RLVs and corresponding structure factors.
        try:
            structure_factors = Diffraction.get_structure_factors(
                unit_cell, form_factors, min_magnitude, max_magnitude
            )
        except ValueError as exc:
            raise ValueError(f"Error calculating structure factors: {exc}") from exc

        # A list which will store the deflection angles and intensities of each peak.
        intensity_peaks = []

        # Iterates through neutron_structure_factors and populates intensity_peaks.
        for reciprocal_lattice_vector, structure_factor in structure_factors:
            try:
                angle = Diffraction.get_deflection_angle(
                    reciprocal_lattice_vector.get_magnitude(), wavelength
                )
            except ValueError as exc:
                raise ValueError(f"Error calculating deflection angle: {exc}") from exc

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
            while i < length and np.isclose(angle, angles[i], rtol=1e-10):
                intensity += intensities[i]
                i += 1

            merged_intensity_peaks.append((angle, intensity))

        # Find the maximum intensity
        angles, intensities = zip(*merged_intensity_peaks)
        max_intensity = max(intensities)

        # Divide all intensities by max_intensity to get relative intensities
        relative_intensities = [x / max_intensity for x in intensities]

        return list(zip(angles, relative_intensities))

    @staticmethod
    def plot_diffraction_pattern(
        unit_cell: UnitCell,
        form_factors: Mapping[int, FormFactorProtocol],
        wavelength: float,
        min_deflection_angle: float,
        max_deflection_angle: float,
        peak_width: float,
    ) -> str:
        """
        Plot diffraction pattern
        ========================

        Plots the diffraction pattern for a given crystal and saves the plot as a .pdf
        file in the results directory.

        Name of .pdf file
        -----------------
            - For neutron diffraction, the .pdf file has the following name:
            "<material>_<NDP>_<date>.pdf", where "NDP" stands for Neutron Diffraction
            Pattern.
            - For X-ray diffraction, the .pdf file has the following name:
            "<material>_<XRDP>_<date>.pdf", where "XRDP" stands for X-Ray Diffraction
            Pattern.

        Parameters
        ----------
            - unit_cell (UnitCell): the unit cell of the chosen crystal.
            - form_factors (Mapping[int, FormFactorProtocol]): a mapping from atomic
            numbers to a class which represents an atomic form factor. Currently, two
            classes implement the form factor protocol, `NeutronFormFactor` and
            `XRayFormFactor`.
            - wavelength (float): the wavelength of incident particles, given in
            nanometers (nm).
            - min_deflection_angle (float), max_deflection_angle (float): these
            parameters specify the range of deflection angles to be plotted.
            - peak_width (float): The width of the intensity peaks. This parameter is
            only used for plotting. A value should be chosen so that all diffraction
            peaks can be observed.

        Returns
        -------
            - (str): The path to the plot.
        """
        try:
            diffraction_peaks = Diffraction.get_diffraction_peaks(
                unit_cell,
                form_factors,
                wavelength,
                min_deflection_angle,
                max_deflection_angle,
            )
        except Exception as exc:
            raise ValueError(f"Error finding diffraction peaks: {exc}") from exc

        # Calculate a sensible number of points
        num_points = np.round(
            10 * (max_deflection_angle - min_deflection_angle) / peak_width
        ).astype(int)

        # Get x coordinates of plotted points.
        x_values = np.linspace(min_deflection_angle, max_deflection_angle, num_points)

        # Get y coordinates of plotted points.
        y_values = np.zeros_like(x_values)

        for angle, intensity in diffraction_peaks:
            y_values += utils.gaussian(x_values, angle, peak_width, intensity)

        # Get today's date and format as a string
        today = datetime.today()
        date_string = today.strftime("%d_%m_%Y")

        # Figure out the diffraction type and filename from form_factors
        if isinstance(form_factors, Mapping) and all(
            isinstance(v, NeutronFormFactor) for v in form_factors.values()
        ):
            diffraction_type = "neutron "
            filename = f"{unit_cell.material}_NDP_{date_string}"
        elif isinstance(form_factors, Mapping) and all(
            isinstance(v, XRayFormFactor) for v in form_factors.values()
        ):
            diffraction_type = "X-ray "
            filename = f"{unit_cell.material}_XRDP_{date_string}"
        else:
            diffraction_type = ""
            filename = f"{unit_cell.material}_DP_{date_string}"

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the data
        ax.plot(x_values, y_values, color="black")

        # Set axis labels
        ax.set_xlabel("Deflection angle (°)", fontsize=11)
        ax.set_ylabel("Relative intensity", fontsize=11)

        # Set title
        ax.set_title(
            f"{unit_cell.material} {diffraction_type}diffraction pattern for λ = {wavelength}nm.",
            fontsize=15,
        )

        # Add grid lines
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Customize the tick marks
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", length=4, color="gray")

        # Add minor ticks
        ax.minorticks_on()

        # Adjust layout to prevent clipping
        fig.tight_layout()

        # Save the figure
        fig.savefig(f"results/{filename}.pdf", format="pdf")

        return f"results/{filename}.pdf"
