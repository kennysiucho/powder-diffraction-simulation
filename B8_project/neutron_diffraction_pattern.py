"""
This module contains functions which calculate the X-ray or neutron diffraction pattern
of a given crystal.

Functions:
TODO: add functions.
"""

import math
import cmath
from B8_project import utils
from B8_project.crystal import UnitCell, ReciprocalLatticeVector


def get_neutron_structure_factor(
    unit_cell: UnitCell,
    neutron_scattering_lengths: dict[int, float],
    reciprocal_lattice_vector: ReciprocalLatticeVector,
):
    """
    Get neutron structure factor
    ============================

    Returns the structure factor of a crystal evaluated at a given reciprocal lattice
    vector.

    An instance of `UnitCell` represents the crystal. The neutron scattering lengths
    are stored in a dictionary which maps atomic number to neutron scattering length.

    Parameters
    ----------
    TODO: add parameters.

    Returns
    -------
    TODO: add returns.
    """
    structure_factor = 0
    for atom in unit_cell.atoms:
        exponent = (2 * math.pi * 1j) * utils.dot_product_tuples(
            reciprocal_lattice_vector.miller_indices, atom.position
        )

        try:
            structure_factor += neutron_scattering_lengths[
                atom.atomic_number
            ] * cmath.exp(exponent)
        except KeyError as exc:
            raise KeyError(
                f"Error reading neutron scattering length dictionary: {exc}"
            ) from exc
    return structure_factor


def get_neutron_structure_factors(
    unit_cell: UnitCell,
    neutron_scattering_lengths: dict[int, float],
    max_magnitude: float,
) -> list[tuple["ReciprocalLatticeVector", float]]:
    """
    Get neutron structure factors
    =============================

    Computes the structure factors for all reciprocal lattice vectors whose magnitudes
    are less than the specified max_magnitude. The function returns a list of tuples,
    where each tuple contains a reciprocal lattice vector and the corresponding
    structure factor.

    Parameters
    ----------
    TODO: add parameters.

    Returns
    -------
    TODO: add returns.
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
        raise ValueError(f"Error generating reciprocal lattice vectors: {exc}") from exc

    # Empty list that will store the structure factors associated with each reciprocal
    # lattice vector.
    structure_factors = []

    # Iterates through reciprocal_lattice_vectors. For each RLV, calculates the
    # structure factor and appends this to structure_factors.
    for reciprocal_lattice_vector in reciprocal_lattice_vectors:
        try:
            structure_factors.append(
                get_neutron_structure_factor(
                    unit_cell, neutron_scattering_lengths, reciprocal_lattice_vector
                )
            )
        except Exception as exc:
            raise ValueError(f"Error computing structure factor: {exc}") from exc

    # Returns a list of tuples, where each tuple contains a reciprocal lattice vector
    # and the corresponding structure factor.
    return list(zip(reciprocal_lattice_vectors, structure_factors))


def get_diffraction_peaks(
    unit_cell: UnitCell, neutron_scattering_lengths: dict[int, float], wavelength: float
) -> list[tuple[float, float]]:
    """
    Get diffraction peaks
    =====================

    Calculates the angles and relative intensities of diffraction peaks. Returns a list
    of tuples, each containing the angle and relative intensity of a peak.

    Parameters
    ----------
    TODO: add parameters.

    Returns
    -------
    TODO: add returns.
    """
    # Calculate maximum magnitude of RLV for scattering to still occur.
    max_magnitude = ((4 * math.pi) / wavelength) - 1e-10

    # Calculate list of RLVs and corresponding structure factors.
    neutron_structure_factors = get_neutron_structure_factors(
        unit_cell, neutron_scattering_lengths, max_magnitude
    )

    # A list of tuples (angle, intensity).
    intensity_peaks = []

    # Iterates through neutron_structure_factors and populates intensity_peaks.
    for reciprocal_lattice_vector, structure_factor in neutron_structure_factors:
        # Calculate sin of the diffraction angle.
        sin_angle = (
            wavelength * reciprocal_lattice_vector.get_magnitude() / (4 * math.pi)
        )

        if sin_angle >= 1 or sin_angle <= -1:
            continue

        angle = math.asin(sin_angle)
        intensity = abs(structure_factor) ** 2

        intensity_peaks.append((angle, intensity))

    # Sorts the intensity peaks by angle, and separate intensity_peaks into two lists.
    intensity_peaks.sort(key=lambda x: x[0])
    angles, intensities = zip(*intensity_peaks)

    merged_peaks = []
    i = 0

    # Iterates over angles. Any angles which are the same are merged, and the
    # intensities are summed.
    length = len(angles)
    while i < length:
        angle = angles[i]
        intensity = intensities[i]

        i += 1
        while i < length and math.isclose(angle, angles[i], rel_tol=1e-10):
            intensity += intensities[i]
            i += 1

        merged_peaks.append((angle, intensity))

    return merged_peaks
