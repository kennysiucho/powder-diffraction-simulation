"""
This module contains functions which calculate the X-ray or neutron diffraction pattern
of a given crystal.

Functions:
TODO: add functions.
"""

import math
import cmath
import utils
from B8_project.crystal_lattice import UnitCell, ReciprocalLatticeVector


def get_neutron_structure_factor(
    unit_cell: UnitCell,
    neutron_scattering_length: dict[int, float],
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
            structure_factor += neutron_scattering_length[
                atom.atomic_number
            ] * cmath.exp(exponent)
        except KeyError as exc:
            raise KeyError(
                f"Error reading neutron scattering length dictionary: {exc}"
            ) from exc
    return structure_factor


def get_diffraction_peaks(
    unit_cell: UnitCell,
    neutron_scattering_length: dict[int, float],
    max_magnitude: float,
) -> dict["ReciprocalLatticeVector", float]:
    """
    Get diffraction peaks
    =====================

    Computes the structure factors for all reciprocal lattice vectors whose magnitudes
    are less than the specified max_magnitude. The function returns a dictionary where
    the keys are instances of `ReciprocalLatticeVectors` and the values are the squared
    magnitudes of the corresponding structure factors.

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
            ReciprocalLatticeVector.get_reciprocal_lattice_vectors_inside_sphere(
                max_magnitude, unit_cell
            )
        )
    except ValueError as exc:
        raise ValueError(f"Error generating reciprocal lattice vectors: {exc}") from exc

    # Empty list that will store the squared magnitude of the structure factor
    # associated with each reciprocal lattice vector.
    structure_factor_squared_magnitudes = []

    # Iterates through reciprocal_lattice_vectors. For each RLV, calculates the squared
    # magnitude of the structure factor and appends this to structure_factor_magnitudes.
    for reciprocal_lattice_vector in reciprocal_lattice_vectors:
        try:
            structure_factor = get_neutron_structure_factor(
                unit_cell, neutron_scattering_length, reciprocal_lattice_vector
            )

            structure_factor_squared_magnitudes.append(abs(structure_factor) ** 2)

        except Exception as exc:
            raise ValueError(f"Error computing structure factor: {exc}") from exc

    # Returns a dictionary which maps reciprocal lattice vectors to the squared
    # magnitude of their structure factors.
    length = len(reciprocal_lattice_vectors)
    return {
        reciprocal_lattice_vectors[i]: structure_factor_squared_magnitudes[i]
        for i in range(length)
    }
