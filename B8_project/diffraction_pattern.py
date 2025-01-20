"""
TODO: add unit tests for functions in this module.
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

        structure_factor += neutron_scattering_length[atom.atomic_number] * cmath.exp(
            exponent
        )

    return structure_factor
