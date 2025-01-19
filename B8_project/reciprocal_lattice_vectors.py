"""
TODO: add unit tests for functions in this module.
"""

import math


class ReciprocalLatticeVector:
    """
    Reciprocal lattice vector
    =========================

    A class to represent a reciprocal lattice vector.

    Attributes
    ----------
        - miller_indices (tuple[float, float, float]): The miller indices (hkl)
        associated with a reciprocal lattice vector.

    Methods
    -------
    TODO: add methods.
    """

    def __init__(self, miller_indices: tuple[int, int, int]):
        """
        Initialize a `ReciprocalLatticeVector` instance
        """
        self.miller_indices = miller_indices

    def __str__(self):
        """
        Return a string representing a `ReciprocalLatticeVector` instance for printing.
        """
        return f"""(h, k, l): ({self.miller_indices[0]}, {self.miller_indices[1]},
        {self.miller_indices[2]})"""

    def __repr__(self):
        """
        Return a string representation of a `ReciprocalLatticeVector` instance.
        """
        return self.__str__()

    def calculate_reciprocal_lattice_vector(
        self, lattice_constants: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """
        Calculate reciprocal lattice vector
        =============================

        Returns the reciprocal lattice vector associated with an instance of
        `ReciprocalLatticeVector`, given the lattice constants of the conventional unit
        cell.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.

        """
        return (
            2 * math.pi * self.miller_indices[0] / lattice_constants[0],
            2 * math.pi * self.miller_indices[1] / lattice_constants[1],
            2 * math.pi * self.miller_indices[2] / lattice_constants[2],
        )
