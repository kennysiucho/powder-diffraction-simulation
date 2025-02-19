"""
Form factor
===========

This module contains classes which represent atomic form factors. Classes which 
represent form factors should implement the interface defined by FormFactorProtocol.

Classes
-------
    - FormFactorProtocol: This protocol defines the interface for any class that 
    represents a form factor.
    - NeutronFormFactor: A class to represent the neutron form factor of an atom.
    - XRayFormFactor: A class to represent the X-ray form factor of an atom.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np
from B8_project.crystal import ReciprocalLatticeVector


@runtime_checkable
class FormFactorProtocol(Protocol):
    """
    Form factor protocol
    ====================

    This protocol defines the interface for any class that represents a form factor.
    Form factor classes must implement the `evaluate_form_factor` method.
    """

    def evaluate_form_factor(
        self, reciprocal_lattice_vector: ReciprocalLatticeVector
    ) -> float:
        """
        Evaluate form factor
        ====================

        Calculates the form factor given a reciprocal lattice vector. The way the form
        factor is calculated varies depending on the class that implements the form
        factor interface.
        """
        ...  # pylint: disable=W2301

    def evaluate_form_factors(self, k_vectors: np.ndarray) -> np.ndarray:
        """
        Evaluate form factors given array of wave vectors. The way the form factor is
        calculated varies depending on the class that implements the form factor
        interface.

        Parameters
        ----------
        k_vectors : (N, 3) np.ndarray
            List of wave vectors (units nm^-1)

        Returns
        -------
        form_factors : (N, ) np.ndarray
            Array containing the calculated form factors for each wave vector.
        """


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
    """

    neutron_scattering_length: float

    def evaluate_form_factor(
        self,
        reciprocal_lattice_vector: ReciprocalLatticeVector,  # pylint: disable=W0613
    ) -> float:
        """
        Evaluate neutron form factor
        ============================

        Returns the neutron scattering length of an instance of `NeutronFormFactor`.
        The neutron scattering length of an atom is proportional to the neutron form
        factor.
        """
        return self.neutron_scattering_length

    def evaluate_form_factors(self, k_vectors: np.ndarray) -> np.ndarray:
        """
        Returns a list of scattering lengths, since the ND form factor only depends on
        the neutron scattering length of an atom.

        Parameters
        ----------
        k_vectors : (N, 3) np.ndarray
            List of wave vectors (units nm^-1)

        Returns
        -------
        form_factors : (N, ) np.ndarray
            Array containing the calculated form factors for each wave vector.
        """
        return np.ones(k_vectors.shape[0]) * self.neutron_scattering_length


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
        - b1, b2, b3, b4 (float): Inversely proportional to the to the squared width of
        Gaussian 1, 2, 3, 4 respectively.
        - c (float): The constant term.
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

    def __post_init__(self):
        self.a = [self.a1, self.a2, self.a3, self.a4]
        self.b = [self.b1, self.b2, self.b3, self.b4]

    def evaluate_form_factor(
        self, reciprocal_lattice_vector: ReciprocalLatticeVector
    ) -> float:
        """
        Evaluate X-ray form factor
        ==========================

        Returns the form factor associated with an instance of `XRayFormFactor`.
        """
        reciprocal_lattice_vector_magnitude = reciprocal_lattice_vector.magnitude()

        form_factor = 0
        for i in range(4):
            form_factor += self.a[i] * np.exp(
                -self.b[i] * (reciprocal_lattice_vector_magnitude / (4 * np.pi)) ** 2
            )

        form_factor += self.c
        return form_factor

    def evaluate_form_factors(self, k_vectors: np.ndarray) -> np.ndarray:
        """
        Returns a list of X-ray form factors given list of wave vectors.

        Parameters
        ----------
        k_vectors : (N, 3) np.ndarray
            List of wave vectors (units nm^-1)

        Returns
        -------
        form_factors : (N, ) np.ndarray
            Array containing the calculated form factors for each wave vector.
        """
        form_factors = np.zeros(k_vectors.shape[0])
        for i in range(4):
            mag_squared = np.sum(np.square(k_vectors), axis=1)
            form_factors += self.a[i] * np.exp(
                -self.b[i] * mag_squared / (4 * np.pi) ** 2
            )
        form_factors += self.c
        return form_factors
