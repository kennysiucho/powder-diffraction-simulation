"""
Form factor
===========

This module contains classes which represent atomic form factors. Classes which 
represent form factors should implement the interface defined by FormFactorProtocol.

Classes
-------
TODO: add classes.
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
