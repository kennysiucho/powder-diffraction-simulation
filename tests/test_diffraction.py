"""
This module contains unit tests for the diffraction module.

TODO: add unit test for _get_structure_factor.
TODO: add unit test for get_diffraction_peaks.
TODO: add unit test for plot_diffraction_pattern.
"""

import numpy as np

from B8_project.crystal import ReciprocalLatticeVector
from B8_project.diffraction import (
    _get_reciprocal_lattice_vector_magnitude,
    _get_deflection_angle,
)


def test_get_reciprocal_lattice_vector_magnitude_normal_operation():
    """
    A unit test for the _get_reciprocal_lattice_vector function. This unit test tests
    normal operation of the function.
    """
    deflection_angle = 90
    wavelength = 1
    result = _get_reciprocal_lattice_vector_magnitude(deflection_angle, wavelength)

    angle = deflection_angle * np.pi / 360
    expected_result = 4 * np.pi * np.sin(angle) / wavelength

    assert np.isclose(result, expected_result, rtol=1e-6)


def test_get_deflection_angle_normal_operation():
    """
    A unit test for the _get_deflection_angle function. This unit test tests
    normal operation of the function.
    """
    rlv = ReciprocalLatticeVector((1, 0, 0), (1, 1, 1))
    wavelength = 1

    result = _get_deflection_angle(rlv, wavelength)

    sin_angle = 0.5
    angle = np.arcsin(sin_angle)
    expected_result = angle * 360 / np.pi

    assert np.isclose(result, expected_result, rtol=1e-6)
