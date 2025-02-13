"""
This module contains unit tests for the form_factor module.
"""

import numpy as np
import numpy.testing as nptest
import pytest
from B8_project.crystal import ReciprocalLatticeVector
from B8_project.form_factor import FormFactorProtocol, NeutronFormFactor, XRayFormFactor


class TestFormFactorProtocol:
    """
    Unit tests for the form factor protocol
    """

    @staticmethod
    def test_neutron_form_factor_implements_protocol():
        """
        A unit test to check if the `NeutronFormFactor` class implements the form
        factor protocol
        """
        neutron_form_factor = NeutronFormFactor(neutron_scattering_length=1.0)
        assert isinstance(neutron_form_factor, FormFactorProtocol)

    @staticmethod
    def test_x_ray_form_factor_implements_protocol():
        """
        A unit test to check if the `XRayFormFactor` class implements the form
        factor protocol
        """
        x_ray_form_factor = XRayFormFactor(1, 1, 1, 1, 1, 1, 1, 1, 1)
        assert isinstance(x_ray_form_factor, FormFactorProtocol)


class TestNeutronFormFactor:
    """
    Unit tests for the `NeutronFormFactor` class.
    """

    @pytest.fixture(name="neutron_form_factor")
    def fixture_neutron_form_factor(self):
        """
        Returns an instance of NeutronFormFactor
        """
        yield NeutronFormFactor(1)

    @staticmethod
    def test_neutron_form_factor_initialization_normal_operation(neutron_form_factor):
        """
        A unit test that tests the initialization of a `NeutronFormFactor` instance.
        This unit test tests initialization with normal attributes.
        """
        assert neutron_form_factor.neutron_scattering_length == 1

    @staticmethod
    def test_form_factor_normal_operation(neutron_form_factor):
        """
        A unit test for the form_factor function. This unit test tests normal
        operation of the function.
        """
        assert (
            neutron_form_factor.evaluate_form_factor(
                ReciprocalLatticeVector((0, 0, 0), (1, 1, 1))
            )
            == 1
        )

    @staticmethod
    def test_evaluate_form_factors(neutron_form_factor):
        """
        A unit test for the evaluate_form_factors function.
        """
        k_vecs = np.random.random((100, 3))
        nptest.assert_allclose(
            neutron_form_factor.evaluate_form_factors(k_vecs),
            np.ones(100)
        )

class TestXRayFormFactor:
    """
    Unit tests for the `XRayFormFactor` class.
    """
    @pytest.fixture(name="xray_form_factor")
    def fixture_xray_form_factor(self):
        """
        Returns an instance of XRayFormFactor
        """
        yield XRayFormFactor(1, 2, 3, 4, 5, 6, 7, 8, 9)

    @staticmethod
    def test_xray_form_factor_initialization_normal_operation(xray_form_factor):
        """
        A unit test that tests the initialization of a `XRayFormFactor` instance.
        This unit test tests initialization with normal attributes.
        """
        assert xray_form_factor.a1 == 1
        assert xray_form_factor.b1 == 2
        assert xray_form_factor.a2 == 3
        assert xray_form_factor.b2 == 4
        assert xray_form_factor.a3 == 5
        assert xray_form_factor.b3 == 6
        assert xray_form_factor.a4 == 7
        assert xray_form_factor.b4 == 8
        assert xray_form_factor.c == 9
        assert xray_form_factor.a == [1, 3, 5, 7]
        assert xray_form_factor.b == [2, 4, 6, 8]

    @staticmethod
    def test_form_factor(xray_form_factor):
        """
        A unit test for the form_factor function. This unit test tests normal
        operation of the function.
        """
        reciprocal_lattice_vector = ReciprocalLatticeVector((1, 0, 0), (1, 1, 1))

        assert np.isclose(reciprocal_lattice_vector.magnitude(), 2 * np.pi, rtol=1e-6)

        result = xray_form_factor.evaluate_form_factor(reciprocal_lattice_vector)

        a = [1, 3, 5, 7]
        b = [2, 4, 6, 8]
        c = 9

        expected_result = 0

        for i in range(4):
            expected_result += a[i] * np.exp(-b[i] / 4)

        expected_result += c

        assert np.isclose(result, expected_result, rtol=1e-6)

    @staticmethod
    def test_evaluate_form_factors(xray_form_factor, mocker):
        """
        A unit test for the evaluate_form_factors function.
        """
        k_vecs = np.random.random((100, 3))
        form_factors = xray_form_factor.evaluate_form_factors(k_vecs)

        # Use evaluate_form_factor to validate result - hence need to mock the
        # magnitude method
        mocker.patch(
            "B8_project.crystal.ReciprocalLatticeVector.magnitude",
            side_effect=np.linalg.norm(k_vecs, axis=1),
        )
        dummy_rlv = ReciprocalLatticeVector((1, 0, 0), (1, 1, 1))
        for i, _ in enumerate(k_vecs):
            nptest.assert_allclose(form_factors[i],
                                   xray_form_factor.evaluate_form_factor(dummy_rlv))
