"""
This module contains unit tests for the utils.py module.
"""

from typing import cast
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pytest
from B8_project import utils
from B8_project.crystal import Atom

RUN_VISUAL_TESTS = False

def test_duplicate_elements_normal_operation():
    """
    A unit test for the duplicate_elements function. This unit test tests normal
    operation of the function.
    """
    assert utils.duplicate_elements([0], 1) == [0]
    assert utils.duplicate_elements(["a"], 2) == ["a", "a"]
    assert utils.duplicate_elements([1, 2, 3], 1) == [1, 2, 3]
    assert utils.duplicate_elements([4, 5, 6], 2) == [4, 4, 5, 5, 6, 6]
    assert utils.duplicate_elements(["a", "b", "c"], 3) == [
        "a",
        "a",
        "a",
        "b",
        "b",
        "b",
        "c",
        "c",
        "c",
    ]


def test_add_tuples_normal_operation():
    """
    A unit test for the add_tuples function. This unit test tests normal operation of
    the function.
    """
    assert utils.add_tuples((0, 0, 0), (0, 0, 0)) == (0, 0, 0)
    assert utils.add_tuples((1, 2, 3), (4, 5, 6)) == (5, 7, 9)
    assert utils.add_tuples((0.5, 1.0, 1.5), (0.25, 0.5, 0.75)) == (0.75, 1.5, 2.25)


def test_dot_product_tuples_normal_operation():
    """
    A unit test for the dot_product_tuples function. This unit test tests normal
    operation of the function.
    """
    assert utils.dot_product_tuples((0, 0, 0), (0, 0, 0)) == 0
    assert utils.dot_product_tuples((1, 2, 3), (4, 5, 6)) == 32
    assert np.isclose(
        utils.dot_product_tuples((0.5, 0.5, 0.5), (0.6, 0.6, 0.6)), 0.9, rtol=1e-6
    )


def test_gaussian_normal_operation():
    """
    A unit test for the gaussian function. This unit test tests normal
    operation of the function.
    """
    x = 5
    mean = 1
    width = 0.5
    amplitude = 2

    assert np.isclose(
        utils.gaussian(x, mean, width, amplitude),
        amplitude * np.exp(-0.5 * ((x - mean) / width) ** 2),
        rtol=1e-6,
    )


def test_random_uniform_unit_vector_is_unit_vector():
    """
    Test that the random unit vectors have length unity.
    """

    n = 500
    for dim in range(2, 5):
        vectors = [utils.random_uniform_unit_vector(dim) for _ in range(n)]
        for vector in vectors:
            mag_squared = 0.0
            for e in vector:
                mag_squared += e**2
            nptest.assert_allclose(mag_squared, 1.0)


def test_random_uniform_unit_vectors_are_unit_vectors():
    """
    Test that the random unit vectors have length unity.
    """

    n = 500
    for dim in range(2, 5):
        vectors = utils.random_uniform_unit_vectors(n, dim)
        for vector in vectors:
            mag_squared = 0.0
            for e in vector:
                mag_squared += e**2
            nptest.assert_allclose(mag_squared, 1.0)


def plot_3d_unit_vectors(vectors, title):
    """
    Plots 3D unit vectors using matplotlib.

    Parameters:
    vectors : np.ndarray
        Array of 3D vectors to plot.
    title : string
        Title of the plot.
    """
    fig = plt.figure()
    ax = cast(Axes3D, fig.add_subplot(projection="3d"))
    xs, ys, zs = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z", fontsize=12)
    ax.set_aspect("auto")
    plt.title(title)
    plt.show()

def test_random_uniform_3d_unit_vector():
    """
    Visual test to check if the generated random unit vectors seem spherically uniform
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Test skipped: visual tests are off.")
    n = 500
    vectors = [utils.random_uniform_unit_vector(3) for _ in range(n)]
    vectors = np.array(vectors)
    plot_3d_unit_vectors(
        vectors, "Visual check for spherical uniformity: random_uniform_unit_vector"
    )


def test_random_uniform_3d_unit_vectors():
    """
    Visual test to check if the generated random unit vectors seem spherically uniform
    """
    if not RUN_VISUAL_TESTS:
        pytest.skip("Test skipped: visual tests are off.")
    n = 500
    vectors = utils.random_uniform_unit_vectors(n, 3)
    plot_3d_unit_vectors(
        vectors, "Visual check for spherical uniformity: random_uniform_unit_vectors"
    )

def test_equals_lists():
    """
    Tests that two lists are equal when they are exactly the same (ordered)
    """
    assert utils.equals([1, 2, 3], [1, 2, 3]) is True
    assert utils.equals([1, 2, 1], [2, 1, 1]) is False
    assert utils.equals([1., 2., 3.], [1., 2., 3.]) is True
    assert utils.equals([1., 2., 3.], [1., 2., 3.0000000001]) is False
    assert utils.equals([1], [1, 2, 3]) is False
    assert utils.equals(["asdf"], ["asdf"]) is True

def test_equals_arrays():
    """
    Tests that two arrays are equal when they are exactly the same (ordered)
    """
    assert utils.equals(np.array([1., 2., 3.]), np.array([1., 2., 3.])) is True
    assert utils.equals(np.array([1., 2., 3.]), np.array([1., 2., 3.0000001])) is False

obj1 = Atom(1, (1, 2, 3))
obj2 = Atom(1, (1, 2, 3))
obj3 = Atom(3, (1, 2, 3))

def test_equals_objects():
    """
    Tests that objects are equal according to __eq__
    """
    assert utils.equals(obj1, obj2) is True
    assert utils.equals([obj1], [obj2]) is True
    assert utils.equals([obj1], [obj3]) is False

def test_equals_different_types():
    """
    Tests that objects of different types that cannot be compared are handled.
    """
    assert utils.equals(1, 1.0) is False
    assert utils.equals(1, "1") is False

def test_is_close_lists():
    """
    Tests output is true when lists are close according to rtol and atol.
    """
    assert utils.is_close([1, 2, 3], [1, 2, 3]) is True
    assert utils.is_close([1, 2, 1], [2, 1, 1]) is False
    assert utils.is_close([1., 2., 3.], [1., 2., 3.]) is True
    assert utils.is_close([1., 2., 3.], [1., 2., 3.0000000001]) is True
    assert utils.is_close([1], [1, 2, 3]) is False

def test_is_close_arrays():
    """
    Tests output is true when arrays are close according to rtol and atol.
    """
    assert utils.is_close(np.array([1., 2., 3.]), np.array([1., 2., 3.])) is True
    assert utils.is_close(np.array([1., 2., 3.]), np.array([1., 2., 3.0000001])) is True
    assert utils.is_close(np.array([1., 2., 3.]), np.array([1., 2., 3.0001])) is False

def test_is_close_different_types():
    """
    Tests that objects of different types that cannot be compared are handled.
    """
    assert utils.equals(1, 1.0) is False
    assert utils.equals(1, "1") is False

def test_have_same_elements_list():
    """
    Tests contents of two lists are compared, regardless of order.
    """
    assert utils.have_same_elements([1, 2, 3], [3, 1, 2]) is True
    assert utils.have_same_elements([[1, 2], [3, 4]], [[1, 2], [3, 4]]) is True
    assert utils.have_same_elements([[1, 2], [3, 4]], [[3, 4], [1, 2]]) is True
    assert utils.have_same_elements([[1., 2.], [3., 4.]],
                                    [[1., 2.], [3., 4.000000001]]) is False
    assert utils.have_same_elements([[1, obj1], [2, obj3]],
                                    [[2, obj3], [1, obj1]]) is True

def test_have_same_elements_array():
    """
    Tests contents of two arrays are compared, regardless of order.
    """
    assert utils.have_same_elements(np.array([[1, 2], [3, 4]]),
                                     np.array([[3, 4], [1, 2]])) is True
    assert utils.have_same_elements(np.array([[1., 2.], [3., 4.]]),
                                    np.array([[3., 4.], [1., 2.]])) is True
    assert utils.have_same_elements(np.array([[1., 2.], [3., 4.00000001]]),
                                    np.array([[3., 4.], [1., 2.]])) is False

def test_have_same_elements_list_close():
    """
    Tests contents of two lists are compared, regardless of order.
    """
    assert utils.have_same_elements([[1, 2], [3, 4]], [[3, 4], [1, 2]],
                                    close=True) is True
    assert utils.have_same_elements([[1., 2.], [3., 4.]],
                                    [[3., 4.000000001], [1., 2.]], close=True) is True
    assert utils.have_same_elements([[1., 2.], [3., 4.]],
                                    [[3., 4.00001], [1., 2.]], close=True) is False

def test_have_same_elements_array_close():
    """
    Tests contents of two arrays are compared, regardless of order.
    """
    assert utils.have_same_elements(np.array([[1., 2.], [3., 4.00000001]]),
                                    np.array([[3., 4.], [1., 2.]]), close=True) is True
    assert utils.have_same_elements(np.array([[1., 2.], [3., 4.0001]]),
                                    np.array([[3., 4.], [1., 2.]]), close=True) is False
