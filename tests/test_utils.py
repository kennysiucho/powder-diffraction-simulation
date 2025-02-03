"""
This module contains unit tests for the utils.py module.
"""

import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from B8_project.utils import (
    duplicate_elements,
    add_tuples,
    dot_product_tuples,
    gaussian,
    random_uniform_unit_vector,
    random_uniform_unit_vectors
)


def test_duplicate_elements_normal_operation():
    """
    A unit test for the duplicate_elements function. This unit test tests normal
    operation of the function.
    """
    assert duplicate_elements([0], 1) == [0]
    assert duplicate_elements(["a"], 2) == ["a", "a"]
    assert duplicate_elements([1, 2, 3], 1) == [1, 2, 3]
    assert duplicate_elements([4, 5, 6], 2) == [4, 4, 5, 5, 6, 6]
    assert duplicate_elements(["a", "b", "c"], 3) == [
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
    assert add_tuples((0, 0, 0), (0, 0, 0)) == (0, 0, 0)
    assert add_tuples((1, 2, 3), (4, 5, 6)) == (5, 7, 9)
    assert add_tuples((0.5, 1.0, 1.5), (0.25, 0.5, 0.75)) == (0.75, 1.5, 2.25)


def test_dot_product_tuples_normal_operation():
    """
    A unit test for the dot_product_tuples function. This unit test tests normal
    operation of the function.
    """
    assert dot_product_tuples((0, 0, 0), (0, 0, 0)) == 0
    assert dot_product_tuples((1, 2, 3), (4, 5, 6)) == 32
    assert np.isclose(
        dot_product_tuples((0.5, 0.5, 0.5), (0.6, 0.6, 0.6)), 0.9, rtol=1e-6
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
        gaussian(x, mean, width, amplitude),
        amplitude * np.exp(-0.5 * ((x - mean) / width) ** 2),
        rtol=1e-6,
    )


def test_random_uniform_unit_vector_is_unit_vector():
    N = 500
    for dim in range(2, 5):
        vectors = [random_uniform_unit_vector(dim) for _ in range(N)]
        for vector in vectors:
            mag_squared = 0.0
            for e in vector:
                mag_squared += e**2
            nptest.assert_allclose(mag_squared, 1.0)


def test_random_uniform_unit_vectors_are_unit_vectors():
    N = 500
    for dim in range(2, 5):
        vectors = random_uniform_unit_vectors(N, dim)
        for vector in vectors:
            mag_squared = 0.0
            for e in vector:
                mag_squared += e**2
            nptest.assert_allclose(mag_squared, 1.0)


def plot_3d_unit_vectors(vectors, title):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    xs, ys, zs = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    plt.title(title)
    plt.show()


# These two tests are merely visual tests - check if the generated random unit vectors seem spherically uniform
# These two tests are merely visual tests - check if the generated random unit
# vectors seem spherically uniform
def test_random_uniform_3d_unit_vector():
    N = 500
    vectors = [random_uniform_unit_vector(3) for _ in range(N)]
    vectors = np.array(vectors)
    plot_3d_unit_vectors(
        vectors, "Visual check for spherical uniformity: random_uniform_unit_vector"
    )


def test_random_uniform_3d_unit_vectors():
    N = 500
    vectors = random_uniform_unit_vectors(N, 3)
    plot_3d_unit_vectors(
        vectors, "Visual check for spherical uniformity: random_uniform_unit_vectors"
    )
