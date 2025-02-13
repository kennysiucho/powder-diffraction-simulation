"""
This module includes useful functions that perform arithmetic operations.

Functions:
TODO: add functions.
"""

import random
import numpy as np

def duplicate_elements(original_list: list, num_duplicates: int) -> list:
    """
    Duplicates each element in the list and place the duplicate(s) next to the original element.

    Parameters
    ----------
        - original_list (list): The original list of elements.
        - num_duplicates (int): The number of duplicates.

    Returns
    -------
        - (list): A new list with each element duplicated.
    """
    return [item for item in original_list for _ in range(num_duplicates)]


def add_tuples(
    tuple_1: tuple[float, float, float], tuple_2: tuple[float, float, float]
) -> tuple[float, float, float]:
    """
    Adds two tuples element-wise. Both tuples should contain 3 floats.

    Parameters
    ----------
        - tuple_1 (tuple[float, float, float]): The first tuple.
        - tuple_2 (tuple[float, float, float]): The second tuple.

    Returns
    -------
        - (tuple[float, float, float]): A new tuple where each element is the sum of
        the corresponding elements of `tuple_1` and `tuple_2`.
    """
    return (tuple_1[0] + tuple_2[0], tuple_1[1] + tuple_2[1], tuple_1[2] + tuple_2[2])


def dot_product_tuples(
    tuple_1: tuple[float, float, float], tuple_2: tuple[float, float, float]
) -> float:
    """
    Computes the dot product of two tuples. Both tuples should have three elements.

    Parameters
    ----------
        - tuple_1 (tuple[float, float, float]): The first tuple.
        - tuple_2 (tuple[float, float, float]): The second tuple.

    Returns
    -------
        - float: The dot product of tuple_1 and tuple_2.
    """
    return (
        (tuple_1[0] * tuple_2[0])
        + (tuple_1[1] * tuple_2[1])
        + (tuple_1[2] * tuple_2[2])
    )


def gaussian(x, mean, width, amplitude):
    """
    Evaluates a Gaussian function at a given coordinate, `x`. The gaussian function is
    given by f(x) = amplitude * exp(0.5 * ((x - mean)/width)^2).
    """
    return amplitude * np.exp(-0.5 * ((x - mean) / width) ** 2)

def random_uniform_unit_vector(dims: int):
    """
    Returns a list of length `dims` representing a unit vector uniformly and randomly
    selected from the unit sphere.
    """
    vec = [random.gauss(0, 1) for i in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]

def random_uniform_unit_vectors(n: int, dims: int):
    """
    Returns a NumPy array of shape `(n, dims)` consisting of `n` unit vectors,
    each uniformly and randomly selected
    from the unit sphere.
    """
    vecs = np.random.normal(size=(n, dims))
    mag = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / mag
