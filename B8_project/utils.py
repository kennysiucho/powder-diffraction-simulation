"""
This module includes useful functions that perform arithmetic operations.
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

def equals(a, b) -> bool:
    """
    Checks if two elements (lists, arrays, objects) are equal.
    """
    if not isinstance(a, type(b)):
        return False
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray): # Compare NumPy arrays
        return np.array_equal(a, b)
    return a == b  # For lists, scalars, strings, and objects

def is_close(a, b, rtol=1e-6, atol=1e-6) -> bool:
    """
    Checks if two objects are close (applicable for lists, arrays, int, float)
    """
    if not isinstance(a, type(b)):
        return False
    if isinstance(a, (list, np.ndarray, int, float)):
        return np.allclose(a, b, rtol=rtol, atol=atol)
    return a == b # for strings, objects

def have_same_elements(list1, list2, close: bool=None, rtol=1e-6, atol=1e-6) -> bool:
    """
    Checks if two lists contain the same elements, regardless of order.

    If close=True, lists can only contain numeric values.
    """
    if len(list1) != len(list2):
        return False
    # Convert elements to a comparable form and sort them
    sorted_list1 = sorted(list1, key=lambda x: str(type(x)) + str(x))
    sorted_list2 = sorted(list2, key=lambda x: str(type(x)) + str(x))

    if close:
        return all(is_close(a, b, rtol, atol) for a, b in zip(sorted_list1,
                                                              sorted_list2))
    return all(equals(a, b) for a, b in zip(sorted_list1, sorted_list2))
