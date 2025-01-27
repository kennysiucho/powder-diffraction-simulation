"""
This module includes useful functions that perform arithmetic operations.

Functions:
TODO: add functions.

Example usage:
TODO: add example usage.
"""

import numpy as np


def duplicate_elements(original_list: list, num_duplicates: int) -> list:
    """
    Duplicate elements
    ==================

    Duplicates each element in the list and place the duplicate(s) next to the original element.
    """
    return [item for item in original_list for _ in range(num_duplicates)]


def add_tuples(
    tuple_1: tuple[float, float, float], tuple_2: tuple[float, float, float]
) -> tuple[float, float, float]:
    """
    Add tuples
    ==========

    Adds two tuples element-wise. Both tuples should contain 3 floats.
    """
    return (tuple_1[0] + tuple_2[0], tuple_1[1] + tuple_2[1], tuple_1[2] + tuple_2[2])


def dot_product_tuples(
    tuple_1: tuple[float, float, float], tuple_2: tuple[float, float, float]
) -> float:
    """
    Dot product tuples
    ==================

    Computes the dot product of two tuples. Both tuples should have three elements.
    """
    return (
        (tuple_1[0] * tuple_2[0])
        + (tuple_1[1] * tuple_2[1])
        + (tuple_1[2] * tuple_2[2])
    )


def gaussian(x, mean, width, amplitude):
    """
    Gaussian
    ========

    Evaluates a Gaussian function at a given coordinate, `x`. The gaussian function is
    given by f(x) = amplitude * exp(0.5 * ((x - mean)/width)^2).
    """
    return amplitude * np.exp(-0.5 * ((x - mean) / width) ** 2)
