"""
This module contains unit tests for the utils.py module.
TODO: add tests for error handling for duplicate_elements.
TODO: add tests for error handling for add_tuples.
TODO: add tests for error handling for dot_product_tuples.
"""

import math
from B8_project.utils import duplicate_elements, add_tuples, dot_product_tuples


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
    assert math.isclose(
        dot_product_tuples((0.5, 0.5, 0.5), (0.6, 0.6, 0.6)), 0.9, rel_tol=1e-6
    )
