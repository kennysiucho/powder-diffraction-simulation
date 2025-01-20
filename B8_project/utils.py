"""
This module includes useful functions for performing arithmetic operations with tuples.

Functions:
TODO: add functions.

Example usage:
TODO: add example usage.
"""


def duplicate_elements(original_list: list, num_duplicates: int) -> list:
    """
    Duplicate elements
    ==================

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
    Add tuples
    ==========

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
    Dot product tuples
    ==========

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
