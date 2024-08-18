"""Created on Aug 18 23:52:19 2024"""

from typing import List, Union

import numpy as np


def get_y_values_at_closest_x(x_array: np.ndarray, y_array: np.ndarray,
                              target_x_values: Union[np.ndarray, List[float]]) -> List[float]:
    """Retrieve y-values from `y_array` corresponding to the `target_x_values`, by finding the closest in `x_array`.

    Parameters
    ----------
    x_array : np.ndarray
        Array of x-values. Should have the same length as y_array.
    y_array : np.ndarray
        Array of y-values corresponding to x_array.
    target_x_values : Union[np.ndarray, List[float]]
        Array or list of x-values for which the closest y-values should be retrieved.

    Returns
    -------
    List[float]
        List of y-values corresponding to the closest x-values from x_array.

    Raises
    ------
    ValueError
        If x_array and y_array have different lengths.
    """
    if len(x_array) != len(y_array):
        raise ValueError("x_array and y_array must have the same length.")

    if isinstance(target_x_values, list):
        target_x_values = np.array(target_x_values)

    if target_x_values.size == 0:
        return []

    closest_indices = np.abs(x_array[:, np.newaxis] - target_x_values).argmin(axis=0)

    return y_array[closest_indices].tolist()
