"""Created on Aug 18 23:52:19 2024"""

from typing import List, Tuple, Union

import numpy as np

# SAFEGUARD:
xy_values = Union[List[float], np.ndarray]
xy_tuple = Tuple[np.ndarray, np.ndarray]
indexType = Union[int, List[int], None]


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


def sanity_check(x_values: xy_values, y_values: xy_values) -> xy_tuple:
    """
    Convert input lists to NumPy arrays if necessary.

    Parameters
    ----------
    x_values : list of float or np.ndarray
        Input x-values that will be converted to a NumPy array if they are in list format.
    y_values : list of float or np.ndarray
        Input y-values that will be converted to a NumPy array if they are in list format.

    Returns
    -------
    x_values : np.ndarray
        The x-values as a NumPy array.
    y_values : np.ndarray
        The y-values as a NumPy array.
    """
    if isinstance(x_values, list):
        x_values = np.array(x_values)

    if isinstance(y_values, list):
        y_values = np.array(y_values)

    return x_values, y_values


def parameter_logic(par_array: np.ndarray, n_par: int, selected_models: indexType) -> np.ndarray:
    """
    Extracts specific parameter values from a given function based on the number of parameters per fit and selected indices.

    Parameters
    ----------
    par_array : np.ndarray
        A 2D array where the first column contains the parameter values.
    n_par : int
        The number of parameters per fit (e.g., amplitude, mu, sigma, etc.).
    selected_models : int, list of int, or None
        Indices of model components to extract.
        - If None, selects all components.
        - If int or list of int, selects the specified components (1-based indexing).

    Returns
    -------
    np.ndarray
        A 2D array containing the selected parameter values for the specified Gaussian components.
    """
    indices = np.array(selected_models) - 1 if selected_models is not None else slice(None)
    return par_array.reshape(-1, n_par)[indices]
