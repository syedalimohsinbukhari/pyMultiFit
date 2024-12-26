"""Created on Aug 18 23:52:19 2024"""
from typing import List, Tuple, Union

import numpy as np
from scipy.optimize import Bounds

from .. import GAUSSIAN, LOG_NORMAL, SKEW_NORMAL

# SAFEGUARD:
xy_values = Union[List[float], np.ndarray]
xy_tuple = Tuple[np.ndarray, np.ndarray]
indexType = Union[int, List[int], None]


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


class DistributionBounds:
    def __init__(self, n_fits):
        self.n_fits = n_fits

    def get_bounds(self, distribution_name):
        """
        Get the bounds for the given distribution type.

        Parameters
        ----------
        distribution_name : str
            Name of the distribution (e.g., "GAUSSIAN", "SKEW_NORMAL").

        Returns
        -------
        Bounds
            Lower and upper bounds for the parameters.
        """
        if distribution_name == GAUSSIAN:
            lb = [0, -np.inf, 0]
            ub = [np.inf, np.inf, np.inf]
        elif distribution_name == SKEW_NORMAL:
            lb = [-np.inf, -np.inf, -np.inf, 0]
            ub = [np.inf, np.inf, np.inf, np.inf]
        elif distribution_name == LOG_NORMAL:
            lb = [0, -np.inf, 0, -np.inf]
            ub = [np.inf, np.inf, np.inf, np.inf]
        else:
            raise ValueError(f"Bounds not defined for distribution '{distribution_name}'")

        # Scale bounds for the number of fits
        return Bounds(lb=lb * self.n_fits, ub=ub * self.n_fits)
