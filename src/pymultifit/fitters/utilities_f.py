"""Created on Aug 18 23:52:19 2024"""

__all__ = ['parameter_logic', 'sanity_check', '_Line', 'model_dict']

from typing import List, Tuple, Union

import numpy as np

from .. import GAUSSIAN, LOG_NORMAL, SKEW_NORMAL, LAPLACE, LINE
from .. import distributions as dist

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
    Extract parameter values from a given function based on the number of parameters per fit and selected indices.

    Parameters
    ----------
    par_array : np.ndarray
        A 2D array where the first column contains the parameter values and the second contains its standard errors.
    n_par : int
        The number of parameters per fit (e.g., amplitude, mu, sigma, etc.).
    selected_models : int, list of int, or None
        Indices of model components to extract.
        - If None, selects all components.
        - If int or list of int, selects the specified components (1-based indexing).

    Returns
    -------
    np.ndarray
        A 2D array containing the selected parameter values for the specified mean and error values for the fit.
    """
    indices = np.array(selected_models) - 1 if selected_models is not None else slice(None)
    return par_array.reshape(-1, n_par)[indices]


class _Line:
    """
    Helper class for the line fitting function.

    This class is intended for internal use only.
    Provides a wrapper for evaluating a linear function with a given slope and intercept.
    """

    def __init__(self, slope: float, intercept: float, normalize: bool = False):
        self.slope = slope
        self.intercept = intercept

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the value of the line function.

        Parameters
        ----------
        x: np.ndarray
            The input array to evaluate the line function.

        Returns
        -------
        np.ndarray
            The value of the line function for the given slope and intercept.
        """
        return dist.line(x=x, slope=self.slope, intercept=self.intercept)


model_dict = {LINE: [_Line, 2],
              GAUSSIAN: [dist.GaussianDistribution, 3],
              LOG_NORMAL: [dist.LogNormalDistribution, 3],
              SKEW_NORMAL: [dist.SkewNormalDistribution, 4],
              LAPLACE: [dist.LaplaceDistribution, 3]}
