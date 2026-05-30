"""Created on Aug 18 23:52:19 2024"""

__all__ = ["parameter_logic", "sanity_check"]

import numpy as np

from ..typing import ArrayLike, NDArray

# SAFEGUARD:
xy_tuple = tuple[NDArray, NDArray]
indexType = int | list[int] | None


def sanity_check(x_values: ArrayLike, y_values: ArrayLike) -> xy_tuple:
    """
    Convert input lists to NumPy arrays if necessary.

    Parameters
    ----------
    x_values :
        Input x-values that will be converted to a NumPy array if they are in list format.
    y_values :
        Input y-values that will be converted to a NumPy array if they are in list format.

    Returns
    -------
    x_values :
        The x-values as a NumPy array.
    y_values :
        The y-values as a NumPy array.
    """
    x_values = np.asarray(a=x_values, dtype=float)
    y_values = np.asarray(a=y_values, dtype=float)

    return x_values, y_values


def parameter_logic(par_array: NDArray, n_par: int, selected_models) -> NDArray:
    """
    Extract parameter values from a given function based on the number of parameters per fit and selected indices.

    Parameters
    ----------
    par_array :
        A 2D array where the first column contains the parameter values and the second contains its standard errors.
    n_par :
        The number of parameters per fit (e.g., amplitude, mu, sigma, etc.).
    selected_models :
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
