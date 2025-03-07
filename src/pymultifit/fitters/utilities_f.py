"""Created on Aug 18 23:52:19 2024"""

__all__ = ['parameter_logic', 'sanity_check', 'plot_fit']

from typing import List, Tuple, Union, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpyez.backend.uPlotting import LinePlot
from mpyez.ezPlotting import plot_xy

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


def plot_fit(x_values, y_values, parameters, n_fits, class_name, _n_fitter, _n_plotter, show_individuals: bool = False,
             x_label: Optional[str] = None, y_label: Optional[str] = None, title: Optional[str] = None,
             data_label: Union[list[str], str] = None, axis: Optional[Axes] = None):
    """
    Plot the fitted models.

    Parameters
    ----------
    show_individuals: bool, optional
        Whether to show individually fitted models or not.
    x_label: str, optional
        The label for the x-axis.
    y_label: str, optional
        The label for the y-axis.
    title: str, optional
        The title for the plot.
    data_label: str, optional
        The label for the data.
    axis: Axes, optional
        Axes to plot instead of the entire figure. Defaults to None.

    Returns
    -------
    plotter
        The plotter handle for the drawn plot.
    """
    if parameters is None:
        raise RuntimeError("Fit not performed yet. Call fit() first.")

    if 1 < len(data_label) <= 2:
        dl, tt = data_label
    elif len(data_label) == 1 or isinstance(data_label, str):
        dl, tt = data_label, 'Total fit'
    else:
        raise ValueError()

    plotter = plot_xy(x_data=x_values, y_data=y_values, data_label=dl, axis=axis)

    plot_xy(x_data=x_values, y_data=_n_fitter(x_values, *parameters),
            x_label=x_label, y_label=y_label, plot_title=title, data_label=tt,
            plot_dictionary=LinePlot(color='k'), axis=plotter)

    if show_individuals:
        _n_plotter(plotter=plotter)

    plotter.set_xlabel(x_label if x_label else 'X')
    plotter.set_ylabel(y_label if y_label else 'Y')
    plotter.set_title(title if title else f'{n_fits} {class_name} fit')
    plt.tight_layout()

    return plotter
