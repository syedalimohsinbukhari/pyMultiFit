"""Created on Jul 18 00:16:01 2024"""

from itertools import chain
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, curve_fit

from .utilities import parameter_logic

# safe keeping class names for spelling mistakes
_gaussian = 'GaussianFitter'
_lNormal = 'LogNormalFitter'
_skNormal = 'SkewedNormalFitter'
_laplace = 'Laplace Fitter'


class BaseFitter:
    """The base class for multi-fitting functionality."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        self.n_fits = n_fits
        self.x_values = x_values
        self.y_values = y_values
        self.max_iterations = max_iterations
        self.n_par = None

        self.params = None
        self.covariance = None

    def _params(self):
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self.params

    def _covariance(self):
        if self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self.covariance

    def _standard_errors(self):
        if self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return np.sqrt(np.diag(self.covariance))

    @staticmethod
    def _fitter(x, params):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _n_fitter(self, x, *params):
        y = np.zeros_like(a=x, dtype=float)
        params = np.reshape(a=params, newshape=(self.n_fits, self.n_par))
        for par in params:
            y += self._fitter(x=x, params=par)
        return y

    def _plot_individual_fitter(self, x, plotter):
        params = np.reshape(a=self.params, newshape=(self.n_fits, self.n_par))
        for i, par in enumerate(params):
            plotter.plot(x, self._fitter(x=x, params=par),
                         '--', label=f'{self.__class__.__name__.replace("Fitter", "")} {i + 1}('
                                     f'{", ".join(self.format_param(i) for i in par)})')

    @staticmethod
    def format_param(value, t_low=0.001, t_high=10_000):
        """Formats the parameter value based on its magnitude."""
        return f'{value:.3E}' if t_high < abs(value) or abs(value) < t_low else f'{value:.3f}'

    def _get_bounds(self):
        """
        Sets the bounds for each parameter based on the model list.

        Returns
        -------
        tuple of array_like
            Lower and upper bounds for the parameters.
        """
        class_name = self.__class__.__name__

        if class_name in [_gaussian, _lNormal, _laplace]:
            return Bounds(lb=[0, -np.inf, 0] * self.n_fits, ub=[np.inf, np.inf, np.inf] * self.n_fits)
        elif class_name == _skNormal:
            return Bounds(lb=[-np.inf, -np.inf, -np.inf, 0] * self.n_fits,
                          ub=[np.inf, np.inf, np.inf, np.inf] * self.n_fits)
        else:
            return Bounds()

    def fit(self, p0):
        """
        Fit the data.

        Parameters
        ----------
        p0: List[int | float | ...]
            A list of initial guesses for the parameters of the Gaussian model.

        Raises
        ------
        ValueError
            If the length of initial parameters does not match the expected count.
        """
        len_guess = len(list(chain(*p0)))
        total_pars = self.n_par * self.n_fits

        if len_guess != total_pars:
            raise ValueError(f"Initial guess length must be {total_pars}.")
        # flatten idea taken from https://stackoverflow.com/a/73000598/3212945
        self.params, self.covariance, *_ = curve_fit(f=self._n_fitter, xdata=self.x_values, ydata=self.y_values,
                                                     p0=np.array(p0).flatten(), maxfev=self.max_iterations, bounds=self._get_bounds())

    def get_fit_values(self):
        """
        Get the fitted values.

        Returns
        -------
        np.ndarray
            An array of fitted values.
        """
        if self.params is None:
            raise RuntimeError('Fit not performed yet. Call fit() first.')
        return self._n_fitter(self.x_values, self.params)

    def get_value_error_pair(self, mean_values=False, std_values=False):
        """
        Get the value/error pair of the fitted values.

        Parameters
        ----------
        mean_values: bool, optional
            Whether to only give the values of the fitted parameters. Defaults to False.
        std_values: bool, optional
            Whether to only give the errors of the fitted parameters. Defaults to False.

        Returns
        -------
        np.ndarray
            An array consisting of only values, only errors or both.
        """
        pairs = np.array([np.array([i, j]) for i, j in zip(self._params(), self._standard_errors())])

        if mean_values and std_values:
            return pairs
        elif mean_values:
            return pairs[:, 0]
        elif std_values:
            return pairs[:, 1]

    def plot_fit(self, show_individual: bool = False, auto_label: bool = False,
                 fig_size: Optional[Tuple[int, int]] = (12, 6), ax: Optional[plt.Axes] = None) -> plt:
        """
        Plot the fitted Skewed Normal functions on the data.

        Parameters
        ----------
        show_individual: bool
            Whether to show individually fitted Skewed Normal functions.
        auto_label: bool, optional
            Whether to auto decorate the plot with x/y labels, title and other plot decorators. Defaults to False.
        fig_size: Tuple[int, int], optional
            Figure size to draw the plot on. Defaults to (12, 6)
        ax: plt.Axes, optional
            Axes to plot instead of the entire figure. Defaults to None.

        Returns
        -------
        plt
            The plotter handle for the drawn plot.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        plotter = ax if ax else plt
        if not ax:
            plt.figure(figsize=fig_size)

        plotter.plot(self.x_values, self.y_values, label='Data')
        plotter.plot(self.x_values, self._n_fitter(self.x_values, *self.params),
                     label='Total Fit', color='k')

        if show_individual:
            self._plot_individual_fitter(x=self.x_values, plotter=plotter)

        if auto_label:
            labels = {'x_label': plotter.set_xlabel if ax else plotter.xlabel,
                      'y_label': plotter.set_ylabel if ax else plotter.ylabel,
                      'title': plotter.set_title if ax else plotter.title}
            labels['x_label']('X')
            labels['y_label']('Y')
            labels['title'](f'{self.n_fits} {self.__class__.__name__} fit')
            plotter.legend(loc='best')
            plotter.tight_layout()

        return plotter

    def get_parameters(self, select: Tuple[int, Any] = None, errors: bool = False):
        """ Extracts specific parameter values from the fitting process.

        Parameters
        ----------
        select : list of int or None, optional
            A list of indices specifying which sub-model to return values for.
            If None, returns values for all sub-models. Defaults to None.
        errors : bool, optional
            Whether to return the standard deviations of the selected parameters. Defaults to False.

        Returns
        -------
        tuple:
            Arrays corresponding to model parameters.

        Notes
        -----
        - The `select` parameter is used to filter the returned values to specific sub-model based on their indices. Indexing starts at 1.
        """

        parameter_mean = self.get_value_error_pair(mean_values=True, std_values=errors)

        if not errors:
            selected = parameter_logic(par_array=parameter_mean,
                                       n_par=self.n_par, selected_models=select)

            return selected[:, range(self.n_par)].T
        else:
            par_list = parameter_mean.reshape(self.n_fits, self.n_par, 2)
            mean = parameter_logic(par_array=par_list[:, :, 0].flatten(),
                                   n_par=self.n_par, selected_models=select)
            std_ = parameter_logic(par_array=par_list[:, :, 1].flatten(),
                                   n_par=self.n_par, selected_models=select)

            return mean[:, range(self.n_par)].T, std_[:, range(self.n_par)].T
