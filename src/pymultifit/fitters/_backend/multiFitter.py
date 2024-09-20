"""Created on Jul 18 00:16:01 2024"""

from itertools import chain
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# safe keeping class names for spelling mistakes
_gaussian = 'GaussianFitter'
_lNormal = 'LogNormalFitter'
_skNormal = 'SkewedNormalFitter'
_laplace = 'Laplace Fitter'


class BaseFitter:
    """The base class for multi-fitting functionality."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
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
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _plot_individual_fitter(self, x, plotter):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def format_param(value, threshold=0.001):
        """Formats the parameter value based on its magnitude."""
        return f'{value:.3E}' if abs(value) < threshold else f'{value:.3f}'

    def _get_bounds(self):
        """
        Sets the bounds for each parameter based on the model list.

        Returns
        -------
        tuple of array_like
            Lower and upper bounds for the parameters.
        """
        lower_bounds, upper_bounds = [], []

        class_name = self.__class__.__name__

        if class_name in [_gaussian, _lNormal, _laplace]:
            lower_bounds.extend([0, -np.inf, 0])
            upper_bounds.extend([np.inf, np.inf, np.inf])
        elif class_name == _skNormal:
            lower_bounds.extend([-np.inf, -np.inf, -np.inf, 0])
            upper_bounds.extend([np.inf, np.inf, np.inf, np.inf])

        return lower_bounds, upper_bounds

    def fit(self, p0):
        """
        Fit the data.

        Parameters
        ----------
        p0: List[int | float | ...]
            A list of initial guesses for the parameters of the Gaussian model.
        """
        len_guess = len(list(chain(*p0)))
        total_pars = self.n_par * self.n_fits

        if len_guess != total_pars:
            raise ValueError(f"Initial guess length must be {3 * self.n_fits}.")
        _ = curve_fit(self._n_fitter, self.x_values, self.y_values, p0=p0,
                      maxfev=self.max_iterations)

        self.params, self.covariance = _[0], _[1]

    def _get_overall_parameter_values(self):
        """
        Returns the overall parameters of the multi-fitter.

        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing the amplitude and mean value of the multi-fitter.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def parameter_extractor(self):
        """Extract the required parameters from the fitters."""
        raise NotImplementedError("This method should be implemented by subclasses.")

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
        return self._n_fitter(self.x_values, *self.params)

    def get_value_error_pair(self, mean_values=False, std_values=False) -> np.ndarray:
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

        return pairs[:, 0] if mean_values else pairs[:, 1] if std_values else pairs

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
            self._plot_individual_fitter(self.x_values, plotter)

        if auto_label:
            labels = {
                'x_label': plotter.set_xlabel if ax else plotter.xlabel,
                'y_label': plotter.set_ylabel if ax else plotter.ylabel,
                'title': plotter.set_title if ax else plotter.title,
            }
            labels['x_label']('X')
            labels['y_label']('Y')
            labels['title'](f'{self.n_fits} {self.__class__.__name__} fit')
            plotter.legend(loc='best')
            plotter.tight_layout()

        return plotter
