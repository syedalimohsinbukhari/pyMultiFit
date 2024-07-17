"""Created on Jul 18 00:16:01 2024"""

from typing import Optional

import numpy as np


class BaseFitter:

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
        self.n_fits = n_fits
        self.x_values = x_values
        self.y_values = y_values
        self.max_iterations = max_iterations

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

    @staticmethod
    def _fitter(x, params):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _n_fitter(self, x, *params):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _plot_individual_fitter(self, x, plotter):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _standard_errors(self):
        if self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return np.sqrt(np.diag(self.covariance))

    def fit(self, p0):
        """
        Fit the data.

        Parameters
        ----------
        p0: List[int | float | ...]
            A list of initial guesses for the parameters of the Gaussian model.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_fit_values(self):
        """
        Get the fitted values.

        Returns
        -------
        np.ndarray
            An array of fitted values.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_value_error_pair(self, only_values=False, only_errors=False):
        """
        Get the value/error pair of the fitted values.

        Parameters
        ----------
        only_values: bool, optional
            Whether to only give the values of the fitted parameters. Defaults to False.
        only_errors: bool, optional
            Whether to only give the errors of the fitted parameters. Defaults to False.

        Returns
        -------
        np.ndarray
            An array consisting of only values, only errors or both.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
