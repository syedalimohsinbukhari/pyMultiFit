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
    def individual_fitter(x, *params):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def fit(self, p0):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_standard_errors(self):
        if self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return np.sqrt(np.diag(self.covariance))

    def get_value_error_pair(self, only_values=False, only_errors=False):
        raise NotImplementedError("This method should be implemented by subclasses.")
