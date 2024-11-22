"""Created on Jul 18 00:25:57 2024"""

import numpy as np

from ._backend.baseFitter import BaseFitter
from ..distributions.gaussian_d import gaussianWA


class GaussianFitter(BaseFitter):
    """A class for fitting multiple Gaussian functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    @staticmethod
    def _fitter(x: np.ndarray, params: list) -> np.array:
        return gaussianWA(*params).pdf(x)
