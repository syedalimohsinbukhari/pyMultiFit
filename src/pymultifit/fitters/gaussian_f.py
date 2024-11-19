"""Created on Jul 18 00:25:57 2024"""

from typing import Any

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

    def _n_fitter(self, x: np.ndarray, *params: Any) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)
        params = np.reshape(params, (self.n_fits, self.n_par))
        for amp, mu, sigma in params:
            y += self._fitter(x=x, params=[amp, mu, sigma])
        return y

    def _plot_individual_fitter(self, x: np.ndarray, plotter):
        params = np.reshape(self.params, (self.n_fits, self.n_par))
        for i, (amp, mu, sigma) in enumerate(params):
            plotter.plot(x, self._fitter(x=x, params=[amp, mu, sigma]),
                         '--', label=f'Gaussian {i + 1}('
                                     f'{self.format_param(amp)}, '
                                     f'{self.format_param(mu)}, '
                                     f'{self.format_param(sigma)})')
