"""Created on Jul 18 13:54:03 2024"""

from typing import Optional

import numpy as np
from scipy.stats import skewnorm

from .backend import BaseFitter


class SkewedNormal(BaseFitter):
    """A class for fitting multiple Skewed Normal functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
        super().__init__(n_fits, x_values, y_values, max_iterations)
        self.n_parameters = 4

    @staticmethod
    def _fitter(x, params):
        return params[0] * skewnorm.pdf(x, params[1], loc=params[2], scale=params[3])

    def _n_fitter(self, x, *params):
        y = np.zeros_like(x)
        for i in range(self.n_fits):
            amp = params[i * 4]
            shape = params[i * 4 + 1]
            loc = params[i * 4 + 2]
            scale = params[i * 4 + 3]
            y += self._fitter(x, [amp, shape, loc, scale])
        return y

    def _plot_individual_fitter(self, x, plotter):
        for i in range(self.n_fits):
            amp = self.params[i * 4]
            shape = self.params[i * 4 + 1]
            loc = self.params[i * 4 + 2]
            scale = self.params[i * 4 + 3]
            plotter.plot(x, self._fitter(x, [amp, shape, loc, scale]), linestyle=':', label=f'Skewed Normal {i + 1}')
