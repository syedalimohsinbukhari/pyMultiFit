"""Created on Jul 18 13:54:03 2024"""

from typing import Optional

import numpy as np
from scipy.stats import skewnorm

from ._backend import BaseFitter


# TODO:
#   Implement `_get_overall_parameter_values`
#   Implement `parameter_extractor`


class SkewedNormalFitter(BaseFitter):
    """A class for fitting multiple Skewed Normal functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
        super().__init__(n_fits, x_values, y_values, max_iterations)
        self.n_par = 3

    @staticmethod
    def _fitter(x, params):
        # return params[0] * skewnorm.pdf(x, params[1], loc=params[2], scale=params[3])
        return skewnorm.pdf(x, params[0], loc=params[1], scale=params[2])

    def _n_fitter(self, x, *params):
        y = np.zeros_like(x)
        for i in range(self.n_fits):
            # amp = params[i * self.n_par]
            shape = params[i * self.n_par]
            loc = params[i * self.n_par + 1]
            scale = params[i * self.n_par + 2]
            y += self._fitter(x, [shape, loc, scale])
        return y

    def _plot_individual_fitter(self, x, plotter):
        for i in range(self.n_fits):
            # amp = self.params[i * self.n_par]
            shape = self.params[i * self.n_par]
            loc = self.params[i * self.n_par + 1]
            scale = self.params[i * self.n_par + 2]
            plotter.plot(x, self._fitter(x, [shape, loc, scale]), linestyle=':', label=f'Skewed Normal {i + 1}')
