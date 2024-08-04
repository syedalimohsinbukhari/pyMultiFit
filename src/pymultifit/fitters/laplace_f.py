"""Created on Jul 20 16:59:14 2024"""

from typing import Optional

import numpy as np

from ._backend import BaseFitter


# TODO:
#   Implement `_get_overall_parameter_values`
#   Implement `parameter_extractor`

class LaplaceFitter(BaseFitter):
    """A class for fitting multiple Laplace distributions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
        super().__init__(n_fits, x_values, y_values, max_iterations)
        self.n_par = 3

    @staticmethod
    def _fitter(x, params):
        return params[0] * np.exp(-abs(x - params[1]) / params[2])

    def _n_fitter(self, x, *params):
        y = np.zeros_like(x)
        for i in range(self.n_fits):
            amp = params[i * self.n_par]
            mu = params[i * self.n_par + 1]
            b = params[i * self.n_par + 2]
            y += self._fitter(x, [amp, mu, b])
        return y

    def _plot_individual_fitter(self, x, plotter):
        for i in range(self.n_fits):
            amp = self.params[i * self.n_par]
            mu = self.params[i * self.n_par + 1]
            b = self.params[i * self.n_par + 2]
            plotter.plot(x, self._fitter(x, [amp, mu, b]), ls=':', label=f'Laplace {i + 1}')
