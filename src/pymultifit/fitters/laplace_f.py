"""Created on Jul 20 16:59:14 2024"""

from typing import Optional

import numpy as np

from ._backend.baseFitter import BaseFitter
from ..distributions.laplace_d import laplace_


# TODO:
#   Implement `_get_overall_parameter_values`
#   Implement `parameter_extractor`

class LaplaceFitter(BaseFitter):
    """A class for fitting multiple Laplace distributions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    @staticmethod
    def _fitter(x, params):
        return laplace_(x, *params, normalize=False)

    def _n_fitter(self, x, *params):
        y = np.zeros_like(x, dtype=float)
        params = np.reshape(params, (self.n_fits, self.n_par))
        for amp, mu, b in params:
            y += self._fitter(x, [amp, mu, b])
        return y

    def _plot_individual_fitter(self, x, plotter):
        params = np.reshape(self.params, (self.n_fits, self.n_par))
        for i, (amp, mu, b) in enumerate(params):
            plotter.plot(x, self._fitter(x, [amp, mu, b]),
                         '--', label=f'Laplace {i + 1}('
                                     f'{self.format_param(amp)}, '
                                     f'{self.format_param(mu)}, '
                                     f'{self.format_param(b)})')
