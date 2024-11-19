"""Created on Jul 18 13:54:03 2024"""

from typing import Optional

import numpy as np
from scipy.stats import skewnorm

from ._backend.baseFitter import BaseFitter


# TODO:
#   Implement `_get_overall_parameter_values`
#   Implement `parameter_extractor`


class SkewedNormalFitter(BaseFitter):
    """A class for fitting multiple Skewed Normal functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 4

    @staticmethod
    def _fitter(x, params):
        return params[0] * skewnorm.pdf(x, params[1], loc=params[2], scale=params[3])

    def _n_fitter(self, x, *params):
        y = np.zeros_like(x, dtype=float)
        params = np.reshape(params, (self.n_fits, self.n_par))
        for amp, shape, loc, scale in params:
            y += self._fitter(x, [amp, shape, loc, scale])
        return y

    def _plot_individual_fitter(self, x, plotter):
        """Plots individual fitted components and displays parameter values."""
        params = np.reshape(self.params, (self.n_fits, self.n_par))

        for i, (amp, shape, loc, scale) in enumerate(params):
            # Plot the fitted curve
            plotter.plot(x, self._fitter(x, [amp, shape, loc, scale]),
                         '--', label=f'SkewNormal {i + 1}('
                                     f'{self.format_param(amp)}, '
                                     f'{self.format_param(shape)}, '
                                     f'{self.format_param(loc)}, '
                                     f'{self.format_param(scale)})')

        plotter.legend()
