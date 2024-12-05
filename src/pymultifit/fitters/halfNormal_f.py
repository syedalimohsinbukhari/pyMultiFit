"""Created on Dec 04 23:29:32 2024"""

from .backend import BaseFitter
from .utilities import sanity_check
from ..distributions.utilities import half_normal_


class HalfNormalFitter(BaseFitter):
    """A class for fitting multiple half-normal functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 2

    @staticmethod
    def _fitter(x, params):
        return half_normal_(x, *params, normalize=False)
