"""Created on Nov 30 11:30:45 2024"""

from .backend import BaseFitter, utilities as utils
from ..distributions import exponential_


class ExponentialFitter(BaseFitter):
    """A class for fitting multiple Exponential functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = utils.sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 2

    @staticmethod
    def _fitter(x, params):
        return exponential_(x, *params, normalize=False)
