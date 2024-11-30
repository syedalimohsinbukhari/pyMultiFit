"""Created on Jul 18 19:01:45 2024"""

from .backend import BaseFitter, utilities as utils
from ..distributions import log_normal_


# TODO:
#   See if `exact_mean` can be reimplemented


class LogNormalFitter(BaseFitter):
    """A class for fitting multiple Log Normal functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = utils.sanity_check(x_values=x_values, y_values=y_values)
        if any(x_values < 0):
            raise ValueError("The LogNormal distribution must have x > 0. "
                             "Use `EPSILON` from the package to get as close to zero as possible.")
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    @staticmethod
    def _fitter(x, params):
        return log_normal_(x, *params, normalize=False)
