"""Created on Nov 30 05:33:38 2024"""

from .backend import BaseFitter, utilities as utils
from ..distributions import norris2005, norris2011


class Norris2005Fitter(BaseFitter):
    """A class for fitting multiple Norris 2005 models to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = utils.sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    @staticmethod
    def _fitter(x, params):
        return norris2005(x, *params, normalize=False)


class Norris2011Fitter(BaseFitter):
    """A class for fitting multiple Norris 2011 models to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = utils.sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    @staticmethod
    def _fitter(x, params):
        return norris2011(x, *params, normalize=False)
