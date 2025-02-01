"""Created on Jul 18 19:01:45 2024"""

from math import inf

from .backend import BaseFitter
from .utilities_f import sanity_check
from ..distributions.utilities_d import log_normal_pdf_


# TODO:
#   See if `exact_mean` can be reimplemented


class LogNormalFitter(BaseFitter):
    """A class for fitting multiple LogNormal distributions to the given data."""

    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)

        self.n_par = 4
        self.pn_par = 3
        self.sn_par = {'loc': 0}

    @staticmethod
    def fit_boundaries():
        lb = (0, -inf, 0, -inf)
        ub = (inf, inf, inf, inf)
        return lb, ub

    @staticmethod
    def fitter(x, params):
        return log_normal_pdf_(x, *params, normalize=False)
