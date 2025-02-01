"""Created on Nov 30 11:30:45 2024"""

from math import inf

from .backend import BaseFitter
from .utilities_f import sanity_check
from ..distributions.utilities_d import exponential_pdf_


class ExponentialFitter(BaseFitter):
    """A class for fitting multiple Exponential distributions to the given data."""

    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3
        self.pn_par = 2
        self.sn_par = {'loc': 0.0}

    @staticmethod
    def fit_boundaries():
        lb = (0, 0, -inf)
        ub = (inf, inf, inf)
        return lb, ub

    @staticmethod
    def fitter(x, params):
        return exponential_pdf_(x, *params, normalize=False)
