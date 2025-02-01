"""Created on Jul 18 00:25:57 2024"""

from math import inf

from .backend import BaseFitter
from .utilities_f import sanity_check
from ..distributions.utilities_d import gaussian_pdf_


class GaussianFitter(BaseFitter):
    """A class for fitting multiple Gaussian distributions to the given data."""

    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    @staticmethod
    def fit_boundaries():
        lb = (0, -inf, 0)
        ub = (inf, inf, inf)
        return lb, ub

    @staticmethod
    def fitter(x, params):
        return gaussian_pdf_(x, *params, normalize=False)
