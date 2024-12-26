"""Created on Nov 30 11:30:45 2024"""

from .backend import BaseFitter
from .utilities import sanity_check
from ..distributions.utilities import exponential_pdf_


class ExponentialFitter(BaseFitter):
    """A class for fitting multiple Exponential functions to the given data."""

    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 2

    @staticmethod
    def _fitter(x, params):
        return exponential_pdf_(x, normalize=False)
