"""Created on Jul 18 13:54:03 2024"""

from .backend import BaseFitter
from .utilities import sanity_check
from ..distributions.utilities import skew_normal_pdf_


class SkewNormalFitter(BaseFitter):
    """A class for fitting multiple Skewed Normal functions to the given data."""

    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 4

    @staticmethod
    def _fitter(x, params):
        return skew_normal_pdf_(x, *params, normalize=False)
