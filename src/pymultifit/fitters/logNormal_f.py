"""Created on Jul 18 19:01:45 2024"""

from .backend import BaseFitter
from .utilities_f import sanity_check
from ..distributions.utilities_d import log_normal_pdf_


# TODO:
#   See if `exact_mean` can be reimplemented


class LogNormalFitter(BaseFitter):
    """Class for multi-LogNormal fitting."""

    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        if any(x_values < 0):
            raise ValueError("The LogNormal distribution must have x > 0. "
                             "Use `EPSILON` from the package to get as close to zero as possible.")
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)

        self.n_par = 4
        self.pn_par = 3
        self.sn_par = {'loc': 0}

    @staticmethod
    def _fitter(x, params):
        return log_normal_pdf_(x, *params, normalize=False)
