"""Created on Jul 18 19:01:45 2024"""

import numpy as np
from numpy.typing import NDArray

from .backend import BaseFitter
from .utilities_f import sanity_check
from .. import Sequences_, lArray
from ..distributions.utilities_d import log_normal_pdf_


# TODO:
#   See if `exact_mean` can be reimplemented


class LogNormalFitter(BaseFitter):
    """A class for fitting multiple LogNormal distributions to the given data."""

    def __init__(
        self,
        x_values: lArray,
        y_values: lArray,
        max_iterations: int = 1000,
    ):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)

        self.n_par = 4
        self.pn_par = 3
        self.sn_par = {"loc": 0}

    @staticmethod
    def fit_boundaries():
        lb = (0, -np.inf, 0, -np.inf)
        ub = (np.inf, np.inf, np.inf, np.inf)
        return lb, ub

    @staticmethod
    def fitter(x: NDArray, params: Sequences_):
        return log_normal_pdf_(x, *params)
