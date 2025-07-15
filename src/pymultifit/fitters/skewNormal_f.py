"""Created on Jul 18 13:54:03 2024"""

import numpy as np

from .backend import BaseFitter
from .utilities_f import sanity_check
from .. import listOrNdArray, Params_
from ..distributions.utilities_d import skew_normal_pdf_


class SkewNormalFitter(BaseFitter):
    """A class for fitting multiple SkewNormal distributions to the given data."""

    def __init__(
        self,
        x_values: listOrNdArray,
        y_values: listOrNdArray,
        max_iterations: int = 1000,
    ):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 4

    def fit_boundaries(self):
        lb = (0, -np.inf, -np.inf, 0)
        ub = (np.inf, np.inf, np.inf, np.inf)
        return lb, ub

    @staticmethod
    def fitter(x, params: Params_):
        return skew_normal_pdf_(x, *params)
