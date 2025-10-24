"""Created on Dec 27 11:31:54 2024"""

import numpy as np

from .backend import BaseFitter
from .utilities_f import sanity_check
from .. import OneDArray, Params_
from ..distributions.utilities_d import gamma_pdf_


class GammaFitter(BaseFitter):
    """A class for fitting multiple Gamma SR functions to the given data."""

    def __init__(self, x_values: OneDArray, y_values: OneDArray, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 4
        self.pn_par = 3
        self.sn_par = {"loc": 0.0}

    def fit_boundaries(self):
        lb = (0, 0, 0, -np.inf)
        ub = (np.inf, np.inf, np.inf, np.inf)
        return lb, ub

    @staticmethod
    def fitter(x, params: Params_):
        return gamma_pdf_(x, *params)
