"""Created on Aug 10 23:37:54 2024"""

import numpy as np

from .backend import BaseFitter
from .utilities_f import sanity_check
from .. import Params_
from ..distributions.utilities_d import line, quadratic, cubic


class LineFitter(BaseFitter):

    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 2

    def fit_boundaries(self):
        lb = (-np.inf, -np.inf)
        ub = (np.inf, np.inf)

        return lb, ub

    @staticmethod
    def fitter(x, params: Params_):
        return line(x, *params)


class QuadraticFitter(BaseFitter):
    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    def fit_boundaries(self):
        lb = (-np.inf, -np.inf, -np.inf)
        ub = (np.inf, np.inf, np.inf)

        return lb, ub

    @staticmethod
    def fitter(x, params: Params_):
        return quadratic(x, *params)


class CubicFitter(BaseFitter):

    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 4

    def fit_boundaries(self):
        lb = (-np.inf, -np.inf, -np.inf, -np.inf)
        ub = (np.inf, np.inf, np.inf, np.inf)

        return lb, ub

    @staticmethod
    def fitter(x, params: Params_):
        return cubic(x, *params)
