"""Created on Jan 13 11:47:55 2025"""

from typing import Tuple, Any

import numpy as np

from .backend import BaseFitter
from .utilities_f import sanity_check
from ..distributions.backend import line


class LineFitter(BaseFitter):

    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 2

    @staticmethod
    def fit_boundaries():
        lb = (-np.inf, -np.inf)
        ub = (np.inf, np.inf)

        return lb, ub

    @staticmethod
    def fitter(x: np.ndarray, params: Tuple[float, Any]):
        return line(x, *params)
