"""Created on Jul 20 16:59:14 2024"""

from typing import Optional

from ._backend.baseFitter import BaseFitter
from ..distributions.laplace_d import laplace_


# TODO:
#   Implement `_get_overall_parameter_values`
#   Implement `parameter_extractor`

class LaplaceFitter(BaseFitter):
    """A class for fitting multiple Laplace distributions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    @staticmethod
    def _fitter(x, params):
        return laplace_(x, *params, normalize=False)
