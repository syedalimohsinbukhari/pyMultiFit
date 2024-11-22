"""Created on Nov 10 00:17:16 2024"""

from ._backend.baseFitter import BaseFitter
from ..distributions.powerLaw_d import powerLawWA


class PowerLawFitter(BaseFitter):

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 2

    @staticmethod
    def _fitter(x, params):
        return powerLawWA(*params).pdf(x)
