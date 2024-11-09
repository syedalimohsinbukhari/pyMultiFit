"""Created on Nov 10 00:17:16 2024"""

import numpy as np

from ._backend.baseFitter import BaseFitter
from ..distributions.powerLaw_d import powerLawWA


class PowerLawFitter(BaseFitter):
    
    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 2
    
    @staticmethod
    def _fitter(x, params):
        return powerLawWA(*params).pdf(x)
    
    def _n_fitter(self, x, *params):
        y = np.zeros_like(x)
        params = np.reshape(params, (self.n_fits, self.n_par))
        for amp, alpha in params:
            y += self._fitter(x=x, params=[amp, alpha])
        return y
    
    def _plot_individual_fitter(self, x, plotter):
        params = np.reshape(self.params, (self.n_fits, self.n_par))
        for i, (amp, alpha) in enumerate(params):
            plotter.plot(x, self._fitter(x=x, params=[amp, alpha]),
                         '--', label=f'PowerLaw {i + 1}('
                                     f'{self.format_param(amp)} '
                                     f'{self.format_param(alpha)})')
    
    def _get_overall_parameter_values(self):
        pass
    
    def parameter_extractor(self):
        pass
