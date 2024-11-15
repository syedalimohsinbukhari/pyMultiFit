"""Created on Nov 15 11:22:13 2024"""

import numpy as np
from pymultifit.fitters._backend.baseFitter import BaseFitter

from .sersic_d import SersicDistribution


class SersicFitter(BaseFitter):
    
    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3
    
    @staticmethod
    def _fitter(x, params):
        return SersicDistribution(*params).pdf(x)
    
    def _n_fitter(self, x, *params):
        y = np.zeros_like(x, dtype=float)
        params = np.reshape(params, (self.n_fits, self.n_par))
        for ef_r, ef_d, n in params:
            y += self._fitter(x=x, params=[ef_r, ef_d, n])
        return y
    
    def _plot_individual_fitter(self, x, plotter):
        params = np.reshape(self.params, (self.n_fits, self.n_par))
        for i, (ef_r, ef_d, n) in enumerate(params):
            plotter.plot(x, self._fitter(x=x, params=[ef_r, ef_d, n]),
                         '--', label=f'Sersic {i + 1}('
                                     f'{self.format_param(ef_r)}, '
                                     f'{self.format_param(ef_d)}, '
                                     f'{self.format_param(n)})')

# class SphericalFitter(BaseFitter):
#
#     def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
#         super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
#         self.n_par = 3
#
#     @staticmethod
#     def _fitter(x, params):
#         return SphericalDistribution(*params).pdf(x)
#
#     def _n_fitter(self, x, *params):
#         y = np.zeros_like(x, dtype=float)
#         params = np.reshape(params, (self.n_fits, self.n_par))
#         for a_, b_, c_ in params:
#             y += self._fitter(x=x, params=[a_, b_, c_])
#         return y
#
#     def _plot_individual_fitter(self, x, plotter):
#         params = np.reshape(self.params, (self.n_fits, self.n_par))
#         for i, (a_, b_, c_) in enumerate(params):
#             plotter.plot(x, self._fitter(x=x, params=[a_, b_, c_]),
#                          '--', label=f'Sersic {i + 1}('
#                                      f'{self.format_param(a_)}, '
#                                      f'{self.format_param(b_)}, '
#                                      f'{self.format_param(c_)})')
