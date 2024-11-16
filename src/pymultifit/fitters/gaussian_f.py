"""Created on Jul 18 00:25:57 2024"""

from typing import Any, List

import numpy as np

from ._backend.baseFitter import BaseFitter
from ..distributions.gaussian_d import gaussianWA


class GaussianFitter(BaseFitter):
    """A class for fitting multiple Gaussian functions to the given data."""
    
    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3
    
    @staticmethod
    def _fitter(x: np.ndarray, params: list) -> np.array:
        return gaussianWA(*params).pdf(x)
    
    def _n_fitter(self, x: np.ndarray, *params: Any) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)
        params = np.reshape(params, (self.n_fits, self.n_par))
        for amp, mu, sigma in params:
            y += self._fitter(x=x, params=[amp, mu, sigma])
        return y
    
    def _plot_individual_fitter(self, x: np.ndarray, plotter):
        params = np.reshape(self.params, (self.n_fits, self.n_par))
        for i, (amp, mu, sigma) in enumerate(params):
            plotter.plot(x, self._fitter(x=x, params=[amp, mu, sigma]),
                         '--', label=f'Gaussian {i + 1}('
                                     f'{self.format_param(amp)}, '
                                     f'{self.format_param(mu)}, '
                                     f'{self.format_param(sigma)})')
    
    @staticmethod
    def format_param(value, threshold1=0.001, threshold2=10_000):
        """Formats the parameter value based on its magnitude."""
        return f'{value:.3E}' if threshold2 < abs(value) or abs(value) < threshold1 else f'{value:.3f}'
    
    def get_parameters(self,
                       amplitude: bool = False, mu: bool = False, sigma: bool = False,
                       gaussian_number: List[int, Any] = None):
        """
        Extracts specific parameter values (amplitude, mean (mu), and standard deviation (sigma))
        from the fitting process.

        Parameters
        ----------
        amplitude : bool, optional
            If True, returns the amplitude values. Defaults to False.
        mu : bool, optional
            If True, returns the mean (mu) values. Defaults to False.
        sigma : bool, optional
            If True, returns the standard deviation (sigma) values. Defaults to False.
        gaussian_number : list of int or None, optional
            A list of indices specifying which Gaussian components to return values for.
            If None, returns values for all components. Defaults to None.

        Returns
        -------
        tuple:
            Three arrays corresponding to amplitude, mu, and sigma values.
            
            If a specific parameter is not requested (i.e., its corresponding argument is False), its array will be empty.

            - `amp_values`: numpy array of amplitude values, or an empty array if `amplitude=False`.
            - `mu_values`: numpy array of mean values (mu), or an empty array if `mu=False`.
            - `sigma_values`: numpy array of standard deviation values, or an empty array if `sigma=False`.

        Notes
        -----
        - The `gaussian_number` parameter is used to filter the returned values to specific Gaussian  components based on their indices. Indexing starts at 1.
        """
        if not (amplitude or mu or sigma):
            return [], [], []
        
        values = self.get_value_error_pair(mean_values=True)
        
        amp_values, mu_values, sigma_values = [], [], []
        
        n_fits, n_par = self.n_fits, self.n_par
        if amplitude:
            amp_values = [values[i * n_par] for i in range(n_fits)]
            if gaussian_number:
                amp_values = [amp_values[i - 1] for i in gaussian_number]
        if mu:
            mu_values = [values[i * n_par + 1] for i in range(n_fits)]
            if gaussian_number:
                mu_values = [mu_values[i - 1] for i in gaussian_number]
        if sigma:
            sigma_values = [values[i * n_par + 2] for i in range(n_fits)]
            if gaussian_number:
                sigma_values = [sigma_values[i - 1] for i in gaussian_number]
        
        return np.array(amp_values), np.array(mu_values), np.array(sigma_values)
