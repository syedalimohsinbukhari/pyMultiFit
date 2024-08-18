"""Created on Jul 18 00:25:57 2024"""

from typing import Any, List, Optional, Tuple

import numpy as np

from ._backend.multiFitter import BaseFitter
from ._backend.utilities import get_y_values_at_closest_x

oBool = Optional[bool]


class GaussianFitter(BaseFitter):
    """A class for fitting multiple Gaussian functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: int = 1000):
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    @staticmethod
    def _fitter(x: np.ndarray, params: list) -> np.array:
        return params[0] * np.exp(-(x - params[1])**2 / (2 * params[2]**2))

    def _n_fitter(self, x: np.ndarray, *params: Any) -> np.ndarray:
        y = np.zeros_like(x)
        params = np.reshape(params, (self.n_fits, self.n_par))
        for amp, mu, sigma in params:
            y += self._fitter(x=x, params=[amp, mu, sigma])
        return y

    def _plot_individual_fitter(self, x: np.ndarray, plotter):
        params = np.reshape(self.params, (self.n_fits, self.n_par))
        for i, (amp, mu, sigma) in enumerate(params):
            plotter.plot(x, self._fitter(x=x, params=[amp, mu, sigma]), ls=':', label=f'Gaussian {i + 1}')

    def _get_overall_parameter_values(self) -> tuple[list, list]:
        _, mu, _ = self.parameter_extractor(mean=True)
        amp = get_y_values_at_closest_x(x_array=self.x_values, y_array=self.get_fit_values(), target_x_values=mu)
        return amp, mu

    def parameter_extractor(self,
                            amplitude: bool = False, mean: bool = False, standard_deviation: bool = False,
                            fit_amplitude: bool = False) -> Tuple[List[float], List[float], List[float]]:
        """
        Extracts parameter values based on provided flags.

        Parameters
        ----------
        amplitude : bool, optional
            Flag to extract amplitude values. Defaults to False.
        mean : bool, optional
            Flag to extract mean (mu) values. Defaults to False.
        standard_deviation : bool, optional
            Flag to extract standard deviation (sigma) values. Defaults to False.
        fit_amplitude : bool, optional
            Flag to extract overall amplitude values. Overwrites the default amplitude selection.
            This will not return the amplitudes of individual fitters, but rather the amplitude of overall fitters.
            Defaults to False.

        Returns
        -------
        Tuple[List[float], List[float], List[float]]
            A tuple containing three lists in the following order:
            - Amplitude values if `amplitude` is True, otherwise an empty list.
            - Mean (mu) values if `mean` is True, otherwise an empty list.
            - Standard deviation (sigma) values if `standard_deviation` is True, otherwise an empty list.
        """
        if not (amplitude or mean or standard_deviation):
            return [], [], []

        values = self.get_value_error_pair(mean_values=True)

        if fit_amplitude:
            amp_values, mu_values = self._get_overall_parameter_values()
            return amp_values, mu_values, []

        amp_values, mu_values, sigma_values = [], [], []

        n_fits, n_par = self.n_fits, self.n_par
        if amplitude:
            amp_values = [values[i * n_par] for i in range(n_fits)]
        if mean:
            mu_values = [values[i * n_par + 1] for i in range(n_fits)]
        if standard_deviation:
            sigma_values = [values[i * n_par + 2] for i in range(n_fits)]

        return amp_values, mu_values, sigma_values
