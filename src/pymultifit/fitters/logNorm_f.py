"""Created on Jul 18 19:01:45 2024"""

from typing import Dict, Optional

import numpy as np

from ._backend import BaseFitter


# TODO:
#   Implement `_get_overall_parameter_values`
#   Implement `parameter_extractor`


class LogNormalFitter(BaseFitter):
    """A class for fitting multiple Log Normal functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000, exact_mean: bool = False):
        if any(x < 0 for x in x_values):
            raise ValueError("The LogNormal distribution must have x > 0.")
        super().__init__(n_fits, x_values, y_values, max_iterations)
        self.n_par = 3
        self.exact_mean = exact_mean

    @staticmethod
    def _fitter(x, params):
        f1 = params[0]
        f2 = np.exp(-(np.log(x) - params[1])**2 / (2 * params[2]**2))
        return f1 * f2

    def _n_fitter(self, x, *params):
        y = np.zeros_like(x)
        for i in range(self.n_fits):
            amp = params[i * self.n_par]
            mu = params[i * self.n_par + 1]
            sigma = params[i * self.n_par + 2]
            if self.exact_mean:
                mu = np.log(mu) - (sigma**2 / 2)

            y += self._fitter(x, [amp, mu, sigma])
        return y

    def _plot_individual_fitter(self, x, plotter):
        for i in range(self.n_fits):
            amp = self.params[i * self.n_par]
            mu = self.params[i * self.n_par + 1]
            sigma = self.params[i * self.n_par + 2]
            if self.exact_mean:
                mu = np.log(mu) - (sigma**2 / 2)
            plotter.plot(x, self._fitter(x, [amp, mu, sigma]), linestyle=':', label=f'LogNormal {i + 1}')

    def _get_overall_parameter_values(self):
        overall_fit = self.get_fit_values()
        _, mu, _ = self.parameter_extractor(mu=True)

        amp = []
        for mu_values in mu:
            closest_index = (np.abs(self.x_values - mu_values)).argmin()
            amplitude_value = overall_fit[closest_index]
            amp.append(amplitude_value)

        return amp, mu

    def parameter_extractor(self,
                            parameter_dictionary: Optional[Dict[str, bool]] = None,
                            amplitude: Optional[bool] = None,
                            mu: Optional[bool] = None,
                            sigma: Optional[bool] = None,
                            overall_amplitude: bool = False):

        if parameter_dictionary is None:
            parameter_dictionary = {}

        amplitude = amplitude if amplitude is not None else parameter_dictionary.get('amp', False)
        mu = mu if mu is not None else parameter_dictionary.get('mu', False)
        sigma = sigma if sigma is not None else parameter_dictionary.get('sigma', False)

        if not (amplitude or mu or sigma):
            return [], [], []

        values = self.get_value_error_pair(mean_values=True)

        if overall_amplitude:
            amp_values, mu_values = self._get_overall_parameter_values()
            sigma_values = []
        else:
            amp_values = [values[_ * self.n_par] for _ in range(self.n_fits)] if amplitude else []
            mu_values = [values[_ * self.n_par + 1] for _ in range(self.n_fits)] if mu else []
            sigma_values = [values[_ * self.n_par + 2] for _ in range(self.n_fits)] if sigma else []

        return amp_values, mu_values, sigma_values
