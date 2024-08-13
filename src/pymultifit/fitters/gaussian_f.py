"""Created on Jul 18 00:25:57 2024"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ._backend import BaseFitter


class GaussianFitter(BaseFitter):
    """A class for fitting multiple Gaussian functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
        super().__init__(n_fits, x_values, y_values, max_iterations)
        self.n_par = 3

    @staticmethod
    def _fitter(x, params) -> np.array:
        return params[0] * np.exp(-(x - params[1])**2 / (2 * params[2]**2))

    def _n_fitter(self, x, *params) -> np.array:
        y = np.zeros_like(x)
        for i in range(self.n_fits):
            amp = params[i * self.n_par]
            mu = params[i * self.n_par + 1]
            sigma = params[i * self.n_par + 2]
            y += self._fitter(x, [amp, mu, sigma])
        return y

    def _plot_individual_fitter(self, x, plotter):
        for i in range(self.n_fits):
            amp = self.params[i * self.n_par]
            mu = self.params[i * self.n_par + 1]
            sigma = self.params[i * self.n_par + 2]
            plotter.plot(x, self._fitter(x, [amp, mu, sigma]), linestyle=':', label=f'Gaussian {i + 1}')

    def _get_overall_parameter_values(self) -> Tuple[List[float], List[float]]:
        overall_fit = self.get_fit_values()
        _, mu, _ = self.parameter_extractor(mu=True)

        amp = []
        for mu_value in mu:
            closest_index = (np.abs(self.x_values - mu_value)).argmin()
            amplitude_value = overall_fit[closest_index]
            amp.append(amplitude_value)

        return amp, mu

    def parameter_extractor(self,
                            parameter_dictionary: Optional[Dict[str, bool]] = None,
                            amplitude: Optional[bool] = None,
                            mu: Optional[bool] = None,
                            sigma: Optional[bool] = None,
                            fit_amplitude: bool = False,
                            ) -> Tuple[List[float], List[float], List[float]]:
        """
        Extracts parameter values based on provided flags.

        Parameters
        ----------
        parameter_dictionary : dict, optional
            A dictionary containing flags for 'amp', 'mu', and 'sigma'. The default is None.
        amplitude : bool, optional
            Flag to extract amplitude values. Defaults to False if not provided.
        mu : bool, optional
            Flag to extract mu values. Defaults to False if not provided.
        sigma : bool, optional
            Flag to extract sigma values. Defaults to False if not provided.
        fit_amplitude : bool
            Flag to extract overall amplitude values. Overwrites the default amplitude selection.
            This will not give back the amplitudes of individual fitters, but rather the amplitude of overall fitters.
            Defaults to False.

        Notes
        -----
            If `overall_amplitude` is true, the function will not return `sigma` values.

        Returns
        -------
        tuple of list of float
            A tuple containing three lists in the following order:
            - Amplitude values if `amplitude` is True, otherwise an empty list.
            - Mu values if `mu` is True, otherwise an empty list.
            - Sigma values if `sigma` is True, otherwise an empty list.
        """

        if parameter_dictionary is None:
            parameter_dictionary = {}

        # Use function parameters if provided, else fall back to dictionary values, and default to False if neither
        # is provided
        amplitude = amplitude if amplitude is not None else parameter_dictionary.get('amp', False)
        mu = mu if mu is not None else parameter_dictionary.get('mu', False)
        sigma = sigma if sigma is not None else parameter_dictionary.get('sigma', False)

        # Guard clause to handle case where no parameters are requested
        if not (amplitude or mu or sigma):
            return [], [], []

        values = self.get_value_error_pair(mean_values=True)

        if fit_amplitude:
            amp_values, mu_values = self._get_overall_parameter_values()
            sigma_values = []
        else:
            amp_values = [values[_ * self.n_par] for _ in range(self.n_fits)] if amplitude else []
            mu_values = [values[_ * self.n_par + 1] for _ in range(self.n_fits)] if mu else []
            sigma_values = [values[_ * self.n_par + 2] for _ in range(self.n_fits)] if sigma else []

        return amp_values, mu_values, sigma_values
