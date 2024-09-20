"""Created on Jul 20 16:59:14 2024"""

from typing import Optional

import numpy as np

from ._backend.multiFitter import BaseFitter
from ._backend.utilities import get_y_values_at_closest_x
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

    def _n_fitter(self, x, *params):
        y = np.zeros_like(x)
        params = np.reshape(params, (self.n_fits, self.n_par))
        for amp, mu, b in params:
            y += self._fitter(x, [amp, mu, b])
        return y

    def _plot_individual_fitter(self, x, plotter):
        params = np.reshape(self.params, (self.n_fits, self.n_par))
        for i, (amp, mu, b) in enumerate(params):
            plotter.plot(x, self._fitter(x, [amp, mu, b]),
                         '--', label=f'Laplace {i + 1}('
                                     f'{self.format_param(amp)}, '
                                     f'{self.format_param(mu)}, '
                                     f'{self.format_param(b)})')

    def _get_overall_parameter_values(self) -> tuple[list, list]:
        _, mu, _ = self.parameter_extractor(mean=True)
        amp = get_y_values_at_closest_x(x_array=self.x_values, y_array=self.get_fit_values(), target_x_values=mu)
        return amp, mu

    def parameter_extractor(self,
                            amplitude: bool = False, mean: bool = False, diversity: bool = False,
                            fit_amplitude: bool = False):
        """
        Extracts parameter values based on provided flags.

        Parameters
        ----------
        amplitude : bool, optional
            Flag to extract amplitude values. Defaults to False.
        mean : bool, optional
            Flag to extract mean (mu) values. Defaults to False.
        diversity : bool, optional
            Flag to extract diversity (b) values. Defaults to False.
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
            - Diversity (b) values if `diversity` is True, otherwise an empty list.
        """
        if not (amplitude or mean or diversity):
            return [], [], []

        values = self.get_value_error_pair(mean_values=True)

        amp_values, mu_values, b_values = [], [], []

        n_fits, n_par = self.n_fits, self.n_par
        if amplitude:
            amp_values = [values[i * n_par] for i in range(n_fits)]
        if mean:
            mu_values = [values[i * n_par + 1] for i in range(n_fits)]
        if diversity:
            b_values = [values[i * n_par + 2] for i in range(n_fits)]

        return amp_values, mu_values, b_values
