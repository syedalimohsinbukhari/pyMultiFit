"""Created on Jul 18 19:01:45 2024"""

from typing import Optional

import numpy as np

from .backend import BaseFitter


class LogNormal(BaseFitter):
    """A class for fitting multiple Log Normal functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000, exact_mean=False):
        if any(x < 0 for x in x_values):
            raise ValueError("The LogNormal distribution must have x > 0.")
        super().__init__(n_fits, x_values, y_values)
        self.n_parameters = 3
        self.exact_mean = exact_mean

    @classmethod
    def from_exact_mean(cls, n_fits, x_values, y_values):
        """
        Create an instance of LogNormal with the option to fit the log-normal distribution to have exact mean values
        provided.

        Parameters
        ----------
        n_fits : int
            Number of fits to perform.
        x_values : list or array-like
            Independent variable values for fitting.
        y_values : list or array-like
            Dependent variable values for fitting.

        Returns
        -------
        LogNormal
            An instance of LogNormal configured to fit the distribution with exact mean values.
        """
        return cls(n_fits, x_values, y_values, True)

    @staticmethod
    def _fitter(x, params):
        f1 = params[0]
        f2 = np.exp(-(np.log(x) - params[1])**2 / (2 * params[2]**2))
        return f1 * f2

    def _n_fitter(self, x, *params):
        y = np.zeros_like(x)
        for i in range(self.n_fits):
            amp = params[i * 3]
            mu = params[i * 3 + 1]
            sigma = params[i * 3 + 2]
            if self.exact_mean:
                mu = np.log(mu) - (sigma**2 / 2)

            y += self._fitter(x, [amp, mu, sigma])
        return y

    def _plot_individual_fitter(self, x, plotter):
        for i in range(self.n_fits):
            amp = self.params[i * 3]
            mu = self.params[i * 3 + 1]
            sigma = self.params[i * 3 + 2]
            if self.exact_mean:
                mu = np.log(mu) - (sigma**2 / 2)
            plotter.plot(x, self._fitter(x, [amp, mu, sigma]), linestyle=':', label=f'LogNormal {i + 1}')
