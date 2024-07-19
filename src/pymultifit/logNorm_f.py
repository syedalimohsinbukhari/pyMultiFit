"""Created on Jul 18 19:01:45 2024"""

import numpy as np

from src.pymultifit.backend import BaseFitter


class LogNormal(BaseFitter):

    def __init__(self, n_fits: int, x_values, y_values, exact_mean=False):
        if any(x < 0 for x in x_values):
            raise ValueError("The LogNormal distribution must have x > 0.")
        super().__init__(n_fits, x_values, y_values)
        self.n_parameters = 3
        self.exact_mean = exact_mean

    @classmethod
    def from_exact_mean(cls, n_fits, x_values, y_values):
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

    def get_fit_values(self) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self._n_fitter(self.x_values, self.params)

    def get_value_error_pair(self, only_values=False, only_errors=False) -> np.ndarray:
        pairs = np.array([np.array([i, j]) for i, j in zip(self._params(), self._standard_errors())])

        return pairs[:, 0] if only_values else pairs[:, 1] if only_errors else pairs
