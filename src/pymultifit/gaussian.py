"""Created on Jul 18 00:25:57 2024"""

from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from .backend.multiFitter import BaseFitter


class MultiGaussian(BaseFitter):

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
        super().__init__(n_fits, x_values, y_values, max_iterations)

    @staticmethod
    def individual_fitter(x, *params):
        return params[0] * np.exp(-(x - params[1])**2 / (2 * params[2]**2))

    def get_fitted_values(self):
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self.n_gaussian(self.x_values, *self.params)

    def n_gaussian(self, x, *params):
        y = np.zeros_like(x)
        for i in range(self.n_fits):
            amp = params[i * 3]
            mu = params[i * 3 + 1]
            sigma = params[i * 3 + 2]
            y += self.individual_fitter(x, *[amp, mu, sigma])
        return y

    def _plot_individual_gaussian(self, x, plotter):
        for i in range(self.n_fits):
            amp = self.params[i * 3]
            mu = self.params[i * 3 + 1]
            sigma = self.params[i * 3 + 2]
            plotter.plot(x, self._gaussian(x, amp, mu, sigma), linestyle=':', label=f'Gaussian {i + 1}')

    def plot_fit(self, show_individual: bool = False, fig_size: Tuple[int, int] = (12, 6), auto_label: bool = False,
                 ax: Optional[plt.Axes] = None):
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        plotter = ax if ax else plt
        if not ax:
            plt.figure(figsize=fig_size)

        plotter.plot(self.x_values, self.y_values, label='Data')
        plotter.plot(self.x_values, self.n_gaussian(self.x_values, *self.params), label='Total Fit', linestyle='--')

        if show_individual:
            self._plot_individual_gaussian(self.x_values, plotter)

        if auto_label:
            labels = {
                'xlabel': plotter.set_xlabel if ax else plotter.xlabel,
                'ylabel': plotter.set_ylabel if ax else plotter.ylabel,
                'title': plotter.set_title if ax else plotter.title,
            }
            labels['xlabel']('X')
            labels['ylabel']('Y')
            labels['title'](f'{self.n_fits} Gaussian fit')
            plotter.legend(loc='best')
            plt.tight_layout()

        return plotter

    def fit(self, p0):
        if len(p0) != 3 * self.n_fits:
            raise ValueError(f"Initial guess length must be {3 * self.n_fits}.")
        _ = curve_fit(self.n_gaussian, self.x_values, self.y_values, p0=p0, maxfev=self.max_iterations)

        self.params = _[0]
        self.covariance = _[1]

    def get_value_error_pair(self, only_values=False, only_errors=False):
        pairs = np.array([np.array([i, j]) for i, j in zip(self._params(), self.get_standard_errors())])

        return pairs[:, 0] if only_values else pairs[:, 1] if only_errors else pairs
