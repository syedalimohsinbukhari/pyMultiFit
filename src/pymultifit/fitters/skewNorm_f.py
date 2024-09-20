"""Created on Jul 18 13:54:03 2024"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import skewnorm

from ._backend.multiFitter import BaseFitter
from ._backend.utilities import get_y_values_at_closest_x


# TODO:
#   Implement `_get_overall_parameter_values`
#   Implement `parameter_extractor`


class SkewedNormalFitter(BaseFitter):
    """A class for fitting multiple Skewed Normal functions to the given data."""

    def __init__(self, n_fits: int, x_values, y_values, max_iterations: Optional[int] = 1000):
        super().__init__(n_fits=n_fits, x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 4

    @staticmethod
    def _fitter(x, params):
        return params[0] * skewnorm.pdf(x, params[1], loc=params[2], scale=params[3])

    def _n_fitter(self, x, *params):
        y = np.zeros_like(x)
        params = np.reshape(params, (self.n_fits, self.n_par))
        for amp, shape, loc, scale in params:
            y += self._fitter(x, [amp, shape, loc, scale])
        return y

    def _plot_individual_fitter(self, x, plotter):
        """Plots individual fitted components and displays parameter values."""
        params = np.reshape(self.params, (self.n_fits, self.n_par))

        for i, (amp, shape, loc, scale) in enumerate(params):
            # Plot the fitted curve
            plotter.plot(x, self._fitter(x, [amp, shape, loc, scale]),
                         '--', label=f'SkewNormal {i + 1}('
                                     f'{self.format_param(amp)}, '
                                     f'{self.format_param(shape)}, '
                                     f'{self.format_param(loc)}, '
                                     f'{self.format_param(scale)})')

        plotter.legend()

    def _get_overall_parameter_values(self) -> Tuple[List[float], List[float], List[float]]:
        _, shape, loc, _ = self.parameter_extractor(location=True)
        amp = get_y_values_at_closest_x(x_array=self.x_values, y_array=self.get_fit_values(), target_x_values=loc)
        return amp, shape, loc

    def parameter_extractor(self,
                            amplitude: bool = False, shape: bool = False, location: bool = False, scale: bool = False,
                            fit_amplitude: bool = False) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Extracts parameter values based on provided flags.

        Parameters
        ----------
        amplitude : bool, optional
            Flag to extract amplitude values. Defaults to False.
        shape : bool, optional
            Flag to extract shape values. Defaults to False.
        location : bool, optional
            Flag to extract location values. Defaults to False.
        scale : bool, optional
            Flag to extract scale values. Defaults to False.
        fit_amplitude : bool, optional
            Flag to extract overall amplitude values. Overwrites the default amplitude selection.
            This will not return the amplitudes of individual fitters, but rather the amplitude of overall fitters.
            Defaults to False.

        Returns
        -------
        Tuple[List[float], List[float], List[float], List[float]]
            A tuple containing three lists in the following order:
            - Amplitude values if `amplitude` is True, otherwise an empty list.
            - Shape values if `shape` is True, otherwise an empty list.
            - Location values if `location` is True, otherwise an empty list.
            - Scale values if `scale` is True, otherwise an empty list.
        """
        if not (amplitude or shape or location or scale):
            return [], [], [], []

        values = self.get_value_error_pair(mean_values=True)

        if fit_amplitude:
            amp_values, shape_values, location_values = self._get_overall_parameter_values()
            return amp_values, shape_values, location_values, []

        amp_values, shape_values, location_values, scale_values = [], [], [], []

        n_fits, n_par = self.n_fits, self.n_par
        if amplitude:
            amp_values = [values[i * n_par] for i in range(n_fits)]
        if shape:
            shape_values = [values[i * n_par + 1] for i in range(n_fits)]
        if location:
            location_values = [values[i * n_par + 2] for i in range(n_fits)]
        if scale:
            scale_values = [values[i * n_par + 3] for i in range(n_fits)]

        return amp_values, shape_values, location_values, scale_values
