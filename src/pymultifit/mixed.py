"""Created on Aug 10 23:08:38 2024"""

import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .distributions import GaussianDistribution
from .others import line


class MixedDataFitter:
    def __init__(self, x_data, y_data, model_list):
        """
        Initializes the MixedDataFitter with data and a list of models.

        Parameters
        ----------
        x_data : array_like
            The x-values for the data.
        y_data : array_like
            The y-values for the data.
        model_list : list of str
            List of models to fit (e.g., ['gaussian', 'gaussian', 'line']).
        """
        self.x_data = x_data
        self.y_data = y_data
        self.model_list = model_list
        self.params = None
        self.covariance = None

        # Validate the model list and create the model function
        self._validate_models()
        self.model_function = self._create_model_function()

    def _validate_models(self):
        """Validate the models in the model list."""
        allowed_models = {'gaussian', 'line'}
        if not all(model in allowed_models for model in self.model_list):
            raise ValueError(f"All models must be one of {allowed_models}.")

    def _create_model_function(self):
        """
        Creates a composite model function based on the specified models.

        Returns
        -------
        callable
            A function that can be used for fitting.
        """

        def _composite_model(x, *params):
            y = np.zeros_like(x)
            param_index = 0

            for model in self.model_list:
                if model == 'gaussian':
                    amp = params[param_index]
                    mu = params[param_index + 1]
                    sigma = params[param_index + 2]
                    y += GaussianDistribution.with_amplitude(amp, mu, sigma).pdf(x)
                    param_index += 3

                elif model == 'line':
                    slope = params[param_index]
                    intercept = params[param_index + 1]
                    y += line(x, slope, intercept)
                    param_index += 2

            return y

        return _composite_model

    def fit(self, p0):
        """
        Fits the model to the data using non-linear least squares.

        Parameters
        ----------
        p0 : array_like
            Initial guess for the fitting parameters.
        """
        if len(p0) != self._expected_param_count():
            raise ValueError(
                f"Initial parameters length {len(p0)} does not match expected count {self._expected_param_count()}.")

        lower_bounds, upper_bounds = self._get_bounds()

        _ = curve_fit(self.model_function, self.x_data, self.y_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        self.params, self.covariance = _[0], _[1]

    def _expected_param_count(self):
        """
        Calculates the expected number of parameters based on the model list.

        Returns
        -------
        int
            The expected number of parameters.
        """
        count = 0
        for model in self.model_list:
            if model == 'gaussian':
                count += 3  # amplitude, mean, standard deviation
            elif model == 'line':
                count += 2  # slope, intercept
        return count

    def _get_bounds(self):
        """
        Sets the bounds for each parameter based on the model list.

        Returns
        -------
        tuple of array_like
            Lower and upper bounds for the parameters.
        """
        lower_bounds = []
        upper_bounds = []

        for model in self.model_list:
            if model == 'gaussian':
                lower_bounds.extend([0, -np.inf, 0])  # amplitude >= 0, mean unrestricted, stddev >= 0
                upper_bounds.extend([np.inf, np.inf, np.inf])
            elif model == 'line':
                lower_bounds.extend([-np.inf, -np.inf])  # slope and intercept unrestricted
                upper_bounds.extend([np.inf, np.inf])

        return lower_bounds, upper_bounds

    def plot(self, plot_individuals=False, auto_label=False):
        """
        Plots the original data, fitted model, and optionally individual components.

        Parameters
        ----------
        plot_individuals : bool, optional
            Whether to plot individual fitted functions, by default False.
        auto_label : bool, optional
            If True, automatically labels the plot with 'X', 'Y', 'MixedFittedData',
            applies a legend, and adjusts layout, by default False.
        """
        if self.y_data is None or self.params is None:
            raise ValueError("Data must be fitted before plotting.")

        plt.plot(self.x_data, self.y_data, '-', label='data')
        plt.plot(self.x_data, self.model_function(self.x_data, *self.params), 'k-', label='fitted')

        if plot_individuals:
            self._plot_individual_components()

        if auto_label:
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('MixedFittedData')
            plt.legend(loc='best')
            plt.tight_layout()

        return plt

    def _plot_individual_components(self):
        """Plots the individual fitted components of the model."""
        param_index = 0
        for model in self.model_list:
            if model == 'gaussian':
                amp = self.params[param_index]
                mu = self.params[param_index + 1]
                sigma = self.params[param_index + 2]
                y_component = GaussianDistribution.with_amplitude(amp, mu, sigma).pdf(self.x_data)
                plt.plot(self.x_data, y_component, '--', label=f'Gaussian({amp:.2f}, {mu:.2f}, {sigma:.2f})')
                param_index += 3

            elif model == 'line':
                slope = self.params[param_index]
                intercept = self.params[param_index + 1]
                y_component = line(self.x_data, slope, intercept)
                plt.plot(self.x_data, y_component, '--', label=f'Line({slope:.2f}, {intercept:.2f})')
                param_index += 2

    def _parameter_extractor(self):
        all_ = self.params
        p_index = 0
        param_dict = {}

        for model in self.model_list:
            if model not in param_dict:
                param_dict[model] = []

            if model == 'line':
                param_dict[model].extend([all_[p_index:p_index + 2]])
                p_index += 2
            elif model == 'gaussian':
                param_dict[model].extend([all_[p_index:p_index + 3]])
                p_index += 3

        return param_dict

    def parameter_extractor(self, model, return_individual_values=False):
        dict_ = self._parameter_extractor()

        if return_individual_values:
            par_dict = dict_[model]
            if model == 'gaussian':
                flattened_list = list(itertools.chain.from_iterable(par_dict))
                if len(flattened_list) > 1:
                    amplitude = flattened_list[::3]
                    mean = flattened_list[1::3]
                    std = flattened_list[2::3]

                    return amplitude, mean, std
                else:
                    return flattened_list
            elif model == 'line':
                return par_dict

    def get_fit_values(self):
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        return self.model_function(self.x_data, *self.params)

    def get_peaks(self):
        return find_peaks(self.get_fit_values())
