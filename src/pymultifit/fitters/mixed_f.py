"""Created on Aug 10 23:08:38 2024"""

import itertools
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ._backend.utilities import sanity_check
from .. import GAUSSIAN, LAPLACE, LINE, LOG_NORMAL, NORMAL, SKEW_NORMAL
from ..distributions import (GaussianDistribution, LaplaceDistribution, line, LogNormalDistribution,
                             SkewedNormalDistribution)


class MixedDataFitter:
    """
    A class to fit a mixture of different models to data.

    Attributes
    ----------
    x_data : array_like
        The x-values for the data.
    y_data : array_like
        The y-values for the data.
    model_list : list of str
        List of models to fit (e.g., ['gaussian', 'gaussian', 'line']).
    params : array_like, optional
        The fitted parameters of the model after fitting.
    covariance : array_like, optional
        The covariance of the fitted parameters.
    model_function : callable
        The composite model function used for fitting.
    """

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
        x_data, y_data = sanity_check(x_values=x_data, y_values=y_data)

        self.x_data = x_data
        self.y_data = y_data
        self.model_list = model_list
        self.params = None
        self.covariance = None

        # Validate the model list and create the model function
        self._validate_models()
        self.model_function = self._create_model_function()

    def _validate_models(self):
        """
        Validate the models in the model list.

        Raises
        ------
        ValueError
            If any model in the model list is not recognized.
        """
        allowed_models = {GAUSSIAN, LINE, LOG_NORMAL, SKEW_NORMAL, LAPLACE}
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
            """
            Compute the composite model.

            Parameters
            ----------
            x : array_like
                The x-values where the model is evaluated.
            params : tuple
                Parameters for the model components.

            Returns
            -------
            y : array_like
                The computed y-values from the composite model.
            """
            y = np.zeros_like(x)
            param_index = 0

            for model in self.model_list:
                if model in [GAUSSIAN, NORMAL]:
                    amplitude, mean, std = params[param_index:param_index + 3]
                    y += GaussianDistribution.with_amplitude(amplitude, mean, std).pdf(x)
                    param_index += 3

                elif model == LINE:
                    slope, intercept = params[param_index:param_index + 2]
                    y += line(x, slope, intercept)
                    param_index += 2

                elif model == SKEW_NORMAL:
                    shape, loc, scale = params[param_index:param_index + 3]
                    y += SkewedNormalDistribution(shape, loc, scale).pdf(x)
                    param_index += 3

                elif model == LOG_NORMAL:
                    amplitude, mean, std = params[param_index:param_index + 3]
                    y += LogNormalDistribution.with_amplitude(amplitude, mean, std).pdf(x)
                    param_index += 3

                elif model == LAPLACE:
                    amplitude, mean, diversity = params[param_index:param_index + 3]
                    y += LaplaceDistribution.with_amplitude(amplitude, mean, diversity).pdf(x)
                    param_index += 3

            return y

        return _composite_model

    def fit(self, p0):
        """
        Fits the model to the data using non-linear least squares.

        Parameters
        ----------
        p0 : array_like
            Initial guess for the fitting parameters.

        Raises
        ------
        ValueError
            If the length of initial parameters does not match the expected count.
        """
        p0_ = list(itertools.chain.from_iterable(p0))
        if len(p0_) != self._expected_param_count():
            raise ValueError(f"Initial parameters length {len(p0)} does not match expected "
                             f"count {self._expected_param_count()}.")

        lower_bounds, upper_bounds = self._get_bounds()

        _ = curve_fit(self.model_function, self.x_data, self.y_data, p0_, bounds=(lower_bounds, upper_bounds))
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
            if model in [GAUSSIAN, NORMAL, LOG_NORMAL, LAPLACE]:
                count += 3
            elif model == LINE:
                count += 2
            elif model == SKEW_NORMAL:
                count += 4

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
            if model in [GAUSSIAN, NORMAL, LOG_NORMAL, LAPLACE]:
                lower_bounds.extend([0, -np.inf, 0])
                upper_bounds.extend([np.inf, np.inf, np.inf])
            elif model == LINE:
                lower_bounds.extend([-np.inf, -np.inf])
                upper_bounds.extend([np.inf, np.inf])
            elif model == SKEW_NORMAL:
                lower_bounds.extend([-np.inf, -np.inf, -np.inf, 0])
                upper_bounds.extend([np.inf, np.inf, np.inf, np.inf])

        return lower_bounds, upper_bounds

    def plot(self, plot_individuals=False, auto_label=False,
             fig_size: Optional[Tuple[int, int]] = (12, 6), ax: Optional[plt.Axes] = None):
        """
        Plots the original data, fitted model, and optionally individual components.

        Parameters
        ----------
        fig_size
        ax
        plot_individuals : bool, optional
            Whether to plot individual fitted functions, by default False.
        auto_label : bool, optional
            If True, automatically labels the plot with 'X', 'Y', 'MixedFittedData',
            applies a legend, and adjusts layout, by default False.

        Returns
        -------
        matplotlib.pyplot
            The plot object.

        Raises
        ------
        ValueError
            If data is not fitted before plotting.
        """
        if self.y_data is None or self.params is None:
            raise ValueError("Data must be fitted before plotting.")

        plotter = ax if ax else plt
        if not ax:
            plt.figure(figsize=fig_size)

        plotter.plot(self.x_data, self.y_data, '-', label='data')
        plotter.plot(self.x_data, self.model_function(self.x_data, *self.params), 'k-', label='fitted')

        if plot_individuals:
            self._plot_individual_components()

        if auto_label:
            plotter.xlabel('X')
            plotter.ylabel('Y')
            plotter.title('MixedFittedData')
            plotter.legend(loc='best')
            plotter.tight_layout()

        return plotter

    def _plot_individual_components(self):
        """Plots the individual fitted components of the model."""
        model_dict = {GAUSSIAN: GaussianDistribution.with_amplitude,
                      LOG_NORMAL: LogNormalDistribution.with_amplitude,
                      SKEW_NORMAL: SkewedNormalDistribution,
                      LAPLACE: LaplaceDistribution.with_amplitude}

        param_index = 0
        for model in self.model_list:
            if model == LINE:
                slope, intercept = self.params[param_index:param_index + 2]
                y_component = line(self.x_data, slope, intercept)
                plt.plot(self.x_data, y_component, '--',
                         label=f'Line({self._format_param(slope)}, {self._format_param(intercept)})')
                param_index += 2

            elif model in model_dict:
                pars = self.params[param_index:param_index + 3]
                y_component = model_dict[model](*pars).pdf(self.x_data)
                plt.plot(self.x_data, y_component, '--',
                         label=f'{model.capitalize()}({self._format_param(pars[0])}, '
                               f'{self._format_param(pars[1])}, {self._format_param(pars[2])})')
                param_index += 3

    @staticmethod
    def _format_param(param):
        """Dynamically formats the parameter based on its value."""
        return f'{param:.4E}' if abs(param) < 0.001 else f'{param:.4f}'

    def _parameter_extractor(self):
        """
        Extracts the parameters for each model in the model list.

        Returns
        -------
        dict
            A dictionary where the keys are model names and the values are lists of parameters.
        """
        all_ = self.params
        p_index = 0
        param_dict = {}

        for model in self.model_list:
            if model not in param_dict:
                param_dict[model] = []

            if model == LINE:
                param_dict[model].extend([all_[p_index:p_index + 2]])
                p_index += 2
            elif model in [GAUSSIAN, NORMAL, LOG_NORMAL, LAPLACE]:
                param_dict[model].extend([all_[p_index:p_index + 3]])
                p_index += 3
            elif model == SKEW_NORMAL:
                param_dict[model].extend(([all_[p_index:p_index + 4]]))
                p_index += 4

        return param_dict

    def parameter_extractor(self, model=None, return_individual_values=True):
        """
        Extracts parameters for a specific model, or for all models if no model is specified.

        Parameters
        ----------
        model : str, optional
            The model name to extract parameters for. If None, extracts parameters for all models. Defaults to None.
        return_individual_values : bool, optional
            If True, returns the parameters in a more detailed format.
            This is automatically set to False if no model is specified. Defaults to True.

        Returns
        -------
        dict, list, or tuple
            If `model` is None, returns a dictionary with model names as keys and lists of parameter sets as values.
            If a specific model is specified, returns the extracted parameters for that model.
        """
        dict_ = self._parameter_extractor()

        if model is None:
            return dict_

        if return_individual_values:
            par_dict = dict_.get(model, [])
            flattened_list = list(itertools.chain.from_iterable(par_dict))
            if model in [GAUSSIAN, NORMAL, LOG_NORMAL, LAPLACE]:
                return flattened_list[::3], flattened_list[1::3], flattened_list[2::3]
            elif model == LINE:
                return [flattened_list[i] for i in range(2)]
            elif model == SKEW_NORMAL:
                return flattened_list[::4], flattened_list[1::4], flattened_list[2::4], flattened_list[3::4]
        else:
            return dict_.get(model, [])

    def get_fit_values(self):
        """
        Gets the y-values from the fitted model.

        Returns
        -------
        array_like
            The fitted y-values from the model.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        return self.model_function(self.x_data, *self.params)

    def get_peaks(self):
        """
        Finds peaks in the fitted model values.

        Returns
        -------
        tuple
            Indices of peaks in the fitted model values.
        """
        return find_peaks(self.get_fit_values())
