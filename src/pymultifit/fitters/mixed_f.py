"""Created on Aug 10 23:08:38 2024"""

import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ..distributions import GaussianDistribution, LogNormalDistribution, SkewedNormalDistribution
from ..others import line


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
        allowed_models = {'gaussian', 'line', 'log_normal', 'skew_normal'}
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
                if model in ['gaussian', 'normal']:
                    amplitude, mean, std = params[param_index:param_index + 3]
                    y += GaussianDistribution.with_amplitude(amplitude, mean, std).pdf(x)
                    param_index += 3

                elif model == 'line':
                    slope, intercept = params[param_index:param_index + 2]
                    y += line(x, slope, intercept)
                    param_index += 2

                elif model in ['skew_normal', 'skewNorm']:
                    shape, loc, scale = params[param_index:param_index + 3]
                    y += SkewedNormalDistribution(shape, loc, scale).pdf(x)
                    param_index += 3

                elif model in ['log_normal', 'logNorm']:
                    amplitude, mean, std = params[param_index:param_index + 3]
                    y += LogNormalDistribution.with_amplitude(amplitude, mean, std).pdf(x)
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
            if model == 'gaussian' or model == 'log_normal' or model == 'skew_normal':
                count += 3
            elif model == 'line':
                count += 2
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
            if model in ['gaussian', 'log_normal']:
                lower_bounds.extend([0, -np.inf, 0])
                upper_bounds.extend([np.inf, np.inf, np.inf])
            elif model == 'line':
                lower_bounds.extend([-np.inf, -np.inf])
                upper_bounds.extend([np.inf, np.inf])
            elif model == 'skew_normal':
                lower_bounds.extend([-np.inf, -np.inf, 0])
                upper_bounds.extend([np.inf, np.inf, np.inf])

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
        model_dict = {
            'gaussian': GaussianDistribution.with_amplitude,
            'log_normal': LogNormalDistribution.with_amplitude,
            'skew_normal': SkewedNormalDistribution
        }

        param_index = 0
        for model in self.model_list:
            if model == 'line':
                slope, intercept = self.params[param_index:param_index + 2]
                y_component = line(self.x_data, slope, intercept)
                plt.plot(self.x_data, y_component, '--', label=f'Line({slope:.2f}, {intercept:.2f})')
                param_index += 2

            elif model in model_dict:
                pars = self.params[param_index:param_index + 3]
                y_component = model_dict[model](*pars).pdf(self.x_data)
                plt.plot(self.x_data, y_component, '--',
                         label=f'{model.capitalize()}({pars[0]:.3E}, {pars[1]:.3f}, {pars[2]:.3f})')
                param_index += 3

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

            if model == 'line':
                param_dict[model].extend([all_[p_index:p_index + 2]])
                p_index += 2
            elif model in ['gaussian', 'log_normal', 'skew_normal']:
                param_dict[model].extend([all_[p_index:p_index + 3]])
                p_index += 3

        return param_dict

    def parameter_extractor(self, model, return_individual_values=False):
        """
        Extracts parameters for a specific model.

        Parameters
        ----------
        model : str
            The model name to extract parameters for.
        return_individual_values : bool, optional
            If True, returns the parameters in a more detailed format, by default False.

        Returns
        -------
        list or tuple
            The extracted parameters for the specified model.
            If `return_individual_values` is True and the model is 'gaussian', returns separate lists for amplitude,
            mean, and standard deviation. Otherwise, returns a list of parameter sets.
        """
        dict_ = self._parameter_extractor()

        if return_individual_values:
            par_dict = dict_.get(model, [])
            if model in ['gaussian', 'log_normal', 'skew_normal', 'laplace']:
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
