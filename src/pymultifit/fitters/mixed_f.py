"""Created on Aug 10 23:08:38 2024"""

import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpyez.backend.uPlotting import LinePlot
from mpyez.ezPlotting import plot_xy
from scipy.optimize import curve_fit

from .utilities_f import sanity_check
from .. import GAUSSIAN, LAPLACE, LINE, LOG_NORMAL, NORMAL, SKEW_NORMAL
from ..distributions import GaussianDistribution, LaplaceDistribution, line, LogNormalDistribution, \
    SkewNormalDistribution


class _Line:
    """
    Helper class for the line fitting function.

    This class is intended for internal use only.
    Provides a wrapper for evaluating a linear function with a given slope and intercept.
    """

    def __init__(self, slope: float, intercept: float):
        self.slope = slope
        self.intercept = intercept

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the value of the line function.

        Parameters
        ----------
        x: np.ndarray
            The input array to evaluate the line function.

        Returns
        -------
        np.ndarray
            The value of the line function for the given slope and intercept.
        """
        return line(x=x, slope=self.slope, intercept=self.intercept)


model_dict = {LINE: [_Line, 2],
              GAUSSIAN: [GaussianDistribution, 3],
              LOG_NORMAL: [LogNormalDistribution, 3],
              SKEW_NORMAL: [SkewNormalDistribution, 4],
              LAPLACE: [LaplaceDistribution, 3]}


class MixedDataFitter:
    """
    A class to fit a mixture of different models to data.

    Attributes
    ----------
    x_values : array_like
        The x-values for the data.
    y_values : array_like
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

    def __init__(self, x_values, y_values, model_list, max_iterations: int = 1000):
        """
        Initializes the MixedDataFitter with data and a list of models.

        Parameters
        ----------
        x_values : array_like
            The x-values for the data.
        y_values : array_like
            The y-values for the data.
        model_list : list of str
            List of models to fit (e.g., ['gaussian', 'gaussian', 'line']).
        max_iterations: int, optional
            The max number of iterations for fitting procedure.
        """
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)

        self.x_values = x_values
        self.y_values = y_values
        self.model_list = model_list
        self.max_iterations = max_iterations
        self.params = None
        self.covariance = None

        # Validate the model list and create the model function
        self._validate_models()
        self.model_function = self._create_model_function()

    def __repr__(self):
        return (f"{self.__class__.__name__}(x_values={self.x_values}, y_values={self.y_values}, "
                f"model_list={self.model_list}, max_iterations={self.max_iterations})")

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
            y = np.zeros_like(x, dtype=float)
            param_index = 0

            for model in self.model_list:
                func, n_par = model_dict[model]
                y += func(*params[param_index:param_index + n_par]).pdf(x=x)
                param_index += n_par

            return y

        return _composite_model

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
            _, n_par = model_dict[model]
            count += n_par

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
                lower_bounds.extend([0, -np.inf, -np.inf, 0])
                upper_bounds.extend([np.inf, np.inf, np.inf, np.inf])

        return lower_bounds, upper_bounds

    def _parameter_extractor(self, values):
        """
        Extracts the parameters for each model in the model list.

        Returns
        -------
        dict
            A dictionary where the keys are model names and the values are lists of parameters.
        """
        p_index = 0
        param_dict = {}

        for model in self.model_list:
            if model not in param_dict:
                param_dict[model] = []

            _, n_pars = model_dict[model]
            param_dict[model].extend([values[p_index:p_index + n_pars]])
            p_index += n_pars

        return param_dict

    def _plot_individual_fitter(self, plotter):
        x = self.x_values
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
        param_index = 0
        for i, model in enumerate(self.model_list):
            color = colors[i % len(colors)]
            m_, p_ = model_dict[model]
            pars = self.params[param_index:param_index + p_]
            y_component = m_(*pars).pdf(x)
            plot_xy(x_data=x, y_data=y_component,
                    x_label='', y_label='', plot_title='',
                    data_label=f'{model.capitalize()} {i + 1}({", ".join(self.format_param(i) for i in pars)})',
                    plot_dictionary=LinePlot(line_style='--', color=color),
                    axis=plotter)
            param_index += p_

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

    def fit(self, p0):
        """
        Fit the data.

        Parameters
        ----------
        p0 : array_like
            Initial guess for the fitting parameters.

        Raises
        ------
        ValueError
            If the length of initial parameters does not match the expected count.
        """
        p0 = list(itertools.chain.from_iterable(p0))
        if len(p0) != self._expected_param_count():
            raise ValueError(
                f"Initial parameters length {len(p0)} does not match expected count {self._expected_param_count()}.")

        self.params, self.covariance, *_ = curve_fit(f=self.model_function, xdata=self.x_values, ydata=self.y_values,
                                                     p0=p0, maxfev=self.max_iterations, bounds=self._get_bounds())

    @staticmethod
    def format_param(value, t_low=0.001, t_high=10_000):
        """Formats the parameter value based on its magnitude."""
        return f'{value:.3E}' if t_high < abs(value) or abs(value) < t_low else f'{value:.3f}'

    def plot_fit(self, show_individuals=False,
                 x_label: Optional[str] = None, y_label: Optional[str] = None, title: Optional[str] = None,
                 data_label: Optional[str] = None,
                 figure_size: tuple = (12, 6)) -> tuple:
        """
        Plots the original data, fitted model, and optionally individual components.

        Parameters
        ----------
        show_individuals : bool, optional
            Whether to plot individual fitted functions, by default False.
        x_label: str
            The label for the x-axis.
        y_label: str
            The label for the y-axis.
        title: str
            The title for the plot.
        data_label: str
            The label for the data.
        figure_size: tuple, optional
            The size of the figure, by default (12, 6).

        Returns
        -------
        tuple
            The figure and axes handle for the drawn plot.
        """
        if self.y_values is None or self.params is None:
            raise ValueError("Data must be fitted before plotting.")

        fig, ax = plt.subplots(figsize=figure_size)
        plotter = plot_xy(self.x_values, self.y_values, data_label=data_label if data_label else 'Data', axis=ax)
        plot_xy(x_data=self.x_values, y_data=self.model_function(self.x_values, *self.params),
                data_label='Total Fit', plot_dictionary=LinePlot(color='k'), axis=plotter)

        if show_individuals:
            self._plot_individual_fitter(plotter=plotter)

        plotter.set_xlabel(x_label if x_label else 'X')
        plotter.set_ylabel(y_label if y_label else 'Y')
        plotter.set_title(title if title else f'{self.__class__.__name__} fit')
        plotter.legend(loc='best')
        fig.tight_layout()

        return fig, plotter

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

        return self.model_function(self.x_values, *self.params)

    def get_parameters(self, model=None, get_errors: bool = False):
        """
        Extracts parameters for a specific model, or for all models if no model is specified.

        Parameters
        ----------
        get_errors : bool, optional
            If True, includes errors in the returned output. Defaults to False.
        model : str, optional
            The model name to extract parameters for. If None, extracts parameters for all models. Defaults to None.

        Returns
        -------
        dict
            A dictionary containing:
            - "parameters": Nested dictionary of parameter values for each model if `get_errors` is True.
            - "errors": Nested dictionary of errors for each model (if `get_errors=True`).
            Otherwise, returns just the parameters directly.
        """
        # Extract parameters and errors if requested
        if not get_errors:
            parameters = self._parameter_extractor(self.params)
            # Return parameters directly if errors are not requested
            return parameters if model is None else parameters.get(model, [])

        # If get_errors is True, extract parameters and errors
        parameters = self._parameter_extractor(self.params)
        errors = self._parameter_extractor(np.sqrt(np.diag(self.covariance)))

        if model is None:
            # Return combined dictionary for all models
            return {"parameters": parameters, "errors": errors}

        # Prepare output for a specific model
        output = {"parameters": {}, "errors": {}}

        keys = ["parameters", "errors"]
        for temp_, key in zip([parameters, errors], keys):
            par_dict = temp_.get(model, [])
            _, n_par = model_dict[model]
            flattened_list = [item for sublist in par_dict for item in sublist.tolist()]
            output[key] = tuple(flattened_list[i::n_par] for i in range(n_par))

        return output
