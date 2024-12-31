"""Created on Aug 10 23:08:38 2024"""

import itertools
from typing import Optional, Tuple, Union, List, Callable

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
    r"""
    Class to fit a mixture of different models to data.

    :param x_values: The x-values for the data.
    :param y_values: The y-values for the data.
    :param model_list: List of models to fit (e.g., `LINE`, `GAUSSIAN`, `LOG_NORMAL`)
    :param max_iterations: The maximum number of iterations for fitting procedure.
    """

    def __init__(self, x_values: Union[List, np.ndarray], y_values: Union[List, np.ndarray],
                 model_list: List[str], max_iterations: int = 1000):
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

    def _create_model_function(self) -> Callable:
        """
        Creates a composite model function based on the specified models.

        :return: A composite model for fitting.
        """

        def _composite_model(x: np.ndarray, *params) -> np.ndarray:
            """
            Compute the composite model.

            Parameters
            ----------
            x : np.ndarray
                The x-values where the model is evaluated.
            params : tuple
                Parameters for the model components.

            Returns
            -------
            y : np.ndarray
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

    def _expected_param_count(self) -> int:
        """
        Calculates the expected number of parameters based on the model list.

        :return: The number of parameters.
        """
        count = 0
        for model in self.model_list:
            _, n_par = model_dict[model]
            count += n_par

        return count

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sets the bounds for each parameter based on the model list.

        :returns: Lower and upper bounds for the parameters.
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

        return np.array(lower_bounds), np.array(upper_bounds)

    def _parameter_extractor(self, values: np.ndarray) -> dict:
        """
        Extracts the parameters for each model in the model list.

        :param values: The values from which the model dictionary is to be extracted.

        :return: A dictionary where the keys are model names and the values are lists of parameters/error values.
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
        """
        Plot the individual fitters function.

        :param plotter: The plotting axis object
        """
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

        :raise ValueError: If any model in the model list is not recognized.
        """
        allowed_models = {GAUSSIAN, LINE, LOG_NORMAL, SKEW_NORMAL, LAPLACE}
        if not all(model in allowed_models for model in self.model_list):
            raise ValueError(f"All models must be one of {allowed_models}.")

    def fit(self, p0: Union[List, np.ndarray]):
        """
        Fit the data.

        :param p0: Initial guess for the fitted parameters.

        :raises ValueError: If the length of the initial guess is not equal to the expected parameter count.
        """
        p0 = list(itertools.chain.from_iterable(p0))
        if len(p0) != self._expected_param_count():
            raise ValueError(
                f"Initial parameters length {len(p0)} does not match expected count {self._expected_param_count()}.")

        self.params, self.covariance, *_ = curve_fit(f=self.model_function, xdata=self.x_values, ydata=self.y_values,
                                                     p0=p0, maxfev=self.max_iterations, bounds=self._get_bounds())

    @staticmethod
    def format_param(value: float, t_low: float = 0.001, t_high: float = 10_000) -> str:
        """
        Formats the parameter value based on its magnitude.

        :param value: The value of the parameter to be formatted.
        :param t_low: The lower bound below which the value is to be formatted.
        :param t_high: The upper bound above which the value is to be formatted.

        :return: The formatted value of the parameter
        """
        return f'{value:.3E}' if t_high < abs(value) or abs(value) < t_low else f'{value:.3f}'

    def plot_fit(self, show_individuals: bool = False,
                 x_label: Optional[str] = None, y_label: Optional[str] = None, title: Optional[str] = None,
                 data_label: Optional[str] = None, figure_size: tuple = (12, 6)) -> tuple:
        """
        Plots the original data, fitted model, and optionally individual components.

        :param show_individuals: Whether to plot individual fitted functions, by default False.
        :param x_label: The label for the x-axis of the plot.
        :param y_label: The label for the y-axis of the plot.
        :param title: The title for the plot.
        :param data_label: The label for the data to be plotted.
        :param figure_size: The size of the figure. Default is (12,6).

        :return: A tuple of figure and axes object for the drawn plot

        :raises ValueError: Raised if the plotting function is called before the fitting is done.
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

    def get_fit_values(self) -> np.ndarray:
        """
        Gets the y-values from the fitted model.

        :return: The y-values from the fitted model

        :raises ValueError: If the model has not been fitted yet.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        return self.model_function(self.x_values, *self.params)

    def get_parameters(self, model: Optional[str] = None, get_errors: bool = False):
        """
        Extracts parameters (and error) values for a specific model, or for all models if no model is specified.

        :param model: Model name to extract parameters for. If unspecified, extracts parameters for all models.
            Defaults to ``None``.
        :param get_errors: If ``True``, includes the errors in the returned output. Defaults to ``False``.

        :return: A dictionary containing:

                - "parameters": Nested dictionary of parameter values for each model if `get_errors` is True.
                - "errors": Nested dictionary of errors for each model (if `get_errors=True`).

                Otherwise, returns just the parameters directly.
        """
        if not get_errors:
            parameters = self._parameter_extractor(self.params)
            return parameters if model is None else parameters.get(model, [])

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
