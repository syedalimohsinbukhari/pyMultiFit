"""Created on Aug 10 23:08:38 2024"""

import itertools
from typing import Optional, Tuple, Union, List, Callable

import matplotlib.pyplot as plt
import numpy as np
from mpyez.backend.uPlotting import LinePlot
from mpyez.ezPlotting import plot_xy
from scipy.optimize import curve_fit

# importing from files to avoid circular import
from .chiSquare_f import ChiSquareFitter
from .exponential_f import ExponentialFitter
from .foldedNormal_f import FoldedNormalFitter
from .gamma_f import GammaFitterSR, GammaFitterSS
from .gaussian_f import GaussianFitter
from .halfNormal_f import HalfNormalFitter
from .laplace_f import LaplaceFitter
from .logNormal_f import LogNormalFitter
from .others import LineFitter
from .skewNormal_f import SkewNormalFitter
from .utilities_f import sanity_check
from .. import (GAUSSIAN, LAPLACE, LINE, LOG_NORMAL, SKEW_NORMAL, CHI_SQUARE, EXPONENTIAL, FOLDED_NORMAL, GAMMA_SR,
                GAMMA_SS, NORMAL, HALF_NORMAL)

# mock initialize the internal classes for auto MixedDataFitter class
fitter_dict = {CHI_SQUARE: ChiSquareFitter,
               EXPONENTIAL: ExponentialFitter,
               FOLDED_NORMAL: FoldedNormalFitter,
               GAMMA_SR: GammaFitterSR,
               GAMMA_SS: GammaFitterSS,
               GAUSSIAN: GaussianFitter,
               NORMAL: GaussianFitter,
               HALF_NORMAL: HalfNormalFitter,
               LAPLACE: LaplaceFitter,
               LOG_NORMAL: LogNormalFitter,
               SKEW_NORMAL: SkewNormalFitter,
               LINE: LineFitter}


class MixedDataFitter:
    r"""
    Class to fit a mixture of different models to data.

    :param x_values: The x-values for the data.
    :param y_values: The y-values for the data.
    :param model_list: List of models to fit (e.g., `LINE`, `GAUSSIAN`, `LOG_NORMAL`)
    :param max_iterations: The maximum number of iterations for fitting procedure.
    """

    def __init__(self, x_values: Union[List, np.ndarray], y_values: Union[List, np.ndarray],
                 model_list: List[str], max_iterations: int = 1000, fitter_dictionary=None):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)

        self.x_values = x_values
        self.y_values = y_values
        self.model_list = model_list
        self.max_iterations = max_iterations
        self.params = None
        self.covariance = None

        self.fitter_dict = fitter_dictionary or fitter_dict

        self._validate_models()
        self.model_function = self._create_model_function()

    def __repr__(self):
        return (f"{self.__class__.__name__}(x_values={self.x_values}, y_values={self.y_values}, "
                f"model_list={self.model_list}, max_iterations={self.max_iterations})")

    def _instantiate_fitter(self, model: str, return_values: Union[str, List[str]] = 'class'):
        """
        Instantiate the fitter for the specified model and return requested values.

        :param model: The model name as a string.
        :param return_values: The specific attribute(s) or instance to return.
                              Options are 'class', 'n_par', and 'bounds'.
                              Can be a string (for one value) or a list of strings.
        :return: The requested values as a single value or a tuple.
        :raises ValueError: If the model is not recognized or return_values are invalid.
        """
        try:
            fitter_instance = self.fitter_dict[model]([], [])
        except KeyError:
            raise ValueError(f"Model '{model}' not recognized. Ensure it is defined in the fitter dictionary.")

        valid_options = {'class', 'n_par', 'bounds'}
        if isinstance(return_values, str):
            return_values = [return_values]  # Convert to list for uniform processing

        if not all(val in valid_options for val in return_values):
            invalid_options = [val for val in return_values if val not in valid_options]
            raise ValueError(f"Invalid return_values {invalid_options}. Expected values: {valid_options}")

        result = []
        for val in return_values:
            if val == 'class':
                result.append(fitter_instance)
            elif val == 'n_par':
                result.append(fitter_instance.n_par)
            elif val == 'bounds':
                result.append(fitter_instance.fit_boundaries())

        return result[0] if len(result) == 1 else tuple(result)

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
            y = np.zeros_like(a=x, dtype=float)
            param_index = 0

            for model in self.model_list:
                model_class, n_par = self._instantiate_fitter(model=model, return_values=['class', 'n_par'])
                y += model_class.fitter(x=x, params=params[param_index:param_index + n_par])
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
            n_par = self._instantiate_fitter(model=model, return_values='n_par')
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
            lb, ub = self._instantiate_fitter(model=model, return_values='bounds')
            lower_bounds.extend(lb)
            upper_bounds.extend(ub)

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

            n_pars = self._instantiate_fitter(model=model, return_values='n_par')
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
            class_model, n_par = self._instantiate_fitter(model=model, return_values=['class', 'n_par'])
            pars = self.params[param_index:param_index + n_par]
            y_component = class_model.fitter(x=x, params=pars)
            plot_xy(x_data=x, y_data=y_component,
                    x_label='', y_label='', plot_title='',
                    data_label=f'{model.capitalize()} {i + 1}({", ".join(self.format_param(i) for i in pars)})',
                    plot_dictionary=LinePlot(line_style='--', color=color),
                    axis=plotter)
            param_index += n_par

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
        # flatten cannot always work here because the mixed fitter might contain variable number of parameters
        p0 = list(itertools.chain.from_iterable(p0))
        if len(p0) != self._expected_param_count():
            raise ValueError(f"Initial parameters length {len(p0)} does not match expected count "
                             f"{self._expected_param_count()}.")

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
        plotter = plot_xy(x_data=self.x_values, y_data=self.y_values,
                          data_label=data_label if data_label else 'Data', axis=ax)
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
            n_pars = self._instantiate_fitter(model=model, return_values='n_par')
            flattened_list = [item for sublist in par_dict for item in sublist.tolist()]
            output[key] = tuple(flattened_list[i::n_pars] for i in range(n_pars))

        return output
