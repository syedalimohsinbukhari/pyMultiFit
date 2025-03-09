"""Created on Aug 10 23:08:38 2024"""

import itertools
from typing import Optional, Tuple, Union, List, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpyez.backend.uPlotting import LinePlot
from mpyez.ezPlotting import plot_xy
from scipy.optimize import Bounds, curve_fit

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
from .utilities_f import sanity_check, _plot_fit
from .. import (epsilon, GAUSSIAN, LAPLACE, LINE, LOG_NORMAL, SKEW_NORMAL, CHI_SQUARE, EXPONENTIAL, FOLDED_NORMAL,
                GAMMA_SR, GAMMA_SS, NORMAL, HALF_NORMAL)

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
                 model_list: List[str], fitter_dictionary=None, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)

        self.x_values = x_values
        self.y_values = y_values
        self.model_list = model_list
        self.max_iterations = max_iterations
        self.params = None
        self.covariance = None

        self.fitter_dict = fitter_dictionary or fitter_dict

        # self._validate_models()
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

    @staticmethod
    def _format_param(value, t_low=0.001, t_high=10_000) -> str:
        r"""
        Formats the parameter value to scientific notation based on its magnitude.

        Parameters
        ----------
        value: float
            The value of the parameter to be formatted.
        t_low: float, optional
            The lower bound below which the formatting should be applied to the value.
            Defaults to 0.001.
        t_high: float, optional
            The upper bound above which the formatting should be applied to the value.
            Defaults to 10,000.

        Returns
        -------
        str:
            A formatted string of the parameter value.
        """
        return f'{value:.3E}' if t_high < abs(value) or abs(value) < t_low else f'{value:.3f}'

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
            fitter_instance = self.fitter_dict[model](x_values=[], y_values=[])
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
                try:
                    result.append(fitter_instance.fit_boundaries())
                except NotImplementedError:
                    # in case the boundaries are not defined, put -np.inf, and np.inf to work with
                    n_par = fitter_instance.n_par
                    result.append([[-np.inf] * n_par, [np.inf] * n_par])

        return result[0] if len(result) == 1 else tuple(result)

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

    def _params(self) -> np.ndarray:
        r"""
        Store the fitted parameters of the fitted model.

        Returns
        -------
        np.ndarray
            The parameters obtained after performing the fit.

        Raises
        ------
        RuntimeError
            If the fit has not been performed yet (i.e., ``self.params`` is ``None``).

        Notes
        -----
        This method assumes that the fitting process assigns values to ``self.params``.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self.params

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
                    data_label=f'{model.capitalize()} {i + 1}({", ".join(self._format_param(i) for i in pars)})',
                    plot_dictionary=LinePlot(line_style='--', color=color),
                    axis=plotter)
            param_index += n_par

    def _standard_errors(self) -> np.ndarray:
        r"""
        Store the standard errors of the fitted parameters.

        Returns
        -------
        np.ndarray
            An array containing the standard errors of the fitted parameters.

        Raises
        ------
        RuntimeError
            If the fit has not been performed yet (i.e., ``self.covariance`` is ``None``).
        """
        if self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return np.sqrt(np.diag(self.covariance))

    def fit(self, p0: Union[List, np.ndarray], frozen: Union[int, List[int]] = None):
        """
        Fit the data.

        :param p0: Initial guess for the fitted parameters.
        :type p0: Union[List, np.ndarray]

        :param frozen: Parameter number of list of parameter numbers to freeze the value of.
        :type frozen: Union[int, List[int]]

        :raises ValueError: If the length of the initial guess is not equal to the expected parameter count.
        """
        # flatten cannot always work here because the mixed fitter might contain variable number of parameters
        p0 = list(itertools.chain.from_iterable(p0))
        if len(p0) != self._expected_param_count():
            raise ValueError(f"Initial parameters length {len(p0)} does not match expected count "
                             f"{self._expected_param_count()}.")

        lb, ub = self._get_bounds()

        if frozen:
            if isinstance(frozen, int):
                frozen = [frozen]
            for par_num in frozen:
                lb[par_num - 1] = p0[par_num - 1] - epsilon
                ub[par_num - 1] = p0[par_num - 1] + epsilon

        self.params, self.covariance, *_ = curve_fit(f=self.model_function, xdata=self.x_values, ydata=self.y_values,
                                                     p0=p0, maxfev=self.max_iterations, bounds=Bounds(lb=lb, ub=ub))

    def get_fitted_curve(self) -> np.ndarray:
        """
        Gets the y-values from the fitted model.

        :return: The y-values from the fitted model

        :raises ValueError: If the model has not been fitted yet.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        return self.model_function(self.x_values, *self.params)

    def get_model_parameters(self, model: Optional[str] = None, errors: bool = False):
        """
        Extracts parameters (and error) values for a specific model, or for all models if no model is specified.

        :param model: Model name to extract parameters for. If unspecified, extracts parameters for all models.
            Defaults to ``None``.
        :param errors: If ``True``, includes the errors in the returned output. Defaults to ``False``.

        :return: A dictionary containing:

                - "parameters": Nested dictionary of parameter values for each model if `get_errors` is True.
                - "errors": Nested dictionary of errors for each model (if `get_errors=True`).

                Otherwise, returns just the parameters directly.
        """

        parameters = self._parameter_extractor(self.params)
        errs = self._parameter_extractor(np.sqrt(np.diag(self.covariance)))

        if not errors:
            return parameters if model is None else parameters.get(model, [])

        if model is None:
            # Return combined dictionary for all models
            return {"parameters": parameters, "errors": errs}

        # Prepare output for a specific model
        output = {"parameters": {}, "errors": {}}

        keys = ["parameters", "errors"]
        n_pars = self._instantiate_fitter(model=model, return_values='n_par')
        for temp_, key in zip([parameters, errs], keys):
            par_dict = temp_.get(model, [])
            if n_pars == 2:
                output[key] = par_dict
            else:
                output[key] = np.array_split(ary=np.array(par_dict).flatten(), indices_or_sections=n_pars)

        return output

    def get_value_error_pair(self, mean_values: bool = True, std_values: bool = False) -> np.ndarray:
        r"""
        Retrieve the value/error pairs for the fitted parameters.

        This method provides the fitted parameter values and their corresponding standard errors as a combined array or
        individually based on the input flags.

        Parameters
        ----------
        mean_values : bool, optional
            If ``True``, return only the values of the fitted parameters.
            Defaults to ``True``.
        std_values : bool, optional
            If ``True``, return only the standard errors of the fitted parameters.
            Defaults to ``False``.

        Returns
        -------
        np.ndarray
            - If ``mean_values`` and ``std_values`` are both ``True``: A 2D array of shape (n_parameters, 2),
                where each row is ``[value, error]``.
            - If ``mean_values`` is ``True`` and ``std_values`` is ``False``: A 1D array of parameter values.
            - If ``std_values`` is ``True`` and ``mean_values`` is ``False``: A 1D array of standard errors.
            - If both flags are ``False``: An error message.

        Raises
        ------
        ValueError
            If both ``mean_values`` and ``std_values`` are ``False``.
        """
        pairs = np.column_stack([self._params(), self._standard_errors()])

        if mean_values and std_values:
            return pairs
        elif mean_values:
            return pairs[:, 0]
        elif std_values:
            return pairs[:, 1]
        else:
            raise ValueError("Either 'mean_values' or 'std_values' must be True.")

    def plot_fit(self, show_individuals: bool = False,
                 x_label: Optional[str] = None, y_label: Optional[str] = None, data_label: Union[list[str], str] = None,
                 title: Optional[str] = None, axis: Optional[Axes] = None):
        """
        Plot the fitted models.

        Parameters
        ----------
        show_individuals: bool, optional
            Whether to show individually fitted models or not.
        x_label: str, optional
            The label for the x-axis.
        y_label: str, optional
            The label for the y-axis.
        title: str, optional
            The title for the plot.
        data_label: str, optional
            The label for the data.
        axis: Axes, optional
            Axes to plot instead of the entire figure. Defaults to None.

        Returns
        -------
        plotter
            The plotter handle for the drawn plot.
        """
        return _plot_fit(x_values=self.x_values, y_values=self.y_values, parameters=self.params,
                         n_fits=len(self.model_list), class_name=self.__class__.__name__,
                         _n_fitter=self.model_function, _n_plotter=self._plot_individual_fitter,
                         show_individuals=show_individuals, x_label=x_label, y_label=y_label, title=title,
                         data_label=data_label, axis=axis)
