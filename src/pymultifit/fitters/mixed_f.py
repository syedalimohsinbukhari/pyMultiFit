"""Created on Aug 10 23:08:38 2024"""

import itertools
import warnings
from typing import Optional, Tuple, Union, List, Callable, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpyez.backend.uPlotting import LinePlot  # type: ignore
from mpyez.ezPlotting import plot_xy  # type: ignore
from scipy.optimize import Bounds, curve_fit

# importing from files to avoid circular import
from .chiSquare_f import ChiSquareFitter
from .exponential_f import ExponentialFitter
from .foldedNormal_f import FoldedNormalFitter
from .gamma_f import GammaFitter
from .gaussian_f import GaussianFitter
from .halfNormal_f import HalfNormalFitter
from .laplace_f import LaplaceFitter
from .logNormal_f import LogNormalFitter
from .polynomial_f import LineFitter
from .skewNormal_f import SkewNormalFitter
from .utilities_f import sanity_check, _plot_fit
from .. import (
    epsilon,
    GAUSSIAN,
    LAPLACE,
    LINE,
    LOG_NORMAL,
    SKEW_NORMAL,
    CHI_SQUARE,
    EXPONENTIAL,
    FOLDED_NORMAL,
    GAMMA,
    NORMAL,
    HALF_NORMAL,
    OneDArray,
    Params_,
)

# mock initialize the internal classes for auto MixedDataFitter class
fitter_dict = {
    CHI_SQUARE: ChiSquareFitter,
    EXPONENTIAL: ExponentialFitter,
    FOLDED_NORMAL: FoldedNormalFitter,
    GAMMA: GammaFitter,
    GAUSSIAN: GaussianFitter,
    NORMAL: GaussianFitter,
    HALF_NORMAL: HalfNormalFitter,
    LAPLACE: LaplaceFitter,
    LOG_NORMAL: LogNormalFitter,
    SKEW_NORMAL: SkewNormalFitter,
    LINE: LineFitter,
}


class MixedDataFitter:
    r"""
    Class to fit a mixture of different models to data.

    :param x_values: The x-values for the data.
    :param y_values: The y-values for the data.
    :param model_list: List of models to fit (e.g., `LINE`, `GAUSSIAN`, `LOG_NORMAL`)
    :param max_iterations: The maximum number of iterations for fitting procedure.
    """

    def __init__(
        self,
        x_values: OneDArray,
        y_values: OneDArray,
        model_list: List[str],
        fitter_dictionary: Optional[dict] = None,
        model_dictionary: Optional[dict] = None,
        max_iterations: int = 1000,
    ):
        # Check if the deprecated parameter was used
        if fitter_dictionary is not None:
            warnings.warn(
                message="`fitter_dictionary` is deprecated and will be removed in a future release. "
                "Use `model_dictionary` instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)

        self.x_values: np.ndarray = x_values
        self.y_values: np.ndarray = y_values
        self.model_list = model_list
        self.max_iterations = max_iterations
        self.params: Any = None
        self.covariance: Any = None

        self.fitter_dict = fitter_dictionary or fitter_dict
        self.fitter_dict = model_dictionary or fitter_dict

        # self._validate_models()
        self.model_function = self._create_model_function()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(x_values={self.x_values}, y_values={self.y_values}, "
            f"model_list={self.model_list}, max_iterations={self.max_iterations})"
        )

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
                model_class = self._instantiate_class(model=model)
                n_par = self._instantiate_n_par(model=model)
                y += model_class.fitter(x=x, params=list(params[param_index : param_index + n_par]))
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
            count += self._instantiate_n_par(model=model)

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
        return f"{value:.3E}" if t_high < abs(value) or abs(value) < t_low else f"{value:.3f}"

    def _get_bounds(self):
        """
        Sets the bounds for each parameter based on the model list.

        :returns: Lower and upper bounds for the parameters.
        """
        lower_bounds = []
        upper_bounds = []

        for model in self.model_list:
            lb, ub = self._instantiate_bounds(model=model)
            lower_bounds.extend(lb)
            upper_bounds.extend(ub)

        return np.array(lower_bounds), np.array(upper_bounds)

    def _instantiate_class(self, model: str):
        try:
            fitter_instance = self.fitter_dict[model](x_values=np.array([]), y_values=np.array([]))
        except KeyError:
            raise ValueError(f"Model '{model}' not recognized. Ensure it is defined in the fitter dictionary.")

        return fitter_instance

    def _instantiate_n_par(self, model: str) -> int:
        return self._instantiate_class(model).n_par

    def _instantiate_bounds(self, model: str) -> tuple[Sequence[float], Sequence[float]]:
        return self._instantiate_class(model).fit_boundaries()

    def _parameter_extractor(self, values: np.ndarray) -> dict:
        """
        Extracts the parameters for each model in the model list.

        :param values: The values from which the model dictionary is to be extracted.

        :return: A dictionary where the keys are model names and the values are lists of parameters/error values.
        """
        p_index = 0
        param_dict: dict = {}

        for model in self.model_list:
            if model not in param_dict:
                param_dict[model] = []

            n_pars = self._instantiate_n_par(model=model)
            param_dict[model].extend([values[p_index : p_index + n_pars]])
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
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]
        param_index = 0
        for i, model in enumerate(self.model_list):
            color = colors[i % len(colors)]
            class_model = self._instantiate_class(model=model)
            n_par = self._instantiate_n_par(model=model)
            pars = self.params[param_index : param_index + n_par]
            y_component = class_model.fitter(x=x, params=pars)
            plot_xy(
                x_data=x,
                y_data=y_component,
                x_label="",
                y_label="",
                plot_title="",
                data_label=f"{model.capitalize()} {i + 1}({', '.join(self._format_param(i) for i in pars)})",
                plot_dictionary=LinePlot(line_style="--", color=color),
                axis=plotter,
            )
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

    def fit(self, p0: Params_, frozen: Optional[Union[int, List[int]]] = None):
        """
        Fit the data.

        :param p0: Initial guess for the fitted parameters.
        :type p0: Union[List[Tuple[int | float, ...]], np.ndarray]

        :param frozen: Parameter number of list of parameter numbers to freeze the value of.
        :type frozen: Union[int, List[int]]

        :raises ValueError: If the length of the initial guess is not equal to the expected parameter count.
        """
        p0_chain = p0.tolist() if isinstance(p0, np.ndarray) else p0

        # flatten cannot always work here because the mixed fitter might contain a variable number of parameters
        p0_chain = list(itertools.chain.from_iterable(p0_chain))
        if len(p0_chain) != self._expected_param_count():
            raise ValueError(
                f"Initial parameters length {len(p0_chain)} does not match expected count "
                f"{self._expected_param_count()}."
            )

        lb, ub = self._get_bounds()

        if frozen:
            if isinstance(frozen, int):
                frozen = [frozen]
            for par_num in frozen:
                lb[par_num - 1] = p0_chain[par_num - 1] - epsilon
                ub[par_num - 1] = p0_chain[par_num - 1] + epsilon

        self.params, self.covariance, *_ = curve_fit(
            f=self.model_function,
            xdata=self.x_values,
            ydata=self.y_values,
            p0=np.array(p0_chain),
            maxfev=self.max_iterations,
            bounds=Bounds(lb=lb, ub=ub),
        )

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
            # Return a combined dictionary for all models
            return {"parameters": parameters, "errors": errs}

        # Prepare output for a specific model
        output: dict = {"parameters": {}, "errors": {}}

        keys = ["parameters", "errors"]
        n_pars = self._instantiate_n_par(model=model)
        for temp_, key in zip([parameters, errs], keys):
            par_dict = temp_.get(model, [])
            if n_pars == 2:
                output[key] = par_dict
            else:
                output[key] = np.array_split(np.asarray(par_dict, dtype=float).flatten(), n_pars)

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

    def plot_fit(
        self,
        show_individuals: bool = False,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        data_label: Optional[str] = None,
        fit_label: Optional[str] = None,
        title: Optional[str] = None,
        axis: Optional[Axes] = None,
    ):
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
        fit_label: str, optional
            The label for the fitted model.
        axis: Axes, optional
            Axes to plot instead of the entire figure. Defaults to None.

        Returns
        -------
        plotter
            The plotter handle for the drawn plot.
        """
        return _plot_fit(
            x_values=self.x_values,
            y_values=self.y_values,
            parameters=self.params,
            n_fits=len(self.model_list),
            class_name=self.__class__.__name__,
            _n_fitter=self.model_function,
            _n_plotter=self._plot_individual_fitter,
            show_individuals=show_individuals,
            x_label=x_label,
            y_label=y_label,
            title=title,
            data_label=data_label,
            fit_label=fit_label,
            axis=axis,
        )
