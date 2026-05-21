"""Created on Aug 10 23:08:38 2024"""

import itertools
import warnings
from typing import Callable, List, Optional, Sequence, Union

import matplotlib.pyplot as plt  # noqa: F401 – kept for any subclass that may reference it
import numpy as np
from matplotlib.axes import Axes  # noqa: F401 – part of public API type hints
from plotez import LinePlotConfig, plot_xy  # noqa: F401 – kept for external callers
from scipy.optimize import Bounds, curve_fit

# importing from files to avoid circular import
from .backend import BaseFitter
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
from .utilities_f import _plot_fit
from .. import (
    CHI_SQUARE,
    EXPONENTIAL,
    FOLDED_NORMAL,
    GAMMA,
    GAUSSIAN,
    HALF_NORMAL,
    LAPLACE,
    LINE,
    LOG_NORMAL,
    NORMAL,
    SKEW_NORMAL,
    epsilon,
)
from ..typing import NDArray, Params_

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


class MixedDataFitter(BaseFitter):

    def __init__(
        self,
        x_values: NDArray,
        y_values: NDArray,
        model_list: List[str],
        fitter_dictionary: dict | None = None,
        model_dictionary: dict | None = None,
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

        # Set model-specific attributes before calling super().__init__()
        self.model_list = model_list
        self.fitter_dict = model_dictionary or fitter_dictionary or fitter_dict

        # Call parent constructor
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)

        # Set n_par to the total parameter count and n_fits to the number of models
        self.n_par = self._expected_param_count()
        self.n_fits = len(model_list)

        # Create the composite model function
        self.model_function = self._create_model_function()

    def __repr__(self) -> str:
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

    def _n_fitter(self, x: np.ndarray, *params) -> NDArray:
        """
        Override parent method to use the composite model function.

        Parameters
        ----------
        x : np.ndarray
            Input array of values for which the composite function is evaluated.
        params : tuple
            A tuple with all parameters to be fitted.

        Returns
        -------
        np.ndarray
            An array containing the composite fitted values for the input ``x``.
        """
        return self.model_function(x, *params)

    def _evaluate_individual_component(self, x: np.ndarray, fit_index: int, params: Params_) -> NDArray:
        """
        Override to evaluate a single model component for CI calculation.

        Parameters
        ----------
        x
            X-values at which to evaluate the model.
        fit_index
            Index of the component model in model_list (0-based).
        params
            Parameters for this specific component.

        Returns
        -------
        NDArray
            Evaluated y-values for this component.
        """
        model = self.model_list[fit_index]
        model_class = self._instantiate_class(model=model)
        return model_class.fitter(x=x, params=list(params))

    def _compute_individual_ci(
        self, mv_parameters: NDArray, x_: NDArray, bounds: list[tuple[int, tuple[float, float, float]]]
    ) -> dict:
        """
        Override to handle mixed models with different parameter counts.

        Parameters
        ----------
        mv_parameters
            Bootstrap parameter samples, shape (n_bootstrap, n_total_params).
        x_
            X-values at which to evaluate.
        bounds
            List of (ci_value, (lower_percentile, median_percentile, upper_percentile)).

        Returns
        -------
        dict
            Dictionary mapping ci_value to list of component CI dicts.
        """
        n_bootstrap = mv_parameters.shape[0]
        curves_ = np.zeros(shape=(n_bootstrap, self.n_fits, x_.shape[0]))

        # Generate curves for each model and bootstrap sample
        for boot_idx, boot_params in enumerate(mv_parameters):
            param_index = 0
            for model_idx, model in enumerate(self.model_list):
                n_par = self._instantiate_n_par(model=model)
                model_params = boot_params[param_index : param_index + n_par]
                curves_[boot_idx, model_idx, :] = self._evaluate_individual_component(x_, model_idx, model_params)
                param_index += n_par

        results = {}
        for ci_val, (lower_p, median_p, upper_p) in bounds:
            individual_results = []

            for fit_idx in range(self.n_fits):
                quantiles = np.quantile(curves_[:, fit_idx, :], [lower_p, median_p, upper_p], axis=0)

                # Validate dimensions
                if quantiles.shape[-1] != len(x_):
                    raise ValueError(
                        f"Dimension mismatch for fit {fit_idx}: x_range has length {len(x_)} but "
                        f"quantiles have shape {quantiles.shape}"
                    )

                individual_results.append({"lower": quantiles[0], "median": quantiles[1], "upper": quantiles[2]})

            results[ci_val] = individual_results

        return results

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

    def _parameter_extractor(self, values: NDArray) -> dict:
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
                plot_config=LinePlotConfig(linestyle="--", color=color),
                axis=plotter,
            )
            param_index += n_par

    def fit(self, p0: Params_, frozen: Optional[Union[int, List[int]]] = None):
        """
        Fit the data.

        :param p0: Initial guess for the fitted parameters.
        :type p0: Union[List[Tuple[int or float, ...]], np.ndarray]

        :param frozen: Parameter number of list of parameter numbers to freeze the value of.
        :type frozen: Union[int, List[int]]

        :raises ValueError: If the length of the initial guess is not equal to the expected parameter count.
        """
        p0_chain = p0.tolist() if isinstance(p0, np.ndarray) else p0
        p0_chain: list

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

        self._plotter = None  # invalidate cached plotter after each fit

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
