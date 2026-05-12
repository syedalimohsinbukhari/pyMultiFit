"""Created on Aug 10 23:08:38 2024"""

import itertools
import warnings
from typing import Callable, List, Optional, Sequence, Union

import matplotlib.pyplot as plt  # noqa: F401 – kept for any subclass that may reference it
import numpy as np
from matplotlib.axes import Axes  # noqa: F401 – part of public API type hints
from plotez import LinePlotConfig, plot_xy  # noqa: F401 – kept for external callers
from scipy.optimize import Bounds, curve_fit
from tqdm import trange

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
    r"""
    Class to fit a mixture of different models to data.

    :param x_values: The x-values for the data.
    :param y_values: The y-values for the data.
    :param model_list: List of models to fit (e.g., `LINE`, `GAUSSIAN`, `LOG_NORMAL`)
    :param max_iterations: The maximum number of iterations for fitting procedure.
    """

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

    def ci_bounds(
        self,
        ci_level: Union[int, list[int]] = 95,
        n_bootstrap: int = 1000,
        plot_it: bool = False,
        overall_ci: bool = True,
        individual_ci: bool = False,
        random_state: Optional[int] = None,
        axis=None,
    ):
        """
        Compute confidence interval (CI) bounds for fitted data using bootstrap resampling.

        Parameters
        ----------
        ci_level : int or list of int, optional
            Confidence interval level(s) as percentages (e.g., 95 for 95% CI). Defaults to 95.
        n_bootstrap : int, optional
            Number of bootstrap samples to generate. Defaults to 1000.
        plot_it : bool, optional
            If True, plots the fitted curve and shaded CI regions. Defaults to False.
        overall_ci : bool, optional
            If True, compute CI bounds for the summed fitted curve. Defaults to True.
        individual_ci : bool, optional
            If True, compute CI bounds for each individual model. Defaults to False.
        axis : matplotlib.axes.Axes, optional
            Axes to plot on. If None and plot_it=True, a new figure is created.
        random_state : int, optional
            Random seed for reproducibility. Can be an integer or None. Defaults to None.
        fast_bootstrap : bool, optional
            If True, use reduced max_iterations for bootstrap (faster, minimal accuracy loss). Defaults to True.
            Recommended for 2-3x speedup.

        Returns
        -------
        dict
            Dictionary with either or both:
            - 'overall_ci_XX': dict with 'lower', 'upper', 'median' bounds for summed fits (if overall_ci=True).
            - 'individual_ci_XX': list of dicts with 'lower', 'upper', 'median' for each model (if individual_ci=True).

        Raises
        ------
        ValueError
            If neither overall_ci nor individual_ci is True.
        RuntimeError
            If fit has not been performed yet.

        Notes
        -----
        This method uses bootstrap resampling of (x, y) data pairs to estimate confidence intervals.
        For each bootstrap sample, the model is refitted and predictions are generated. The CI bounds
        are computed from the percentiles of the bootstrap distribution.

        Using fast_bootstrap=True (default) provides 2-3x speedup by reducing max_iterations for
        bootstrap samples, which converge faster due to good initial guesses from the original fit.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        if not overall_ci and not individual_ci:
            raise ValueError("At least one of `overall_ci` or `individual_ci` must be True.")

        # Initialize a random number generator for reproducibility
        rng = np.random.default_rng(random_state)

        bootstrap_max_iter = self.max_iterations

        # Handle single or multiple CI levels
        ci_levels = [ci_level] if isinstance(ci_level, int) else ci_level

        # Store original parameters as a flattened list for refitting
        original_params = self.params
        n_samples = len(self.x_values)

        # Create a p0 structure for refitting (list of tuples per model)
        p0_list = []
        param_index = 0
        for model in self.model_list:
            n_par = self._instantiate_n_par(model=model)
            p0_list.append(tuple(original_params[param_index : param_index + n_par]))
            param_index += n_par

        # Storage for bootstrap predictions
        bootstrap_overall = []
        bootstrap_individual = []

        # Perform bootstrap resampling
        successful_bootstraps = 0
        for _ in trange(n_bootstrap):
            # Resample indices with replacement using RNG
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
            x_boot = self.x_values[bootstrap_indices]
            y_boot = self.y_values[bootstrap_indices]

            try:
                # Create a temporary fitter instance for the bootstrap sample
                temp_fitter = MixedDataFitter(
                    x_values=x_boot,
                    y_values=y_boot,
                    model_list=self.model_list,
                    model_dictionary=self.fitter_dict,
                    max_iterations=bootstrap_max_iter,
                )

                # Refit using original parameters as an initial guess
                temp_fitter.fit(p0=p0_list)

                # Generate predictions on original x_values
                if overall_ci:
                    overall_pred = temp_fitter.model_function(self.x_values, *temp_fitter.params)
                    bootstrap_overall.append(overall_pred)

                if individual_ci:
                    # Extract predictions for each individual model
                    individual_preds = []
                    param_idx = 0
                    for model in self.model_list:
                        model_class = self._instantiate_class(model=model)
                        n_par = self._instantiate_n_par(model=model)
                        model_params = temp_fitter.params[param_idx : param_idx + n_par]
                        individual_pred = model_class.fitter(x=self.x_values, params=list(model_params))
                        individual_preds.append(individual_pred)
                        param_idx += n_par
                    bootstrap_individual.append(individual_preds)

                successful_bootstraps += 1

            except (RuntimeError, ValueError):
                # Skip failed fits
                continue

        # Convert to arrays
        if overall_ci:
            bootstrap_overall = np.array(bootstrap_overall)
        if individual_ci:
            bootstrap_individual = np.array(bootstrap_individual)

        # Report success rate if some failed
        if successful_bootstraps < n_bootstrap:
            print(f"Warning: Only {successful_bootstraps}/{n_bootstrap} bootstrap samples succeeded.")

        if successful_bootstraps == 0:
            raise RuntimeError("All bootstrap samples failed. Try adjusting initial parameters or max_iterations.")

        # Compute confidence intervals
        results = {}

        for ci in ci_levels:
            lower_percentile = (100 - ci) / 2
            upper_percentile = 100 - lower_percentile

            if overall_ci:
                results[f"overall_ci_{ci}"] = {
                    "lower": np.percentile(bootstrap_overall, lower_percentile, axis=0),
                    "upper": np.percentile(bootstrap_overall, upper_percentile, axis=0),
                    "median": np.percentile(bootstrap_overall, 50, axis=0),
                }

            if individual_ci:
                results[f"individual_ci_{ci}"] = []
                for j in range(self.n_fits):
                    results[f"individual_ci_{ci}"].append(
                        {
                            "lower": np.percentile(bootstrap_individual[:, j], lower_percentile, axis=0),
                            "upper": np.percentile(bootstrap_individual[:, j], upper_percentile, axis=0),
                            "median": np.percentile(bootstrap_individual[:, j], 50, axis=0),
                        }
                    )

        # Plot if requested
        if plot_it:
            self.plotter.plot_ci_bounds(results, ci_levels, overall_ci, individual_ci, axis)

        return results
