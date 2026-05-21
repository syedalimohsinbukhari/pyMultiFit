"""Created on Jul 18 00:16:01 2024"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from typing import Any
from warnings import warn

import numpy as np
from matplotlib.axes import Axes
from numpy.random import Generator
from scipy.optimize import Bounds, curve_fit

from ..utilities_f import parameter_logic, sanity_check
from ... import epsilon
from ..._plot import FitPlotter
from ...typing import ArrayLike, NDArray, Params_


class BaseFitter:
    """The base class for multi-fitting functionality."""

    _plotter: FitPlotter | None

    def __init__(self, x_values: ArrayLike, y_values: ArrayLike, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        self.x_values = x_values
        self.y_values = y_values
        self.max_iterations = max_iterations

        self.n_par: int = 0
        self.pn_par: int = self.n_par
        self.sn_par: dict = {}

        self.n_fits: int = 0
        self.params: NDArray | None = None
        self.covariance: NDArray | None = None

    @property
    def plotter(self) -> FitPlotter:
        """Return a cached :class:`~pymultifit._plot.FitPlotter` for this fitter.

        The cache is invalidated automatically each time :meth:`fit` is called, so the plotter always reflects the
        most recent fitted parameters.
        """
        if self._plotter is None:
            self._plotter: FitPlotter = FitPlotter(self)
        return self._plotter

    def _adjust_parameters(self, p0: Params_) -> Params_:
        """
        Adjust input parameters to include defaults for secondary parameters if missing.

        Parameters
        ----------
        p0 :
            A list of initial guesses for the parameters.

        Returns
        -------
        Params_
            Adjusted parameter list with default values for missing secondary parameters.
        """
        adjusted_p0 = []
        for params in p0:
            # Too few parameters
            if len(params) < self.pn_par:
                raise ValueError(f"Each parameter set must have at least {self.pn_par} primary parameters.")

            primary_params = params[: self.pn_par]
            provided_secondary_params = params[self.pn_par :]

            secondary_params = dict(self.sn_par)
            for key, value in zip(self.sn_par.keys(), provided_secondary_params):
                secondary_params[key] = value

            adjusted_params = list(primary_params) + list(secondary_params.values())

            if len(adjusted_params) != self.n_par:
                raise ValueError(f"Adjusted parameter set must have {self.n_par} total parameters.")

            adjusted_p0.append(adjusted_params)

        return adjusted_p0

    def _fit_preprocessing(self, p0: Params_, frozen: list[bool] | None) -> tuple[NDArray, NDArray, NDArray]:
        """
        Process frozen parameters and adjust bounds.

        Parameters
        ----------
        p0 :
            A list of initial guesses for the parameters of the models. For example, [(1, 1, 0), (3, 3, 2)].
        frozen :
            A list of booleans indicating which parameters are frozen.
            For example, [False, False, True] for 3 parameters.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray[np.floating]]:
            Adjusted lower and upper bounds, and flattened initial guesses.
        """
        # Get initial boundaries
        try:
            lb, ub = self.fit_boundaries()
        except NotImplementedError:
            # if they're not implemented, self-imposes -inf + inf boundaries
            lb = np.repeat(-np.inf, repeats=self.n_fits)
            ub = np.repeat(np.inf, repeats=self.n_fits)

        # Resize bounds to match total parameters
        lb = np.resize(lb, new_shape=self.n_par * self.n_fits)
        ub = np.resize(ub, new_shape=self.n_par * self.n_fits)

        # Validate frozen length
        if frozen is None:
            frozen: list[bool] = [False] * self.n_par

        if len(frozen) != self.n_par:
            raise ValueError("The length of 'frozen' must match the number of parameters per model.")

        # Repeat frozen mask for all models
        frozen: list[bool] = frozen * self.n_fits

        # Flatten initial guesses
        p0_flat = np.array(p0).flatten()

        # Adjust bounds for frozen parameters
        for i, is_frozen in enumerate(frozen):
            if is_frozen:
                lb[i] = p0_flat[i] - epsilon
                ub[i] = p0_flat[i] + epsilon

        return lb, ub, p0_flat

    @staticmethod
    def _format_param(value, t_low: float = 0.001, t_high: float = 10_000.0) -> str:
        """
        Formats the parameter value to scientific notation based on its magnitude.

        Parameters
        ----------
        value :
            The value of the parameter to be formatted.
        t_low :
            The lower bound below which the formatting should be applied to the value. Defaults to 0.001.
        t_high :
            The upper bound above which the formatting should be applied to the value. Defaults to 10,000.

        Returns
        -------
        str :
            A formatted string of the parameter value.
        """
        return f"{value:.3E}" if t_high < abs(value) or abs(value) < t_low else f"{value:.3f}"

    def _n_fitter(self, x: ArrayLike, *params: Params_) -> NDArray:
        """
        Perform N-fitting by summing over multiple parameter sets.

        Parameters
        ----------
        x :
            Input array of values for which the composite function is evaluated.
        params :
            A tuple with all parameters to be fitted in an array of size (``self.n_fits, self.n_par``) where:

            - ``self.n_fits`` is the number of individual fits.
            - ``self.n_par`` is the number of parameters per fit.

        Returns
        -------
        NDArray
            An array containing the composite fitted values for the input ``x``.
        """
        y = np.zeros_like(x, dtype=float)
        parameters: NDArray = np.reshape(np.array(params), newshape=(self.n_fits, self.n_par))
        for par in parameters:
            y += self.fitter(x=x, params=par.tolist())
        return y

    def _params(self) -> NDArray:
        """
        Store the fitted parameters of the fitted model.

        Returns
        -------
        NDArray
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

    def _standard_errors(self) -> NDArray:
        """
        Store the standard errors of the fitted parameters.

        Returns
        -------
        NDArray
            An array containing the standard errors of the fitted parameters.

        Raises
        ------
        RuntimeError
            If the fit has not been performed yet (i.e., ``self.covariance`` is ``None``).
        """
        if self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return np.sqrt(np.diag(self.covariance))

    def dry_run(self, axis: Axes | None = None):
        """
        Plot the x and y data for a quick visual inspection of the data.

        Parameters
        ----------
        axis:
            The axis to plot the data on.
        """
        self.plotter.dry_run(axis=axis)

    def fit(self, p0: Params_, frozen: list[bool] | None = None):
        """
        Fit the data.

        Parameters
        ----------
        p0:
            A list of initial guesses for the parameters of the models.
            For example, [(1, 1, 0), (3, 3, 2)].
        frozen:
            A list of booleans indicating whether each parameter is frozen.
            For example, [False, False, True] for 3 parameters.
        """
        self.n_fits = len(p0)
        len_guess = len(list(chain(*p0)))
        total_pars = self.n_par * self.n_fits

        if len_guess != total_pars:
            p0 = self._adjust_parameters(p0)

        lb, ub, p0_flat = self._fit_preprocessing(p0=p0, frozen=frozen)

        self.params, self.covariance, *_ = curve_fit(
            f=self._n_fitter,
            xdata=self.x_values,
            ydata=self.y_values,
            p0=p0_flat,
            maxfev=self.max_iterations,
            bounds=Bounds(lb=lb, ub=ub),
        )
        self._plotter = None  # invalidate cached plotter after each fit

    def _fit_boundaries(self) -> tuple[list[float], list[float]]:
        """Defines the internal distribution boundaries to be used by fitter."""
        ub = np.repeat(np.inf, repeats=self.n_par).tolist()
        lb = np.repeat(-np.inf, repeats=self.n_par).tolist()

        return lb, ub

    def fit_boundaries(self) -> tuple[list[float], list[float]]:
        """Defines the distribution boundaries to be used by fitter."""
        return self._fit_boundaries()

    @staticmethod
    def fitter(x, params: Params_):
        """
        Fitter function for multi-fitting.

        Parameters
        ----------
        x :
            The x-array on which the fitting is to be performed.
        params :
            An array of parameters to fit.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _evaluate_individual_component(self, x: ArrayLike, fit_index: int, params: Params_) -> NDArray:
        """
        Evaluate a single model component for CI calculation.

        This method is used by ci_bounds() for individual_ci calculation.
        Override in subclasses if special handling is needed (e.g., MixedDataFitter).

        Parameters
        ----------
        x
            X-values at which to evaluate the model.
        fit_index
            Index of the component model (0-based).
        params
            Parameters for this specific component.

        Returns
        -------
        NDArray
            Evaluated y-values for this component.
        """
        return self.fitter(x, params)

    def get_fitted_curve(self) -> NDArray:
        """
        Get the fitted values of the model.

        Returns
        -------
        NDArray
            An array of fitted values.

        Raises
        ------
        RuntimeError
            If the fit has not been performed yet.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self._n_fitter(self.x_values, *self.params)

    def get_residuals(self) -> NDArray:
        """
        Get the residuals (difference between data and fitted model).

        Returns
        -------
        NDArray
            An array of residual values (y_data - y_fitted).

        Raises
        ------
        RuntimeError
            If the fit has not been performed yet.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        fitted_curve = self.get_fitted_curve()

        return self.y_values - fitted_curve

    def get_model_parameters(self, select: tuple[int, Any] | None = None, errors: bool = False):
        """
        Extract specific parameter values or their uncertainties from the fitting process.

        Parameters
        ----------
        select
            A list of indices specifying which submodels to extract parameters for.
            Indexing starts at 1.
            If ``None``, parameters for all submodels are returned.
            Defaults to None.
        errors
            If ``True``, both the parameter values and their standard errors are returned.
            Defaults to ``False``.

        Returns
        -------
        NDArray | tuple[NDArray, NDArray[np.floating]]
            Parameter values (and optionally uncertainties) for the selected submodels.

        Raises
        ------
        ValueError
            If ``select`` contains invalid indices or is incompatible with the model structure.

        Notes
        -----
            - The ``select`` parameter allows filtering by specific submodel indices. If ``None``, all submodels are used.
            - When ``errors=True``, parameter means and uncertainties are returned as separate arrays of identical shape.
        """
        parameter_mean = self.get_value_error_pair(mean_values=True, std_values=errors)

        if not errors:
            selected = parameter_logic(par_array=parameter_mean, n_par=self.n_par, selected_models=select)

            return selected[:, range(self.n_par)].T
        else:
            par_list = parameter_mean.reshape(self.n_fits, self.n_par, 2)
            mean = parameter_logic(par_array=par_list[:, :, 0].flatten(), n_par=self.n_par, selected_models=select)
            std_ = parameter_logic(par_array=par_list[:, :, 1].flatten(), n_par=self.n_par, selected_models=select)

            return mean[:, range(self.n_par)].T, std_[:, range(self.n_par)].T

    def get_value_error_pair(
        self, mean_values: bool = True, std_values: bool = False
    ) -> NDArray | tuple[NDArray, NDArray]:
        """
        Retrieve the value/error pairs for the fitted parameters.

        Parameters
        ----------
        mean_values :
            If ``True``, return only the values of the fitted parameters. Defaults to ``True``.
        std_values :
            If ``True``, return only the standard errors of the fitted parameters. Defaults to ``False``.

        Returns
        -------
        NDArray | tuple[NDArray, NDArray]
            A 2D array containing the parameter values and their standard errors.

        Notes
        -----
            - If ``mean_values`` and ``std_values`` are both ``True``:
                A 2D array of shape (n_parameters, 2), where each row is ``[value, error]``.
            - If ``mean_values`` is ``True`` and ``std_values`` is ``False``:
                A 1D array of parameter values.
            - If ``std_values`` is ``True`` and ``mean_values`` is ``False``:
                A 1D array of standard errors.
            - If both flags are ``False``: An error message.

        Raises
        ------
        ValueError
            If both ``mean_values`` and ``std_values`` are ``False``.
        """
        pairs: NDArray = np.column_stack([self._params(), self._standard_errors()])

        if mean_values and std_values:
            return pairs
        elif mean_values:
            return pairs[:, 0]
        elif std_values:
            return pairs[:, 1]
        else:
            raise ValueError("Either 'mean_values' or 'std_values' must be True.")

    def ci_bounds(
        self,
        ci_levels: float | tuple[float] | list[float],
        n_bootstrap: int = 5_000,
        overall_ci: bool = True,
        individual_ci: bool = False,
        seed: int | None = None,
        rng_engine: Generator | None = None,
        x_range: ArrayLike | None = None,
    ) -> dict:
        """
        Calculate bootstrap confidence intervals for fitted model.

        Parameters
        ----------
        ci_levels
            Confidence interval level(s) as percentage (e.g., 95 or [68, 95, 99]).
            Can be float, tuple, or list.
        n_bootstrap
            Number of bootstrap samples to generate. Defaults to 5000.
        overall_ci
            If ``True``, compute confidence intervals for the overall composite fit.
            Defaults to ``True``.
        individual_ci
            If ``True``, compute confidence intervals for each individual component fit.
            Defaults to ``False``.
        seed
            Random seed for reproducibility. Either ``seed`` or ``rng_engine`` must be provided.
        rng_engine
            NumPy random generator instance. Either ``seed`` or ``rng_engine`` must be provided.
        x_range
            X-values at which to evaluate confidence intervals.
            If ``None``, uses 1000 points spanning the original x_values range.

        Returns
        -------
        dict
            Dictionary containing CI results with keys:
            - ``"x_range"``: X-values used for CI evaluation
            - ``"overall_ci_95"``: Dict with ``"lower"``, ``"median"``, ``"upper"`` arrays (if overall_ci=True)
            - ``"individual_ci_95"``: List of dicts, one per fit (if individual_ci=True)

        Raises
        ------
        ValueError
            If neither ``overall_ci`` nor ``individual_ci`` is ``True``, or if x_range dimensions don't match.
        """

        def _ci_to_percentiles(_ci_lvls: float | Iterable[float]) -> list[tuple[int, tuple[float, float, float]]]:
            """Convert CI levels to (ci_value, (lower, median, upper)) tuples."""
            _bounds: list[tuple[int, tuple[float, float, float]]] = []

            if isinstance(_ci_lvls, float | int):
                _ci_lvls: list[float] = [float(_ci_lvls)]
            elif isinstance(_ci_lvls, tuple):
                _ci_lvls: list[float] = list(_ci_lvls)

            for ci in _ci_lvls:
                ci_original = int(ci) if ci > 1 else int(ci * 100)
                ci = ci / 100 if ci > 1 else ci

                if not (0 < ci < 1):
                    raise ValueError(f"Invalid confidence interval: {ci}. Must be between 0 and 1 (or 0 and 100).")

                alpha = 1.0 - ci
                lower = alpha / 2.0
                upper = 1.0 - lower

                _bounds.append((ci_original, (lower, 0.5, upper)))

            return _bounds

        # Validate at least one CI type is requested
        if not overall_ci and not individual_ci:
            raise ValueError("At least one of 'overall_ci' or 'individual_ci' must be True.")

        # Setup x range for evaluation
        x_ = np.asarray(x_range) if x_range is not None else np.linspace(*self.x_values[[0, -1]], 1000)

        # Get fitted parameters and covariance
        mean_ = self.params
        cov_matrix = self.covariance

        # Generate bootstrap samples
        _rng = _sanitize_generator(rng_engine=rng_engine, seed=seed)
        mv_parameters = _rng.multivariate_normal(mean=mean_, cov=cov_matrix, size=n_bootstrap)

        # Convert CI levels to percentiles with original CI values
        bounds = _ci_to_percentiles(ci_levels)

        # Initialize results dictionary
        results = {"x_range": x_}

        # Compute overall CI
        if overall_ci:
            curves_ = np.array([self._n_fitter(x_, *j) for j in mv_parameters])

            for ci_val, (lower_p, median_p, upper_p) in bounds:
                quantiles = np.quantile(curves_, [lower_p, median_p, upper_p], axis=0)

                # Validate dimensions
                if quantiles.shape[-1] != len(x_):
                    raise ValueError(
                        f"Dimension mismatch: x_range has length {len(x_)} but "
                        f"quantiles have shape {quantiles.shape}"
                    )

                results[f"overall_ci_{ci_val}"] = {"lower": quantiles[0], "median": quantiles[1], "upper": quantiles[2]}

        # Compute individual CI
        if individual_ci:
            individual_ci_results = self._compute_individual_ci(mv_parameters, x_, bounds)
            for ci_val in individual_ci_results:
                results[f"individual_ci_{ci_val}"] = individual_ci_results[ci_val]

        return results

    def _compute_individual_ci(
        self, mv_parameters: NDArray, x_: NDArray, bounds: list[tuple[int, tuple[float, float, float]]]
    ) -> dict:
        """
        Compute individual component confidence intervals.

        This method can be overridden by subclasses (e.g., MixedDataFitter)
        that have different parameter structures.

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
        # Total parameters across all fits
        n_total_params = mv_parameters.shape[1]
        params_per_fit = n_total_params // self.n_fits
        params = mv_parameters.reshape((-1, self.n_fits, params_per_fit))
        curves_ = np.zeros(shape=(params.shape[0], self.n_fits, x_.shape[0]))

        # Generate curves for each fit and bootstrap sample
        for j_idx, j in enumerate(params):
            for i_idx, i in enumerate(j):
                curves_[j_idx, i_idx, :] = self._evaluate_individual_component(x_, i_idx, i)

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

    def plot_fit(
        self,
        show_individuals: bool = False,
        x_label: str = "X",
        y_label: str = "Y",
        plot_title: str = "Plot",
        data_label: str = "Data",
        fit_label: str = "Total Fit",
        axis: Axes | None = None,
    ) -> Axes:
        # Emit a clear deprecation warning for callers (stacklevel=2 points to the user's call site)
        warn(
            "BaseFitter.plot_fit is deprecated and will be removed in a future release. "
            "Please use the fitter's plotter API instead, e.g. `fitter.plotter.plot_fit(...)`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.plotter.plot_fit(
            show_individuals=show_individuals,
            x_label=x_label,
            y_label=y_label,
            plot_title=plot_title,
            data_label=data_label,
            fit_label=fit_label,
            axis=axis,
        )


def _sanitize_generator(rng_engine: Generator | None, seed: int | None) -> Generator:
    if seed is None and rng_engine is None:
        raise ValueError("Either 'seed' or 'rng_engine' must be provided.")

    if seed is not None and rng_engine is not None:
        raise ValueError("Only one of 'seed' or 'rng_engine' should be provided.")

    if rng_engine is not None:
        return rng_engine

    return np.random.default_rng(seed)
