"""Created on Jul 18 00:16:01 2024"""

from __future__ import annotations

import warnings
from itertools import chain
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from numpy.random import Generator
from scipy.optimize import Bounds, curve_fit

from ._ci_backend import compute_ci_bounds, compute_individual_ci_base
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
            Length must equal ``n_par`` (all parameters) or ``pn_par`` (primary parameters only — secondary
            parameters such as ``loc`` are automatically treated as unfrozen).
            When ``pn_par`` length is given, a :class:`UserWarning` is emitted to flag the auto-padding.
            For example, for a distribution with ``n_par=4`` and ``pn_par=3``, passing ``[False, False, True]``
            is equivalent to ``[False, False, True, False]``.

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

        # Validate and normalise frozen mask
        if frozen is None:
            frozen: list[bool] = [False] * self.n_par
        elif len(frozen) == self.pn_par:
            # Auto-pad: secondary params (loc/scale) default to unfrozen
            warnings.warn(
                f"'frozen' has length {self.pn_par} (pn_par), which is shorter than n_par={self.n_par}. "
                f"The {self.n_par - self.pn_par} secondary parameter(s) (e.g. loc/scale) are being auto-padded as "
                f"False (unfrozen). Pass a mask of length {self.n_par} to make this explicit.",
                UserWarning,
                stacklevel=3,
            )
            frozen = list(frozen) + [False] * (self.n_par - self.pn_par)
        elif len(frozen) != self.n_par:
            raise ValueError(
                f"'frozen' length ({len(frozen)}) must equal n_par ({self.n_par}) " f"or pn_par ({self.pn_par})."
            )

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

    def dry_run(self, axis: Axes | None = None, is_scatter: bool = False):
        """
        Plot the x and y data for a quick visual inspection of the data.

        Parameters
        ----------
        axis :
            The axis to plot the data on.
        is_scatter :
            If ``True``, the data will be plotted as a scatter plot.
            If ``False``, the data will be plotted as a line plot.
            Defaults to ``False``.
        """
        self.plotter.dry_run(axis=axis, is_scatter=is_scatter)

    def fit(self, p0: Params_, frozen: list[bool] | None = None):
        """
        Fit the data.

        Parameters
        ----------
        p0 :
            A list of initial guesses for the parameters of the models.
        frozen :
            A list of booleans indicating whether each parameter is frozen.
        """
        if isinstance(p0[0], float):
            # flat list — use n_par to split
            if len(p0) % self.n_par != 0:
                raise ValueError(f"p0 length {len(p0)} not divisible by n_par={self.n_par}")
            p0 = np.asarray(p0).reshape(-1, self.n_par)
        else:
            # already structured as list-of-guesses — pass through
            p0 = [np.asarray(g) for g in p0]

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
        x :
            X-values at which to evaluate the model.
        fit_index :
            Index of the component model (0-based).
        params :
            Parameters for this specific component.

        Returns
        -------
        NDArray :
            Evaluated y-values for this component.
        """
        return self.fitter(x=x, params=params)

    def get_fitted_curve(self) -> NDArray:
        """
        Get the fitted values of the model.

        Returns
        -------
        NDArray :
            An array of fitted values.

        Raises
        ------
        RuntimeError :
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
        NDArray :
            An array of residual values (y_data - y_fitted).

        Raises
        ------
        RuntimeError :
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
        select :
            A list of indices specifying which submodels to extract parameters for.
            Indexing starts at 1.
            If ``None``, parameters for all submodels are returned.
            Defaults to None.
        errors :
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
            parameter_mean: NDArray
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
        plot: bool = False,
        axis: Axes | None = None,
    ) -> dict:
        """Compute bootstrap confidence intervals for the fitted model.

        Parameters
        ----------
        ci_levels :
            CI level(s) as a percentage (e.g., 95 or [68, 95, 99]).
        n_bootstrap :
            Bootstrap samples. Defaults to 5 000.
        overall_ci :
            Compute CI for the overall composite fit. Defaults to ``True``.
        individual_ci :
            Compute CI for each component. Defaults to ``False``.
        seed :
            Random seed (mutually exclusive with *rng_engine*).
        rng_engine :
            NumPy Generator instance (mutually exclusive with *seed*).
        x_range :
            X-values at which to evaluate the CI.
            Defaults to 1 000 points spanning the data range.
        plot :
            When ``True``, plot the CI bands immediately after computing them.
            Defaults to ``False``.
        axis :
            Target axes for the optional plot (ignored when ``plot=False``).

        Returns
        -------
        dict
            ``{"x_range": ..., "overall_ci_<level>": {...}, "individual_ci_<level>": [...]}``
        """
        results = compute_ci_bounds(
            fitter_object=self,
            ci_levels=ci_levels,
            n_bootstrap=n_bootstrap,
            overall_ci=overall_ci,
            individual_ci=individual_ci,
            seed=seed,
            rng_engine=rng_engine,
            x_range=x_range,
        )

        if plot:
            self.plotter.plot_ci_bounds(
                ci_levels=ci_levels, results=results, overall_ci=overall_ci, individual_ci=individual_ci, axis=axis
            )

        return results

    def _compute_individual_ci(
        self, x_: ArrayLike, mv_parameters: ArrayLike, bounds: list[tuple[int, tuple[float, float, float]]]
    ) -> dict:
        return compute_individual_ci_base(fitter_object=self, mv_parameters=mv_parameters, x_=x_, bounds=bounds)

    def plot_fit(
        self,
        show_individuals: bool = False,
        x_label: str = "X",
        y_label: str = "Y",
        plot_title: str = "Plot",
        data_label: str = "Data",
        fit_label: str = "Total Fit",
        is_scatter: bool = False,
        axis: Axes | None = None,
    ) -> Axes:
        return self.plotter.plot_fit(
            show_individuals=show_individuals,
            x_label=x_label,
            y_label=y_label,
            plot_title=plot_title,
            data_label=data_label,
            fit_label=fit_label,
            is_scatter=is_scatter,
            axis=axis,
        )
