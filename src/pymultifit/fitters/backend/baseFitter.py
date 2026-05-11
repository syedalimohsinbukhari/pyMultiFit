"""Created on Jul 18 00:16:01 2024"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import Bounds, curve_fit

from ... import _UNSET, epsilon
from ..._plot import FitPlotter
from ...typing import ArrayLike, NDArray, Params_
from ..utilities_f import parameter_logic, sanity_check


class BaseFitter:
    """The base class for multi-fitting functionality."""

    _plotter: FitPlotter | None

    def __init__(self, x_values: NDArray, y_values: NDArray, max_iterations: int = 1000):
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

    def _covariance(self) -> NDArray:
        """
        Store the covariance matrix of the fitted model.

        Returns
        -------
        NDArray
            An array containing the covariance matrix of the fitted model.
        """
        if self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self.covariance

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

    def _n_fitter(self, x: ArrayLike, *params: Params_) -> np.ndarray:
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
        parameters: np.ndarray = np.reshape(np.array(params), newshape=(self.n_fits, self.n_par))
        for par in parameters:
            y += self.fitter(x=x, params=par.tolist())
        return y

    def _params(self) -> np.ndarray:
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

    def _fit_boundaries(self) -> tuple[Sequence[float], Sequence[float]]:
        """Defines the internal distribution boundaries to be used by fitter."""
        ub = np.repeat(np.inf, repeats=self.n_par).tolist()
        lb = np.repeat(-np.inf, repeats=self.n_par).tolist()

        return lb, ub

    def fit_boundaries(self) -> tuple[Sequence[float], Sequence[float]]:
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

    def plot_fit(
        self,
        show_individuals: bool = False,
        x_label: str = "X",
        y_label: str = "Y",
        data_label: str = "Data",
        fit_label: str = "Total Fit",
        plot_title: str = "",
        axis: Axes | None = None,
    ) -> Axes:
        return self.plotter.plot_fit(
            show_individuals=show_individuals,
            x_label=x_label,
            y_label=y_label,
            plot_title=plot_title,
            data_label=data_label,
            fit_label=fit_label,
            axis=axis,
        )

    def plot_residuals(self, x_label: str = "", y_label: str = "", title: str = "", axis: Axes | None = None):
        return self.plotter.plot_residuals(x_label=x_label, y_label=y_label, plot_title=title, axis=axis)

    def plot_fit_and_residuals(
        self,
        show_individuals: bool = False,
        x_label: str | None = None,
        y_label: str | None = None,
        data_label: str | None = None,
        fit_label: str | None = None,
        title: str | None = None,
    ):
        """
        Plot the fitted model and residuals in a 2-panel figure.

        :param show_individuals: Whether to show individually fitted models or not.
        :param x_label: The label for the x-axis.
        :param y_label: The label for the y-axis for the fit plot.
        :param title: The overall title for the figure.
        :param data_label: The label for the data.
        :param fit_label: The label for the fitted model.

        :return: A tuple of (figure, (ax1, ax2)) where ax1 is the fit plot and ax2 is the residuals plot.
        :rtype: tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        """
        return self.plotter.plot_fit_and_residuals(
            show_individuals=show_individuals,
            x_label=x_label,
            y_label=y_label,
            plot_title=title,
            data_label=data_label,
            fit_label=fit_label,
        )

    def ci_bounds(
        self,
        ci_level: int | Sequence[int] = 95,
        n_bootstrap: int = 1000,
        plot_it: bool = False,
        overall_ci: bool = True,
        individual_ci: bool = False,
        random_state: int | None = None,
        axis=None,
    ):
        """
        Compute confidence interval (CI) bounds for fitted data using bootstrap resampling.

        :param ci_level: Confidence interval level(s) as percentages (e.g., 95 for 95% CI). Defaults to 95.
        :param n_bootstrap: Number of bootstrap samples to generate. Defaults to 1000.
        :param plot_it: If True, plots the fitted curve and shaded CI regions. Defaults to False.
        :param overall_ci: If True, compute CI bounds for the summed fitted curve. Defaults to True.
        :param individual_ci: If True, compute CI bounds for each fitter. Defaults to False.
        :param random_state: Random seed for reproducibility. Can be an integer or None. Defaults to None.
        :param axis: Axes to plot on. If None and plot_it=True, a new figure is created.

        :returns: Either a dictionary or a list of dictionary
        :rtype: dict | list[dict]

        :raises ValueError: If neither overall_ci nor individual_ci is True.
        :raises RuntimeError: If fit has not been performed yet.

        .. note::
            - ``overall_ci_XX``: dict with 'lower', 'upper', 'median' bounds for summed fits (if overall_ci=True).
            - ``individual_ci_XX``: list of dicts with 'lower', 'upper', 'median' for each fitter (if individual_ci=True).
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

        # Store original parameters for refitting
        original_params = np.reshape(self.params, (self.n_fits, self.n_par))
        n_samples = len(self.x_values)

        # Storage for bootstrap predictions
        bootstrap_overall = []
        bootstrap_individual = []

        # Perform bootstrap resampling
        successful_bootstraps = 0
        for i in range(n_bootstrap):
            # Resample indices with replacement using RNG
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
            x_boot = self.x_values[bootstrap_indices]
            y_boot = self.y_values[bootstrap_indices]

            try:
                # Create a temporary fitter instance for a bootstrap sample
                temp_fitter = self.__class__(x_values=x_boot, y_values=y_boot, max_iterations=bootstrap_max_iter)

                # Refit using original parameters as an initial guess
                temp_fitter.fit(p0=original_params.tolist())

                # Generate predictions on original x_values
                if overall_ci:
                    overall_pred = temp_fitter._n_fitter(self.x_values, *temp_fitter.params)
                    bootstrap_overall.append(overall_pred)

                if individual_ci:
                    boot_params = np.reshape(temp_fitter.params, (self.n_fits, self.n_par))
                    individual_preds = np.array(
                        [temp_fitter.fitter(x=self.x_values, params=list(par)) for par in boot_params]
                    )
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
