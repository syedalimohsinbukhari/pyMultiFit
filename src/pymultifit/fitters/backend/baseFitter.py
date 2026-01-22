"""Created on Jul 18 00:16:01 2024"""

from itertools import chain
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpyez.backend.uPlotting import LinePlot  # type: ignore
from mpyez.ezPlotting import plot_xy  # type: ignore
from numpy.typing import NDArray
from scipy.optimize import Bounds, curve_fit

from ..utilities_f import _plot_fit, parameter_logic, sanity_check
from ... import epsilon, OneDArray, Params_


class BaseFitter:
    """The base class for multi-fitting functionality."""

    def __init__(self, x_values: OneDArray, y_values: OneDArray, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        self.x_values: np.ndarray = x_values
        self.y_values: np.ndarray = y_values
        self.max_iterations = max_iterations

        self.n_par: int = 0
        self.pn_par: int = self.n_par
        self.sn_par: dict = {}

        self.n_fits: int = 0
        self.params = None
        self.covariance = None

    def _adjust_parameters(self, p0: Params_):
        """
        Adjust input parameters to include defaults for secondary parameters if missing.

        Parameters
        ----------
        p0: List[List[float]]
            A list of initial guesses for the parameters.

        Returns
        -------
        adjusted_p0: List[List[float]]
            Adjusted parameter list with default values for missing secondary parameters.
        """
        adjusted_p0 = []
        for params in p0:
            # Too few parameters
            if len(params) < self.pn_par:
                raise ValueError(f"Each parameter set must have at least {self.pn_par} primary parameters.")

            primary_params = params[: self.pn_par]
            provided_secondary_params = params[self.pn_par:]

            secondary_params = dict(self.sn_par)
            for key, value in zip(self.sn_par.keys(), provided_secondary_params):
                secondary_params[key] = value

            adjusted_params = list(primary_params) + list(secondary_params.values())

            if len(adjusted_params) != self.n_par:
                raise ValueError(f"Adjusted parameter set must have {self.n_par} total parameters.")

            adjusted_p0.append(adjusted_params)

        return adjusted_p0

    def _covariance(self):
        """
        Store the covariance matrix of the fitted model.

        Returns
        -------
        np.ndarray
            An array containing the covariance matrix of the fitted model.
        """
        if self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self.covariance

    def _fit_preprocessing(self, p0, frozen):
        """
        Process frozen parameters and adjust bounds.

        Parameters
        ----------
        p0: Sequences_
            A list of initial guesses for the parameters of the models.
            For example, [(1, 1, 0), (3, 3, 2)].
        frozen: List[bool]
            A list of booleans indicating which parameters are frozen.
            For example, [False, False, True] for 3 parameters.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Adjusted lower and upper bounds, and flattened initial guesses.
        """
        # Get initial boundaries
        try:
            lb, ub = self.fit_boundaries()
        except NotImplementedError:
            # if they're not implemented, self-imposes -inf + inf boundaries
            lb = np.repeat(-np.inf, self.n_fits)
            ub = np.repeat(np.inf, self.n_fits)

        # Resize bounds to match total parameters
        lb = np.resize(lb, self.n_par * self.n_fits)
        ub = np.resize(ub, self.n_par * self.n_fits)

        # Validate frozen length
        if frozen is None:
            frozen = [False] * self.n_par

        if len(frozen) != self.n_par:
            raise ValueError("The length of 'frozen' must match the number of parameters per model.")

        # Repeat frozen mask for all models
        frozen = frozen * self.n_fits

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

    def _n_fitter(self, x: NDArray, *params: Params_) -> np.ndarray:
        r"""
        Perform N-fitting by summing over multiple parameter sets.

        Parameters
        ----------
        x : np.ndarray
            Input array of values for which the composite function is evaluated.
        params : tuple
            A tuple with all parameters to be fitted in an array of size (``self.n_fits, self.n_par``) where:
            - ``self.n_fits`` is the number of individual fits.
            - ``self.n_par`` is the number of parameters per fit.

        Returns
        -------
        np.ndarray
            An array containing the composite fitted values for the input ``x``.
        """
        y = np.zeros_like(x, dtype=float)
        parameters: np.ndarray = np.reshape(np.array(params), newshape=(self.n_fits, self.n_par))
        for par in parameters:
            y += self.fitter(x=x, params=par.tolist())
        return y

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
        r"""
        Plot individual fits from the composite fitter.

        Parameters
        ----------
        plotter : matplotlib.axes.Axes
            The axis object where the plots will be rendered.

        Notes
        -----
        - ``self.params`` must contain the fitted parameters reshaped into (``self.n_fits``, ``self.n_par``).
        - Each plot will be labeled with the class name and the index of the fit, along with the formatted parameters.
        """
        x = self.x_values
        params = np.reshape(self.params, (self.n_fits, self.n_par))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]
        for i, par in enumerate(params):
            color = colors[i % len(colors)]
            plot_xy(
                x_data=x,
                y_data=self.fitter(x=x, params=list(par)),
                data_label=f"{self.__class__.__name__.replace('Fitter', '')} {i + 1}("
                           f"{', '.join(self._format_param(i) for i in par)})",
                plot_dictionary=LinePlot(line_style="--", color=color),
                axis=plotter,
                x_label="",
                y_label="",
                plot_title="",
            )

    def _standard_errors(self) -> NDArray:
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

    def dry_run(self, axis=None):
        """
        Plot the x and y data for a quick visual inspection of the data.

        Parameters
        ----------
        axis
            The axis to plot the data on.
        """
        plot_xy(x_data=self.x_values, y_data=self.y_values, axis=axis)

    def fit(self, p0: Params_, frozen: Optional[List[bool]] = None):
        """
        Fit the data.

        Parameters
        ----------
        p0: Sequences_
            A list of initial guesses for the parameters of the models.
            For example, [(1, 1, 0), (3, 3, 2)].
        frozen: List[bool]
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

    def _fit_boundaries(self) -> Tuple[Sequence[float], Sequence[float]]:
        """Defines the internal distribution boundaries to be used by fitter."""
        ub = np.repeat(np.inf, self.n_par).tolist()
        lb = np.repeat(-np.inf, self.n_par).tolist()
        return lb, ub

    def fit_boundaries(self) -> Tuple[Sequence[float], Sequence[float]]:
        """Defines the distribution boundaries to be used by fitter."""
        return self._fit_boundaries()

    @staticmethod
    def fitter(x, params: Params_) -> OneDArray:
        """
        Fitter function for multi-fitting.

        Parameters
        ----------
        x: np.ndarray
            The x-array on which the fitting is to be performed.
        params: Params_
            An array of parameters to fit.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_fitted_curve(self) -> OneDArray:
        """
        Get the fitted values of the model.

        Returns
        -------
        np.ndarray
            An array of fitted values.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self._n_fitter(self.x_values, self.params)

    def get_residuals(self) -> np.ndarray:
        """
        Get the residuals (difference between data and fitted model).

        Returns
        -------
        np.ndarray
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

    def get_model_parameters(self, select: Optional[Tuple[int, Any]] = None, errors: bool = False):
        r"""
        Extract specific parameter values or their uncertainties from the fitting process.

        This method allows for retrieving the fitted parameters or their corresponding standard errors for specific
         submodels, or for all sub-models if no selection is provided.

        Parameters
        ----------
        select : list of int or None, optional
            A list of indices specifying which sub-models to extract parameters for. Indexing starts at 1.
            If ``None``, parameters for all sub-models are returned. Defaults to None.
        errors : bool, optional
            If ``True``, both the parameter values and their standard errors are returned.
            Defaults to ``False``.

        Returns
        -------
        np.ndarray or tuple of np.ndarray

           * If ``errors`` is ``False``:
                - A 2D array of shape `(n_parameters, selected_models)` with parameter values for the selected models.

           * If ``errors`` is ``True``: A tuple of two 2D arrays:
                - The first array contains the parameter values of shape `(n_parameters, selected_models)`.
                - The second array contains the standard errors of the parameters, with the same shape.

        Notes
        -----
        - The ``select`` parameter allows filtering by specific sub-model indices. If ``None``, use all sub-models.
        - When ``errors`` is ``True``, both the parameter means and their uncertainties are returned as separate arrays.

        Raises
        ------
        ValueError
            If the input ``select`` is not a valid list of indices or is incompatible with the model structure.
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

    def get_value_error_pair(self, mean_values: bool = True, std_values: bool = False) -> np.ndarray:
        r"""
        Retrieve the value/error pairs for the fitted parameters.

        This method provides the fitted parameter values and their corresponding standard errors as a combined array or
        individually based on the input flags.

        Parameters
        ----------
        mean_values : bool, optional.
            If ``True``, return only the values of the fitted parameters.
            Defaults to ``True``.
        std_values : bool, optional.
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
        pairs: np.ndarray = np.column_stack([self._params(), self._standard_errors()])

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
        show_individuals: bool, optional.
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
        return _plot_fit(
            x_values=self.x_values,
            y_values=self.y_values,
            parameters=self.params,
            n_fits=self.n_fits,
            class_name=self.__class__.__name__,
            _n_fitter=self._n_fitter,
            _n_plotter=self._plot_individual_fitter,
            show_individuals=show_individuals,
            x_label=x_label,
            y_label=y_label,
            title=title,
            data_label=data_label,
            fit_label=fit_label,
            axis=axis,
        )

    def plot_residuals(
            self,
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
            title: Optional[str] = None,
            axis: Optional[Axes] = None,
    ):
        """
        Plot the residuals (data - fitted model).

        Parameters
        ----------
        x_label: str, optional
            The label for the x-axis.
        y_label: str, optional
            The label for the y-axis.
        title: str, optional
            The title for the plot.
        axis: Axes, optional
            Axes to plot instead of the entire figure. Defaults to None.

        Returns
        -------
        plotter
            The plotter handle for the drawn plot.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        residuals = self.get_residuals()

        plotter = plot_xy(
            x_data=self.x_values,
            y_data=residuals,
            data_label="Residuals",
            axis=axis,
            plot_dictionary=LinePlot(alpha=0.75),
        )

        # Add a horizontal line at y=0
        plotter2: Axes = plotter[0] if isinstance(plotter, list) else plotter
        plotter2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        plotter2.set_xlabel(x_label if x_label else "X")
        plotter2.set_ylabel(y_label if y_label else "Residuals")
        plotter2.set_title(title if title else f"{self.n_fits} {self.__class__.__name__} residuals")
        plt.tight_layout()

        return plotter2

    def plot_fit_and_residuals(
            self,
            show_individuals: bool = False,
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
            data_label: Optional[str] = None,
            fit_label: Optional[str] = None,
            title: Optional[str] = None,
    ):
        """
        Plot the fitted model and residuals in a 2-panel figure.

        Parameters
        ----------
        show_individuals: bool, optional
            Whether to show individually fitted models or not.
        x_label: str, optional
            The label for the x-axis.
        y_label: str, optional
            The label for the y-axis for the fit plot.
        title: str, optional
            The overall title for the figure.
        data_label: str, optional
            The label for the data.
        fit_label: str, optional
            The label for the fitted model.

        Returns
        -------
        tuple
            A tuple of (figure, (ax1, ax2)) where ax1 is the fit plot and ax2 is the residuals plot.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Plot the fit
        self.plot_fit(
            show_individuals=show_individuals,
            x_label="",
            y_label=y_label,
            data_label=data_label,
            fit_label=fit_label,
            title=title,
            axis=ax1,
        )

        # Plot the residuals
        self.plot_residuals(
            x_label=x_label,
            y_label="Residuals",
            title="",
            axis=ax2,
        )

        plt.tight_layout()
        return fig, (ax1, ax2)

    def ci_bounds(self, ci_level: Union[int, list[int]] = 95, n_bootstrap: int = 1000,
                  plot_it: bool = False, overall_ci: bool = True, individual_ci: bool = False,
                  axis=None, random_state: Optional[int] = None, fast_bootstrap: bool = True):
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
            If True, compute CI bounds for each individual fitter. Defaults to False.
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
            - 'individual_ci_XX': list of dicts with 'lower', 'upper', 'median' for each fitter (if individual_ci=True).

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

        # Initialize random number generator for reproducibility
        rng = np.random.default_rng(random_state)

        # Optimize max_iterations for bootstrap if fast_bootstrap is enabled
        if fast_bootstrap:
            # Use 1/4 of original iterations (bootstrap converges faster with good initial guess)
            bootstrap_max_iter = max(100, self.max_iterations // 4)
        else:
            bootstrap_max_iter = self.max_iterations

        # Handle single or multiple CI levels
        ci_levels = [ci_level] if isinstance(ci_level, int) else ci_level

        # Store original parameters for refitting
        original_params = np.reshape(self.params, (self.n_fits, self.n_par))
        n_samples = len(self.x_values)

        # Storage for bootstrap predictions
        if overall_ci:
            bootstrap_overall = []
        if individual_ci:
            bootstrap_individual = []

        # Perform bootstrap resampling
        successful_bootstraps = 0
        for i in range(n_bootstrap):
            # Resample indices with replacement using RNG
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
            x_boot = self.x_values[bootstrap_indices]
            y_boot = self.y_values[bootstrap_indices]

            try:
                # Create temporary fitter instance for bootstrap sample
                temp_fitter = self.__class__(x_values=x_boot, y_values=y_boot,
                                             max_iterations=bootstrap_max_iter)

                # Refit using original parameters as initial guess
                temp_fitter.fit(p0=original_params.tolist())

                # Generate predictions on original x_values
                if overall_ci:
                    overall_pred = temp_fitter._n_fitter(self.x_values, *temp_fitter.params)
                    bootstrap_overall.append(overall_pred)

                if individual_ci:
                    boot_params = np.reshape(temp_fitter.params, (self.n_fits, self.n_par))
                    individual_preds = np.array([
                        temp_fitter.fitter(x=self.x_values, params=list(par))
                        for par in boot_params
                    ])
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
                results[f'overall_ci_{ci}'] = {
                    'lower': np.percentile(bootstrap_overall, lower_percentile, axis=0),
                    'upper': np.percentile(bootstrap_overall, upper_percentile, axis=0),
                    'median': np.percentile(bootstrap_overall, 50, axis=0),
                }

            if individual_ci:
                results[f'individual_ci_{ci}'] = []
                for j in range(self.n_fits):
                    results[f'individual_ci_{ci}'].append({
                        'lower': np.percentile(bootstrap_individual[:, j], lower_percentile, axis=0),
                        'upper': np.percentile(bootstrap_individual[:, j], upper_percentile, axis=0),
                        'median': np.percentile(bootstrap_individual[:, j], 50, axis=0),
                    })

        # Plot if requested
        if plot_it:
            self._plot_ci_bounds(results, ci_levels, overall_ci, individual_ci, axis)

        return results

    def _plot_ci_bounds(self, results: dict, ci_levels: list, overall_ci: bool,
                        individual_ci: bool, axis=None):
        """
        Plot confidence interval bounds.

        Parameters
        ----------
        results : dict
            Dictionary containing CI bounds from ci_bounds method.
        ci_levels : list
            List of confidence levels to plot.
        overall_ci : bool
            Whether overall CI was computed.
        individual_ci : bool
            Whether individual CIs were computed.
        axis : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        """
        if axis is None:
            fig, axis = plt.subplots(figsize=(10, 6))

        # Plot overall CI
        if overall_ci:
            for idx, ci in enumerate(ci_levels):
                ci_data = results[f'overall_ci_{ci}']
                axis.fill_between(
                    self.x_values,
                    ci_data['lower'],
                    ci_data['upper'],
                    alpha=1 * (ci / 100),
                    color='gray',
                    label=f'{ci}% CI (overall)'
                )

        # Plot individual CIs
        if individual_ci:
            for idx, ci in enumerate(ci_levels):
                ci_data = results[f'individual_ci_{ci}']
                for idx2, fit in enumerate(ci_data):
                    axis.fill_between(
                        self.x_values,
                        fit['lower'],
                        fit['upper'],
                        alpha=1 * (ci / 100),
                        color='gray', label=f'{ci}% CI (fit {idx2 + 1})' if idx2 == 0 else '',
                    )

        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_title('Bootstrap Confidence Intervals')
        axis.legend()
        axis.grid(True, alpha=0.3)
        plt.tight_layout()

        return axis
