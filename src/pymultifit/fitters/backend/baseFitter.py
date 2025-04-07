"""Created on Jul 18 00:16:01 2024"""

import itertools
from itertools import chain
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from custom_inherit import doc_inherit
from matplotlib.axes import Axes
from mpyez.backend.uPlotting import LinePlot
from mpyez.ezPlotting import plot_xy
from scipy.optimize import curve_fit

from .errorHandling import BoundaryInconsistentWithGuess
from ..utilities_f import parameter_logic, _plot_fit
from ... import MPL_COLORS, doc_style, epsilon, listOfTuplesOrFloatsOrArray


class BaseFitter:
    """The base class for multi-fitting functionality."""

    def __init__(self, x_values: Union[list, np.ndarray], y_values: Union[list, np.ndarray],
                 max_iterations: int = 1000):
        self.x_values = x_values
        self.y_values = y_values
        self.max_iterations = max_iterations

        self.n_par = None
        self.pn_par = self.n_par
        self.sn_par = {}

        self.p0 = None

        self.n_fits = None
        self.params = None
        self.covariance = None

    def _adjust_parameters(self, p0: List[Tuple[float, ...]]):
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

            primary_params = params[:self.pn_par]
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

    def _fit_preprocessing(self, frozen: dict):
        """
        Process frozen parameters and adjust bounds.

        Parameters
        ----------
        frozen: Dict[int, List[int]]
            A dictionary where keys are indices of model fits (1-based), and values are lists of parameter indices
            (1-based) to freeze.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Adjusted lower and upper bounds, and flattened initial guesses.
        """
        try:
            lb, ub = self.fit_boundaries()
        except NotImplementedError:
            lb = np.full(shape=self.n_fits, fill_value=-np.inf)
            ub = np.full(shape=self.n_fits, fill_value=np.inf)

        # Ensure bounds are NumPy arrays and resize properly
        lb = np.resize(a=lb, new_shape=(self.n_fits, self.n_par))
        ub = np.resize(a=ub, new_shape=(self.n_fits, self.n_par))

        p0_array = np.array(self.p0)

        if frozen:
            # Process frozen parameters
            for key, values in frozen.items():
                values = values if isinstance(values, list) else [values]
                for val in values:
                    param_value = p0_array[key - 1, val - 1]
                    lb[key - 1, val - 1] = param_value - epsilon
                    ub[key - 1, val - 1] = param_value + epsilon

        return lb.flatten(), ub.flatten(), p0_array.flatten()

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

    def _n_fitter(self, x: np.ndarray, *params: Iterable) -> np.ndarray:
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
        y = np.zeros_like(a=x, dtype=float)
        params = np.reshape(a=np.array(params), newshape=(self.n_fits, self.n_par))
        for par in params:
            y += self.fitter(x=x, params=par)
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
        params = np.reshape(a=self.params, newshape=(self.n_fits, self.n_par))
        colors = MPL_COLORS[1:]
        for i, par in enumerate(params):
            color = colors[i % len(colors)]
            plot_xy(x_data=x, y_data=self.fitter(x=x, params=par),
                    data_label=f'{self.__class__.__name__.replace("Fitter", "")} {i + 1}('
                               f'{", ".join(self._format_param(i) for i in par)})',
                    plot_dictionary=LinePlot(line_style='--', color=color), axis=plotter, x_label='', y_label='',
                    plot_title='')

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

    def dry_run(self, axis=None):
        """
        Plot the x and y data for a quick visual inspection of the data.

        Parameters
        ----------
        axis:
            The axis to plot the data on.
        """
        plot_xy(x_data=self.x_values, y_data=self.y_values, axis=axis)

    def fit(self, p0: listOfTuplesOrFloatsOrArray, frozen: dict = None):
        """
        Fit the data.

        Parameters
        ----------
        p0: listOfTuplesOrArray
            A list of initial guesses for the parameters of the models.
            Each element is a tuple or array representing the parameters for a single fit.
            Example: [(1, 1, 0), (3, 3, 2)].

        frozen: Dict[int, List[int]], optional
            A dictionary specifying which parameters should be frozen during fitting.
            - Keys are 1-based indices corresponding to the fits.
            - Values are lists of 1-based parameter indices to freeze for that fit.
            - Example: `{1: [2, 3], 2: [1]}` freezes parameter 2 and 3 for the first and parameter 1 for the second fit.

        Returns
        -------
        None
            Performs the fitting process with the given constraints.
        """
        c_name = self.__class__.__name__
        self.p0 = p0

        # int -> list converter check
        if isinstance(p0[0], int | float):
            self.p0 = [p0]

        self.n_fits = len(self.p0) if c_name != 'MixedDataFitter' else 1
        len_guess = len(list(chain(*self.p0)))

        if not c_name == 'MixedDataFitter':
            total_pars = self.n_par * self.n_fits
        else:
            total_pars = len_guess

        lb, ub, p0_flat = self._fit_preprocessing(frozen=frozen)

        if len(lb) != total_pars and c_name != 'MixedDataFitter':
            self.p0 = self._adjust_parameters(self.p0)

        length_ = len(self.p0[0]) if c_name != 'MixedDataFitter' else self._expected_param_count()

        if ub.shape[0] != length_ * self.n_fits:
            raise BoundaryInconsistentWithGuess(f"{ub.shape[0]} != {length_ * self.n_fits}.")

        # remove the Bound parameter for the library to work with previous version of `scipy`
        self.params, self.covariance, *_ = curve_fit(f=self._n_fitter,
                                                     xdata=self.x_values, ydata=self.y_values,
                                                     p0=p0_flat, maxfev=self.max_iterations,
                                                     bounds=(lb, ub))

    @staticmethod
    def fit_boundaries():
        """Defines the distribution boundaries to be used by fitter."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def fitter(x: np.ndarray, params: Tuple[float, Any]):
        """
        Fitter function for multi-fitting.

        Parameters
        ----------
        x: np.ndarray
            The x-array on which the fitting is to be performed.
        params: List[float]
            A list of parameters to fit.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_fitted_curve(self) -> np.ndarray:
        """
        Get the fitted values of the model.

        Returns
        -------
        np.ndarray
            An array of fitted values.
        """
        if self.params is None:
            raise RuntimeError('Fit not performed yet. Call fit() first.')
        return self._n_fitter(self.x_values,
                              *self.params if self.__class__.__name__ == 'MixedDataFitter' else self.params)

    def get_model_parameters(self, select: Tuple[int, Any] = None, errors: bool = False):
        r"""
        Extract specific parameter values or their uncertainties from the fitting process.

        This method allows for retrieving the fitted parameters or their corresponding standard errors for specific
         sub-models, or for all sub-models if no selection is provided.

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

    @doc_inherit(parent=_plot_fit, style=doc_style)
    def plot_fit(self, show_individuals: bool = False, ci: Union[int, list[int]] = None,
                 x_label: Optional[str] = None, y_label: Optional[str] = None, data_label: Union[list[str], str] = None,
                 title: Optional[str] = None, axis: Optional[Axes] = None,
                 data_color: str = MPL_COLORS[0], grid: bool = True):
        """
        Plot the fitted models.

        Returns
        -------
        plotter
            The plotter handle for the drawn plot.
        """
        axes = _plot_fit(x_values=self.x_values, y_values=self.y_values, parameters=self.params, n_fits=self.n_fits,
                         class_name=self.__class__.__name__, _n_fitter=self._n_fitter,
                         _n_plotter=self._plot_individual_fitter, show_individuals=show_individuals, x_label=x_label,
                         y_label=y_label, title=title, data_label=data_label, axis=axis,
                         data_color=data_color, grid=grid)
        if ci:
            self.ci_bounds(ci_level=ci, individual_ci=show_individuals, overall_ci=True, plot_it=True, axis=axes)

        return axes

    def residuals(self):
        return self.y_values - self.get_fitted_curve()

    def mse_rmse(self):
        residuals = self.y_values - self.get_fitted_curve()

        mse = np.mean(residuals**2)  # Mean squared error
        r_mse = np.sqrt(mse)  # Root mean squared error

        return mse, r_mse

    def ci_bounds(self, ci_level: Union[int, list[int]] = 3, plot_it: bool = False, overall_ci: bool = False,
                  individual_ci: bool = False, axis=None):
        """
        Compute confidence interval (CI) bounds for fitted data.

        Parameters:
        - ci_level (int or list of int, optional): Confidence interval multiplier(s) (default: 3σ).
        - plot_it (bool, optional): If True, plots the fitted curve and shaded CI regions.
        - overall_ci (bool, optional): If True, compute CI bounds for the summed fitted curve.
        - individual_ci (bool, optional): If True, compute CI bounds for each individual fitter.

        Returns:
        - Dict with either or both:
          - 'summed_fit': Confidence intervals for summed fits (if overall_ci=True).
          - 'individual_fits': Confidence intervals for individual fitters (if individual_ci=True).
        """
        if not overall_ci and not individual_ci:
            raise ValueError("At least one of `overall_ci` or `individual_ci` must be True.")

        ci_levels = [ci_level] if isinstance(ci_level, int) else ci_level
        num_samples = 5
        num_x = len(self.x_values)
        n_fits = self.n_fits if self.__class__.__name__ != 'MixedDataFitter' else len(self.model_list)
        is_mixed = False if self.__class__.__name__ != 'MixedDataFitter' else True

        mc_iter_values, mc_individual_values = self.run_mc_sampling(num_samples, num_x,
                                                                    n_fits, overall_ci,
                                                                    individual_ci,
                                                                    is_mixed)

        # Compute results
        results = {}

        fitted_curve = self.get_fitted_curve()
        mean_fit = np.mean(mc_iter_values, axis=0)
        std_fit = np.std(mc_iter_values, axis=0)

        mean_ind_fit = np.mean(mc_individual_values, axis=0)
        std_ind_fit = np.std(mc_individual_values, axis=0)

        # Compute overall CI bounds
        if overall_ci:
            results["summed_fit"] = [[fitted_curve, mean_fit - (level * std_fit), mean_fit + (level * std_fit)]
                                     for level in ci_levels]

        # Compute individual fitter CI bounds
        if individual_ci:
            individual_ci_results = []
            for fitter_idx in range(n_fits):
                mean_individual = np.mean(mc_individual_values[:, fitter_idx, :], axis=0)
                std_individual = np.std(mc_individual_values[:, fitter_idx, :], axis=0)
                print(mc_individual_values[:, fitter_idx, :])
                individual_ci_results.append([
                    [mean_individual - (level * std_individual),
                     mean_individual + (level * std_individual)]
                    for level in ci_levels
                ])
            results["individual_fits"] = individual_ci_results

        if plot_it:
            if overall_ci:
                for level in ci_levels:
                    lower, upper = mean_fit - level * std_fit, mean_fit + level * std_fit
                    axis.fill_between(self.x_values, lower, upper, color='k', alpha=0.25)

            if individual_ci:
                for level, fitter_idx in zip(ci_levels, range(n_fits)):
                    color_cycle = itertools.cycle(MPL_COLORS[1:])
                    lower, upper = mean_ind_fit - (level * std_ind_fit), mean_ind_fit + (level * std_ind_fit)
                    [axis.fill_between(self.x_values, i, j, color=next(color_cycle), alpha=0.25)
                     for i, j in zip(lower, upper)]

        return results

    def run_mc_sampling(self, num_samples, num_x, n_fits, overall_ci=True,
                        individual_ci=True, is_mixed_model=False):
        mc_iter_values = np.zeros((num_samples, num_x))
        mc_individual_values = np.zeros((num_samples, n_fits, num_x))

        for i in range(num_samples):
            sampled_params = np.array([np.random.normal(*p)
                                       for p in self.get_value_error_pair(std_values=True)])

            if is_mixed_model:
                individual_fits = 0
                par_count = 0

                for model_idx, model in enumerate(self.model_list):
                    model_class, n_par = self._instantiate_fitter(model=model, return_values=['class', 'n_par'])
                    part_fit = model_class.fitter(self.x_values, sampled_params[par_count:par_count + n_par])
                    individual_fits += part_fit
                    par_count += n_par

                    if individual_ci:
                        mc_individual_values[i, model_idx, :] = part_fit

                if overall_ci:
                    mc_iter_values[i, :] = individual_fits
            else:
                sampled_params = sampled_params.reshape(self.n_fits, self.n_par)
                individual_fits = np.array([self.fitter(self.x_values, params)
                                            for params in sampled_params])

                if overall_ci:
                    mc_iter_values[i, :] = np.sum(individual_fits, axis=0)

                if individual_ci:
                    mc_individual_values[i, :, :] = individual_fits

        return mc_iter_values, mc_individual_values
