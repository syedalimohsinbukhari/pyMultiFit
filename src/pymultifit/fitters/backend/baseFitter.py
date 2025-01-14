"""Created on Jul 18 00:16:01 2024"""

from itertools import chain
from typing import Any, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpyez.backend.uPlotting import LinePlot
from mpyez.ezPlotting import plot_xy
from scipy.optimize import Bounds, curve_fit

from ..utilities_f import parameter_logic
from ... import listOfTuplesOrArray, epsilon


class BaseFitter:
    """The base class for multi-fitting functionality."""

    def __init__(self, x_values: list | np.ndarray, y_values: list | np.ndarray, max_iterations: int = 1000):
        self.x_values = x_values
        self.y_values = y_values
        self.max_iterations = max_iterations

        self.n_par = None
        self.pn_par = self.n_par
        self.sn_par = {}

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

    def _fit_preprocessing(self, p0: listOfTuplesOrArray, frozen: List[bool]):
        """
        Process frozen parameters and adjust bounds.

        Parameters
        ----------
        p0: listOfTuplesOrArray
            A list of initial guesses for the parameters of the models.
            For example: [(1, 1, 0), (3, 3, 2)].
        frozen: List[bool]
            A list of booleans indicating which parameters are frozen.
            For example: [False, False, True] for 3 parameters.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Adjusted lower and upper bounds, and flattened initial guesses.
        """
        # Get initial boundaries
        try:
            lb, ub = self.fit_boundaries()
        except NotImplementedError:
            # if they're not implemented, self impose -inf + inf boundaries
            lb = [-np.inf] * self.n_fits
            ub = [np.inf] * self.n_fits

        # Resize bounds to match total parameters
        lb = np.resize(a=lb, new_shape=self.n_par * self.n_fits)
        ub = np.resize(a=ub, new_shape=self.n_par * self.n_fits)

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
        - `self.params` must contain the fitted parameters reshaped into (``self.n_fits``, ``self.n_par``).
        - Each plot will be labeled with the class name and the index of the fit, along with the formatted parameters.
        """
        x = self.x_values
        params = np.reshape(a=self.params, newshape=(self.n_fits, self.n_par))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
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

    def fit(self, p0: listOfTuplesOrArray, frozen: List[bool] = None):
        """
        Fit the data.

        Parameters
        ----------
        p0: listOfTuplesOrArray
            A list of initial guesses for the parameters of the models.
            For example: [(1, 1, 0), (3, 3, 2)].
        frozen: List[bool]
            A list of booleans indicating whether each parameter is frozen.
            For example: [False, False, True] for 3 parameters.
        """
        self.n_fits = len(p0)
        len_guess = len(list(chain(*p0)))
        total_pars = self.n_par * self.n_fits

        if len_guess != total_pars:
            p0 = self._adjust_parameters(p0)

        lb, ub, p0_flat = self._fit_preprocessing(p0=p0, frozen=frozen)

        self.params, self.covariance, *_ = curve_fit(f=self._n_fitter,
                                                     xdata=self.x_values, ydata=self.y_values,
                                                     p0=p0_flat, maxfev=self.max_iterations,
                                                     bounds=Bounds(lb=lb, ub=ub))

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

    def get_fit_values(self) -> np.ndarray:
        """
        Get the fitted values of the model.

        Returns
        -------
        np.ndarray
            An array of fitted values.
        """
        if self.params is None:
            raise RuntimeError('Fit not performed yet. Call fit() first.')
        return self._n_fitter(self.x_values, self.params)

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

    def plot_fit(self, show_individual: bool = False,
                 x_label: Optional[str] = None, y_label: Optional[str] = None, title: Optional[str] = None,
                 data_label: Optional[str] = None, axis: Optional[Axes] = None):
        """
        Plot the fitted models.

        Parameters
        ----------
        show_individual: bool, optional
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
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        plotter = plot_xy(x_data=self.x_values, y_data=self.y_values,
                          data_label=data_label if data_label else 'Data', axis=axis)

        plot_xy(x_data=self.x_values, y_data=self._n_fitter(self.x_values, *self.params),
                x_label=x_label, y_label=y_label, plot_title=title, data_label='Total Fit',
                plot_dictionary=LinePlot(color='k'), axis=plotter)

        if show_individual:
            self._plot_individual_fitter(plotter=plotter)

        plotter.set_xlabel(x_label if x_label else 'X')
        plotter.set_ylabel(y_label if y_label else 'Y')
        plotter.set_title(title if title else f'{self.n_fits} {self.__class__.__name__} fit')
        plt.tight_layout()

        return plotter
