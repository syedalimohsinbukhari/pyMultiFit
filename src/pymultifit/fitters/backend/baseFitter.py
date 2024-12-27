"""Created on Jul 18 00:16:01 2024"""

from itertools import chain
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpyez.backend.uPlotting import LinePlot
from mpyez.ezPlotting import plot_xy
from scipy.optimize import Bounds, curve_fit

from ..utilities_f import parameter_logic
from ...generators import listOfTuplesOrArray

# safe keeping class names for spelling mistakes
_gaussian = 'GaussianFitter'
_lNormal = 'LogNormalFitter'
_skNormal = 'SkewNormalFitter'
_laplace = 'LaplaceFitter'
_gammaSR = 'GammaSRFitter'
_gammaSS = 'GammaSSFitter'


class BaseFitter:
    """The base class for multi-fitting functionality."""

    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        self.x_values = x_values
        self.y_values = y_values
        self.max_iterations = max_iterations

        self.n_par = None
        self.pn_par = self.n_par
        self.sn_par = {}

        self.n_fits = None
        self.params = None
        self.covariance = None

    def _adjust_parameters(self, p0):
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
        if self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self.covariance

    @staticmethod
    def _format_param(value, t_low=0.001, t_high=10_000):
        """Formats the parameter value based on its magnitude."""
        return f'{value:.3E}' if t_high < abs(value) or abs(value) < t_low else f'{value:.3f}'

    def _n_fitter(self, x, *params):
        y = np.zeros_like(a=x, dtype=float)
        params = np.reshape(a=params, newshape=(self.n_fits, self.n_par))
        for par in params:
            y += self.fitter(x=x, params=par)
        return y

    def _params(self):
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return self.params

    def _plot_individual_fitter(self, plotter):
        x = self.x_values
        params = np.reshape(a=self.params, newshape=(self.n_fits, self.n_par))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
        for i, par in enumerate(params):
            color = colors[i % len(colors)]
            plot_xy(x_data=x, y_data=self.fitter(x=x, params=par),
                    data_label=f'{self.__class__.__name__.replace("Fitter", "")} {i + 1}('
                               f'{", ".join(self._format_param(i) for i in par)})',
                    plot_dictionary=LinePlot(line_style='--', color=color), axis=plotter, x_label='', y_label='', plot_title='')

    def _standard_errors(self):
        if self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")
        return np.sqrt(np.diag(self.covariance))

    def fit(self, p0):
        """
        Fit the data.

        Parameters
        ----------
        p0: List[int | float | ...]
            A list of initial guesses for the parameters of the Gaussian model.

        Raises
        ------
        ValueError
            If the length of initial parameters does not match the expected count.
        """
        self.n_fits = len(p0)
        len_guess = len(list(chain(*p0)))
        total_pars = self.n_par * self.n_fits

        if len_guess != total_pars:
            p0 = self._adjust_parameters(p0)

        # get the boundaries
        lb, ub = self.fit_boundaries()
        # repeat the boundary logic for n_fits
        lb = np.resize(a=lb, new_shape=total_pars)
        ub = np.resize(a=ub, new_shape=total_pars)

        boundary_ = Bounds(lb=lb, ub=ub)

        # flatten idea taken from https://stackoverflow.com/a/73000598/3212945
        self.params, self.covariance, *_ = curve_fit(f=self._n_fitter, xdata=self.x_values, ydata=self.y_values,
                                                     p0=np.array(p0).flatten(), maxfev=self.max_iterations, bounds=boundary_)

    @staticmethod
    def fit_boundaries():
        """Defines the distribution boundaries to be used by fitter."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def fitter(x: np.array, params: listOfTuplesOrArray):
        """
        Fitter function for multi-fitting.

        Parameters
        ----------
        x: np.array
            The x-array on which the fitting is to be performed.
        params: listOfTuplesOrArray
            A list of parameters to fit.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_fit_values(self):
        """
        Get the fitted values.

        Returns
        -------
        np.ndarray
            An array of fitted values.
        """
        if self.params is None:
            raise RuntimeError('Fit not performed yet. Call fit() first.')
        return self._n_fitter(self.x_values, self.params)

    def get_parameters(self, select: Tuple[int, Any] = None, errors: bool = False):
        """ Extracts specific parameter values from the fitting process.

        Parameters
        ----------
        select : list of int or None, optional
            A list of indices specifying which sub-model to return values for.
            If None, returns values for all sub-models. Defaults to None.
        errors : bool, optional
            Whether to return the standard deviations of the selected parameters. Defaults to False.

        Returns
        -------
        tuple:
            Arrays corresponding to model parameters.

        Notes
        -----
        - The `select` parameter is used to filter the returned values to specific sub-model based on their indices. Indexing starts at 1.
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

    def get_value_error_pair(self, mean_values=False, std_values=False):
        """
        Get the value/error pair of the fitted values.

        Parameters
        ----------
        mean_values: bool, optional
            Whether to only give the values of the fitted parameters. Defaults to False.
        std_values: bool, optional
            Whether to only give the errors of the fitted parameters. Defaults to False.

        Returns
        -------
        np.ndarray
            An array consisting of only values, only errors or both.
        """
        pairs = np.column_stack([self._params(), self._standard_errors()])

        if mean_values and std_values:
            return pairs
        elif mean_values:
            return pairs[:, 0]
        elif std_values:
            return pairs[:, 1]

    def plot_fit(self, show_individual: bool = False,
                 x_label: Optional[str] = None, y_label: Optional[str] = None, title: Optional[str] = None, data_label: Optional[str] = None,
                 axis: Optional[Axes] = None) -> plt:
        """
        Plot the fitted Skewed Normal functions on the data.

        Parameters
        ----------
        show_individual: bool
            Whether to show individually fitted Skewed Normal functions.
        x_label: str
            The label for the x-axis.
        y_label: str
            The label for the y-axis.
        title: str
            The title for the plot.
        data_label: str
            The label for the data.
        axis: Axes, optional
            Axes to plot instead of the entire figure. Defaults to None.

        Returns
        -------
        plt
            The plotter handle for the drawn plot.
        """
        if self.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

        plotter = plot_xy(x_data=self.x_values, y_data=self.y_values, data_label=data_label if data_label else 'Data', axis=axis)

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
