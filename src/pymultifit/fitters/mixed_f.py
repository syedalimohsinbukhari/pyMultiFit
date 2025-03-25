"""Created on Aug 10 23:08:38 2024"""

from itertools import chain
from typing import Union, List, Callable, Tuple, Iterable

import numpy as np
from mpyez.backend.uPlotting import LinePlot
from mpyez.ezPlotting import plot_xy

from .backend import BaseFitter
# importing from files to avoid circular import
from .chiSquare_f import ChiSquareFitter
from .exponential_f import ExponentialFitter
from .foldedNormal_f import FoldedNormalFitter
from .gamma_f import GammaFitterSR, GammaFitterSS
from .gaussian_f import GaussianFitter
from .halfNormal_f import HalfNormalFitter
from .laplace_f import LaplaceFitter
from .logNormal_f import LogNormalFitter
from .polynomial_f import LineFitter
from .skewNormal_f import SkewNormalFitter
from .utilities_f import sanity_check
from .. import (GAUSSIAN, LAPLACE, LINE, LOG_NORMAL, SKEW_NORMAL, CHI_SQUARE, EXPONENTIAL, FOLDED_NORMAL,
                GAMMA_SR, GAMMA_SS, NORMAL, HALF_NORMAL, epsilon, MPL_COLORS)

# mock initialize the internal classes for auto MixedDataFitter class
fitter_dict = {CHI_SQUARE: ChiSquareFitter,
               EXPONENTIAL: ExponentialFitter,
               FOLDED_NORMAL: FoldedNormalFitter,
               GAMMA_SR: GammaFitterSR,
               GAMMA_SS: GammaFitterSS,
               GAUSSIAN: GaussianFitter,
               NORMAL: GaussianFitter,
               HALF_NORMAL: HalfNormalFitter,
               LAPLACE: LaplaceFitter,
               LOG_NORMAL: LogNormalFitter,
               SKEW_NORMAL: SkewNormalFitter,
               LINE: LineFitter}


class MixedDataFitter(BaseFitter):
    r"""
    Class to fit a mixture of different models to data.

    :param x_values: The x-values for the data.
    :param y_values: The y-values for the data.
    :param model_list: List of models to fit (e.g., `LINE`, `GAUSSIAN`, `LOG_NORMAL`)
    :param max_iterations: The maximum number of iterations for fitting procedure.
    """

    def __init__(self, x_values: Union[List, np.ndarray], y_values: Union[List, np.ndarray],
                 model_list: List[str], fitter_dictionary=None, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)

        self.x_values = x_values
        self.y_values = y_values
        self.model_list = model_list
        self.max_iterations = max_iterations
        self.params = None
        self.covariance = None

        self.p0 = None

        self.fitter_dict = fitter_dictionary or fitter_dict
        self.model_function = self._create_model_function()

        # placed in last because the expected parameter count cannot be obtained unless the models are instantiated.
        self.n_par = self._expected_param_count()

    def __repr__(self):
        return (f"{self.__class__.__name__}(x_values={self.x_values}, y_values={self.y_values}, "
                f"model_list={self.model_list}, max_iterations={self.max_iterations})")

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
            y = np.zeros_like(a=x, dtype=float)
            param_index = 0

            for model in self.model_list:
                model_class, n_par = self._instantiate_fitter(model=model, return_values=['class', 'n_par'])
                y += model_class.fitter(x=x, params=params[param_index:param_index + n_par])
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
            n_par = self._instantiate_fitter(model=model, return_values='n_par')
            count += n_par

        return count

    def _n_fitter(self, x: np.ndarray, *params: Iterable) -> np.ndarray:
        return self.model_function(x, *params)

    def fit_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sets the bounds for each parameter based on the model list.

        :returns: Lower and upper bounds for the parameters.
        """
        lower_bounds = []
        upper_bounds = []

        for model in self.model_list:
            lb, ub = self._instantiate_fitter(model=model, return_values='bounds')
            lower_bounds.extend(lb)
            upper_bounds.extend(ub)

        return np.array(lower_bounds), np.array(upper_bounds)

    def _plot_individual_fitter(self, plotter):
        """
        Plot the individual fitters function.

        :param plotter: The plotting axis object
        """
        x = self.x_values
        colors = MPL_COLORS[1:]
        param_index = 0
        for i, model in enumerate(self.model_list):
            color = colors[i % len(colors)]
            class_model, n_par = self._instantiate_fitter(model=model, return_values=['class', 'n_par'])
            pars = self.params[param_index:param_index + n_par]
            y_component = class_model.fitter(x=x, params=pars)
            plot_xy(x_data=x, y_data=y_component,
                    x_label='', y_label='', plot_title='',
                    data_label=f'{model.capitalize()} {i + 1}({", ".join(self._format_param(i) for i in pars)})',
                    plot_dictionary=LinePlot(line_style='--', color=color),
                    axis=plotter)
            param_index += n_par

    def _instantiate_fitter(self, model: str, return_values: Union[str, List[str]] = 'class'):
        """
        Instantiate the fitter for the specified model and return requested values.

        :param model: The model name as a string.
        :param return_values: The specific attribute(s) or instance to return.
                              Options are 'class', 'n_par', and 'bounds'.
                              Can be a string (for one value) or a list of strings.
        :return: The requested values as a single value or a tuple.
        :raises ValueError: If the model is not recognized or return_values are invalid.
        """
        try:
            fitter_instance = self.fitter_dict[model](x_values=[], y_values=[])
        except KeyError:
            raise ValueError(f"Model '{model}' not recognized. Ensure it is defined in the fitter dictionary.")

        valid_options = {'class', 'n_par', 'bounds'}
        if isinstance(return_values, str):
            return_values = [return_values]  # Convert to list for uniform processing

        if not all(val in valid_options for val in return_values):
            invalid_options = [val for val in return_values if val not in valid_options]
            raise ValueError(f"Invalid return_values {invalid_options}. Expected values: {valid_options}")

        result = []
        for val in return_values:
            if val == 'class':
                result.append(fitter_instance)
            elif val == 'n_par':
                result.append(fitter_instance.n_par)
            elif val == 'bounds':
                try:
                    result.append(fitter_instance.fit_boundaries())
                except NotImplementedError:
                    # in case the boundaries are not defined, put -np.inf, and np.inf to work with
                    n_par = fitter_instance.n_par
                    result.append([[-np.inf] * n_par, [np.inf] * n_par])

        return result[0] if len(result) == 1 else tuple(result)

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

        par_count = [self.fitter_dict[i]([], []).n_par
                     for i in self.model_list]

        if frozen:
            # Process frozen parameters
            for key, values in frozen.items():
                values = values if isinstance(values, list) else [values]
                for val in values:
                    param_value = self.p0[key - 1][val - 1]
                    lb[sum(par_count[:key - 1]) + (val - 1)] = param_value - epsilon
                    ub[sum(par_count[:key - 1]) + (val - 1)] = param_value + epsilon

        return lb, ub, list(chain.from_iterable(self.p0))
