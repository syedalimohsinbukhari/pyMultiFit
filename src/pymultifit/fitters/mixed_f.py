"""Created on Aug 10 23:08:38 2024"""

import itertools
import warnings
from typing import Callable, Sequence

import numpy as np
from matplotlib.axes import Axes  # noqa: F401 – part of public API type hints
from plotez import LinePlotConfig, plot_xy  # noqa: F401 – kept for external callers
from scipy.optimize import Bounds, curve_fit

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
from .backend import BaseFitter, compute_individual_ci_mixed
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

    def __init__(
        self,
        x_values: NDArray,
        y_values: NDArray,
        model_list: list[str] | None = None,
        fitter_dictionary: dict | None = None,
        model_dictionary: dict | None = None,
        max_iterations: int = 1_000,
    ):
        # Check if the deprecated parameter was used
        if fitter_dictionary is not None:
            warnings.warn(
                message="`fitter_dictionary` is deprecated and will be removed in a future release. "
                "Use `model_dictionary` instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        resolved_dict = model_dictionary or fitter_dictionary or None

        # Infer model_list from model_dictionary keys when not explicitly provided
        if model_list is None:
            if resolved_dict is not None:
                model_list = list(resolved_dict.keys())
            else:
                raise ValueError("`model_list` must be provided when `model_dictionary` is not given.")
        elif resolved_dict is not None and list(resolved_dict.keys()) != model_list:
            warnings.warn(
                message="`model_list` and `model_dictionary` keys differ. "
                "`model_list` takes precedence; consider omitting it and relying on `model_dictionary` keys.",
                category=UserWarning,
                stacklevel=2,
            )

        # Set model-specific attributes before calling super().__init__()
        self.model_list = model_list
        self.fitter_dict = resolved_dict or fitter_dict

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

        Returns
        -------
        Callable :
            A composite model for fitting.
        """

        def _composite_model(x: np.ndarray, *params) -> np.ndarray:
            """
            Compute the composite model.

            Parameters
            ----------
            x :
                The x-values where the model is evaluated.
            params :
                Parameters for the model components.

            Returns
            -------
            y :
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

    def _n_fitter(self, x: NDArray, *params: Params_) -> NDArray:
        """
        Override the parent method to use the composite model function.

        Parameters
        ----------
        x :
            Input array of values for which the composite function is evaluated.
        params :
            A tuple with all parameters to be fitted.

        Returns
        -------
        NDArray :
            An array containing the composite fitted values for the input ``x``.
        """
        return self.model_function(x, *params)

    def _compute_individual_ci(
        self, x_: NDArray, mv_parameters: NDArray, bounds: list[tuple[int, tuple[float, float, float]]]
    ) -> dict:
        return compute_individual_ci_mixed(fitter_object=self, mv_parameters=mv_parameters, x_=x_, bounds=bounds)

    def _get_bounds(self) -> tuple[NDArray, NDArray]:
        """
        Sets the bounds for each parameter based on the model list.

        Returns
        -------
        tuple[NDArray, NDArray] :
            Lower and upper bounds for the parameters.
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
            raise ValueError(f"Model '{model}' is not recognized. " f"Ensure it is defined in the fitter dictionary.")

        return fitter_instance

    def _instantiate_n_par(self, model: str) -> int:
        return self._instantiate_class(model).n_par

    def _instantiate_bounds(self, model: str) -> tuple[Sequence[float], Sequence[float]]:
        return self._instantiate_class(model).fit_boundaries()

    def _component_param_offsets(self) -> list[int]:
        """Return the flat parameter offset for each component in model_list."""
        offsets = []
        offset = 0
        for model in self.model_list:
            offsets.append(offset)
            offset += self._instantiate_n_par(model=model)
        return offsets

    def _parameter_extractor(self, values: NDArray) -> dict:
        """
        Extracts the parameters for each model in the model list.

        Parameters
        ----------
        values :
            The values from which the model dictionary is to be extracted.

        Returns
        -------
        dict :
            A dictionary where the keys are model names and the values are lists of parameters/error values.
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

    def fit(self, p0: Params_, frozen: dict[int, list[bool]] | None = None):
        """
        Fit the data.

        Parameters
        ----------
        p0 :
            Initial guess for the fitted parameters.
            Must be a list of per-component guesses: ``[(p1, p2, ...), ...]``.
        frozen :
            A sparse dict mapping **0-based component indices** to a per-parameter boolean mask.
            Components not listed in the dict are treated as fully unfrozen.
            Each inner list must have length equal to the component's ``n_par`` or ``pn_par``
            (if ``pn_par`` length is given, secondary parameters such as ``loc`` are auto-padded with ``False``,
            and a :class:`UserWarning` is emitted to flag this).


        Examples
        --------
        - Freeze ``sigma`` in the first component and ``loc`` in the second::

            frozen = {0: [False, False, True], 1: [False, False, False, True]}

        - With 30 components and only one to freeze::

            frozen = {7: [False, True, False]}

        Raises
        ------
        TypeError :
            If ``p0`` is not a list of per-component sequences.
        ValueError :
            If the total length of ``p0`` does not match the expected parameter count, or if a frozen mask length is
            incompatible with its component's parameter count.
        """
        p0_chain = p0.tolist() if isinstance(p0, np.ndarray) else p0
        p0_chain: list

        if not all(isinstance(g, (tuple, list, np.ndarray)) for g in p0_chain):
            raise TypeError("MixedDataFitter requires p0 as a list of per-component guesses: [(p1, p2, ...), ...]")

        p0_chain = list(itertools.chain.from_iterable(p0_chain))
        if len(p0_chain) != self._expected_param_count():
            raise ValueError(
                f"The length of the initial guess ({len(p0_chain)}) does not match the expected parameter count "
                f"({self._expected_param_count()})."
            )

        lb, ub = self._get_bounds()

        if frozen is not None:
            # Build a flat bool mask over all parameters
            flat_frozen: list[bool] = [False] * self._expected_param_count()
            param_offsets = self._component_param_offsets()

            for comp_idx, mask in frozen.items():
                if comp_idx < 0 or comp_idx >= len(self.model_list):
                    raise ValueError(
                        f"frozen key {comp_idx} is out of range for model_list of length {len(self.model_list)}."
                    )
                model = self.model_list[comp_idx]
                comp_instance = self._instantiate_class(model)
                n_par = comp_instance.n_par
                pn_par = comp_instance.pn_par

                if len(mask) == pn_par:
                    warnings.warn(
                        f"frozen[{comp_idx}] has length {pn_par} (pn_par), which is shorter than n_par={n_par}"
                        f" for model '{model}'. The {n_par - pn_par} secondary parameter(s) (e.g. loc/scale) are being "
                        f"auto-padded as False (unfrozen). Pass a mask of length {n_par} to make this explicit.",
                        UserWarning,
                        stacklevel=2,
                    )
                    mask = list(mask) + [False] * (n_par - pn_par)
                elif len(mask) != n_par:
                    raise ValueError(
                        f"frozen[{comp_idx}] length ({len(mask)}) must equal n_par ({n_par}) or pn_par ({pn_par}) for "
                        f"model '{model}'."
                    )

                offset = param_offsets[comp_idx]
                for j, is_frozen in enumerate(mask):
                    if is_frozen:
                        flat_frozen[offset + j] = True

            for i, is_frozen in enumerate(flat_frozen):
                if is_frozen:
                    lb[i] = p0_chain[i] - epsilon
                    ub[i] = p0_chain[i] + epsilon

        self.params, self.covariance, *_ = curve_fit(
            f=self.model_function,
            xdata=self.x_values,
            ydata=self.y_values,
            p0=np.array(p0_chain),
            maxfev=self.max_iterations,
            bounds=Bounds(lb=lb, ub=ub),
        )

        self._plotter = None  # invalidate cached plotter after each fit

    def get_model_parameters(self, model: str | None = None, errors: bool = False):
        """
        Extracts parameters (and error) values for a specific model, or for all models if no model is specified.

        Parameters
        ----------
        model :
            Model name to extract parameters for.
            If unspecified, extracts parameters for all models.
            Defaults to ``None``.
        errors :
            If ``True``, includes the errors in the returned output.
            Defaults to ``False``.

        Returns
        -------
        dict :
            A dictionary containing:
                - "parameters": Nested dictionary of parameter values for each model.
                - "errors": Nested dictionary of errors for each model (if ``get_errors=True``).
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
                output[key] = np.array_split(np.asarray(par_dict, dtype=float).flatten(), indices_or_sections=n_pars)

        return output
