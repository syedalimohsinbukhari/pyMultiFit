"""Created on May 21 00:00:00 2026

Post-fit confidence-interval computation backend.

All functions here are pure numerical routines — no matplotlib dependency.
They are called by :class:`~pymultifit.fitters.backend.BaseFitter` and
:class:`~pymultifit.fitters.MixedDataFitter`, and re-exported for use by
the plotting layer when it needs to compute CI on-the-fly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator

from ...typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from pymultifit.fitters import MixedDataFitter
    from pymultifit.fitters.backend import BaseFitter


# ---------------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------------


def _sanitize_generator(rng_engine: Generator | None, seed: int | None) -> Generator:
    """Return a NumPy random Generator from either a seed or an existing engine.

    Parameters
    ----------
    rng_engine :
        Pre-constructed NumPy Generator.  Mutually exclusive with *seed*.
    seed :
        Integer seed used to construct a new Generator.  Mutually exclusive
        with *rng_engine*.

    Raises
    ------
    ValueError
        If both or neither of *rng_engine* / *seed* are provided.
    """
    if seed is None and rng_engine is None:
        raise ValueError("Either 'seed' or 'rng_engine' must be provided.")
    if seed is not None and rng_engine is not None:
        raise ValueError("Only one of 'seed' or 'rng_engine' should be provided.")
    return rng_engine if rng_engine is not None else np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# CI level normalisation
# ---------------------------------------------------------------------------


def _ci_to_percentiles(ci_lvls: float | int | Iterable[float]) -> list[tuple[int, tuple[float, float, float]]]:
    """Normalise CI levels to ``(ci_integer, (lower_p, 0.5, upper_p))`` tuples.

    Accepts both percentage form (95, 68.5) and decimal form (0.95, 0.685).
    Mixed lists such as ``[68, 0.95, 99]`` are handled correctly.

    Parameters
    ----------
    ci_lvls :
        A single CI level or an iterable of levels.

    Returns
    -------
    list[tuple[int, tuple[float, float, float]]]
        Each element is ``(ci_as_int, (lower_quantile, 0.5, upper_quantile))``.

    Raises
    ------
    ValueError
        If any normalised value is not strictly between 0 and 1.
    """
    bounds: list[tuple[int, tuple[float, float, float]]] = []

    if isinstance(ci_lvls, float | int):
        ci_lvls = [float(ci_lvls)]
    elif isinstance(ci_lvls, tuple):
        ci_lvls = list(ci_lvls)

    for ci in ci_lvls:
        ci_original = int(ci) if ci > 1 else int(ci * 100)
        ci = ci / 100 if ci > 1 else ci

        if not (0 < ci < 1):
            raise ValueError(f"Invalid confidence interval: {ci}. Must be between 0 and 1 (or 0 and 100).")

        alpha = 1.0 - ci
        lower = alpha / 2.0
        upper = 1.0 - lower

        bounds.append((ci_original, (lower, 0.5, upper)))

    return bounds


# ---------------------------------------------------------------------------
# Shared quantile → results converter
# ---------------------------------------------------------------------------


def _curves_to_ci_results(
    curves_: NDArray, n_fits: int, x_: NDArray, bounds: list[tuple[int, tuple[float, float, float]]]
) -> dict:
    """Convert a ``(n_bootstrap, n_fits, n_x)`` curves array into a CI results dict.

    Parameters
    ----------
    curves_ :
        Bootstrap curves of shape ``(n_bootstrap, n_fits, len(x_))``.
    n_fits :
        Number of individual component fits.
    x_ :
        X-values (used only for dimension validation).
    bounds :
        Output of :func:`_ci_to_percentiles`.

    Returns
    -------
    dict
        ``{ci_value: [{"lower": ..., "median": ..., "upper": ...}, ...]}``

    Raises
    ------
    ValueError
        On shape mismatch between *quantiles* and *x_*.
    """
    results: dict = {}
    for ci_val, (lower_p, median_p, upper_p) in bounds:
        individual_results = []
        for fit_idx in range(n_fits):
            quantiles = np.quantile(curves_[:, fit_idx, :], [lower_p, median_p, upper_p], axis=0)
            if quantiles.shape[-1] != len(x_):
                raise ValueError(
                    f"Dimension mismatch for fit {fit_idx}: x_range has length {len(x_)} "
                    f"but quantiles have shape {quantiles.shape}"
                )
            individual_results.append({"lower": quantiles[0], "median": quantiles[1], "upper": quantiles[2]})
        results[ci_val] = individual_results
    return results


# ---------------------------------------------------------------------------
# Per-component CI strategies (one per fitter type)
# ---------------------------------------------------------------------------


def compute_individual_ci_base(
    fitter_object: "BaseFitter",
    mv_parameters: ArrayLike,
    x_: ArrayLike,
    bounds: list[tuple[int, tuple[float, float, float]]],
) -> dict:
    """Compute per-component CIs for a uniform ``BaseFitter`` (equal params per fit).

    Parameters
    ----------
    fitter_object :
        A fitted ``BaseFitter`` subclass.
    mv_parameters :
        Bootstrap samples, shape ``(n_bootstrap, n_total_params)``.
    x_ :
        Evaluation x-values.
    bounds :
        Output of :func:`_ci_to_percentiles`.

    Returns
    -------
    dict
        ``{ci_value: [{"lower": ..., "median": ..., "upper": ...}, ...]}``
    """
    mv_parameters, x_ = np.asarray(mv_parameters), np.asarray(x_)

    n_total_params = mv_parameters.shape[1]
    params_per_fit = n_total_params // fitter_object.n_fits
    params = mv_parameters.reshape((-1, fitter_object.n_fits, params_per_fit))
    curves_ = np.zeros(shape=(params.shape[0], fitter_object.n_fits, x_.shape[0]))

    for j_idx, j in enumerate(params):
        for i_idx, i in enumerate(j):
            curves_[j_idx, i_idx, :] = fitter_object._evaluate_individual_component(x_, i_idx, i)

    return _curves_to_ci_results(curves_=curves_, n_fits=fitter_object.n_fits, x_=x_, bounds=bounds)


def compute_individual_ci_mixed(
    fitter_object: "MixedDataFitter",
    mv_parameters: NDArray,
    x_: NDArray,
    bounds: list[tuple[int, tuple[float, float, float]]],
) -> dict:
    """Compute per-component CIs for a ``MixedDataFitter`` (variable params per model).

    Parameters
    ----------
    fitter_object :
        A fitted ``MixedDataFitter`` instance.
    mv_parameters :
        Bootstrap samples, shape ``(n_bootstrap, n_total_params)``.
    x_ :
        Evaluation x-values.
    bounds :
        Output of :func:`_ci_to_percentiles`.

    Returns
    -------
    dict
        ``{ci_value: [{"lower": ..., "median": ..., "upper": ...}, ...]}``
    """
    n_bootstrap = mv_parameters.shape[0]
    curves_ = np.zeros(shape=(n_bootstrap, fitter_object.n_fits, x_.shape[0]))

    for boot_idx, boot_params in enumerate(mv_parameters):
        param_index = 0
        for model_idx, model in enumerate(fitter_object.model_list):
            n_par = fitter_object._instantiate_n_par(model=model)
            model_params = boot_params[param_index : param_index + n_par]
            curves_[boot_idx, model_idx, :] = fitter_object._evaluate_individual_component(x_, model_idx, model_params)
            param_index += n_par

    return _curves_to_ci_results(curves_=curves_, n_fits=fitter_object.n_fits, x_=x_, bounds=bounds)


# ---------------------------------------------------------------------------
# Top-level CI computation
# ---------------------------------------------------------------------------


def compute_ci_bounds(
    fitter_object: "BaseFitter | MixedDataFitter",
    ci_levels: float | int | tuple | list,
    n_bootstrap: int = 5_000,
    overall_ci: bool = True,
    individual_ci: bool = False,
    seed: int | None = None,
    rng_engine: Generator | None = None,
    x_range: ArrayLike | None = None,
) -> dict:
    """Compute bootstrap confidence intervals for a fitted model.

    Parameters
    ----------
    fitter_object :
        A fitted fitter exposing ``params``, ``covariance``, ``x_values``,
        ``_n_fitter``, and ``_compute_individual_ci``.
    ci_levels :
        CI level(s) as a percentage (e.g., 95 or [68, 95, 99]).
        Decimal form (0.95) is also accepted.
    n_bootstrap :
        Number of multivariate-normal bootstrap samples. Defaults to 5 000.
    overall_ci :
        Compute CI for the overall composite fit. Defaults to ``True``.
    individual_ci :
        Compute CI for each individual component. Defaults to ``False``.
    seed :
        Random seed.  Mutually exclusive with *rng_engine*.
    rng_engine :
        NumPy Generator instance.  Mutually exclusive with *seed*.
    x_range :
        X-values at which to evaluate the CI.
        Defaults to 1 000 evenly-spaced points spanning the data range.

    Returns
    -------
    dict
        ``{"x_range": ..., "overall_ci_<level>": {...}, "individual_ci_<level>": [...]}``

    Raises
    ------
    ValueError
        If neither ``overall_ci`` nor ``individual_ci`` is ``True``.
    """
    if not overall_ci and not individual_ci:
        raise ValueError("At least one of 'overall_ci' or 'individual_ci' must be True.")

    x_ = np.asarray(x_range) if x_range is not None else np.linspace(*np.asarray(fitter_object.x_values)[[0, -1]], 1000)

    _rng = _sanitize_generator(rng_engine=rng_engine, seed=seed)
    mv_parameters = _rng.multivariate_normal(mean=fitter_object.params, cov=fitter_object.covariance, size=n_bootstrap)

    bounds = _ci_to_percentiles(ci_levels)
    results: dict = {"x_range": x_}

    bounds: Iterable

    if overall_ci:
        curves_ = np.array([fitter_object._n_fitter(x_, *j) for j in mv_parameters])
        for ci_val, (lower_p, median_p, upper_p) in bounds:
            quantiles = np.quantile(curves_, [lower_p, median_p, upper_p], axis=0)
            if quantiles.shape[-1] != len(x_):
                raise ValueError(
                    f"Dimension mismatch: x_range has length {len(x_)} but quantiles have shape {quantiles.shape}"
                )
            results[f"overall_ci_{ci_val}"] = {"lower": quantiles[0], "median": quantiles[1], "upper": quantiles[2]}

    if individual_ci:
        individual_ci_results = fitter_object._compute_individual_ci(x_=x_, mv_parameters=mv_parameters, bounds=bounds)
        for ci_val in individual_ci_results:
            results[f"individual_ci_{ci_val}"] = individual_ci_results[ci_val]

    return results
