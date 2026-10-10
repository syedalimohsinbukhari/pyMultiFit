"""Created on May 21 00:00:00 2026

Bootstrap confidence intervals, computed from a :class:`~pymultifit.result.FitResult`."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.random import Generator

from .result import FitResult
from .typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------------


def _sanitize_generator(rng_engine: Generator | None, seed: int | None) -> Generator:
    """Return a NumPy random Generator from either a seed or an existing engine.

    Parameters
    ----------
    rng_engine :
        Pre-constructed NumPy Generator.
        Mutually exclusive with seed.
    seed :
        Integer seed used to construct a new Generator.
        Mutually exclusive with rng_engine.

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
    """Normalize CI levels to ``(ci_integer, (lower_p, 0.5, upper_p))`` tuples.

    Notes
    -----
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

    if isinstance(ci_lvls, int | float | np.integer | np.floating):
        ci_lvls = [float(ci_lvls)]
    elif isinstance(ci_lvls, tuple):
        ci_lvls = list(ci_lvls)

    for ci in ci_lvls:
        ci_original = int(round(ci)) if ci > 1 else int(round(ci * 100))
        ci = ci / 100 if ci > 1 else ci

        if not (0 < ci < 1):
            raise ValueError(f"Invalid confidence interval: {ci}. Must be between 0 and 1 (or 0 and 100).")

        alpha = 1.0 - ci
        lower = alpha / 2.0
        upper = 1.0 - lower

        bounds.append((ci_original, (lower, 0.5, upper)))

    return bounds


def ci_level_labels(levels: float | int | Iterable[float]) -> list[int]:
    """Integer percentage labels of CI / PI levels, e.g. ``[0.68, 95.0]`` -> ``[68, 95]``.

    These are the integers used in the keys of the dictionary returned by :func:`compute_ci_bounds`
    (``overall_ci_<label>``, ``individual_ci_<label>``).

    Parameters
    ----------
    levels :
        A single level or an iterable of levels, in percentage (95, 95.0) or decimal (0.95) form.

    Raises
    ------
    ValueError
        If any level is not strictly between 0 and 100 percent.
    """
    return [label for label, _ in _ci_to_percentiles(levels)]


def ci_level_percents(levels: float | int | Iterable[float]) -> list[float]:
    """Exact percentages of CI / PI levels, e.g. ``[0.68, 99.7]`` -> ``[68.0, 99.7]``.

    :func:`ci_level_labels` rounds to whole percents (``99.7`` becomes ``100``), which is right for the keys of the
    dictionary of :func:`compute_ci_bounds` but not for a quantity computed from the level, like a prediction interval,
    or for a legend text.

    Parameters
    ----------
    levels :
        A single level or an iterable of levels, in percentage (95, 95.0) or decimal (0.95) form.

    Raises
    ------
    ValueError
        If any level is not strictly between 0 and 100 percent.
    """
    return [round((upper - lower) * 100, 9) for _, (lower, _, upper) in _ci_to_percentiles(levels)]


# ---------------------------------------------------------------------------
# Shared quantile → results converter
# ---------------------------------------------------------------------------


def _curves_to_ci_results(
    x_: NDArray, n_fits: int, curves_: NDArray, bounds: list[tuple[int, tuple[float, float, float]]]
) -> dict:
    """Convert a ``(n_bootstrap, n_fits, n_x)`` curves array into a CI results dict.

    Parameters
    ----------
    x_ :
        X-values (used only for dimension validation).
    n_fits :
        Number of individual component fits.
    curves_ :
        Bootstrap curves of shape ``(n_bootstrap, n_fits, len(x_))``.
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
# Per-component CIs
# ---------------------------------------------------------------------------


def compute_individual_ci(
    result: FitResult,
    mv_parameters: NDArray,
    x_: NDArray,
    bounds: list[tuple[int, tuple[float, float, float]]],
) -> dict:
    """Compute per-component CIs from bootstrapped flat parameter vectors.

    Works for any mix of components, since each one is evaluated on its own slice of the parameter vector.

    Parameters
    ----------
    result :
        A fitted :class:`~pymultifit.result.FitResult`.
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
    pairs = list(zip(result.components, result.slices()))
    curves_ = np.zeros(shape=(mv_parameters.shape[0], result.n_fits, x_.shape[0]))

    for boot_idx, boot_params in enumerate(mv_parameters):
        for comp_idx, (comp, sl) in enumerate(pairs):
            curves_[boot_idx, comp_idx, :] = comp.func(x_, list(boot_params[sl]))

    return _curves_to_ci_results(x_=x_, n_fits=result.n_fits, curves_=curves_, bounds=bounds)


# ---------------------------------------------------------------------------
# Top-level CI computation
# ---------------------------------------------------------------------------


def compute_ci_bounds(
    result: FitResult,
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
    result :
        A fitted :class:`~pymultifit.result.FitResult`, e.g. from ``fitter.to_result()``.
    ci_levels :
        CI level(s) as a percentage (e.g., 95 or [68, 95, 99]).
        Decimal form (0.95) is also accepted.
    n_bootstrap :
        Number of multivariate-normal bootstrap samples. Defaults to 5 000.
    overall_ci :
        Compute CI for the overall composite fit. Defaults to ``True``.
    individual_ci :
        Compute CI for each component. Defaults to ``False``.
    seed :
        Random seed.  Mutually exclusive with *rng_engine*.
    rng_engine :
        NumPy Generator instance.  Mutually exclusive with *seed*.
    x_range :
        X-values at which to evaluate the CI.
        Defaults to 1 000 evenly spaced points spanning the data range.

    Returns
    -------
    dict
        ``{"x_range": ..., "overall_ci_<level>": {...}, "individual_ci_<level>": [...]}``

    Raises
    ------
    ValueError
        If neither ``overall_ci`` nor ``individual_ci`` is ``True``.
    RuntimeError
        If the fit has not been performed yet.
    """
    if not overall_ci and not individual_ci:
        raise ValueError("At least one of 'overall_ci' or 'individual_ci' must be True.")

    result.require_fit()

    x_ = np.asarray(x_range) if x_range is not None else np.linspace(*np.asarray(result.x)[[0, -1]], num=1_000)

    _rng = _sanitize_generator(rng_engine=rng_engine, seed=seed)
    mv_parameters = _rng.multivariate_normal(mean=result.params, cov=result.covariance, size=n_bootstrap)

    bounds = _ci_to_percentiles(ci_levels)
    results: dict = {"x_range": x_}

    if overall_ci:
        curves_ = np.array([result.model(x_, params) for params in mv_parameters])
        for ci_val, (lower_p, median_p, upper_p) in bounds:
            quantiles = np.quantile(curves_, q=[lower_p, median_p, upper_p], axis=0)
            if quantiles.shape[-1] != len(x_):
                raise ValueError(
                    f"Dimension mismatch: x_range has length {len(x_)} but quantiles have shape {quantiles.shape}"
                )
            results[f"overall_ci_{ci_val}"] = {"lower": quantiles[0], "median": quantiles[1], "upper": quantiles[2]}

    if individual_ci:
        individual_ci_results = compute_individual_ci(result, mv_parameters=mv_parameters, x_=x_, bounds=bounds)
        for ci_val in individual_ci_results:
            results[f"individual_ci_{ci_val}"] = individual_ci_results[ci_val]

    return results
