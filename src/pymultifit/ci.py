"""Created on May 21 00:00:00 2026

Bootstrap confidence intervals, computed from a :class:`~pymultifit.result.FitResult`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.random import Generator

from .result import FitResult
from .typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Level normalisation
# ---------------------------------------------------------------------------


def _parse_levels(levels: float | int | Iterable[float]) -> list[tuple[float, float]]:
    """Normalise interval levels to ``(percent, fraction)`` pairs, ascending and without duplicates.

    Accepts the percentage form (95, 68.5) and the decimal form (0.95, 0.685), mixed lists such as ``[68, 0.95, 99]``
    included. ``percent`` is rounded to nine decimals, so ``0.29`` gives ``29.0``.

    Raises
    ------
    ValueError
        If a level is not strictly between 0 and 100 percent.
    """
    if isinstance(levels, int | float | np.integer | np.floating):
        levels = [levels]

    parsed: dict[float, float] = {}
    for level in levels:
        fraction = level / 100 if level > 1 else level

        if not (0 < fraction < 1):
            raise ValueError(f"Invalid confidence interval: {fraction}. Must be between 0 and 1 (or 0 and 100).")

        parsed[round(float(fraction) * 100, 9)] = float(fraction)

    return sorted(parsed.items())


def normalize_levels(levels: float | int | Iterable[float]) -> tuple[float, ...]:
    """Exact percentages of interval levels, ascending and without duplicates.

    Used by the confidence and the prediction intervals alike, e.g. ``[0.68, 95, 99.7]`` gives ``(68.0, 95.0, 99.7)``.

    Parameters
    ----------
    levels :
        A single level or an iterable of levels, in percentage (95, 95.0) or decimal (0.95) form.

    Raises
    ------
    ValueError
        If any level is not strictly between 0 and 100 percent.
    """
    return tuple(percent for percent, _ in _parse_levels(levels))


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


def _read_only(values: ArrayLike) -> NDArray:
    snapshot = np.array(values, dtype=float, copy=True)
    snapshot.flags.writeable = False
    return snapshot


@dataclass(frozen=True, eq=False)
class Band:
    """The pointwise lower bound, median and upper bound of an interval.

    Parameters
    ----------
    lower, median, upper :
        Arrays of the same length, the interval evaluated at :attr:`ConfidenceIntervals.x`. They are read-only copies.
    """

    lower: NDArray
    median: NDArray
    upper: NDArray

    def __post_init__(self) -> None:
        for name in ("lower", "median", "upper"):
            object.__setattr__(self, name, _read_only(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class ConfidenceIntervals:
    """Bootstrap confidence intervals of a fit, as returned by :func:`compute_ci_bounds`.

    It describes itself (the levels, the parts that were computed and the x-values they were evaluated at), so it can
    be handed to :meth:`~pymultifit.plot.FitPlotter.FitPlotter.plot_confidence_intervals` as it is. Only the bands are
    kept, not the bootstrap curves: new levels or a new ``x_range`` mean computing again, which a seeded ``rng_engine``
    makes reproducible.

    Parameters
    ----------
    x :
        The x-values at which the intervals were evaluated.
    levels :
        The exact levels in percent, ascending, e.g. ``(68.0, 95.0, 99.7)``.
    n_bootstrap :
        Number of bootstrap samples the bands come from. The Monte-Carlo noise of a band edge shrinks with its square
        root.
    overall :
        ``{level: Band}`` of the total model, ``None`` if it was not computed.
    individual :
        ``{level: (Band, ...)}`` with one band per component, ``None`` if it was not computed.
    """

    x: NDArray
    levels: tuple[float, ...]
    n_bootstrap: int
    overall: Mapping[float, Band] | None = None
    individual: Mapping[float, tuple[Band, ...]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _read_only(self.x))
        object.__setattr__(self, "levels", tuple(float(level) for level in self.levels))
        if self.overall is not None:
            object.__setattr__(self, "overall", MappingProxyType(dict(self.overall)))
        if self.individual is not None:
            object.__setattr__(self, "individual", MappingProxyType({k: tuple(v) for k, v in self.individual.items()}))

    def __repr__(self) -> str:
        return (
            f"ConfidenceIntervals(levels={self.levels}, overall={self.overall is not None}, "
            f"individual={self.individual is not None}, n_x={self.x.size}, n_bootstrap={self.n_bootstrap})"
        )

    def band(self, level: float, component: int | None = None) -> Band:
        """The band of one level, of the total model or of a single component.

        Parameters
        ----------
        level :
            One level, in percentage or decimal form (``95`` and ``0.95`` are the same).
        component :
            Zero-based component index. ``None`` for the total model.

        Raises
        ------
        ValueError
            If the part asked for was not computed, or more than one level is given.
        KeyError
            If the level was not computed.
        """
        percents = normalize_levels(level)
        if len(percents) != 1:
            raise ValueError("Give exactly one level.")
        (percent,) = percents

        part, name = (self.overall, "overall_ci") if component is None else (self.individual, "individual_ci")
        if part is None:
            raise ValueError(
                f"The {'overall' if component is None else 'individual'} interval was not computed ({name}=False)."
            )
        if percent not in part:
            raise KeyError(f"Level {percent:g} % was not computed, the levels are {self.levels}.")

        return part[percent] if component is None else part[percent][component]

    def as_dict(self) -> dict:
        """The flat dictionary layout: ``"x_range"``, ``"overall_ci_<level>"`` and ``"individual_ci_<level>"``.

        ``<level>`` is the level in percent without a trailing zero (``95``, ``99.7``). The overall entries are
        ``{"lower", "median", "upper"}`` dictionaries, the individual ones a list of them, one per component.
        """
        out: dict = {"x_range": self.x}

        def _entry(band: Band) -> dict:
            return {"lower": band.lower, "median": band.median, "upper": band.upper}

        for level, band in (self.overall or {}).items():
            out[f"overall_ci_{level:g}"] = _entry(band)
        for level, bands in (self.individual or {}).items():
            out[f"individual_ci_{level:g}"] = [_entry(band) for band in bands]

        return out


# ---------------------------------------------------------------------------
# Top-level CI computation
# ---------------------------------------------------------------------------


def compute_ci_bounds(
    result: FitResult,
    ci_levels: float | int | tuple | list,
    n_bootstrap: int = 5_000,
    overall_ci: bool = True,
    individual_ci: bool = False,
    rng_engine: Generator | int | None = None,
    x_range: ArrayLike | None = None,
) -> ConfidenceIntervals:
    """Compute bootstrap confidence intervals for a fitted model.

    Parameter vectors are drawn from the multivariate normal distribution given by the fitted parameters and their
    covariance, every component is evaluated once for every sample, the total model is the sum of the components, and
    the intervals are pointwise quantiles of the curves.

    Parameters
    ----------
    result :
        A fitted :class:`~pymultifit.result.FitResult`, e.g. from ``fitter.to_result()``.
    ci_levels :
        CI level(s) as a percentage (e.g., 95 or [68, 95, 99]).
        Decimal form (0.95) is also accepted. Ask for all the levels at once, the extra ones are almost free.
    n_bootstrap :
        Number of multivariate-normal bootstrap samples. Defaults to 5 000.
    overall_ci :
        Compute CI for the overall composite fit. Defaults to ``True``.
    individual_ci :
        Compute CI for each component. Defaults to ``False``.
    rng_engine :
        A NumPy :class:`~numpy.random.Generator`, or an integer seed for a new one. ``None`` (the default) draws fresh
        entropy, so every call differs. To get reproducible and still independent results over several calls, create one
        generator and pass the same object to all of them, a new generator from the same integer would repeat the draws.
    x_range :
        X-values at which to evaluate the CI.
        Defaults to 1 000 evenly spaced points spanning the data range.

    Returns
    -------
    ConfidenceIntervals
        The bands, together with the levels, the x-values and the number of samples they belong to.

    Raises
    ------
    ValueError
        If neither ``overall_ci`` nor ``individual_ci`` is ``True``, if a level is invalid, or if ``n_bootstrap`` is
        below 1 or ``x_range`` is not a non-empty 1-D array.
    RuntimeError
        If the fit has not been performed yet.

    Notes
    -----
    The amplitude of an unnormalised distribution is the height of its peak *on the array it is evaluated on*, so a
    component's value at an ``x`` depends on the other ``x`` evaluated with it. A given ``x_range`` is therefore
    evaluated together with the fitted data's ``x``, and only the ``x_range`` part is kept: the scale then comes from the
    grid the amplitudes were fitted on, wherever the window is. Without it a window that leaves out a component's peak
    would rescale that component's tail to the full amplitude. The default grid spans the data range and needs no help.
    """
    if not overall_ci and not individual_ci:
        raise ValueError("At least one of 'overall_ci' or 'individual_ci' must be True.")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}.")

    result.require_fit()
    parsed = _parse_levels(ci_levels)

    data_x = np.asarray(result.x, dtype=float)
    if x_range is None:
        x_ = np.linspace(data_x.min(), data_x.max(), num=1_000)
        x_eval = x_
    else:
        x_ = np.asarray(x_range, dtype=float)
        if x_.ndim != 1 or x_.size == 0:
            raise ValueError("x_range must be a non-empty 1-D array.")
        x_eval = np.concatenate([x_, data_x])

    rng = np.random.default_rng(rng_engine)
    mv_parameters = rng.multivariate_normal(mean=result.params, cov=result.covariance, size=n_bootstrap)

    n = x_.size
    pairs = list(zip(result.components, result.slices()))
    total = np.zeros((n_bootstrap, n)) if overall_ci else None
    each = np.empty((n_bootstrap, len(pairs), n)) if individual_ci else None

    # every component is evaluated once per sample, the total is accumulated as the model does (left to right), and the
    # per-component curves are only kept when they are asked for
    for boot_idx, boot_params in enumerate(mv_parameters):
        for comp_idx, (comp, sl) in enumerate(pairs):
            curve = np.asarray(comp.func(x_eval, list(boot_params[sl])), dtype=float)[:n]
            if overall_ci:
                total[boot_idx] += curve
            if individual_ci:
                each[boot_idx, comp_idx] = curve

    quantiles = []
    for _, fraction in parsed:
        lower = (1.0 - fraction) / 2.0
        quantiles += [lower, 0.5, 1.0 - lower]

    overall = individual = None
    if overall_ci:
        q = np.quantile(total, quantiles, axis=0)
        overall = {percent: Band(*q[3 * i : 3 * i + 3]) for i, (percent, _) in enumerate(parsed)}
    if individual_ci:
        q = np.quantile(each, quantiles, axis=0)
        individual = {
            percent: tuple(Band(*q[3 * i : 3 * i + 3, comp_idx]) for comp_idx in range(len(pairs)))
            for i, (percent, _) in enumerate(parsed)
        }

    return ConfidenceIntervals(
        x=x_,
        levels=tuple(percent for percent, _ in parsed),
        n_bootstrap=n_bootstrap,
        overall=overall,
        individual=individual,
    )
