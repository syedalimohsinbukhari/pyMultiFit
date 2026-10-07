"""Created on Oct 06 2026

The distribution utility functions must be warning-free by construction: they evaluate only what is defined (or use
functions that return the correct ``inf`` / ``nan`` silently), rather than hide NumPy's warnings behind a decorator.
Outside the support, at its edges and far in the tails, none of them may emit a ``RuntimeWarning``.
"""

import inspect
import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from ...pymultifit.distributions import utilities_d as U

_POLYNOMIALS = {"line", "quadratic", "cubic"}
NAMES = sorted(
    name
    for name, obj in vars(U).items()
    if inspect.isfunction(obj)
    and obj.__module__ == U.__name__
    and (re.fullmatch(r"[a-zA-Z_]+_(log_)?(pdf|cdf)_", name) or name in _POLYNOMIALS)
    and list(inspect.signature(obj).parameters)[0] == "x"
)

# below, at and above the edges of every support, plus far tails (finite: inf / nan inputs may warn like any NumPy math)
X = np.array([-1e3, -10, -1, -0.5, 0, 1e-12, 0.25, 0.5, 1, 1.5, 2, 5, 50, 1e3, 1e6])
POSITIVE = [0.5, 1.0, 2.0, 10.0]
LOCATIONS = [-1.0, 0.0, 1.0]
COEFFICIENTS = [-2.0, 0.0, 1.0, 5.0]
Q_VALUES = [0.5, 0.9, 1.0, 1.1, 1.5, 1.9]  # the q-exponential is only defined for q < 2, both q < 1 and 1 < q < 2 matter


def _grid(parameter: str) -> list[float]:
    if parameter in ("loc", "low"):
        return LOCATIONS
    if parameter in ("a", "b", "c", "d", "slope", "intercept"):
        return COEFFICIENTS
    if parameter == "q":
        return Q_VALUES
    return POSITIVE


def test_the_functions_were_discovered():
    assert len(NAMES) > 75, NAMES


def test_the_suppression_decorator_is_gone():
    import pymultifit

    package = Path(pymultifit.__file__).parent
    users = [str(p.relative_to(package)) for p in package.rglob("*.py") if "suppress_numpy_warnings" in p.read_text()]
    assert users == []
    assert not hasattr(pymultifit, "suppress_numpy_warnings")


@pytest.mark.parametrize("name", NAMES)
def test_no_numpy_warnings(name):
    function = getattr(U, name)
    signature = inspect.signature(function)
    parameters = [p for p in signature.parameters if p not in ("x", "amplitude", "normalize")]
    combinations = list(itertools.product(*(_grid(p) for p in parameters)))
    rng = np.random.default_rng(0)
    if len(combinations) > 80:
        combinations = [combinations[i] for i in rng.choice(len(combinations), 80, replace=False)]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for combination in combinations:
            for normalize in (False, True):
                kwargs = dict(zip(parameters, combination))
                if "normalize" in signature.parameters:
                    kwargs["normalize"] = normalize
                try:
                    function(X.copy(), **kwargs)
                except RuntimeWarning as warning:  # pragma: no cover - only reached on a regression
                    pytest.fail(f"{name}({kwargs}) warned: {warning}")


@pytest.mark.parametrize("q", Q_VALUES)
@pytest.mark.parametrize("x", [-1e3, -3.0, -1.0, -1e-9, 0.0, 1e-9, 1.0, 1e6])
def test_q_exponential_log_pdf_is_silent_outside_the_support(q, x):
    """np.log1p(-u) warned for q > 1 and x < 0 (u > 1 there) until the scipy log1p was used."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        U.q_exponential_log_pdf_(np.array([x]), q=q, rate=1.0)


@pytest.mark.parametrize("std", [0.5, 1.0, 26.0, 27.0, 30.0, 100.0, 1e3, 1e155])
@pytest.mark.parametrize("mu", [1.0, 1e100, 1e155, 1e200, 1e308])
def test_log_normal_stats_overflows_silently_to_inf(std, mu):
    """exp(std**2) and the products overflow for large std or mu: the right answer is inf, not a warning or an exception."""
    from ...pymultifit.distributions import LogNormalDistribution

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = LogNormalDistribution(std=std, mu=mu, loc=0.5).stats()

    assert not any(np.isnan(v) for v in stats.values())
    if std >= 30:
        assert stats["mean"] == stats["variance"] == stats["std"] == np.inf


def test_log_normal_stats_still_matches_scipy():
    import scipy.stats as ss

    from ...pymultifit.distributions import LogNormalDistribution

    for std, mu, loc in [(0.5, 1.0, 0.0), (1.0, 2.0, 0.5), (3.0, 0.5, -1.0)]:
        stats = LogNormalDistribution(std=std, mu=mu, loc=loc).stats()
        reference = ss.lognorm(std, scale=mu, loc=loc)
        assert stats["mean"] == pytest.approx(reference.mean(), rel=1e-12)
        assert stats["variance"] == pytest.approx(reference.var(), rel=1e-12)


# ``stats()`` of every distribution class, over a wide but realistic parameter grid (negative, zero, tiny and large values).
# The shared ``btf.stats`` helper only draws 500 random triples from [-100, 100] and compares them with scipy: it neither
# turns warnings into errors (scipy warns itself in the same call) nor reaches the degenerate values below.
STATS_VALUES = [-10.0, -1.0, 0.0, 1e-6, 1e-3, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 50.0, 100.0, 1e3, 1e6]


def _distribution_classes():
    from ...pymultifit import distributions as plain
    from ...pymultifit.distributions import generalized
    from ...pymultifit.distributions.backend import BaseDistribution

    found = {}
    for module in (plain, generalized):
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and issubclass(obj, BaseDistribution) and obj is not BaseDistribution:
                if "stats" in vars(obj):
                    found[name] = obj
    return found


DISTRIBUTIONS = _distribution_classes()


def test_every_distribution_with_stats_was_discovered():
    assert len(DISTRIBUTIONS) >= 17, sorted(DISTRIBUTIONS)
    assert {"StudentsTDistribution", "LogNormalDistribution", "SkewNormalDistribution"} <= set(DISTRIBUTIONS)


@pytest.mark.parametrize("name", sorted(DISTRIBUTIONS))
def test_stats_has_no_warnings_and_no_exceptions(name):
    """Valid or not, a parameter set gives numbers (NaN / inf where undefined), never a warning or a Python exception.

    Invalid parameters the constructor already rejects are skipped.
    """
    cls = DISTRIBUTIONS[name]
    parameters = [p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "amplitude", "normalize")]
    combinations = list(itertools.product(STATS_VALUES, repeat=len(parameters)))
    rng = np.random.default_rng(0)
    if len(combinations) > 1500:
        combinations = [combinations[i] for i in rng.choice(len(combinations), 1500, replace=False)]

    checked = 0
    for combination in combinations:
        kwargs = dict(zip(parameters, combination))
        try:
            instance = cls(**kwargs)
        except Exception:  # noqa: BLE001 - the constructor rejected invalid parameters
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                instance.stats()
            except Exception as error:  # noqa: BLE001
                pytest.fail(f"{name}({kwargs}).stats() raised {type(error).__name__}: {error}")
        checked += 1

    assert checked > 50, f"only {checked} parameter sets were valid, the grid does not exercise {name}"


def test_stats_of_degenerate_but_valid_parameters():
    """Each of these used to raise ZeroDivisionError or return nan while the parameters are valid."""
    from ...pymultifit.distributions import (
        JohnsonSUDistribution,
        LogNormalDistribution,
        ScaledInverseChiSquareDistribution,
        SkewNormalDistribution,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # a skew normal with shape 0 is a normal distribution
        stats = SkewNormalDistribution(shape=0.0, location=1.0, scale=2.0).stats()
        assert (stats["mean"], stats["mode"], stats["variance"], stats["std"]) == (1.0, 1.0, 4.0, 2.0)
        # the mean of a scaled inverse chi-square is infinite for df <= 2, the variance for df <= 4
        assert ScaledInverseChiSquareDistribution(df=2.0, scale=1.0).stats()["mean"] == np.inf
        stats = ScaledInverseChiSquareDistribution(df=4.0, scale=1.0).stats()
        assert np.isfinite(stats["mean"]) and stats["variance"] == stats["std"] == np.inf
        # Johnson SU with gamma = 0 is symmetric about xi even when exp(1 / (2 delta^2)) overflows
        stats = JohnsonSUDistribution(gamma=0.0, delta=1e-3, xi=2.0, lambda_=50.0).stats()
        assert stats["mean"] == stats["median"] == 2.0
        # tiny std with a huge mu: the variance is huge, not nan
        assert LogNormalDistribution(std=1e-10, mu=1e200, loc=10.0).stats()["variance"] == np.inf
