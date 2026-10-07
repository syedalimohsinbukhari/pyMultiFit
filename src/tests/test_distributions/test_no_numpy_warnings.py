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
