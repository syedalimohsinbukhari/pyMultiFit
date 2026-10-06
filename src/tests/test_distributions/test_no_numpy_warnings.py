"""Created on Oct 06 2026

The distribution utility functions must be warning-free by construction: they evaluate only what is defined (or use
functions that return the correct ``inf`` / ``nan`` silently), rather than hide NumPy's warnings behind a decorator.
Outside the support, at its edges and far in the tails, none of them may emit a ``RuntimeWarning``.
"""

import inspect
import itertools
import re
import warnings

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


def _grid(parameter: str) -> list[float]:
    if parameter in ("loc", "low"):
        return LOCATIONS
    if parameter in ("a", "b", "c", "d", "slope", "intercept"):
        return COEFFICIENTS
    return POSITIVE


def test_the_functions_were_discovered():
    assert len(NAMES) > 75, NAMES


def test_the_suppression_decorator_is_not_used():
    assert "suppress_numpy_warnings" not in inspect.getsource(U)


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
