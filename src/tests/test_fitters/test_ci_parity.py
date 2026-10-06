"""Created on Oct 06 2026

Characterization tests for the confidence-interval / fitted-curve output.

The expected arrays in ``data/ci_oracle.npz`` were generated from the implementation that existed *before* the plotter
was decoupled from the fitters (``FitResult`` refactor). They go through the public fitter API only, so they must keep
passing, unchanged, across that refactor.

Regenerate (only when a numerical change is intended)::

    python -c "from src.tests.test_fitters.test_ci_parity import write_oracle; write_oracle()"
"""

from pathlib import Path

import numpy as np
import pytest

from ...pymultifit import GAUSSIAN, LINE
from ...pymultifit.fitters import GaussianFitter
from ...pymultifit.fitters.mixed_f import MixedDataFitter
from ...pymultifit.generators import multi_gaussian, multiple_models

ORACLE = Path(__file__).parent / "data" / "ci_oracle.npz"

# ``curve_fit`` is only reproducible to optimizer precision across platforms / BLAS / SciPy builds (~1e-8 relative on
# values that are close to zero, e.g. residuals), so the oracle is compared at 1e-6. A real regression, such as a wrong
# parameter slice or a changed RNG order, shifts the arrays by orders of magnitude more than that.
RTOL = 1e-6
ATOL = 1e-6

_G1 = (10.0, -5.0, 1.5)
_G2 = (6.0, 3.0, 2.0)
_LINE = (0.5, 2.0)
_CI_LEVELS = [68, 95]
_CI_KWARGS = {"ci_levels": _CI_LEVELS, "n_bootstrap": 200, "overall_ci": True, "individual_ci": True, "seed": 123}


def _noise(n: int) -> np.ndarray:
    return np.random.default_rng(7).normal(0, 0.1, n)


def _base_fitter() -> GaussianFitter:
    x = np.linspace(-15, 15, 400)
    y = multi_gaussian(x, params=[_G1, _G2], noise_level=0.0) + _noise(x.size)
    fitter = GaussianFitter(x, y)
    fitter.fit(p0=[_G1, _G2])
    return fitter


def _mixed_fitter() -> MixedDataFitter:
    x = np.linspace(-50, 50, 600)
    y = multiple_models(x, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.0)
    fitter = MixedDataFitter(x, y + _noise(x.size), model_list=[LINE, GAUSSIAN, GAUSSIAN])
    fitter.fit(p0=[_LINE, _G1, _G2])
    return fitter


_BUILDERS = {"base": _base_fitter, "mixed": _mixed_fitter}


def _collect(name: str) -> dict[str, np.ndarray]:
    """Flatten everything the public API returns for a fitter into ``{key: array}``."""
    fitter = _BUILDERS[name]()
    out = {"fitted_curve": fitter.get_fitted_curve(), "residuals": fitter.get_residuals(), "params": fitter.params}

    results = fitter.confidence_intervals(**_CI_KWARGS)
    out["x_range"] = results["x_range"]
    for level in _CI_LEVELS:
        for key in ("lower", "median", "upper"):
            out[f"overall_{level}_{key}"] = results[f"overall_ci_{level}"][key]
        for idx, comp in enumerate(results[f"individual_ci_{level}"]):
            for key in ("lower", "median", "upper"):
                out[f"individual_{level}_{idx}_{key}"] = comp[key]

    return {f"{name}/{k}": np.asarray(v) for k, v in out.items()}


def write_oracle() -> None:
    ORACLE.parent.mkdir(exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name in _BUILDERS:
        payload.update(_collect(name))
    np.savez(ORACLE, **payload)


@pytest.mark.parametrize("name", list(_BUILDERS))
def test_public_output_matches_oracle(name):
    current = _collect(name)
    expected = np.load(ORACLE)
    keys = [k for k in expected.files if k.startswith(f"{name}/")]

    assert sorted(keys) == sorted(current), "set of returned arrays changed"
    for key in keys:
        np.testing.assert_allclose(current[key], expected[key], rtol=RTOL, atol=ATOL, err_msg=key)
